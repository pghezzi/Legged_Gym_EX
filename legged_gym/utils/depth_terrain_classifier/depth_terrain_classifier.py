"""Lightweight incremental depth-terrain classification.

This module implements an end-to-end pipeline designed for small labeled datasets
and inexpensive 10 Hz deployment:

    depth image + IMU
        -> orientation-conditioned depth residual
        -> compact Sobel/Laplacian/occupancy feature vector
        -> standardized PCA projection
        -> one or more prototypes per class
        -> optional temporal probability smoothing

The classifier supports adding new terrain classes between evaluations without
retraining the depth feature extractor or refitting PCA. Engineered feature vectors,
not raw depth images, are retained for exact per-class prototype refreshes and an
optional later PCA refit.

Dependencies
------------
Only PyTorch and the Python standard library are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Hashable, List, Mapping, Sequence

import torch
import torch.nn.functional as F


# =============================================================================
# Prediction and evaluation containers
# =============================================================================


@dataclass
class TerrainPrediction:
    """Prediction returned for one deployment-time depth frame."""

    label: str
    instantaneous_label: str
    probabilities: torch.Tensor
    labels: list[str]
    distances: torch.Tensor
    raw_features: torch.Tensor


@dataclass
class EvaluationResult:
    """Dependency-free classification metrics."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: torch.Tensor
    labels: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


# =============================================================================
# Depth + IMU feature extractor
# =============================================================================


@torch.jit.script
def _masked_quantile_dynamic(x: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    # Scripting preserves empty-mask branches when the enclosing model is traced.
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for index in range(x.shape[0]):
        values = x[index][mask[index]]
        if values.numel() == 0:
            outputs.append(torch.zeros((), dtype=x.dtype, device=x.device))
        else:
            outputs.append(torch.quantile(values, q))
    return torch.stack(outputs)


@torch.jit.script
def _masked_topk_mean_dynamic(
    x: torch.Tensor, mask: torch.Tensor, fraction: float,
) -> torch.Tensor:
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for index in range(x.shape[0]):
        values = x[index][mask[index]]
        if values.numel() == 0:
            outputs.append(torch.zeros((), dtype=x.dtype, device=x.device))
        else:
            # The number of valid pixels, and therefore k, changes between frames.
            k = max(1, int(round(values.numel() * fraction)))
            outputs.append(torch.topk(values, k=k, largest=True, sorted=False).values.mean())
    return torch.stack(outputs)


class SobelDepthTerrainFeatureExtractor:
    """Extract a compact, terrain-oriented feature vector from depth and IMU data.

    The extractor avoids point-cloud construction. It instead uses standard image
    processing operations on a small depth image:

    * bilinear downsampling after invalid-value filling;
    * an optional roll/pitch-conditioned flat-ground reference image;
    * 3x3 Gaussian smoothing before Sobel derivatives;
    * Sobel x/y gradient statistics and organized row-edge statistics;
    * Laplacian and multi-scale high-pass roughness statistics;
    * far/invalid occupancy for gaps and close upper-center occupancy for overhead
      obstacles;
    * roll, pitch, and body angular velocity from the IMU.

    The selected features are especially useful for rough ground, stairs, pit exits,
    gaps, and overhead obstacles, but they remain generic geometric descriptors rather
    than terrain-specific hard-coded rules.

    Parameters
    ----------
    output_size:
        Internal image resolution. The default 32x48 is intentionally small.
    crop:
        Fractional crop (top, bottom, left, right) applied before resizing.
    min_depth, max_depth:
        Valid depth range in meters.
    far_depth:
        Depth beyond which a pixel contributes to the far/gap occupancy feature.
    close_depth:
        Absolute close-return threshold used for obstacle occupancy.
    close_residual_threshold:
        A pixel closer than its expected flat-ground depth by this amount also counts
        as a close obstacle return.
    sobel_edge_threshold:
        Threshold in depth units for strong Sobel edges.
    near_fraction:
        Fraction of image rows at the bottom treated as near terrain.
    center_fraction:
        Fraction of image columns treated as the robot's forward corridor.
    orientation_scale:
        Scale used to normalize roll/pitch features.
    angular_velocity_scale:
        Scale used to normalize angular-velocity features.
    """

    # Features are intentionally kept compact. Every feature is scalar and cheap.
    FEATURE_NAMES = (
        # Depth support / free-space / obstacle occupancy
        "valid_fraction",
        "far_or_invalid_fraction",
        "invalid_boundary_density",
        "depth_iqr",
        "near_far_depth_delta",
        "upper_center_close_fraction",
        "center_close_fraction",
        # Multi-scale roughness: strong for gravel/rubble and other uneven surfaces
        "fine_roughness",
        "near_fine_roughness",
        "coarse_roughness",
        "roughness_scale_ratio",
        # Standard Sobel edge statistics
        "sobel_magnitude_mean",
        "sobel_magnitude_p90",
        "horizontal_edge_energy",
        "vertical_edge_energy",
        "horizontal_edge_density",
        "horizontal_edge_topk",
        "row_edge_repetition",
        "strongest_row_edge",
        "horizontal_edge_dominance",
        "positive_row_transition",
        "negative_row_transition",
        # Curvature / high-frequency surface structure
        "laplacian_abs_mean",
        "laplacian_p90",
        "row_profile_curvature",
        # IMU context. Yaw is omitted because terrain appearance should be yaw-invariant.
        "imu_roll",
        "imu_pitch",
        "imu_omega_x",
        "imu_omega_y",
        "imu_omega_z",
        # Spatial 2x3 grid features. The original 30 features above retain their
        # indices; these 24 features are appended in row-major patch order.
        # The six patches cover the entire retained image without overlap.
        "grid_top_left_mean_residual_depth",
        "grid_top_left_fine_roughness",
        "grid_top_left_sobel_magnitude_mean",
        "grid_top_left_far_or_invalid_fraction",
        "grid_top_center_mean_residual_depth",
        "grid_top_center_fine_roughness",
        "grid_top_center_sobel_magnitude_mean",
        "grid_top_center_far_or_invalid_fraction",
        "grid_top_right_mean_residual_depth",
        "grid_top_right_fine_roughness",
        "grid_top_right_sobel_magnitude_mean",
        "grid_top_right_far_or_invalid_fraction",
        "grid_bottom_left_mean_residual_depth",
        "grid_bottom_left_fine_roughness",
        "grid_bottom_left_sobel_magnitude_mean",
        "grid_bottom_left_far_or_invalid_fraction",
        "grid_bottom_center_mean_residual_depth",
        "grid_bottom_center_fine_roughness",
        "grid_bottom_center_sobel_magnitude_mean",
        "grid_bottom_center_far_or_invalid_fraction",
        "grid_bottom_right_mean_residual_depth",
        "grid_bottom_right_fine_roughness",
        "grid_bottom_right_sobel_magnitude_mean",
        "grid_bottom_right_far_or_invalid_fraction",
    )

    def __init__(
        self,
        output_size: tuple[int, int] = (32, 48),
        crop: tuple[float, float, float, float] = (0.10, 1.0, 0.05, 0.95),
        min_depth: float = 0.10,
        max_depth: float = 5.0,
        far_depth: float = 3.0,
        close_depth: float = 0.75,
        close_residual_threshold: float = 0.25,
        depth_scale: float | None = None,
        sobel_edge_threshold: float = 0.035,
        topk_fraction: float = 0.10,
        near_fraction: float = 0.35,
        center_fraction: float = 0.50,
        orientation_scale: float = 0.70,
        angular_velocity_scale: float = 4.0,
        reference_ridge: float = 1e-4,
        device: str | torch.device = "cpu",
        eps: float = 1e-6,
    ):
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.crop = tuple(float(v) for v in crop)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.far_depth = float(far_depth)
        self.close_depth = float(close_depth)
        self.close_residual_threshold = float(close_residual_threshold)
        self.depth_scale = float(depth_scale or (max_depth - min_depth))
        self.sobel_edge_threshold = float(sobel_edge_threshold)
        self.topk_fraction = float(topk_fraction)
        self.near_fraction = float(near_fraction)
        self.center_fraction = float(center_fraction)
        self.orientation_scale = float(orientation_scale)
        self.angular_velocity_scale = float(angular_velocity_scale)
        self.reference_ridge = float(reference_ridge)
        self.device = torch.device(device)
        self.eps = float(eps)

        if not (0.0 <= self.crop[0] < self.crop[1] <= 1.0):
            raise ValueError("crop vertical bounds must satisfy 0 <= top < bottom <= 1")
        if not (0.0 <= self.crop[2] < self.crop[3] <= 1.0):
            raise ValueError("crop horizontal bounds must satisfy 0 <= left < right <= 1")
        if self.min_depth >= self.max_depth:
            raise ValueError("min_depth must be smaller than max_depth")
        if not (0.0 < self.near_fraction < 1.0):
            raise ValueError("near_fraction must be in (0, 1)")
        if not (0.0 < self.center_fraction <= 1.0):
            raise ValueError("center_fraction must be in (0, 1]")
        if not (0.0 < self.topk_fraction <= 1.0):
            raise ValueError("topk_fraction must be in (0, 1]")
        if self.orientation_scale <= 0.0 or self.angular_velocity_scale <= 0.0:
            raise ValueError("IMU normalization scales must be positive")

        # Fixed, normalized image-processing kernels. Registering them as ordinary
        # tensors keeps this class lightweight and independent of nn.Module state.
        self.gaussian_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1, 3, 3) / 16.0
        self.sobel_x_kernel = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1, 3, 3) / 8.0
        self.sobel_y_kernel = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1, 3, 3) / 8.0
        self.laplacian_kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1, 3, 3)

        # Coefficients for an orientation-conditioned nominal flat-ground image:
        # expected_depth = b0 + b_roll * roll + b_pitch * pitch.
        # Shape is [3, H, W]. None means calibration has not been performed.
        self.reference_coefficients: torch.Tensor | None = None

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_NAMES)

    # -------------------------------------------------------------------------
    # Flat-ground reference calibration
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def fit_reference_model(
        self,
        flat_depth_images: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
    ) -> torch.Tensor:
        """Fit a nominal flat-ground depth image conditioned on roll and pitch.

        Flat-ground calibration is optional but recommended. A small ridge-regression
        model is fit independently for every downsampled pixel using a shared design
        matrix [1, roll, pitch]. This compensates for camera/body orientation without
        constructing a point cloud.

        If orientation is omitted, the model reduces to a robust static reference image.
        """
        depth = self._as_depth_batch(flat_depth_images)
        depth, valid = self._crop_fill_resize(depth)
        n, h, w = depth.shape
        rpy = self._as_imu_batch(orientation_rpy, n, 3, "orientation_rpy")

        # Fill residual invalid calibration pixels by the per-pixel median across the
        # calibration set. Remaining all-invalid locations use the global median.
        masked = depth.masked_fill(~valid, float("nan"))
        pixel_median = torch.nanmedian(masked, dim=0).values
        global_median = torch.nanmedian(masked)
        if not torch.isfinite(global_median):
            raise ValueError("Flat-ground calibration contains no valid depth values.")
        pixel_median = torch.where(torch.isfinite(pixel_median), pixel_median, global_median)
        y = torch.where(valid, depth, pixel_median.unsqueeze(0)).reshape(n, h * w)

        # Normalizing roll/pitch improves the condition number of the tiny regression.
        a = torch.stack(
            [
                torch.ones(n, device=self.device),
                rpy[:, 0] / self.orientation_scale,
                rpy[:, 1] / self.orientation_scale,
            ],
            dim=1,
        )

        # With no meaningful orientation variation, fit only the intercept to prevent
        # unstable slope coefficients.
        if n < 3 or torch.std(a[:, 1:]) < 1e-5:
            coefficients = torch.zeros(3, h * w, device=self.device)
            coefficients[0] = torch.median(y, dim=0).values
        else:
            eye = torch.eye(3, dtype=torch.float32, device=self.device)
            eye[0, 0] = 0.0  # Do not regularize the intercept.
            lhs = a.T @ a + self.reference_ridge * eye
            rhs = a.T @ y
            coefficients = torch.linalg.solve(lhs, rhs)

        self.reference_coefficients = coefficients.reshape(3, h, w).detach()
        return self.reference_coefficients.clone()

    def clear_reference_model(self) -> None:
        self.reference_coefficients = None

    # -------------------------------------------------------------------------
    # Public feature extraction
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def extract(
        self,
        depth_image: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
    ) -> torch.Tensor:
        """Extract one feature vector with shape [feature_dim]."""
        features = self.extract_batch(depth_image, orientation_rpy, angular_velocity)
        if features.shape[0] != 1:
            raise ValueError(f"extract() expected one image, received {features.shape[0]}")
        return features[0]

    @torch.inference_mode()
    def extract_batch(
        self,
        depth_images: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
    ) -> torch.Tensor:
        """Extract [B, feature_dim] features from depth and synchronized IMU data."""
        depth = self._as_depth_batch(depth_images)
        depth, valid = self._crop_fill_resize(depth)
        b, h, w = depth.shape
        # The DepthWaQ exporter traces a fixed image shape. Older PyTorch
        # tracers expose these shape entries as scalar tensors, whereas this
        # feature extractor intentionally uses them in static Python layout
        # calculations (regions, crops, and pooling windows).
        b = int(b.item()) if torch.is_tensor(b) else int(b)
        h = int(h.item()) if torch.is_tensor(h) else int(h)
        w = int(w.item()) if torch.is_tensor(w) else int(w)
        rpy = self._as_imu_batch(orientation_rpy, b, 3, "orientation_rpy")
        omega = self._as_imu_batch(angular_velocity, b, 3, "angular_velocity")

        reference = self._expected_reference(depth, rpy)
        residual = depth - reference

        # Valid neighborhoods prevent ordinary terrain gradients from being polluted by
        # invalid measurements. Gap information is represented separately through the
        # occupancy and invalid-boundary features below.
        valid_f = valid.to(depth.dtype)
        valid_neighborhood = F.avg_pool2d(
            valid_f[:, None], kernel_size=3, stride=1, padding=1
        )[:, 0] > 0.999

        # Standard practice: smooth before differentiating. This greatly reduces sensor
        # speckle while preserving step, ledge, and obstacle boundaries.
        smooth = F.conv2d(residual[:, None], self.gaussian_kernel, padding=1)
        sobel_x = F.conv2d(smooth, self.sobel_x_kernel, padding=1)[:, 0]
        sobel_y = F.conv2d(smooth, self.sobel_y_kernel, padding=1)[:, 0]
        sobel_mag = torch.sqrt(sobel_x.square() + sobel_y.square() + self.eps)

        # Laplacian captures high-frequency curvature and is complementary to Sobel.
        laplacian = F.conv2d(smooth, self.laplacian_kernel, padding=1)[:, 0]

        # Multi-scale Gaussian/average residuals provide robust roughness cues.
        fine_local = F.avg_pool2d(residual[:, None], 3, stride=1, padding=1)[:, 0]
        coarse_local = F.avg_pool2d(residual[:, None], 7, stride=1, padding=3)[:, 0]
        fine_hp = (residual - fine_local).abs()
        coarse_hp = (residual - coarse_local).abs()

        # Regions: image bottom is near terrain, and the middle columns are the forward
        # locomotion corridor. Upper-center close returns are useful for low ceilings or
        # obstacles that require crawling.
        near_start = max(0, h - max(1, int(round(h * self.near_fraction))))
        near_slice = slice(near_start, h)
        far_slice = slice(0, max(1, h // 3))
        upper_slice = slice(0, max(1, int(round(0.40 * h))))
        center_width = max(1, int(round(w * self.center_fraction)))
        center_left = max(0, (w - center_width) // 2)
        center_slice = slice(center_left, min(w, center_left + center_width))

        # ---------------------------------------------------------------------
        # Occupancy and broad depth-layout features
        # ---------------------------------------------------------------------
        valid_fraction = valid_f.mean(dim=(1, 2))
        far_or_invalid_fraction = ((depth >= self.far_depth) | (~valid)).float().mean(dim=(1, 2))

        # Sobel on the binary validity image captures the boundary of missing support.
        valid_smooth = F.conv2d(valid_f[:, None], self.gaussian_kernel, padding=1)
        valid_gx = F.conv2d(valid_smooth, self.sobel_x_kernel, padding=1)[:, 0]
        valid_gy = F.conv2d(valid_smooth, self.sobel_y_kernel, padding=1)[:, 0]
        invalid_boundary_density = (
            torch.sqrt(valid_gx.square() + valid_gy.square() + self.eps) > 0.10
        ).float().mean(dim=(1, 2))

        depth_iqr = self._masked_quantile(depth, valid, 0.75) - self._masked_quantile(depth, valid, 0.25)
        depth_iqr = depth_iqr / self.depth_scale
        near_depth = self._masked_mean(depth[:, near_slice, :], valid[:, near_slice, :])
        far_depth = self._masked_mean(depth[:, far_slice, :], valid[:, far_slice, :])
        near_far_depth_delta = (near_depth - far_depth) / self.depth_scale

        # A close return may be absolutely near the camera or substantially closer than
        # the orientation-conditioned expected ground surface.
        close_mask = valid & (
            (depth < self.close_depth) | (residual < -self.close_residual_threshold)
        )
        upper_center_close_fraction = close_mask[:, upper_slice, center_slice].float().mean(dim=(1, 2))
        center_close_fraction = close_mask[:, :, center_slice].float().mean(dim=(1, 2))

        # ---------------------------------------------------------------------
        # Roughness features
        # ---------------------------------------------------------------------
        fine_roughness = self._masked_mean(fine_hp, valid) / self.depth_scale
        near_fine_roughness = self._masked_mean(
            fine_hp[:, near_slice, :], valid[:, near_slice, :]
        ) / self.depth_scale
        coarse_roughness = self._masked_mean(coarse_hp, valid) / self.depth_scale
        roughness_scale_ratio = fine_roughness / coarse_roughness.clamp_min(self.eps)

        # ---------------------------------------------------------------------
        # Sobel edge and organized ledge features
        # ---------------------------------------------------------------------
        abs_x = sobel_x.abs()
        abs_y = sobel_y.abs()
        sobel_magnitude_mean = self._masked_mean(sobel_mag, valid_neighborhood) / self.depth_scale
        sobel_magnitude_p90 = self._masked_quantile(sobel_mag, valid_neighborhood, 0.90) / self.depth_scale
        horizontal_edge_energy = self._masked_mean(abs_y, valid_neighborhood) / self.depth_scale
        vertical_edge_energy = self._masked_mean(abs_x, valid_neighborhood) / self.depth_scale
        horizontal_edge_density = (
            (abs_y > self.sobel_edge_threshold) & valid_neighborhood
        ).float().sum(dim=(1, 2)) / valid_neighborhood.float().sum(dim=(1, 2)).clamp_min(1.0)
        horizontal_edge_topk = self._masked_topk_mean(
            abs_y, valid_neighborhood, self.topk_fraction
        ) / self.depth_scale

        # Horizontal structures extending across image columns are characteristic of
        # stairs. A gap/pit lip tends to produce one dominant row; stairs produce several.
        row_edge_strength = self._masked_mean_along_width(abs_y, valid_neighborhood)
        row_edge_repetition = (row_edge_strength > self.sobel_edge_threshold).float().mean(dim=1)
        strongest_row_edge = row_edge_strength.max(dim=1).values / self.depth_scale
        horizontal_edge_dominance = horizontal_edge_energy / vertical_edge_energy.clamp_min(self.eps)

        # Keep edge sign because an upward obstacle/pit wall and a downward gap can have
        # similar absolute edge strength but opposite depth ordering.
        positive_row_transition = self._masked_topk_mean(
            F.relu(sobel_y), valid_neighborhood, self.topk_fraction
        ) / self.depth_scale
        negative_row_transition = self._masked_topk_mean(
            F.relu(-sobel_y), valid_neighborhood, self.topk_fraction
        ) / self.depth_scale

        # ---------------------------------------------------------------------
        # Curvature and row-profile structure
        # ---------------------------------------------------------------------
        laplacian_abs = laplacian.abs()
        laplacian_abs_mean = self._masked_mean(laplacian_abs, valid_neighborhood) / self.depth_scale
        laplacian_p90 = self._masked_quantile(laplacian_abs, valid_neighborhood, 0.90) / self.depth_scale

        # A second derivative of the row-mean residual captures repeated terracing and
        # broad concave/convex changes without requiring explicit 3-D geometry.
        row_profile = self._masked_mean_along_width(residual, valid)
        if h >= 3:
            row_second = row_profile[:, 2:] - 2.0 * row_profile[:, 1:-1] + row_profile[:, :-2]
            row_profile_curvature = row_second.abs().mean(dim=1) / self.depth_scale
        else:
            row_profile_curvature = torch.zeros(b, device=self.device)

        # ---------------------------------------------------------------------
        # IMU features
        # ---------------------------------------------------------------------
        imu_roll = rpy[:, 0] / self.orientation_scale
        imu_pitch = rpy[:, 1] / self.orientation_scale
        imu_omega = omega / self.angular_velocity_scale

        # ---------------------------------------------------------------------
        # Spatial 2x3 grid features
        # ---------------------------------------------------------------------
        # Divide the complete retained image into two row bands and three column
        # bands. Integer boundaries ensure that every pixel belongs to exactly one
        # patch, including when the image dimensions are not divisible by 2 or 3.
        # At the default internal size of 32x48, all six patches are 16x16.
        row_bounds = (0, h // 2, h)
        col_bounds = (0, w // 3, (2 * w) // 3, w)
        grid_features: list[torch.Tensor] = []

        for row_index in range(2):
            row_slice = slice(row_bounds[row_index], row_bounds[row_index + 1])
            for col_index in range(3):
                col_slice = slice(col_bounds[col_index], col_bounds[col_index + 1])

                patch_valid = valid[:, row_slice, col_slice]
                patch_valid_neighborhood = valid_neighborhood[:, row_slice, col_slice]

                # Signed mean residual retains whether this part of the scene is
                # systematically closer or farther than expected flat ground.
                patch_mean_residual = self._masked_mean(
                    residual[:, row_slice, col_slice], patch_valid
                ) / self.depth_scale

                # Fine high-pass energy captures spatially localized gravel, rubble,
                # footholds, and other small-scale surface irregularity.
                patch_fine_roughness = self._masked_mean(
                    fine_hp[:, row_slice, col_slice], patch_valid
                ) / self.depth_scale

                # Sobel magnitude is orientation-agnostic edge strength, useful for
                # localized steps, ledges, obstacle boundaries, and crawl-under gaps.
                patch_sobel_magnitude = self._masked_mean(
                    sobel_mag[:, row_slice, col_slice], patch_valid_neighborhood
                ) / self.depth_scale

                # Far/invalid occupancy retains missing support and open-space cues
                # locally rather than averaging them across the complete image.
                patch_far_or_invalid = (
                    (depth[:, row_slice, col_slice] >= self.far_depth)
                    | (~patch_valid)
                ).float().mean(dim=(1, 2))

                grid_features.extend(
                    [
                        patch_mean_residual,
                        patch_fine_roughness,
                        patch_sobel_magnitude,
                        patch_far_or_invalid,
                    ]
                )

        features = torch.stack(
            [
                valid_fraction,
                far_or_invalid_fraction,
                invalid_boundary_density,
                depth_iqr,
                near_far_depth_delta,
                upper_center_close_fraction,
                center_close_fraction,
                fine_roughness,
                near_fine_roughness,
                coarse_roughness,
                roughness_scale_ratio,
                sobel_magnitude_mean,
                sobel_magnitude_p90,
                horizontal_edge_energy,
                vertical_edge_energy,
                horizontal_edge_density,
                horizontal_edge_topk,
                row_edge_repetition,
                strongest_row_edge,
                horizontal_edge_dominance,
                positive_row_transition,
                negative_row_transition,
                laplacian_abs_mean,
                laplacian_p90,
                row_profile_curvature,
                imu_roll,
                imu_pitch,
                imu_omega[:, 0],
                imu_omega[:, 1],
                imu_omega[:, 2],
                *grid_features,
            ],
            dim=1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

    # -------------------------------------------------------------------------
    # Tensor preparation and masked reductions
    # -------------------------------------------------------------------------

    def _as_depth_batch(self, depth: torch.Tensor | Sequence) -> torch.Tensor:
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        elif depth.ndim == 3:
            pass
        elif depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        else:
            raise ValueError(
                f"Expected depth shape [H,W], [B,H,W], or [B,1,H,W], got {tuple(depth.shape)}"
            )
        return depth

    def _as_imu_batch(
        self,
        value: torch.Tensor | Sequence | None,
        batch_size: int,
        width: int,
        name: str,
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(batch_size, width, dtype=torch.float32, device=self.device)
        x = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            if x.numel() != width:
                raise ValueError(f"{name} must contain {width} values")
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != width:
            raise ValueError(f"Expected {name} shape [B,{width}], got {tuple(x.shape)}")
        if x.shape[0] == 1 and batch_size > 1:
            x = x.expand(batch_size, -1)
        if x.shape[0] != batch_size:
            raise ValueError(f"{name} batch size {x.shape[0]} does not match depth batch {batch_size}")
        return x

    def _crop_fill_resize(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h, w = depth.shape
        # During tracing on older PyTorch versions, symbolic shape entries may
        # arrive here as scalar tensors. Cropping is fixed configuration, so
        # materialize the traced input dimensions as Python integers.
        h = int(h.item()) if torch.is_tensor(h) else int(h)
        w = int(w.item()) if torch.is_tensor(w) else int(w)
        # ``crop`` is static configuration. It is normally a tuple of floats,
        # but TorchScript tracing can materialize saved scalar configuration as
        # zero-dimensional tensors. Convert it back before applying Python's
        # integer crop arithmetic.
        crop = tuple(float(value.item()) if torch.is_tensor(value) else float(value)
                     for value in self.crop)
        top = min(h - 1, max(0, int(round(crop[0] * h))))
        bottom = min(h, max(top + 1, int(round(crop[1] * h))))
        left = min(w - 1, max(0, int(round(crop[2] * w))))
        right = min(w, max(left + 1, int(round(crop[3] * w))))
        depth = depth[:, top:bottom, left:right]

        valid = torch.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth)
        clipped = torch.nan_to_num(depth, nan=0.0, posinf=self.max_depth, neginf=self.min_depth)
        clipped = clipped.clamp(self.min_depth, self.max_depth)

        # Fill invalid pixels before bilinear resizing so invalid numeric values do not
        # bleed into valid terrain. The validity image is resized separately by nearest
        # neighbor and retained for gap/measurement-support features.
        valid_f = valid.float()
        valid_sum = valid_f.sum(dim=(1, 2), keepdim=True)
        image_mean = (clipped * valid_f).sum(dim=(1, 2), keepdim=True) / valid_sum.clamp_min(1.0)
        image_mean = torch.where(valid_sum > 0, image_mean, torch.full_like(image_mean, self.max_depth))
        filled = torch.where(valid, clipped, image_mean)

        filled = F.interpolate(
            filled[:, None], size=self.output_size, mode="bilinear", align_corners=False
        )[:, 0]
        valid = F.interpolate(
            valid_f[:, None], size=self.output_size, mode="nearest"
        )[:, 0] > 0.5
        return filled, valid

    def _expected_reference(self, depth: torch.Tensor, rpy: torch.Tensor) -> torch.Tensor:
        if self.reference_coefficients is not None:
            coeff = self.reference_coefficients.to(depth.device, depth.dtype)
            return (
                coeff[0].unsqueeze(0)
                + coeff[1].unsqueeze(0) * (rpy[:, 0] / self.orientation_scale)[:, None, None]
                + coeff[2].unsqueeze(0) * (rpy[:, 1] / self.orientation_scale)[:, None, None]
            )

        # Calibration-free fallback: subtract a linear row profile estimated from the
        # image. This removes the dominant perspective slope while retaining localized
        # steps, ledges, roughness, gaps, and overhead obstacles.
        row_mean = depth.mean(dim=2)
        height = depth.shape[1]
        height = int(height.item()) if torch.is_tensor(height) else int(height)
        edge_rows = max(1, height // 8)
        top = row_mean[:, :edge_rows].mean(dim=1)
        bottom = row_mean[:, -edge_rows:].mean(dim=1)
        t = torch.linspace(0.0, 1.0, height, device=depth.device, dtype=depth.dtype)
        profile = top[:, None] + (bottom - top)[:, None] * t[None, :]
        return profile[:, :, None].expand_as(depth)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(x.dtype)
        dims = tuple(range(1, x.ndim))
        return (x * mask_f).sum(dim=dims) / mask_f.sum(dim=dims).clamp_min(1.0)

    @staticmethod
    def _masked_mean_along_width(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(x.dtype)
        return (x * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)

    def _masked_quantile(self, x: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
        return _masked_quantile_dynamic(x, mask, q)

    def _masked_topk_mean(self, x: torch.Tensor, mask: torch.Tensor, fraction: float) -> torch.Tensor:
        return _masked_topk_mean_dynamic(x, mask, fraction)
    
    def save(self, path: str | Path) -> None:
        """Save the extractor's configuration and flat-ground calibration state."""
        torch.save(
            {
                "output_size": self.output_size,
                "crop": self.crop,
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
                "far_depth": self.far_depth,
                "close_depth": self.close_depth,
                "close_residual_threshold": self.close_residual_threshold,
                "depth_scale": self.depth_scale,
                "sobel_edge_threshold": self.sobel_edge_threshold,
                "topk_fraction": self.topk_fraction,
                "near_fraction": self.near_fraction,
                "center_fraction": self.center_fraction,
                "orientation_scale": self.orientation_scale,
                "angular_velocity_scale": self.angular_velocity_scale,
                "reference_ridge": self.reference_ridge,
                "eps": self.eps,
                "reference_coefficients": (
                    self.reference_coefficients.cpu()
                    if self.reference_coefficients is not None
                    else None
                ),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "SobelDepthTerrainFeatureExtractor":
        """Load an extractor from a saved checkpoint, restoring calibration if present."""
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        extractor = cls(
            output_size=checkpoint["output_size"],
            crop=checkpoint["crop"],
            min_depth=checkpoint["min_depth"],
            max_depth=checkpoint["max_depth"],
            far_depth=checkpoint["far_depth"],
            close_depth=checkpoint["close_depth"],
            close_residual_threshold=checkpoint["close_residual_threshold"],
            depth_scale=checkpoint["depth_scale"],
            sobel_edge_threshold=checkpoint["sobel_edge_threshold"],
            topk_fraction=checkpoint["topk_fraction"],
            near_fraction=checkpoint["near_fraction"],
            center_fraction=checkpoint["center_fraction"],
            orientation_scale=checkpoint["orientation_scale"],
            angular_velocity_scale=checkpoint["angular_velocity_scale"],
            reference_ridge=checkpoint["reference_ridge"],
            device=device,
            eps=checkpoint["eps"],
        )

        reference = checkpoint["reference_coefficients"]
        extractor.reference_coefficients = (
            reference.to(extractor.device) if reference is not None else None
        )
        return extractor


# =============================================================================
# Standardized PCA + multiple-prototype incremental classifier
# =============================================================================


class IncrementalPCAPrototypeClassifier:
    """Standardized PCA classifier with one or more prototypes per class.

    Key design choices
    ------------------
    * Raw engineered features are z-score standardized before PCA.
    * PCA can optionally whiten the retained components.
    * Each class is represented by ``num_prototypes`` k-means centroids in PCA space.
    * New classes can be added with the PCA transform frozen. Only the new class is
      clustered; old classes are untouched.
    * Small engineered feature vectors are retained by class. This allows exact
      prototype refreshes and an optional exact PCA refit without storing depth images.
    * No explicit unknown class or rejection threshold is used.
    """

    def __init__(
        self,
        feature_dim: int,
        pca_dim: int = 8,
        num_prototypes: int = 1,
        metric_type: str = "euclidean",
        metric_regularization: float = 1e-3,
        temperature: float = 1.0,
        whiten: bool = True,
        prior_mode: str = "uniform",
        kmeans_iterations: int = 30,
        kmeans_restarts: int = 3,
        random_seed: int = 0,
        device: str | torch.device = "cpu",
        eps: float = 1e-6,
    ):
        self.feature_dim = int(feature_dim)
        self.pca_dim = int(pca_dim)
        self.num_prototypes = int(num_prototypes)
        self.metric_type = str(metric_type)
        self.metric_regularization = float(metric_regularization)
        self.temperature = float(temperature)
        self.whiten = bool(whiten)
        self.prior_mode = str(prior_mode)
        self.kmeans_iterations = int(kmeans_iterations)
        self.kmeans_restarts = int(kmeans_restarts)
        self.random_seed = int(random_seed)
        self.device = torch.device(device)
        self.eps = float(eps)

        if self.num_prototypes < 1:
            raise ValueError("num_prototypes must be >= 1")
        if self.metric_type not in {"euclidean", "diag_mahalanobis", "full_mahalanobis"}:
            raise ValueError("Unsupported metric_type")
        if self.prior_mode not in {"uniform", "empirical"}:
            raise ValueError("prior_mode must be 'uniform' or 'empirical'")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

        # Frozen representation parameters after initial fitting.
        self.scaler_mean: torch.Tensor | None = None
        self.scaler_std: torch.Tensor | None = None
        self.pca_components: torch.Tensor | None = None  # [K,D]
        self.pca_eigenvalues: torch.Tensor | None = None  # [K]

        # Engineered feature vectors retained by class. These are tiny compared with
        # depth images and make later prototype or PCA refits exact and simple.
        self.class_features: dict[str, torch.Tensor] = {}
        self.class_ids: list[str] = []
        self.class_prototype_counts: dict[str, int] = {}
        self.class_prototypes: dict[str, torch.Tensor] = {}  # each [P_c,K]

        # Shared distance metric in PCA space.
        self.metric_covariance: torch.Tensor | None = None
        self.metric_precision: torch.Tensor | None = None
        self.metric_diagonal: torch.Tensor | None = None

    # -------------------------------------------------------------------------
    # Fitting and incremental class management
    # -------------------------------------------------------------------------

    def fit(self, features: torch.Tensor | Sequence, labels: Sequence[str]) -> None:
        """Fit the initial PCA transform and initial terrain classes."""
        x = self._validate_features(features)
        labels = [str(v) for v in labels]
        if len(labels) != x.shape[0]:
            raise ValueError("labels length must match the feature batch")
        if x.shape[0] < 2:
            raise ValueError("At least two samples are needed")

        self.class_features = {}
        self.class_ids = []
        for class_id in self._ordered_unique(labels):
            mask = torch.tensor([label == class_id for label in labels], device=self.device)
            self.class_ids.append(class_id)
            self.class_features[class_id] = x[mask].clone()
            self.class_prototype_counts[class_id] = self.num_prototypes

        self._fit_representation(self._all_features())
        self._refresh_all_prototypes()
        self._refresh_metric()

    def add_class(
        self,
        class_id: str,
        features: torch.Tensor | Sequence,
        *,
        num_prototypes: int | None = None,
        refit_pca: bool = False,
    ) -> None:
        """Add a new class between evaluations.

        By default, the scaler/PCA transform stays frozen and only the new class is
        clustered. ``refit_pca=True`` is available when new data is poorly represented
        by the original PCA subspace; it refreshes all prototypes from stored features.
        """
        self._check_fitted()
        class_id = str(class_id)
        if class_id in self.class_features:
            raise ValueError(f"Class '{class_id}' already exists; use update_class()")
        x = self._validate_features(features)
        if x.shape[0] == 0:
            raise ValueError("Cannot add an empty class")

        self.class_ids.append(class_id)
        self.class_features[class_id] = x.clone()
        self.class_prototype_counts[class_id] = int(num_prototypes or self.num_prototypes)

        if refit_pca:
            self._fit_representation(self._all_features())
            self._refresh_all_prototypes()
        else:
            self._refresh_class_prototypes(class_id)
        self._refresh_metric()

    def update_class(
        self,
        class_id: str,
        features: torch.Tensor | Sequence,
        *,
        num_prototypes: int | None = None,
        refit_pca: bool = False,
    ) -> None:
        """Append labeled examples and recluster only the selected class by default."""
        class_id = str(class_id)
        if class_id not in self.class_features:
            self.add_class(
                class_id,
                features,
                num_prototypes=num_prototypes,
                refit_pca=refit_pca,
            )
            return
        x = self._validate_features(features)
        self.class_features[class_id] = torch.cat([self.class_features[class_id], x], dim=0)
        if num_prototypes is not None:
            self.class_prototype_counts[class_id] = int(num_prototypes)

        if refit_pca:
            self._fit_representation(self._all_features())
            self._refresh_all_prototypes()
        else:
            self._refresh_class_prototypes(class_id)
        self._refresh_metric()

    def refit_pca_from_stored_features(self, pca_dim: int | None = None) -> None:
        """Exactly refit scaler/PCA and all prototypes from retained feature vectors."""
        self._check_has_classes()
        if pca_dim is not None:
            self.pca_dim = int(pca_dim)
        self._fit_representation(self._all_features())
        self._refresh_all_prototypes()
        self._refresh_metric()

    def set_num_prototypes(self, num_prototypes: int, class_id: str | None = None) -> None:
        """Change prototype count globally or for one class, then recluster."""
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be >= 1")
        if class_id is None:
            self.num_prototypes = int(num_prototypes)
            for name in self.class_ids:
                self.class_prototype_counts[name] = int(num_prototypes)
            self._refresh_all_prototypes()
        else:
            if class_id not in self.class_features:
                raise KeyError(class_id)
            self.class_prototype_counts[class_id] = int(num_prototypes)
            self._refresh_class_prototypes(class_id)

    # -------------------------------------------------------------------------
    # Representation and scoring
    # -------------------------------------------------------------------------

    def transform(self, features: torch.Tensor | Sequence) -> torch.Tensor:
        self._check_fitted()
        x = self._validate_features(features)
        x_standard = (x - self.scaler_mean) / self.scaler_std
        z = x_standard @ self.pca_components.T
        if self.whiten:
            z = z / torch.sqrt(self.pca_eigenvalues + self.eps)
        return z

    @torch.inference_mode()
    def predict(self, features: torch.Tensor | Sequence) -> list[str]:
        labels, _, _ = self.predict_proba(features)
        return labels

    @torch.inference_mode()
    def predict_proba(
        self, features: torch.Tensor | Sequence
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Return labels, known-class probabilities, and class distances."""
        distances = self.compute_class_distances(features)
        priors = self._class_prior_tensor()
        logits = -distances / self.temperature + torch.log(priors).unsqueeze(0)
        probabilities = F.softmax(logits, dim=1)
        indices = torch.argmax(probabilities, dim=1)
        labels = [self.class_ids[i] for i in indices.tolist()]
        return labels, probabilities, distances

    @torch.inference_mode()
    def compute_class_distances(self, features: torch.Tensor | Sequence) -> torch.Tensor:
        """Compute minimum prototype distance for each class: [B,C]."""
        self._check_ready()
        z = self.transform(features)
        class_distances: list[torch.Tensor] = []
        for class_id in self.class_ids:
            prototypes = self.class_prototypes[class_id]
            diff = z[:, None, :] - prototypes[None, :, :]
            prototype_distances = self._distance(diff)
            class_distances.append(prototype_distances.min(dim=1).values)
        return torch.stack(class_distances, dim=1)

    @torch.inference_mode()
    def pca_reconstruction_error(self, features: torch.Tensor | Sequence) -> torch.Tensor:
        """Return standardized-space PCA reconstruction MSE for each sample."""
        self._check_fitted()
        x = self._validate_features(features)
        standardized = (x - self.scaler_mean) / self.scaler_std
        projected = standardized @ self.pca_components.T
        reconstructed = projected @ self.pca_components
        return (standardized - reconstructed).square().mean(dim=1)

    # -------------------------------------------------------------------------
    # Internal PCA, prototypes, metric, and k-means
    # -------------------------------------------------------------------------

    def _fit_representation(self, x: torch.Tensor) -> None:
        self.scaler_mean = x.mean(dim=0)
        self.scaler_std = x.std(dim=0, unbiased=True).clamp_min(1e-4)
        standardized = (x - self.scaler_mean) / self.scaler_std
        covariance = standardized.T @ standardized / max(x.shape[0] - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        max_dim = min(self.pca_dim, self.feature_dim, max(1, x.shape[0] - 1))
        self.pca_eigenvalues = eigenvalues[order][:max_dim].clamp_min(self.eps)
        self.pca_components = eigenvectors[:, order][:, :max_dim].T.contiguous()

    def _refresh_all_prototypes(self) -> None:
        for class_id in self.class_ids:
            self._refresh_class_prototypes(class_id)

    def _refresh_class_prototypes(self, class_id: str) -> None:
        z = self.transform(self.class_features[class_id])
        requested = self.class_prototype_counts.get(class_id, self.num_prototypes)
        k = min(max(1, int(requested)), z.shape[0])
        self.class_prototypes[class_id] = self._kmeans(z, k)

    def _refresh_metric(self) -> None:
        z = self.transform(self._all_features())
        k = z.shape[1]
        eye = torch.eye(k, dtype=torch.float32, device=self.device)
        if z.shape[0] <= 1:
            covariance = eye
        else:
            centered = z - z.mean(dim=0)
            covariance = centered.T @ centered / (z.shape[0] - 1)

        if self.metric_type == "euclidean":
            covariance = eye
            precision = eye
            diagonal = torch.ones(k, device=self.device)
        elif self.metric_type == "diag_mahalanobis":
            diagonal = torch.diagonal(covariance).clamp_min(self.eps) + self.metric_regularization
            covariance = torch.diag(diagonal)
            precision = torch.diag(1.0 / diagonal)
        else:
            covariance = covariance + self.metric_regularization * eye
            # pinv is safer than inv for small-data covariance estimates.
            precision = torch.linalg.pinv(covariance)
            diagonal = torch.diagonal(covariance)

        self.metric_covariance = covariance
        self.metric_precision = precision
        self.metric_diagonal = diagonal

    def _distance(self, diff: torch.Tensor) -> torch.Tensor:
        # diff: [B,P,K] -> [B,P]
        if self.metric_type == "euclidean":
            return diff.square().sum(dim=-1)
        if self.metric_type == "diag_mahalanobis":
            return (diff.square() / self.metric_diagonal[None, None, :]).sum(dim=-1)
        return torch.einsum("bpk,kl,bpl->bp", diff, self.metric_precision, diff)

    def _kmeans(self, z: torch.Tensor, k: int) -> torch.Tensor:
        if k == 1:
            return z.mean(dim=0, keepdim=True)

        best_centers: torch.Tensor | None = None
        best_inertia = float("inf")
        for restart in range(self.kmeans_restarts):
            generator = torch.Generator(device=z.device)
            generator.manual_seed(self.random_seed + restart)
            centers = self._kmeans_plus_plus(z, k, generator)

            for _ in range(self.kmeans_iterations):
                distances = torch.cdist(z, centers).square()
                assignment = torch.argmin(distances, dim=1)
                new_centers = centers.clone()
                for cluster in range(k):
                    members = z[assignment == cluster]
                    if members.shape[0] > 0:
                        new_centers[cluster] = members.mean(dim=0)
                    else:
                        # Reinitialize an empty cluster at the currently worst-fit point.
                        nearest = distances.min(dim=1).values
                        new_centers[cluster] = z[torch.argmax(nearest)]
                if torch.allclose(new_centers, centers, atol=1e-5, rtol=1e-4):
                    centers = new_centers
                    break
                centers = new_centers

            inertia = torch.cdist(z, centers).square().min(dim=1).values.sum().item()
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.clone()

        assert best_centers is not None
        return best_centers

    @staticmethod
    def _kmeans_plus_plus(
        z: torch.Tensor, k: int, generator: torch.Generator
    ) -> torch.Tensor:
        n = z.shape[0]
        first = torch.randint(n, (1,), generator=generator, device=z.device).item()
        centers = [z[first]]
        closest_sq = torch.cdist(z, centers[0].unsqueeze(0)).square().squeeze(1)
        for _ in range(1, k):
            total = closest_sq.sum()
            if total <= 1e-12:
                index = torch.randint(n, (1,), generator=generator, device=z.device).item()
            else:
                index = torch.multinomial(closest_sq / total, 1, generator=generator).item()
            centers.append(z[index])
            new_sq = torch.cdist(z, z[index].unsqueeze(0)).square().squeeze(1)
            closest_sq = torch.minimum(closest_sq, new_sq)
        return torch.stack(centers)

    # -------------------------------------------------------------------------
    # Utilities and serialization
    # -------------------------------------------------------------------------

    def _class_prior_tensor(self) -> torch.Tensor:
        if self.prior_mode == "uniform":
            values = torch.ones(len(self.class_ids), device=self.device)
        else:
            values = torch.tensor(
                [self.class_features[c].shape[0] for c in self.class_ids],
                dtype=torch.float32,
                device=self.device,
            )
        return values / values.sum().clamp_min(self.eps)

    def _all_features(self) -> torch.Tensor:
        self._check_has_classes()
        return torch.cat([self.class_features[c] for c in self.class_ids], dim=0)

    def _validate_features(self, features: torch.Tensor | Sequence) -> torch.Tensor:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features [N,{self.feature_dim}], got {tuple(x.shape)}"
            )
        return x

    def _check_has_classes(self) -> None:
        if not self.class_ids:
            raise ValueError("No classes have been fitted")

    def _check_fitted(self) -> None:
        if self.pca_components is None or self.scaler_mean is None:
            raise ValueError("Classifier representation is not fitted")

    def _check_ready(self) -> None:
        self._check_fitted()
        self._check_has_classes()
        if any(c not in self.class_prototypes for c in self.class_ids):
            raise ValueError("Class prototypes are incomplete")

    @staticmethod
    def _ordered_unique(values: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "feature_dim": self.feature_dim,
                "pca_dim": self.pca_dim,
                "num_prototypes": self.num_prototypes,
                "metric_type": self.metric_type,
                "metric_regularization": self.metric_regularization,
                "temperature": self.temperature,
                "whiten": self.whiten,
                "prior_mode": self.prior_mode,
                "kmeans_iterations": self.kmeans_iterations,
                "kmeans_restarts": self.kmeans_restarts,
                "random_seed": self.random_seed,
                "eps": self.eps,
            },
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "pca_components": self.pca_components,
            "pca_eigenvalues": self.pca_eigenvalues,
            "class_features": self.class_features,
            "class_ids": self.class_ids,
            "class_prototype_counts": self.class_prototype_counts,
            "class_prototypes": self.class_prototypes,
            "metric_covariance": self.metric_covariance,
            "metric_precision": self.metric_precision,
            "metric_diagonal": self.metric_diagonal,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.scaler_mean = state["scaler_mean"].to(self.device)
        self.scaler_std = state["scaler_std"].to(self.device)
        self.pca_components = state["pca_components"].to(self.device)
        self.pca_eigenvalues = state["pca_eigenvalues"].to(self.device)
        self.class_features = {k: v.to(self.device) for k, v in state["class_features"].items()}
        self.class_ids = list(state["class_ids"])
        self.class_prototype_counts = dict(state["class_prototype_counts"])
        self.class_prototypes = {k: v.to(self.device) for k, v in state["class_prototypes"].items()}
        self.metric_covariance = state["metric_covariance"].to(self.device)
        self.metric_precision = state["metric_precision"].to(self.device)
        self.metric_diagonal = state["metric_diagonal"].to(self.device)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the extractor's configuration and fitted reference model."""
        torch.save(
            {
                "output_size": self.output_size,
                "crop": self.crop,
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
                "far_depth": self.far_depth,
                "close_depth": self.close_depth,
                "close_residual_threshold": self.close_residual_threshold,
                "depth_scale": self.depth_scale,
                "sobel_edge_threshold": self.sobel_edge_threshold,
                "topk_fraction": self.topk_fraction,
                "near_fraction": self.near_fraction,
                "center_fraction": self.center_fraction,
                "orientation_scale": self.orientation_scale,
                "angular_velocity_scale": self.angular_velocity_scale,
                "reference_ridge": self.reference_ridge,
                "eps": self.eps,
                "reference_coefficients": self.reference_coefficients,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "SobelDepthTerrainFeatureExtractor":
        """Load an extractor from a saved checkpoint, including any fitted
        flat-ground reference model."""
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        extractor = cls(
            output_size=checkpoint["output_size"],
            crop=checkpoint["crop"],
            min_depth=checkpoint["min_depth"],
            max_depth=checkpoint["max_depth"],
            far_depth=checkpoint["far_depth"],
            close_depth=checkpoint["close_depth"],
            close_residual_threshold=checkpoint["close_residual_threshold"],
            depth_scale=checkpoint["depth_scale"],
            sobel_edge_threshold=checkpoint["sobel_edge_threshold"],
            topk_fraction=checkpoint["topk_fraction"],
            near_fraction=checkpoint["near_fraction"],
            center_fraction=checkpoint["center_fraction"],
            orientation_scale=checkpoint["orientation_scale"],
            angular_velocity_scale=checkpoint["angular_velocity_scale"],
            reference_ridge=checkpoint["reference_ridge"],
            device=device,
            eps=checkpoint["eps"],
        )

        reference_coefficients = checkpoint["reference_coefficients"]
        if reference_coefficients is not None:
            extractor.reference_coefficients = reference_coefficients.to(device)

        return extractor


# =============================================================================
# Temporal filtering
# =============================================================================


class TemporalProbabilityFilter:
    """EMA probability smoothing with label-switch hysteresis."""

    def __init__(self, alpha: float = 0.45, switch_frames: int = 2):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0,1]")
        if switch_frames < 1:
            raise ValueError("switch_frames must be >= 1")
        self.alpha = float(alpha)
        self.switch_frames = int(switch_frames)
        self.reset()

    def reset(self) -> None:
        self.filtered_probabilities: torch.Tensor | None = None
        self.labels: list[str] | None = None
        self.committed_label: str | None = None
        self.candidate_label: str | None = None
        self.candidate_count = 0

    def update(self, probabilities: torch.Tensor, labels: Sequence[str]) -> tuple[str, torch.Tensor]:
        probs = probabilities.detach().flatten()
        labels = list(labels)
        if probs.numel() != len(labels):
            raise ValueError("Probability count does not match label count")

        # Adding a class changes probability layout, so start a fresh temporal state.
        if self.labels != labels or self.filtered_probabilities is None:
            self.labels = labels
            self.filtered_probabilities = probs.clone()
            self.committed_label = labels[int(torch.argmax(probs))]
            self.candidate_label = None
            self.candidate_count = 0
            return self.committed_label, self.filtered_probabilities.clone()

        self.filtered_probabilities.mul_(1.0 - self.alpha).add_(probs, alpha=self.alpha)
        self.filtered_probabilities /= self.filtered_probabilities.sum().clamp_min(1e-8)
        best = labels[int(torch.argmax(self.filtered_probabilities))]

        if best == self.committed_label:
            self.candidate_label = None
            self.candidate_count = 0
        elif best == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = best
            self.candidate_count = 1

        if self.candidate_count >= self.switch_frames:
            self.committed_label = best
            self.candidate_label = None
            self.candidate_count = 0

        assert self.committed_label is not None
        return self.committed_label, self.filtered_probabilities.clone()


# =============================================================================
# Unified depth terrain classifier and experiment helpers
# =============================================================================


class DepthTerrainClassifier:
    """Unified training, incremental update, experiment, and deployment interface."""

    def __init__(
        self,
        *,
        extractor: SobelDepthTerrainFeatureExtractor | None = None,
        pca_dim: int = 8,
        num_prototypes: int = 1,
        metric_type: str = "euclidean",
        metric_regularization: float = 1e-3,
        temperature: float = 1.0,
        whiten: bool = True,
        prior_mode: str = "uniform",
        temporal_alpha: float = 0.45,
        switch_frames: int = 2,
        kmeans_iterations: int = 30,
        kmeans_restarts: int = 3,
        random_seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.extractor = extractor or SobelDepthTerrainFeatureExtractor(device=self.device)
        self.classifier = IncrementalPCAPrototypeClassifier(
            feature_dim=self.extractor.feature_dim,
            pca_dim=pca_dim,
            num_prototypes=num_prototypes,
            metric_type=metric_type,
            metric_regularization=metric_regularization,
            temperature=temperature,
            whiten=whiten,
            prior_mode=prior_mode,
            kmeans_iterations=kmeans_iterations,
            kmeans_restarts=kmeans_restarts,
            random_seed=random_seed,
            device=self.device,
        )
        self.temporal_filter = TemporalProbabilityFilter(temporal_alpha, switch_frames)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.extractor.FEATURE_NAMES

    # -------------------------------------------------------------------------
    # Calibration, extraction, initial fitting, and incremental updates
    # -------------------------------------------------------------------------

    def fit_reference_model(
        self,
        flat_depth_images: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
    ) -> torch.Tensor:
        return self.extractor.fit_reference_model(flat_depth_images, orientation_rpy)

    @torch.inference_mode()
    def extract_features(
        self,
        depth_images: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
    ) -> torch.Tensor:
        return self.extractor.extract_batch(depth_images, orientation_rpy, angular_velocity)

    def fit_initial(
        self,
        depth_images: torch.Tensor | Sequence,
        labels: Sequence[str],
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
    ) -> torch.Tensor:
        """Fit PCA and initial classes, e.g. rough and stairs."""
        features = self.extract_features(depth_images, orientation_rpy, angular_velocity)
        self.classifier.fit(features, labels)
        self.temporal_filter.reset()
        return features

    def add_class(
        self,
        class_id: str,
        depth_images: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
        *,
        num_prototypes: int | None = None,
        refit_pca: bool = False,
    ) -> torch.Tensor:
        """Add a new terrain class between evaluations, e.g. pit or gap."""
        features = self.extract_features(depth_images, orientation_rpy, angular_velocity)
        self.classifier.add_class(
            class_id,
            features,
            num_prototypes=num_prototypes,
            refit_pca=refit_pca,
        )
        self.temporal_filter.reset()
        return features

    def update_class(
        self,
        class_id: str,
        depth_images: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
        *,
        num_prototypes: int | None = None,
        refit_pca: bool = False,
    ) -> torch.Tensor:
        features = self.extract_features(depth_images, orientation_rpy, angular_velocity)
        self.classifier.update_class(
            class_id,
            features,
            num_prototypes=num_prototypes,
            refit_pca=refit_pca,
        )
        self.temporal_filter.reset()
        return features

    # -------------------------------------------------------------------------
    # Deployment inference
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def predict_depth(
        self,
        depth_image: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
        *,
        temporal: bool = True,
    ) -> TerrainPrediction:
        features = self.extractor.extract(depth_image, orientation_rpy, angular_velocity)
        labels, probabilities, distances = self.classifier.predict_proba(features.unsqueeze(0))
        instantaneous = labels[0]
        if temporal:
            filtered_label, filtered_probs = self.temporal_filter.update(
                probabilities[0], self.classifier.class_ids
            )
        else:
            filtered_label, filtered_probs = instantaneous, probabilities[0]
        return TerrainPrediction(
            label=filtered_label,
            instantaneous_label=instantaneous,
            probabilities=filtered_probs,
            labels=list(self.classifier.class_ids),
            distances=distances[0],
            raw_features=features,
        )

    @torch.inference_mode()
    def predict_depth_batch(
        self,
        depth_images: torch.Tensor | Sequence,
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Stateless batch inference; temporal filtering is intentionally omitted."""
        features = self.extract_features(depth_images, orientation_rpy, angular_velocity)
        return self.classifier.predict_proba(features)

    def reset_temporal_filter(self) -> None:
        """Call at episode boundaries or after discontinuous camera motion."""
        self.temporal_filter.reset()

    # -------------------------------------------------------------------------
    # Evaluation and hyperparameter-search helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def default_hyperparameter_grid() -> dict[str, Sequence[Any]]:
        """A compact search grid appropriate for small terrain datasets."""
        return {
            "pca_dim": (4, 6, 8, 10, 12),
            "num_prototypes": (1, 2, 3),
            "metric_type": ("euclidean", "diag_mahalanobis"),
            "metric_regularization": (1e-3, 1e-2),
            "temperature": (0.5, 1.0, 2.0),
            "whiten": (True, False),
        }

    @staticmethod
    def evaluate_predictions(
        true_labels: Sequence[str],
        predicted_labels: Sequence[str],
        label_order: Sequence[str] | None = None,
    ) -> EvaluationResult:
        true_labels = [str(v) for v in true_labels]
        predicted_labels = [str(v) for v in predicted_labels]
        if len(true_labels) != len(predicted_labels):
            raise ValueError("true and predicted label lengths differ")
        labels = list(label_order or dict.fromkeys(true_labels + predicted_labels))
        index = {label: i for i, label in enumerate(labels)}
        confusion = torch.zeros(len(labels), len(labels), dtype=torch.int64)
        for truth, prediction in zip(true_labels, predicted_labels):
            confusion[index[truth], index[prediction]] += 1

        total = confusion.sum().item()
        accuracy = float(torch.diagonal(confusion).sum().item() / max(total, 1))
        recalls: list[float] = []
        f1s: list[float] = []
        for i in range(len(labels)):
            tp = confusion[i, i].item()
            fn = confusion[i].sum().item() - tp
            fp = confusion[:, i].sum().item() - tp
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            recalls.append(recall)
            f1s.append(f1)
        return EvaluationResult(
            accuracy=accuracy,
            balanced_accuracy=sum(recalls) / max(len(recalls), 1),
            macro_f1=sum(f1s) / max(len(f1s), 1),
            confusion_matrix=confusion,
            labels=labels,
        )

    @classmethod
    def cross_validate_hyperparameters(
        cls,
        features: torch.Tensor | Sequence,
        labels: Sequence[str],
        *,
        param_grid: Mapping[str, Sequence[Any]] | None = None,
        n_splits: int = 5,
        random_seed: int = 0,
        scoring: str = "balanced_accuracy",
        device: str | torch.device = "cpu",
    ) -> list[dict[str, Any]]:
        """Run dependency-free stratified k-fold model selection on extracted features.

        Feature extraction should be performed once before calling this method. This
        keeps experiments fast and ensures every configuration sees identical inputs.
        Results are sorted from best to worst by ``scoring``.
        """
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        labels = [str(v) for v in labels]
        if x.ndim != 2 or x.shape[0] != len(labels):
            raise ValueError("features and labels are inconsistent")
        if scoring not in {"accuracy", "balanced_accuracy", "macro_f1"}:
            raise ValueError("Unsupported scoring metric")

        folds = cls._stratified_folds(labels, n_splits, random_seed)
        grid = dict(param_grid or cls.default_hyperparameter_grid())
        keys = list(grid)
        results: list[dict[str, Any]] = []

        for values in product(*(grid[key] for key in keys)):
            params = dict(zip(keys, values))
            fold_metrics: list[EvaluationResult] = []
            for fold_index in range(len(folds)):
                val_indices = folds[fold_index]
                train_indices = torch.cat(
                    [folds[j] for j in range(len(folds)) if j != fold_index]
                )
                train_labels = [labels[i] for i in train_indices.tolist()]
                val_labels = [labels[i] for i in val_indices.tolist()]

                model = IncrementalPCAPrototypeClassifier(
                    feature_dim=x.shape[1],
                    device=device,
                    random_seed=random_seed + fold_index,
                    **params,
                )
                model.fit(x[train_indices], train_labels)
                predictions = model.predict(x[val_indices])
                fold_metrics.append(
                    cls.evaluate_predictions(val_labels, predictions, model.class_ids)
                )

            result = dict(params)
            result.update(
                {
                    "accuracy": sum(m.accuracy for m in fold_metrics) / len(fold_metrics),
                    "balanced_accuracy": sum(m.balanced_accuracy for m in fold_metrics) / len(fold_metrics),
                    "macro_f1": sum(m.macro_f1 for m in fold_metrics) / len(fold_metrics),
                }
            )
            results.append(result)

        results.sort(key=lambda item: item[scoring], reverse=True)
        return results

    def run_hyperparameter_search(
        self,
        depth_images: torch.Tensor | Sequence,
        labels: Sequence[str],
        orientation_rpy: torch.Tensor | Sequence | None = None,
        angular_velocity: torch.Tensor | Sequence | None = None,
        *,
        param_grid: Mapping[str, Sequence[Any]] | None = None,
        n_splits: int = 5,
        random_seed: int = 0,
        scoring: str = "balanced_accuracy",
    ) -> list[dict[str, Any]]:
        """Extract features once, then run stratified classifier model selection."""
        features = self.extract_features(depth_images, orientation_rpy, angular_velocity)
        return self.cross_validate_hyperparameters(
            features,
            labels,
            param_grid=param_grid,
            n_splits=n_splits,
            random_seed=random_seed,
            scoring=scoring,
            device=self.device,
        )

    def search_temporal_hyperparameters(
        self,
        features: torch.Tensor | Sequence,
        labels: Sequence[str],
        sequence_ids: Sequence[Hashable],
        *,
        alphas: Sequence[float] = (0.25, 0.45, 0.65, 1.0),
        switch_frames: Sequence[int] = (1, 2, 3),
        scoring: str = "balanced_accuracy",
    ) -> list[dict[str, Any]]:
        """Tune temporal smoothing on ordered, already-extracted validation sequences.

        Samples must be ordered in time. The filter is reset whenever ``sequence_id``
        changes, which prevents probability leakage between episodes.
        """
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        labels = [str(v) for v in labels]
        sequence_ids = list(sequence_ids)
        if not (x.shape[0] == len(labels) == len(sequence_ids)):
            raise ValueError("features, labels, and sequence_ids must have equal length")
        _, probabilities, _ = self.classifier.predict_proba(x)
        results: list[dict[str, Any]] = []

        for alpha, switch_count in product(alphas, switch_frames):
            temporal_filter = TemporalProbabilityFilter(alpha, switch_count)
            predictions: list[str] = []
            previous_sequence: Hashable | None = None
            for i, sequence in enumerate(sequence_ids):
                if i == 0 or sequence != previous_sequence:
                    temporal_filter.reset()
                prediction, _ = temporal_filter.update(
                    probabilities[i], self.classifier.class_ids
                )
                predictions.append(prediction)
                previous_sequence = sequence
            metrics = self.evaluate_predictions(labels, predictions, self.classifier.class_ids)
            result = {
                "temporal_alpha": float(alpha),
                "switch_frames": int(switch_count),
                **metrics.as_dict(),
            }
            results.append(result)
        results.sort(key=lambda item: item[scoring], reverse=True)
        return results

    @staticmethod
    def _stratified_folds(
        labels: Sequence[str], n_splits: int, random_seed: int
    ) -> list[torch.Tensor]:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        generator = torch.Generator().manual_seed(random_seed)
        by_class: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            by_class.setdefault(label, []).append(index)
        minimum = min(len(indices) for indices in by_class.values())
        if minimum < n_splits:
            raise ValueError(
                f"Every class needs at least n_splits samples; smallest class has {minimum}"
            )

        fold_lists: list[list[int]] = [[] for _ in range(n_splits)]
        for indices in by_class.values():
            tensor = torch.tensor(indices, dtype=torch.long)
            tensor = tensor[torch.randperm(len(indices), generator=generator)]
            for position, index in enumerate(tensor.tolist()):
                fold_lists[position % n_splits].append(index)
        return [torch.tensor(sorted(items), dtype=torch.long) for items in fold_lists]

    # -------------------------------------------------------------------------
    # Save/load helpers
    # -------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save extractor calibration, classifier state, and temporal configuration."""
        payload = {
            "extractor_reference_coefficients": self.extractor.reference_coefficients,
            "classifier": self.classifier.state_dict(),
            "temporal_alpha": self.temporal_filter.alpha,
            "switch_frames": self.temporal_filter.switch_frames,
        }
        torch.save(payload, Path(path))

    def load(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        reference = payload["extractor_reference_coefficients"]
        self.extractor.reference_coefficients = (
            None if reference is None else reference.to(self.device)
        )
        self.classifier.load_state_dict(payload["classifier"])
        self.temporal_filter = TemporalProbabilityFilter(
            payload["temporal_alpha"], payload["switch_frames"]
        )


__all__ = [
    "DepthTerrainClassifier",
    "EvaluationResult",
    "IncrementalPCAPrototypeClassifier",
    "SobelDepthTerrainFeatureExtractor",
    "TemporalProbabilityFilter",
    "TerrainPrediction",
]
