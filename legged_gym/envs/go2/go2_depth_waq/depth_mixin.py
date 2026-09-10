import torch
import torch.nn.functional as F


class DepthMixin:
    def _init_depth_processing(self):
        if not self.cfg.sensor.add_depth:
            self.depth_sensor_output = None
            return

        cfg = self.cfg.sensor.depth_camera_config
        self.depth_noise_profile = getattr(
            cfg, "depth_noise_profile", "legacy"
        )
        if self.depth_noise_profile not in ("parkour_aligned", "legacy"):
            raise ValueError(
                "depth_noise_profile must be 'parkour_aligned' or 'legacy', "
                f"got {self.depth_noise_profile!r}"
            )
        if hasattr(cfg, "resized_resolution"):
            self.depth_output_resolution = tuple(cfg.resized_resolution)
        else:
            height, width = cfg.resolution
            top, bottom = cfg.crop_top_bottom
            left, right = cfg.crop_left_right
            self.depth_output_resolution = (
                height - top - bottom,
                width - left - right,
            )

        self.depth_sensor_output = torch.zeros(
            (
                self.num_envs,
                cfg.num_history,
                *self.depth_output_resolution,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        buffer_length = max(2, int(cfg.latency_range[1] / self.dt) + 2)
        self.depth_sensor_obs_buffer = torch.zeros(
            (
                buffer_length,
                self.num_envs,
                *self.depth_output_resolution,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.depth_sensor_obs_write_idx = 0
        self.depth_sensor_latency = torch.empty(
            self.num_envs, device=self.device
        ).uniform_(*cfg.latency_range)
        self.depth_sensor_delayed_frames = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.depth_sensor_obs_refreshed = False
        self.depth_env_ids = torch.arange(
            self.num_envs, device=self.device
        )
        self._sky_artifact_values = torch.as_tensor(
            cfg.sky_artifacts_values,
            device=self.device,
            dtype=torch.float32,
        )
        self._stereo_full_block_values = torch.as_tensor(
            cfg.stereo_full_block_values,
            device=self.device,
            dtype=torch.float32,
        )

    def get_depth_observations(self):
        return self.depth_sensor_output

    def _pre_depth_step(self):
        if self.cfg.sensor.add_depth:
            self.depth_sensor_obs_refreshed = False

    def _update_depth_observations(self):
        if not self.cfg.sensor.add_depth or self.depth_sensor_obs_refreshed:
            return

        cfg = self.cfg.sensor.depth_camera_config
        processed = self._process_depth_images(
            self.simulator.depth_images[:, 0]
        )

        # Keep the latency buffer in-place and advance a write index instead
        # of rolling the whole tensor every depth update.
        write_idx = self.depth_sensor_obs_write_idx
        self.depth_sensor_obs_buffer[write_idx] = processed
        buffer_length = self.depth_sensor_obs_buffer.shape[0]
        self.depth_sensor_obs_write_idx = (write_idx + 1) % buffer_length

        refresh_steps = max(1, int(cfg.refresh_duration / self.dt))
        refresh = (
            self.episode_length_buf % refresh_steps
        ) == 0
        requested_delay = (self.depth_sensor_latency / self.dt).long().clamp(
            0, buffer_length - 1
        )
        self.depth_sensor_delayed_frames = torch.where(
            refresh,
            requested_delay,
            self.depth_sensor_delayed_frames + 1,
        )

        # Select a delayed frame only on this environment's camera refresh
        # tick. The selected output is held unchanged between refreshes while
        # delayed_frames continues to track its increasing age.
        latest_idx = (self.depth_sensor_obs_write_idx - 1) % buffer_length
        frame_indices = (
            latest_idx - requested_delay
        ) % buffer_length
        refresh_env_ids = self.depth_env_ids[refresh]
        latest = self.depth_sensor_obs_buffer[
            frame_indices[refresh], refresh_env_ids
        ]
        if self.depth_sensor_output.shape[1] > 1:
            self.depth_sensor_output[refresh_env_ids, 1:] = (
                self.depth_sensor_output[refresh_env_ids, :-1].clone()
            )
        self.depth_sensor_output[refresh_env_ids, 0] = latest
        self.depth_sensor_obs_refreshed = True

    def _resample_depth_latency(self):
        if not self.cfg.sensor.add_depth:
            return
        cfg = self.cfg.sensor.depth_camera_config
        resample_steps = max(1, int(cfg.latency_resampling_time / self.dt))
        mask = (
            self.episode_length_buf % resample_steps
        ) == 0
        if mask.any():
            new_latency = torch.empty_like(self.depth_sensor_latency).uniform_(
                *cfg.latency_range
            )
            self.depth_sensor_latency[mask] = new_latency[mask]

    def _reset_depth_buffers(self, env_ids):
        if not self.cfg.sensor.add_depth:
            return
        if env_ids.numel() == 0:
            return
        self.depth_sensor_obs_buffer[:, env_ids] = 0.0
        self.depth_sensor_output[env_ids] = 0.0
        self.depth_sensor_delayed_frames[env_ids] = 0
        cfg = self.cfg.sensor.depth_camera_config
        self.depth_sensor_latency[env_ids] = torch.empty(
            env_ids.numel(), device=self.device
        ).uniform_(*cfg.latency_range)

    @torch.no_grad()
    def _process_depth_images(self, depth):
        cfg = self.cfg.sensor.depth_camera_config
        depth = depth.clone()
        depth = self._add_stereo_noise(depth)
        depth = self._add_sky_artifacts(depth)
        depth = depth.clamp(cfg.near_clip, cfg.far_clip)
        depth = (depth - cfg.near_clip) / max(
            cfg.far_clip - cfg.near_clip, 1e-6
        )

        top, bottom = cfg.crop_top_bottom
        left, right = cfg.crop_left_right
        height, width = depth.shape[-2:]
        depth = depth[
            ...,
            top : height - bottom if bottom else height,
            left : width - right if right else width,
        ]
        if tuple(depth.shape[-2:]) != self.depth_output_resolution:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=self.depth_output_resolution,
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
        return depth.clamp(0.0, 1.0)

    def _add_stereo_noise(self, depth):
        cfg = self.cfg.sensor.depth_camera_config
        if cfg.stereo_min_distance <= 0:
            return depth

        if self.depth_noise_profile == "legacy":
            return self._add_stereo_noise_legacy(depth)

        # Match Robot Parkour: masks are classified before either additive
        # noise draw, and the middle band excludes both far and too-close.
        far_mask = depth > cfg.stereo_far_distance
        too_close = depth < cfg.stereo_min_distance
        near_mask = ~far_mask & ~too_close
        depth.add_(
            torch.rand_like(depth)
            .mul_(cfg.stereo_far_noise_std)
            .mul_(far_mask)
        )
        depth.add_(
            torch.rand_like(depth)
            .mul_(cfg.stereo_near_noise_std)
            .mul_(near_mask)
        )

        vertical_block = self._recognize_top_down_too_close(too_close)
        full_block = vertical_block & too_close
        half_block = ~vertical_block & too_close
        value_order = torch.randperm(
            self._stereo_full_block_values.numel(), device=self.device
        )
        for pixel_value in self._stereo_full_block_values[value_order]:
            patch = self._sample_spatial_artifact_mask(
                depth.shape,
                cfg.stereo_full_block_artifacts_prob,
                cfg.stereo_full_block_height_mean_std,
                cfg.stereo_full_block_width_mean_std,
            )
            # The reference overwrites every full-block pixel on every pass:
            # the sampled patch receives this value and its complement zero.
            full_values = patch.to(depth.dtype).mul_(pixel_value)
            depth = torch.where(full_block, full_values, depth)

        sparks = torch.rand_like(depth) < cfg.stereo_half_block_spark_prob
        half_values = sparks.to(depth.dtype).mul_(
            cfg.stereo_half_block_value
        )
        depth = torch.where(half_block, half_values, depth)
        return depth

    def _add_stereo_noise_legacy(self, depth):
        """Exact pre-alignment stereo behavior for checkpoint reproduction."""
        cfg = self.cfg.sensor.depth_camera_config
        far_mask = depth > cfg.stereo_far_distance
        near_mask = (depth >= cfg.stereo_min_distance) & ~far_mask
        if far_mask.any() and cfg.stereo_far_noise_std > 0:
            far_noise = torch.randn_like(depth).abs().mul_(
                cfg.stereo_far_noise_std
            )
            depth = torch.where(far_mask, depth + far_noise, depth)
        if near_mask.any() and cfg.stereo_near_noise_std > 0:
            near_noise = torch.randn_like(depth).mul_(
                cfg.stereo_near_noise_std
            )
            depth = torch.where(near_mask, depth + near_noise, depth)
        too_close = depth < cfg.stereo_min_distance
        if too_close.any():
            spark = torch.rand_like(depth) < cfg.stereo_half_block_spark_prob
            depth[too_close & spark] = cfg.far_clip
            depth[too_close & ~spark] = cfg.near_clip
        return depth

    @staticmethod
    def _recognize_top_down_too_close(too_close_mask):
        return too_close_mask.sum(dim=-2, keepdim=True) > (
            too_close_mask.shape[-2] * 0.6
        )

    @staticmethod
    def _recognize_top_down_seeing_sky(too_far_mask):
        height = too_far_mask.shape[-2]
        row = torch.arange(height, device=too_far_mask.device).view(
            1, height, 1
        )
        return too_far_mask.cumsum(dim=-2) > row

    def _sample_spatial_artifact_mask(
        self,
        shape,
        probability,
        height_mean_std,
        width_mean_std,
    ):
        """Rasterize randomly sized patches for a full ``[N,H,W]`` batch."""
        n, height, width = shape
        seeds = torch.rand(shape, device=self.device) < probability
        coordinates = seeds.nonzero(as_tuple=False)
        count = coordinates.shape[0]
        sizes_h = (
            height_mean_std[0]
            + torch.randn(count, device=self.device) * height_mean_std[1]
        ).clamp(0.0, float(height))
        sizes_w = (
            width_mean_std[0]
            + torch.randn(count, device=self.device) * width_mean_std[1]
        ).clamp(0.0, float(width))

        centers_h = coordinates[:, 1].to(torch.float32)
        centers_w = coordinates[:, 2].to(torch.float32)
        top = (centers_h - sizes_h * 0.5).floor().clamp(0, height).long()
        bottom = (centers_h + sizes_h * 0.5).ceil().clamp(0, height).long()
        left = (centers_w - sizes_w * 0.5).floor().clamp(0, width).long()
        right = (centers_w + sizes_w * 0.5).ceil().clamp(0, width).long()
        bottom = torch.maximum(bottom, (top + 1).clamp_max(height))
        right = torch.maximum(right, (left + 1).clamp_max(width))

        # Difference-array rectangle filling is O(NHW + patches), remains on
        # device, and avoids Robot Parkour's Python loop over environments.
        stride_n = (height + 1) * (width + 1)
        stride_h = width + 1
        batch = coordinates[:, 0]
        difference = torch.zeros(
            n * stride_n, dtype=torch.float32, device=self.device
        )

        def scatter(rows, cols, value):
            indices = batch * stride_n + rows * stride_h + cols
            difference.scatter_add_(
                0, indices, torch.full_like(indices, value, dtype=torch.float32)
            )

        scatter(top, left, 1.0)
        scatter(bottom, left, -1.0)
        scatter(top, right, -1.0)
        scatter(bottom, right, 1.0)
        difference = difference.view(n, height + 1, width + 1)
        return (
            difference.cumsum(dim=1).cumsum(dim=2)[:, :height, :width]
            > 0
        )

    def _add_sky_artifacts(self, depth):
        cfg = self.cfg.sensor.depth_camera_config
        probability = (
            getattr(cfg, "legacy_sky_artifacts_prob", 0.001)
            if self.depth_noise_profile == "legacy"
            else cfg.sky_artifacts_prob
        )
        if probability <= 0:
            return depth
        if self.depth_noise_profile == "legacy":
            return self._add_sky_artifacts_legacy(depth, probability)

        possible_sky = depth > cfg.sky_artifacts_far_distance
        sky = self._recognize_top_down_seeing_sky(possible_sky)
        value_order = torch.randperm(
            self._sky_artifact_values.numel(), device=self.device
        )
        for pixel_value in self._sky_artifact_values[value_order]:
            patch = self._sample_spatial_artifact_mask(
                depth.shape,
                probability,
                cfg.sky_artifacts_height_mean_std,
                cfg.sky_artifacts_width_mean_std,
            )
            depth = torch.where(
                sky & patch, pixel_value.to(depth.dtype), depth
            )
        return depth

    def _add_sky_artifacts_legacy(self, depth, probability):
        cfg = self.cfg.sensor.depth_camera_config
        sky = depth > cfg.sky_artifacts_far_distance
        artifacts = (
            torch.rand_like(depth) < probability
        ) & sky
        if artifacts.any():
            values = self._sky_artifact_values.to(dtype=depth.dtype)
            choices = torch.randint(
                values.numel(), depth.shape, device=depth.device
            )
            depth[artifacts] = values[choices[artifacts]]
        return depth
