"""Regression checks for data-dependent feature reductions in TorchScript exports."""

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from legged_gym.scripts.export_depth_terrain_classifier import _FeatureExport, _trace_classifier
from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import (
    SobelDepthTerrainFeatureExtractor,
)


def test_masked_statistics_match_eager_reductions():
    extractor = SobelDepthTerrainFeatureExtractor()
    values = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0] = False
    mask[1, ::2] = False
    quantiles, topk = [], []
    for image, valid in zip(values, mask):
        selected = image[valid]
        quantiles.append(torch.quantile(selected, 0.75) if selected.numel() else torch.tensor(0.))
        count = max(1, round(selected.numel() * 0.2))
        topk.append(selected.topk(count).values.mean() if selected.numel() else torch.tensor(0.))
    torch.testing.assert_close(extractor._masked_quantile(values, mask, 0.75), torch.stack(quantiles))
    torch.testing.assert_close(extractor._masked_topk_mean(values, mask, 0.2), torch.stack(topk))


def test_feature_export_preserves_statistics_for_changing_masks():
    extractor = SobelDepthTerrainFeatureExtractor(
        output_size=(48, 64), min_depth=0.02, max_depth=1.,
        far_depth=0.6, close_depth=0.15, close_residual_threshold=0.05,
        sobel_edge_threshold=0.007,
    )
    extractor.reference_coefficients = torch.zeros(3, 48, 64)
    standardizer = SimpleNamespace(mean=torch.zeros(extractor.feature_dim),
                                  std=torch.ones(extractor.feature_dim))
    module = _FeatureExport(nn.Identity(), extractor, standardizer)
    exported = _trace_classifier(module, 48, 64)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'features.pt'
        exported.save(str(path))
        restored = torch.jit.load(str(path))
    depth = torch.linspace(0.05, 0.95, 48 * 64).reshape(1, 48, 64)
    rpy, omega = torch.tensor([[0.1, -0.2, 0.3]]), torch.tensor([[0.2, 0.3, 0.1]])
    for partial in (False, True):
        if partial:
            depth[:, ::3] = 0
            depth[:, :, ::4] = 1
        expected = module(depth, rpy, omega)
        torch.testing.assert_close(restored(depth, rpy, omega), expected)
        for name in ('depth_iqr', 'sobel_magnitude_p90', 'horizontal_edge_topk', 'laplacian_p90'):
            assert expected[0, extractor.FEATURE_NAMES.index(name)] > 0


def test_export_validation_rejects_frozen_empty_frame_branch():
    class UnsafeExport(nn.Module):
        def forward(self, depth, rpy, omega):
            if depth.count_nonzero().item() == 0:
                return torch.zeros(1, 4)
            return depth.mean().expand(1, 4)

    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'validation case'):
        _trace_classifier(UnsafeExport(), 48, 64)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_masked_statistics_match_eager_reductions,
        test_feature_export_preserves_statistics_for_changing_masks,
        test_export_validation_rejects_frozen_empty_frame_branch,
    ))
