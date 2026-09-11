"""Export a trained DepthWaQ terrain classifier as a self-contained TorchScript bundle.

The bundle deliberately exposes the same input contract for both architectures:
``(depth[B,H,W], orientation_rpy[B,3], angular_velocity[B,3]) -> logits[B,C]``.
This lets the robot deployment select feature/raw models without importing the
training package or reconstructing Python checkpoints.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import torch
from torch import nn

from legged_gym.scripts.depth_data_pipeline.train_feature_nn import TerrainDepthFeatureClassifierNN
from legged_gym.scripts.depth_data_pipeline.train_raw_depth_nn import TerrainDepthClassifierNN
from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    FeatureStandardizer,
    NeuralClassifierAdapter,
)


class _RawDepthExport(nn.Module):
    def __init__(self, model: nn.Module, robot_state_dim: int):
        super().__init__()
        self.model = model.eval()
        self.robot_state_dim = int(robot_state_dim)

    def forward(self, depth, orientation_rpy, angular_velocity):
        if self.robot_state_dim:
            state = torch.cat((orientation_rpy[:, :2], angular_velocity[:, :3]), dim=1)
            depth = torch.cat((depth.flatten(1), state), dim=1)
        else:
            depth = depth.unsqueeze(1)
        return self.model(depth)


class _FeatureExport(nn.Module):
    """Trace-only wrapper; extractor operations are embedded in the result."""
    def __init__(self, model, extractor, standardizer):
        super().__init__()
        self.model = model.eval()
        self.extractor = extractor
        self.register_buffer("mean", standardizer.mean.detach().float())
        self.register_buffer("std", standardizer.std.detach().float())

    def forward(self, depth, orientation_rpy, angular_velocity):
        features = self.extractor.extract_batch(depth, orientation_rpy, angular_velocity)
        return self.model((features - self.mean) / self.std)


def _model_from_artifacts(architecture: str, checkpoint: Path, args_path: Path):
    model_args = dict(torch.load(args_path, map_location="cpu", weights_only=False))
    cls_name = model_args.pop("cls")
    expected = "TerrainDepthClassifierNN" if architecture == "raw_depth_nn" else "TerrainDepthFeatureClassifierNN"
    if cls_name != expected:
        raise ValueError(f"Expected {expected}, found {cls_name} in {args_path}")
    cls = TerrainDepthClassifierNN if architecture == "raw_depth_nn" else TerrainDepthFeatureClassifierNN
    model = cls(**model_args)
    adapter = NeuralClassifierAdapter.load(checkpoint, model, device="cpu")
    return adapter, model_args


def export_classifier(*, architecture, checkpoint, model_args_path, output,
                      extractor_path=None, standardizer_path=None,
                      selector_mode="instantaneous", ema_alpha=0.6,
                      change_patience=1, stable_stay=0.9):
    """Export one checkpoint and return its deployment manifest."""
    checkpoint, model_args_path, output = map(Path, (checkpoint, model_args_path, output))
    adapter, model_args = _model_from_artifacts(architecture, checkpoint, model_args_path)
    if architecture == "raw_depth_nn":
        module = _RawDepthExport(adapter.model, model_args.get("robot_state_dim", 0))
        height, width = model_args["depth_image_resolution"]
    else:
        if extractor_path is None or standardizer_path is None:
            raise ValueError("feature_nn requires extractor_path and standardizer_path")
        extractor = SobelDepthTerrainFeatureExtractor.load(extractor_path, device="cpu")
        standardizer = FeatureStandardizer.load(standardizer_path)
        module = _FeatureExport(adapter.model, extractor, standardizer)
        height, width = 48, 64
    example = (torch.zeros(1, height, width), torch.zeros(1, 3), torch.zeros(1, 3))
    with torch.inference_mode(), warnings.catch_warnings():
        # The engineered extractor uses Python shape validation and B=1
        # per-frame statistics. Those appear as TracerWarnings even though
        # this deployment artifact deliberately has a fixed [1, 48, 64]
        # contract. Keep trace checking below; silence only this expected
        # warning class, not ordinary export/runtime failures.
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        exported = torch.jit.trace(module, example, strict=False)
        reference, actual = module(*example), exported(*example)
    if not torch.allclose(reference, actual, atol=1e-5, rtol=1e-5):
        raise RuntimeError("TorchScript export does not match the source classifier")
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(output))
    manifest = {
        "format": "depthwaq-terrain-selector-v1", "architecture": architecture,
        "class_ids": [str(value) for value in adapter.class_ids],
        "input_shape": [int(height), int(width)], "selector_mode": selector_mode,
        "ema": {"ema_alpha": float(ema_alpha), "change_patience": int(change_patience)},
        "bayes": {"stable_stay": float(stable_stay)},
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("raw_depth_nn", "feature_nn"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-args", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, help="Required for feature_nn")
    parser.add_argument("--standardizer", type=Path, help="Required for feature_nn")
    parser.add_argument("--selector-mode", choices=("instantaneous", "ema", "bayes"), default="instantaneous")
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--change-patience", type=int, default=1)
    parser.add_argument("--stable-stay", type=float, default=0.9)
    args = parser.parse_args()

    if args.architecture == "feature_nn" and (args.extractor is None or args.standardizer is None):
        parser.error("feature_nn requires --extractor and --standardizer")
    export_classifier(
        architecture=args.architecture, checkpoint=args.checkpoint, model_args_path=args.model_args,
        output=args.output, extractor_path=args.extractor, standardizer_path=args.standardizer,
        selector_mode=args.selector_mode, ema_alpha=args.ema_alpha,
        change_patience=args.change_patience, stable_stay=args.stable_stay)
    print(f"Exported {args.architecture} terrain selector to {args.output}")


if __name__ == "__main__":
    main()
