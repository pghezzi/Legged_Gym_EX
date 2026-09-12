#!/usr/bin/env python3
"""Export single- or multi-FFT Depth-WaQ checkpoints as TorchScript.

Examples:
    python legged_gym/scripts/depth_waq_exporter.py single \
        --checkpoint /path/to/model_10000.pt \
        --args-file /path/to/current_actor_args.pt

    python legged_gym/scripts/depth_waq_exporter.py multi \
        --checkpoint /path/to/base/model.pt \
        --checkpoint /path/to/gap/model.pt \
        --checkpoint /path/to/stairs/model.pt \
        --checkpoint /path/to/pit/model.pt \
        --args-file /path/to/current_actor_args.pt

The export directory contains ``policy.pt``, ``DepthCNN.pt``, ``FeaturesWaQ.pt``,
and ``manifest.json``. In multi mode, ``swap(-1)`` selects the base policy and
``swap(0)``, ``swap(1)``, ... select the FFT specialists.
"""

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SingleDepthCNNExporterWaQ(torch.nn.Module):
    """Depth encoder for a standalone policy; intentionally has no swap/list."""

    def __init__(self, visual_encoder: torch.nn.Module):
        super().__init__()
        self.visual_encoder = copy.deepcopy(visual_encoder)
    
    @torch.jit.export
    def swap(self, index: int):
        pass

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        return self.visual_encoder(depth_image)


class SinglePolicyFeaturesWaQ(torch.nn.Module):
    """Actor/VAE for a standalone policy; intentionally has no swap/list."""

    def __init__(self, actor: torch.nn.Module, vae: torch.nn.Module):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.vae = copy.deepcopy(vae)
    
    @torch.jit.export
    def swap(self, index: int):
        pass

    def forward(
        self,
        observations: torch.Tensor,
        obs_history: torch.Tensor,
        visual_latent: torch.Tensor,
    ) -> torch.Tensor:
        mean_out = self.vae.inference(obs_history)
        return self.actor(torch.cat((observations, mean_out, visual_latent), dim=-1))


class SinglePolicyExporterDepthWaQ(torch.nn.Module):
    """Minimal combined JIT for exactly one Depth-WaQ checkpoint."""

    def __init__(self, actor_critic: torch.nn.Module):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
        self.visual_encoder = copy.deepcopy(actor_critic.visual_encoder)

    def forward(
        self,
        observations: torch.Tensor,
        obs_history: torch.Tensor,
        depth_image: torch.Tensor,
    ) -> torch.Tensor:
        mean_out = self.vae.inference(obs_history)
        visual_latent = self.visual_encoder(depth_image)
        return self.actor(torch.cat((observations, mean_out, visual_latent), dim=-1))

    @torch.jit.export
    def swap(self, index: int):
        pass

    @torch.jit.unused
    def split_cnn(self):
        return (
            SingleDepthCNNExporterWaQ(self.visual_encoder),
            SinglePolicyFeaturesWaQ(self.actor, self.vae),
        )


class DepthCNNExporterWaQ(torch.nn.Module):
    """The depth-encoder half of one or more switchable policies."""

    def __init__(self, visual_encoders: Sequence[torch.nn.Module]):
        super().__init__()
        self.visual_encoders = torch.nn.ModuleList(
            [copy.deepcopy(encoder) for encoder in visual_encoders]
        )
        self.visual_encoder = self.visual_encoders[0]

    @torch.jit.export
    def swap(self, index: int):
        selected = index + 1
        for i, encoder in enumerate(self.visual_encoders):
            if i == selected:
                self.visual_encoder = encoder

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        return self.visual_encoder(depth_image)


class PolicyExporterFeaturesWaQ(torch.nn.Module):
    """The actor/VAE half of one or more switchable policies."""

    def __init__(self, actors: Sequence[torch.nn.Module], vaes: Sequence[torch.nn.Module]):
        super().__init__()
        self.actors = torch.nn.ModuleList([copy.deepcopy(actor) for actor in actors])
        self.vaes = torch.nn.ModuleList([copy.deepcopy(vae) for vae in vaes])
        self.actor = self.actors[0]
        self.vae = self.vaes[0]

    @torch.jit.export
    def swap(self, index: int):
        selected = index + 1
        for i, (actor, vae) in enumerate(zip(self.actors, self.vaes)):
            if i == selected:
                self.actor = actor
                self.vae = vae

    def forward(
        self,
        observations: torch.Tensor,
        obs_history: torch.Tensor,
        visual_latent: torch.Tensor,
    ) -> torch.Tensor:
        mean_out = self.vae.inference(obs_history)
        return self.actor(torch.cat((observations, mean_out, visual_latent), dim=-1))


class PolicyExporterDepthWaQ(torch.nn.Module):
    """Combined Depth-WaQ policy with optional FFT specialist switching."""

    def __init__(self, actor_critics: Sequence[torch.nn.Module]):
        super().__init__()
        if not actor_critics:
            raise ValueError("At least one checkpoint is required.")
        self.actors = torch.nn.ModuleList(
            [copy.deepcopy(actor_critic.actor) for actor_critic in actor_critics]
        )
        self.vaes = torch.nn.ModuleList(
            [copy.deepcopy(actor_critic.vae) for actor_critic in actor_critics]
        )
        self.visual_encoders = torch.nn.ModuleList(
            [copy.deepcopy(actor_critic.visual_encoder) for actor_critic in actor_critics]
        )
        self.actor = self.actors[0]
        self.vae = self.vaes[0]
        self.visual_encoder = self.visual_encoders[0]
        self.num_of_loras = len(self.actors) - 1

    @torch.jit.export
    def swap(self, index: int):
        selected = index + 1
        for i, (actor, vae, encoder) in enumerate(
            zip(self.actors, self.vaes, self.visual_encoders)
        ):
            if i == selected:
                self.actor = actor
                self.vae = vae
                self.visual_encoder = encoder

    def forward(
        self,
        observations: torch.Tensor,
        obs_history: torch.Tensor,
        depth_image: torch.Tensor,
    ) -> torch.Tensor:
        mean_out = self.vae.inference(obs_history)
        visual_latent = self.visual_encoder(depth_image)
        return self.actor(torch.cat((observations, mean_out, visual_latent), dim=-1))

    @torch.jit.unused
    def split_cnn(self):
        return (
            DepthCNNExporterWaQ(self.visual_encoders),
            PolicyExporterFeaturesWaQ(self.actors, self.vaes),
        )


def load_checkpoint(checkpoint_file: Path, args_file: Path) -> torch.nn.Module:
    """Load one Depth-WaQ actor-critic checkpoint on CPU."""
    from rsl_rl.modules import ActorCriticDreamWaQDepth

    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    if not args_file.is_file():
        raise FileNotFoundError(f"Actor arguments do not exist: {args_file}")

    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    saved_args = torch.load(args_file, map_location="cpu")
    model = ActorCriticDreamWaQDepth(**saved_args["args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_args_files(checkpoints: Sequence[Path], args_files: Sequence[Path]) -> List[Path]:
    if len(args_files) == 1:
        return [args_files[0]] * len(checkpoints)
    if len(args_files) == len(checkpoints):
        return list(args_files)
    raise ValueError(
        "Pass exactly one --args-file (shared by every checkpoint), or one "
        "--args-file for each --checkpoint."
    )


def write_manifest(
    output_dir: Path,
    mode: str,
    checkpoints: Sequence[Path],
    args_files: Sequence[Path],
) -> None:
    """Record the deployment contract without embedding it in TorchScript."""
    manifest = {
        "format": "depth-waq-jit-v1",
        "mode": mode,
        "policy_count": len(checkpoints),
        "specialist_count": len(checkpoints) - 1 if mode == "multi" else 0,
        "artifacts": {
            "combined": "policy.pt",
            "depth_encoder": "DepthCNN.pt",
            "actor": "FeaturesWaQ.pt",
        },
        "inputs": {
            "combined": ["observations", "obs_history", "depth_image"],
            "depth_encoder": ["depth_image"],
            "actor": ["observations", "obs_history", "visual_latent"],
        },
        "switching": (
            {"base_index": -1, "specialist_indices": list(range(len(checkpoints) - 1))}
            if mode == "multi"
            else None
        ),
        "sources": [
            {"checkpoint": str(checkpoint.resolve()), "args_file": str(args_file.resolve())}
            for checkpoint, args_file in zip(checkpoints, args_files)
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def export_policy(
    actor_critics: Sequence[torch.nn.Module],
    mode: str,
    output_dir: Path,
    checkpoints: Sequence[Path],
    args_files: Sequence[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "single":
        exporter = SinglePolicyExporterDepthWaQ(actor_critics[0]).cpu().eval()
    else:
        exporter = PolicyExporterDepthWaQ(actor_critics).cpu().eval()
    torch.jit.script(exporter).save(str(output_dir / "policy.pt"))

    cnn, features = exporter.split_cnn()
    torch.jit.script(cnn.cpu().eval()).save(str(output_dir / "DepthCNN.pt"))
    torch.jit.script(features.cpu().eval()).save(str(output_dir / "FeaturesWaQ.pt"))
    write_manifest(output_dir, mode, checkpoints, args_files)

    print(f"Exported {len(actor_critics)} policy/policies to: {output_dir}")
    print(f"  combined: {output_dir / 'policy.pt'}")
    print(f"  split CNN: {output_dir / 'DepthCNN.pt'}")
    print(f"  split actor: {output_dir / 'FeaturesWaQ.pt'}")
    print(f"  manifest: {output_dir / 'manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("single", "multi"), help="JIT policy type to export")
    parser.add_argument(
        "--checkpoint", "-c", action="append", required=True, type=Path,
        help="Checkpoint path. Repeat for every base/specialist policy in multi mode.",
    )
    parser.add_argument(
        "--args-file", "-a", action="append", required=True, type=Path,
        help="current_actor_args.pt path. Supply once to share it, or once per checkpoint.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        help="Directory for policy.pt, DepthCNN.pt, and FeaturesWaQ.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "single" and len(args.checkpoint) != 1:
        raise ValueError("single mode requires exactly one --checkpoint.")
    if args.mode == "multi" and len(args.checkpoint) < 2:
        raise ValueError("multi mode requires a base checkpoint and at least one specialist.")

    args_files = resolve_args_files(args.checkpoint, args.args_file)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "exported" / f"{timestamp}_{args.mode}_fft"

    actor_critics = [
        load_checkpoint(checkpoint, args_file)
        for checkpoint, args_file in zip(args.checkpoint, args_files)
    ]
    export_policy(actor_critics, args.mode, output_dir, args.checkpoint, args_files)


if __name__ == "__main__":
    main()
