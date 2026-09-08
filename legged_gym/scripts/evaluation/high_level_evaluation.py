"""Evaluate terrain classification, skill selection, and multi-terrain navigation."""
from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.utils.terrain_vars import TERRAIN_KEYS

import argparse
import copy
import csv
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch



CANONICAL_CLASSES = ["rough", "gap", "pit", "stairs"]
SKILL_SPEEDS = {"rough": 0.8, "gap": 1.5, "pit": 1.2, "stairs": 1.2}
SKILL_LORA = {"rough": -1, "gap": 0, "stairs": 1, "pit": 2}
METHOD_TO_APPROACH = {
    "feature_nn_deterministic": ("feature_nn", "deterministic"),
    "raw_depth_nn_deterministic": ("raw_depth_nn", "deterministic"),
    "feature_nn_mc": ("feature_nn", "mc"),
    "raw_depth_nn_mc": ("raw_depth_nn", "mc"),
}
PAPER_METHODS = (
    "oracle", "feature_instantaneous", "feature_ema", "feature_bayes",
    "raw_depth_instantaneous", "raw_depth_ema", "raw_depth_bayes", "distilled",
)
PAPER_METHOD_SPECS = {
    "feature_instantaneous": ("feature_nn", "instantaneous"),
    "feature_ema": ("feature_nn", "ema"),
    "feature_bayes": ("feature_nn", "bayes"),
    "raw_depth_instantaneous": ("raw_depth_nn", "instantaneous"),
    "raw_depth_ema": ("raw_depth_nn", "ema"),
    "raw_depth_bayes": ("raw_depth_nn", "bayes"),
}
DIFFICULTY_LEVELS = {
    "easy": {"normalized": 0.25, "pit_depth": 0.20,
             "stair_magnitude": 0.10, "gap_width": 0.40},
    "nominal": {"normalized": 0.50, "pit_depth": 0.35,
                "stair_magnitude": 0.20, "gap_width": 0.60},
    "hard": {"normalized": 0.75, "pit_depth": 0.50,
             "stair_magnitude": 0.30, "gap_width": 0.80},
}

# Keep evaluation randomization identical to low_level_evaluation.py.
EVAL_DOMAIN_RANDOMIZATION_RANGES = {
    "friction_range": [1.0, 1.0],
    "com_pos_x_range": [-0.0, 0.0],
    "com_pos_y_range": [-0.0, 0.0],
    "com_pos_z_range": [-0.0, 0.0],
    "kp_range": [1.0, 1.0],
    "kd_range": [1.0, 1.0],
    "motor_strength_range": [1.0, 1.0],
    "ctrl_delay_step_range": [0, 0],
    "joint_armature_range": [0.020, 0.020],
    "joint_friction_range": [0.015, 0.015],
    "joint_damping_range": [0.275, 0.275],
    "camera_com_displacement_range": [0.0, 0.0, 0.0],
    "camera_euler_offset_range": [0.0, 0.0, 0.0],
}
EVAL_DOMAIN_RANDOMIZATION_ENABLED = {
    "randomize_friction": True,
    "randomize_restitution": False,
    "randomize_base_mass": False,
    "randomize_com_displacement": False,
    "randomize_ctrl_delay": False,
    "randomize_pd_gain": False,
    "randomize_motor_strength": False,
    "randomize_joint_armature": True,
    "randomize_joint_friction": True,
    "randomize_joint_damping": True,
    "push_robots": False,
    "push_links": False,
    "randomize_camera_pos": False,
    "randomize_camera_euler": False,
}


def configure_runtime_device(args):
    """Normalize the evaluator device while retaining physical-GPU masking."""
    if args.cpu:
        args.gpu = "cpu"
        args.device = "cpu"
        return args
    requested = str(args.gpu).lower()
    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested == "cuda":
        requested = "cuda:0"
    if not requested.startswith("cuda:") or not requested.split(":", 1)[1].isdigit():
        raise ValueError("--gpu must be cuda, cuda:N, or a numeric GPU index")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_ids = [value.strip() for value in visible.split(",") if value.strip()]
        index = int(requested.split(":", 1)[1])
        if index < len(visible_ids):
            requested = f"cuda:{index}"
        elif str(index) in visible_ids:
            requested = f"cuda:{visible_ids.index(str(index))}"
        else:
            raise ValueError(f"GPU {index} is unavailable under CUDA_VISIBLE_DEVICES={visible}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested.split(":", 1)[1]
        requested = "cuda:0"
    args.gpu = requested
    args.device = requested
    return args


def init_genesis(args, gs):
    configure_runtime_device(args)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    if not args.cpu:
        torch.cuda.set_device(torch.device(args.gpu))


def canonicalize_label(label):
    if torch.is_tensor(label):
        label = label.item()
    if isinstance(label, (int, np.integer)):
        if not 0 <= int(label) < len(TERRAIN_KEYS):
            raise ValueError(f"Unknown terrain label id {label}")
        label = TERRAIN_KEYS[int(label)]
    value = str(label).lower()
    if value in ("random_uniform", "pyramid_sloped", "rough", "baseline"):
        return "rough"
    if value in ("stairs", "upwards_stairs"):
        return "stairs"
    if value in ("pit", "center_platform", "climb"):
        return "pit"
    if value in ("gap", "leap"):
        return "gap"
    raise ValueError(f"Cannot map terrain label {label!r} to a skill class")


def label_to_lora(label):
    return SKILL_LORA[canonicalize_label(label)]


class PerTerrainPolicySet:
    """Preload one independently swapped policy per skill for batched dispatch."""

    def __init__(self, jit_path, device):
        self.policies = {}
        for lora_id in sorted(set(SKILL_LORA.values())):
            policy = torch.jit.load(jit_path, map_location=device)
            policy.swap(lora_id)
            self.policies[lora_id] = policy

    def act(self, obs_buf, obs_history, depth, assigned_lora):
        actions = None
        for lora_id, policy in self.policies.items():
            indices = (assigned_lora == lora_id).nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            sub_actions = policy(
                obs_buf[indices].detach(), obs_history[indices].detach(), depth[indices].detach()
            )
            if actions is None:
                actions = torch.zeros(
                    obs_buf.shape[0], sub_actions.shape[-1],
                    device=sub_actions.device, dtype=sub_actions.dtype,
                )
            actions[indices] = sub_actions
        if actions is None:
            raise RuntimeError("No environments were assigned to a known skill policy")
        return actions


def get_viewed_terrain_idx(env, look_ahead_frac=0.75):
    """Return the existing look-ahead terrain-cell lookup used by the oracle."""
    look_ahead_dist = 4.0 * look_ahead_frac
    base_pos = env.simulator.base_pos
    heading = env.heading.to(base_pos.device)
    look_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    look_point = base_pos[:, :2] + look_dir * look_ahead_dist
    query_point = torch.where(
        (base_pos[:, 2] < 0.25).unsqueeze(-1), base_pos[:, :2], look_point
    )
    origins = env.simulator._terrain_origins.to(base_pos.device)
    num_cols = origins.shape[1]
    distances = torch.cdist(query_point, origins[..., :2].reshape(-1, 2))
    indices = torch.argmin(distances, dim=-1)
    row_col = torch.stack([indices // num_cols, indices % num_cols], dim=-1)
    return indices, row_col


def get_ground_truth_labels(env, look_ahead_frac=0.75):
    """Decode raw labels solely from the environment terrain-label grid."""
    _, row_col = get_viewed_terrain_idx(env, look_ahead_frac)
    row_col = row_col.detach().cpu()
    label_ids = env.simulator._terrain.labels[row_col[:, 0], row_col[:, 1]]
    raw = [TERRAIN_KEYS[int(label_id)] for label_id in label_ids]
    return raw, [canonicalize_label(label) for label in raw]


def get_ground_truth_label_for_env(env, env_id, look_ahead_frac=0.75):
    """Single-environment equivalent used only by the batch-1 latency benchmark."""
    look_ahead_dist = 4.0 * look_ahead_frac
    base_pos = env.simulator.base_pos[env_id:env_id + 1]
    heading = env.heading[env_id:env_id + 1].to(base_pos.device)
    look_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    look_point = base_pos[:, :2] + look_dir * look_ahead_dist
    query_point = torch.where(
        (base_pos[:, 2] < 0.25).unsqueeze(-1), base_pos[:, :2], look_point)
    origins = env.simulator._terrain_origins.to(base_pos.device)
    num_cols = origins.shape[1]
    index = torch.argmin(
        torch.cdist(query_point, origins[..., :2].reshape(-1, 2)), dim=-1)[0]
    row, column = int((index // num_cols).item()), int((index % num_cols).item())
    raw = TERRAIN_KEYS[int(env.simulator._terrain.labels[row, column])]
    return raw, canonicalize_label(raw)


class RuntimeClassifier:
    def __init__(self, classifier, approach, deployment, extractor=None, standardizer=None):
        self.classifier = classifier
        self.approach = approach
        self.extractor = extractor
        self.standardizer = standardizer
        self.class_ids = list(classifier.class_ids)
        self.inference_mode = deployment["inference_mode"]
        self.mc_samples = int(deployment.get("mc_samples", 1))
        self.filter_temperature = float(deployment["T_filter"])

    def predict(self, depth, euler, angular_velocity):
        if self.approach == "raw_depth_nn":
            from legged_gym.scripts.depth_data_pipeline.util_func import pack_raw_depth_state_inputs
            if getattr(self.classifier.model, "robot_state_dim", 0):
                inputs = pack_raw_depth_state_inputs(depth, euler, angular_velocity)
            else:
                inputs = depth.unsqueeze(1) if depth.ndim == 3 else depth
        else:
            inputs = self.extractor.extract_batch(depth, euler, angular_velocity)
            if self.standardizer is not None:
                inputs = self.standardizer.transform(inputs)
        from legged_gym.scripts.depth_data_pipeline.util_func import (
            collect_neural_logits_batched, probabilities_and_uncertainty,
        )
        logits, _ = collect_neural_logits_batched(
            self.classifier, inputs, batch_size=inputs.shape[0],
            mc_samples=self.mc_samples, mc_dropout=self.inference_mode == "mc",
            cache_device=self.classifier.device)
        q_filter, _, _, _ = probabilities_and_uncertainty(logits, self.filter_temperature)
        q_event, _, _, mi = probabilities_and_uncertainty(logits, 1.0)
        return q_filter.detach(), q_event.detach(), mi.detach()

    def predict_deterministic(self, depth, euler, angular_velocity, temperature=1.0):
        """Return native-order logits/probabilities with one deterministic batch forward."""
        if self.approach == "raw_depth_nn":
            from legged_gym.scripts.depth_data_pipeline.util_func import pack_raw_depth_state_inputs
            inputs = (pack_raw_depth_state_inputs(depth, euler, angular_velocity)
                      if getattr(self.classifier.model, "robot_state_dim", 0)
                      else depth.unsqueeze(1) if depth.ndim == 3 else depth)
        else:
            inputs = self.extractor.extract_batch(depth, euler, angular_velocity)
            if self.standardizer is not None:
                inputs = self.standardizer.transform(inputs)
        self.classifier.model.eval()
        with torch.inference_mode():
            logits = self.classifier.model(self.classifier._prepare(inputs))
        probabilities = torch.softmax(logits / float(temperature), dim=1)
        return logits.detach(), probabilities.detach()

    def predict_deterministic_timed(self, depth, euler, angular_velocity, temperature=1.0):
        """Deterministic inference with synchronized, non-overlapping stage timings."""
        timings = {
            "depth_preprocess_ms": 0.0, "oracle_lookup_ms": 0.0,
            "feature_extraction_ms": 0.0, "standardization_ms": 0.0,
            "classifier_ms": 0.0,
        }
        if self.approach == "raw_depth_nn":
            from legged_gym.scripts.depth_data_pipeline.util_func import pack_raw_depth_state_inputs
            started = _timing_start(self.classifier.device)
            inputs = (pack_raw_depth_state_inputs(depth, euler, angular_velocity)
                      if getattr(self.classifier.model, "robot_state_dim", 0)
                      else depth.unsqueeze(1) if depth.ndim == 3 else depth)
            timings["depth_preprocess_ms"] = _timing_stop(started, self.classifier.device)
        else:
            started = _timing_start(self.classifier.device)
            inputs = self.extractor.extract_batch(depth, euler, angular_velocity)
            timings["feature_extraction_ms"] = _timing_stop(started, self.classifier.device)
            if self.standardizer is not None:
                started = _timing_start(self.classifier.device)
                inputs = self.standardizer.transform(inputs)
                timings["standardization_ms"] = _timing_stop(started, self.classifier.device)
        self.classifier.model.eval()
        started = _timing_start(self.classifier.device)
        with torch.inference_mode():
            logits = self.classifier.model(self.classifier._prepare(inputs))
            probabilities = torch.softmax(logits / float(temperature), dim=1)
        timings["classifier_ms"] = _timing_stop(started, self.classifier.device)
        return logits.detach(), probabilities.detach(), timings


def _classifier_artifacts(classifier_dir):
    root = Path(classifier_dir)
    classifier_file = root / "classifier.pt"
    if not classifier_file.is_file():
        raise FileNotFoundError(f"Missing classifier checkpoint: {classifier_file}")
    return root, classifier_file


def _feature_parts(root, device):
    from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import (
        SobelDepthTerrainFeatureExtractor,
    )
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        FeatureStandardizer,
    )

    extractor_file = root / "extractor.pt"
    if not extractor_file.is_file():
        raise FileNotFoundError(f"Missing engineered-feature extractor: {extractor_file}")
    extractor = SobelDepthTerrainFeatureExtractor.load(extractor_file, device=device)
    standardizer_file = root / "standardizer.pt"
    standardizer = (
        FeatureStandardizer.load(standardizer_file) if standardizer_file.is_file() else None
    )
    return extractor, standardizer


def _load_neural(suite_dir, approach, device):
    from legged_gym.scripts.depth_data_pipeline.train_feature_nn import TerrainDepthFeatureClassifierNN
    from legged_gym.scripts.depth_data_pipeline.train_raw_depth_nn import TerrainDepthClassifierNN
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        NeuralClassifierAdapter,
    )
    architecture, mode = METHOD_TO_APPROACH[approach]
    root = Path(suite_dir)
    run_root = root / architecture if (root / architecture).is_dir() else root
    deployment_file = run_root / f"deployment_{mode}.json"
    if not deployment_file.is_file():
        raise FileNotFoundError(f"Missing selected deployment manifest: {deployment_file}")
    with deployment_file.open(encoding="utf-8") as stream:
        deployment = json.load(stream)
    artifact = lambda value: Path(value) if Path(value).is_absolute() else run_root / value
    checkpoint, args_file = artifact(deployment["model_path"]), artifact(deployment["model_args_path"])
    raw_depth = architecture == "raw_depth_nn"
    if not args_file.is_file():
        raise FileNotFoundError(f"Missing neural model arguments: {args_file}")
    model_args = dict(torch.load(args_file, map_location="cpu", weights_only=False))
    class_name = model_args.pop("cls")
    model_types = {
        "TerrainDepthFeatureClassifierNN": TerrainDepthFeatureClassifierNN,
        "TerrainDepthClassifierNN": TerrainDepthClassifierNN,
    }
    expected = "TerrainDepthClassifierNN" if raw_depth else "TerrainDepthFeatureClassifierNN"
    if class_name != expected:
        raise ValueError(f"Expected {expected} model arguments, found {class_name}")
    model = model_types[class_name](**model_args)
    classifier = NeuralClassifierAdapter.load(checkpoint, model, device=device)
    if raw_depth:
        runtime = RuntimeClassifier(classifier, architecture, deployment)
    else:
        extractor_path = artifact(deployment["extractor_path"])
        standardizer_path = artifact(deployment["standardizer_path"])
        from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor
        from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import FeatureStandardizer
        runtime = RuntimeClassifier(
            classifier, architecture, deployment,
            SobelDepthTerrainFeatureExtractor.load(extractor_path, device=device),
            FeatureStandardizer.load(standardizer_path))
    return runtime, deployment


def _load_paper_classifier(paper_offline_dir, architecture, seed, device):
    """Load one frozen Experiment 1--2 classifier and its preprocessing artifacts."""
    from legged_gym.scripts.depth_data_pipeline.train_feature_nn import TerrainDepthFeatureClassifierNN
    from legged_gym.scripts.depth_data_pipeline.train_raw_depth_nn import TerrainDepthClassifierNN
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        FeatureStandardizer, NeuralClassifierAdapter,
    )
    from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import (
        SobelDepthTerrainFeatureExtractor,
    )

    root = Path(paper_offline_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if int(seed) not in manifest.get("model_seeds", []):
        raise ValueError(f"classifier seed {seed} is absent from {manifest_path}")
    expected_ema = {"ema_alpha": 0.6, "change_patience": 1}
    expected_bayes = {
        "T_filter": 1.0, "stable_stay": 0.90, "evidence_power": 1.0,
        "observation_mix": 0.0, "adaptive_evidence": False,
    }
    expected_models = {
        "feature_nn": {"dropout_p": 0.0, "weight_decay": 1e-5},
        "raw_depth_nn": {"dropout_p": 0.20, "weight_decay": 1e-5},
    }
    if manifest.get("fixed_model_configurations", {}).get(architecture) != expected_models[architecture]:
        raise ValueError(f"paper offline {architecture} configuration is not the frozen model")
    if manifest.get("fixed_ema_configuration") != expected_ema:
        raise ValueError("paper offline EMA configuration does not match the frozen evaluation")
    for key, expected in expected_bayes.items():
        if manifest.get("fixed_bayes_configuration", {}).get(key) != expected:
            raise ValueError(f"paper offline Bayes parameter {key} is not {expected!r}")

    artifact_root = root / "artifacts" / architecture
    checkpoint = artifact_root / f"seed_{seed}" / "classifier.pt"
    args_path = artifact_root / f"seed_{seed}" / "nn_model_args.pt"
    if not checkpoint.is_file() or not args_path.is_file():
        raise FileNotFoundError(f"missing seeded classifier artifacts below {artifact_root}")
    model_args = dict(torch.load(args_path, map_location="cpu", weights_only=False))
    class_name = model_args.pop("cls")
    model_types = {
        "TerrainDepthFeatureClassifierNN": TerrainDepthFeatureClassifierNN,
        "TerrainDepthClassifierNN": TerrainDepthClassifierNN,
    }
    model = model_types[class_name](**model_args)
    classifier = NeuralClassifierAdapter.load(checkpoint, model, device=device)
    if list(classifier.class_ids) != list(manifest.get("class_ordering", classifier.class_ids)):
        raise ValueError("seeded classifier class order differs from the paper manifest")
    deployment = {
        "architecture": architecture, "inference_mode": "deterministic", "mc_samples": 1,
        "dropout_p": model_args.get("dropout_p", 0.0),
        "model_path": str(checkpoint), "model_args_path": str(args_path),
        **manifest["fixed_bayes_configuration"],
        "ema": manifest["fixed_ema_configuration"],
    }
    if architecture == "feature_nn":
        extractor_path = artifact_root / "extractor.pt"
        standardizer_path = artifact_root / "standardizer.pt"
        runtime = RuntimeClassifier(
            classifier, architecture, deployment,
            SobelDepthTerrainFeatureExtractor.load(extractor_path, device=device),
            FeatureStandardizer.load(standardizer_path))
        deployment.update(extractor_path=str(extractor_path),
                          standardizer_path=str(standardizer_path))
    else:
        runtime = RuntimeClassifier(classifier, architecture, deployment)
    return runtime, deployment, manifest


CLASSIFIER_LOADERS = {name: _load_neural for name in METHOD_TO_APPROACH}


def resolve_classifier_approach(args):
    return args.classifier_approach


def _load_bayes_template(path, device):
    from legged_gym.scripts.depth_data_pipeline.sequential_terrain_filter_extensions import CandidateReleaseBayesianTerrainFilter
    if not path.is_file():
        raise FileNotFoundError(f"Missing Bayes filter checkpoint: {path}")
    result = CandidateReleaseBayesianTerrainFilter.load(path, device=device)
    result.reset()
    return result


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate JIT skill selection and terrain classification on fixed tracks"
    )
    parser.add_argument("--task", default="go2_depth_waq_lora")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--episodes_per_track", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=100000, help="safety cap only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--difficulty_level", choices=tuple(DIFFICULTY_LEVELS),
                        help="named frozen paper difficulty (overrides --difficulty)")
    parser.add_argument("--finish_margin", type=float, default=0.25)
    parser.add_argument("--look_ahead_frac", type=float, default=0)
    parser.add_argument("--classify_every", type=int, default=5)
    parser.add_argument("--latency_warmup_updates", type=int, default=20)
    parser.add_argument(
        "--selector_mode", choices=("oracle", "instantaneous", "bayes", "baseline"),
        default="bayes",
    )
    parser.add_argument(
        "--classifier_approach", choices=tuple(CLASSIFIER_LOADERS)
    )
    parser.add_argument("--classifier_dir", "--classifier_suite", dest="classifier_dir")
    parser.add_argument("--jit", "--policy_jit", dest="jit")
    parser.add_argument("--paper_method", choices=PAPER_METHODS,
                        help="run one frozen closed-loop paper condition")
    parser.add_argument("--paper_offline_dir",
                        help="output directory from evaluate_paper_offline_experiments_1_2.py")
    parser.add_argument("--classifier_seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--distilled_jit",
                        help="exported unified depth-DreamWaQ policy for the distilled method")
    parser.add_argument("--fixed_forward_command", type=float, default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--result_name", default=None,
                        help="stable output stem (used by resumable paper orchestration)")
    
    #not used args
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Optional motion file",
    )
    parser.add_argument(
        "--num_student",
        type=int,
        default=None,
        help="Number of student environments/agents",
    )
    args = parser.parse_args()
    if args.cpu:
        args.gpu = "cpu"
    if args.num_envs < 10 or args.num_envs % 10:
        parser.error("--num_envs must be 10 or a balanced multiple of 10")
    if (args.episodes_per_track < 1 or args.num_steps < 1 or args.classify_every < 1
            or args.latency_warmup_updates < 0):
        parser.error("episode quota, step cap, and classify interval must be positive")
    if not 0.0 <= args.difficulty <= 1.0:
        parser.error("--difficulty must be in [0,1]")
    if not 0.0 <= args.look_ahead_frac <= 1.0:
        parser.error("--look_ahead_frac must be in [0,1]")
    if args.paper_method:
        if args.difficulty_level is None:
            parser.error("--paper_method requires --difficulty_level")
        if args.fixed_forward_command is None:
            args.fixed_forward_command = 1.0
        if args.paper_method in PAPER_METHOD_SPECS:
            if args.paper_offline_dir is None or args.classifier_seed is None:
                parser.error("learned paper methods require --paper_offline_dir and --classifier_seed")
            if args.jit is None:
                parser.error("learned paper methods require --jit")
        elif args.paper_method == "oracle" and args.jit is None:
            parser.error("oracle requires --jit")
        elif args.paper_method == "distilled" and args.distilled_jit is None:
            parser.error("distilled requires --distilled_jit")
    elif args.classifier_approach is None or args.classifier_dir is None or args.jit is None:
        parser.error("legacy mode requires --classifier_approach, --classifier_dir, and --jit")
    return args


def _eval_terrain_value(value, difficulty):
    return float(eval(str(value), {"__builtins__": {}}, {"np": np, "difficulty": difficulty}))


def make_track_layout(env_cfg, args):
    difficulty_level = getattr(args, "difficulty_level", None)
    level = DIFFICULTY_LEVELS.get(difficulty_level)
    difficulty = level["normalized"] if level is not None else args.difficulty
    gap_size = level["gap_width"] if level is not None else 0.30 + 0.70 * difficulty
    pit_depth = level["pit_depth"] if level is not None else 0.25 + 0.25 * difficulty
    stair_height = (level["stair_magnitude"] if level is not None
                    else 0.10 + 0.30 * difficulty)
    rough_cfg = env_cfg.terrain.terrain_curriculum_difficulty["random_uniform_params"]
    rough_values = {
        key: _eval_terrain_value(value, difficulty) for key, value in rough_cfg.items()
    }
    # The LoRA rough-terrain training range is [-0.12, 0.12].  Its endpoints
    # are constants rather than difficulty expressions, so use interior
    # 25/50/75% amplitudes for the named paper levels.
    if level is not None and "difficulty" not in str(rough_cfg.get("min_height", "")):
        rough_values["min_height"] = -abs(float(rough_cfg["min_height"])) * difficulty
    if level is not None and "difficulty" not in str(rough_cfg.get("max_height", "")):
        rough_values["max_height"] = abs(float(rough_cfg["max_height"])) * difficulty
    definitions = {
        "random_uniform": {
            "type": "terrain_utils.random_uniform_terrain", **rough_values,
        },
        "gap": {
            "type": "terrain_utils.gap_terrain", "gap_size": gap_size,
            "platform_size": env_cfg.terrain.platform_size,
        },
        "pit": {
            "type": "terrain_utils.pit_terrain", "depth": pit_depth,
            "platform_size": env_cfg.terrain.platform_size,
        },
        "upwards_stairs": {
            "type": "terrain_utils.pyramid_stairs_terrain", "step_width": 0.4,
            "step_height": stair_height, "platform_size": env_cfg.terrain.platform_size,
        },
        "stairs": {
            "type": "terrain_utils.pyramid_stairs_terrain", "step_width": 0.4,
            "step_height": -stair_height, "platform_size": env_cfg.terrain.platform_size,
        },
    }
    rng = random.Random(args.seed)
    columns = []
    for column in range(10):
        sequence = list(definitions)
        rng.shuffle(sequence)
        columns.append(sequence)
    terrain_map = [copy.deepcopy(definitions[columns[col][row]]) for row in range(5) for col in range(10)]
    logged = []
    for column, sequence in enumerate(columns):
        logged.append({
            "track_id": column,
            "sequence": sequence,
            "cells": [
                {"row": row, "raw_label": name,
                 "parameters": {k: v for k, v in definitions[name].items() if k != "type"}}
                for row, name in enumerate(sequence)
            ],
        })
    args.resolved_difficulty = {
        "name": difficulty_level, "normalized": difficulty,
        "pit_depth": pit_depth, "stair_magnitude": stair_height,
        "gap_width": gap_size, "rough_parameters": rough_values,
    }
    return terrain_map, logged


def override_configs_multiterrain(env_cfg, args):
    env_cfg.seed = args.seed
    env_cfg.env.num_envs = args.num_envs
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = args.num_envs
    if hasattr(env_cfg.viewer, "rendered_envs_idx"):
        env_cfg.viewer.rendered_envs_idx = list(range(min(args.num_envs, 10)))
    env_cfg.init_state.yaw_random_scale = 0.0
    env_cfg.commands.curriculum = False
    if hasattr(env_cfg.commands, "custom_command_curriculum"):
        env_cfg.commands.custom_command_curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.resampling_time = 1.0e9
    env_cfg.commands.ranges.lin_vel_x = [0.8, 1.5]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    
    env_cfg.asset.terminate_after_contacts_on = []

    for name, value in EVAL_DOMAIN_RANDOMIZATION_RANGES.items():
        setattr(env_cfg.domain_rand, name, list(value))
    # The mixed tracks include the obstacle skills, matching the low-level
    # evaluator's non-rough branch. These are inert when their flags are off.
    env_cfg.domain_rand.added_mass_range = [-1.0, 2.0]
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.max_push_vel_xy = 0.5
    for name, enabled in EVAL_DOMAIN_RANDOMIZATION_ENABLED.items():
        setattr(env_cfg.domain_rand, name, enabled)

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.custom_selected = True
    terrain_map, args.track_layout = make_track_layout(env_cfg, args)
    env_cfg.terrain.terrain_map = terrain_map
    if not 0.0 <= args.finish_margin < env_cfg.terrain.terrain_length:
        raise ValueError("finish margin must be smaller than one sub-terrain length")


def assign_tracks_and_reset(env, track_ids):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.simulator.terrain_levels[:] = 0
    env.simulator.terrain_types[:] = track_ids
    unchanged = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.simulator.update_terrain_curriculum(env_ids, unchanged, unchanged)
    env.reset_idx(env_ids)
    env.compute_observations()
    return env_ids


def unpack_observations(value, expected):
    if not isinstance(value, (tuple, list)) or len(value) != expected:
        raise RuntimeError(
            f"Task must provide the depth-WaQ {'observation' if expected == 6 else 'step'} "
            f"signature with {expected} entries; received {type(value).__name__}"
        )
    return value


class TerminalCapture:
    def __init__(self, env):
        self.env = env
        self.enabled = False
        self.states = {}
        original = env.reset_idx

        def reset_idx(env_ids):
            if self.enabled:
                for env_id in env_ids.detach().cpu().tolist():
                    self.states[env_id] = {
                        "position": env.simulator.base_pos[env_id].detach().clone(),
                        "timeout": bool(env.time_out_buf[env_id].item()),
                        "episode_steps": int(env.episode_length_buf[env_id].item()),
                    }
            return original(env_ids)

        env.reset_idx = reset_idx

    def begin(self):
        self.states.clear()
        self.enabled = True

    def end(self):
        self.enabled = False


class ClassificationStats:
    def __init__(self):
        self.matrices = {
            "instantaneous": np.zeros((4, 4), dtype=np.int64),
            "bayes": np.zeros((4, 4), dtype=np.int64),
        }
        self.selected_correct = 0
        self.selected_total = 0
        self.switch_count = 0
        self.delays = []
        self.ticks = []

    def update(self, truth, instantaneous, bayes, selected, record):
        truth_index = CANONICAL_CLASSES.index(truth)
        self.matrices["instantaneous"][truth_index, CANONICAL_CLASSES.index(instantaneous)] += 1
        self.matrices["bayes"][truth_index, CANONICAL_CLASSES.index(bayes)] += 1
        self.selected_correct += int(selected == truth)
        self.selected_total += 1
        self.ticks.append(record)

    def summary(self):
        result = {}
        for name, matrix in self.matrices.items():
            support = matrix.sum(axis=1)
            correct = np.diag(matrix)
            result[name] = {
                "accuracy": float(correct.sum() / matrix.sum()) if matrix.sum() else float("nan"),
                "per_class_accuracy": {
                    label: float(correct[i] / support[i]) if support[i] else float("nan")
                    for i, label in enumerate(CANONICAL_CLASSES)
                },
                "support": {label: int(support[i]) for i, label in enumerate(CANONICAL_CLASSES)},
                "confusion_matrix": matrix.tolist(),
                "class_order": CANONICAL_CLASSES,
            }
        result.update({
            "selected_skill_accuracy": (
                self.selected_correct / self.selected_total if self.selected_total else float("nan")
            ),
            "selected_skill_total": self.selected_total,
            "switch_count": self.switch_count,
            "terrain_transition_detection_delay_steps": (
                float(np.mean(self.delays)) if self.delays else None
            ),
            "terrain_transition_detection_delays": self.delays,
        })
        return result


def _force_commands(env, selected_skills, fixed_command, heading_gain=0.4, max_yaw_rate=1, lookahead=5.0):
    if fixed_command is None:
        speeds = torch.tensor(
            [SKILL_SPEEDS[skill] for skill in selected_skills],
            device=env.device, dtype=env.commands.dtype,
        )
    else:
        speeds = torch.full(
            (env.num_envs,), fixed_command, device=env.device, dtype=env.commands.dtype
        )
    env.commands[:, 0] = speeds
    env.commands[:, 1] = 0.0

    # Lateral offset from track center (y only) -- never reference center_x directly
    center_y = env.simulator.env_origins[:, 1].to(env.commands.dtype)
    current_y = env.simulator.base_pos[:, 1].to(env.commands.dtype)
    lateral_error = center_y - current_y  # +y means center is to the left

    # Bearing to a lookahead point straight ahead + laterally corrected,
    # expressed relative to current heading (small-angle style, no backward flips)
    target_heading_offset = torch.atan2(lateral_error, torch.full_like(lateral_error, lookahead))

    heading_error = wrap_to_pi(target_heading_offset)
    yaw_rate_cmd = (heading_gain * heading_error).clamp(-max_yaw_rate, max_yaw_rate)
    env.commands[:, 2] = yaw_rate_cmd

    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0

    return lateral_error


def _depth_valid(depth):
    flattened = depth.reshape(depth.shape[0], -1)
    return torch.isfinite(flattened).all(dim=1) & (flattened.abs().sum(dim=1) > 0)


def _summary_rows(rows, track_id=None):
    subset = rows if track_id is None else [row for row in rows if row["track_id"] == track_id]
    count = len(subset)
    return {
        "track_id": track_id,
        "completed_episodes": count,
        "success_rate": float(np.mean([r["success"] for r in subset])) if count else float("nan"),
        "mean_forward_distance_m": (
            float(np.mean([r["max_forward_distance_m"] for r in subset])) if count else float("nan")
        ),
        "course_completion_count": sum(r["termination_reason"] == "course_complete" for r in subset),
        "timeout_success_count": sum(r["termination_reason"] == "timeout" for r in subset),
        "instantaneous_correct": sum(r["instantaneous_correct"] for r in subset),
        "instantaneous_total": sum(r["instantaneous_total"] for r in subset),
        "bayes_correct": sum(r["bayes_correct"] for r in subset),
        "bayes_total": sum(r["bayes_total"] for r in subset),
    }


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _timing_start(device):
    _synchronize(device)
    return time.perf_counter()


def _timing_stop(started, device):
    _synchronize(device)
    return 1000.0 * (time.perf_counter() - started)


def _timing_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "std": None, "median": None, "p95": None}
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
    }


def _make_paper_bayes_filters(class_ids, config, count, device):
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        BayesianTerrainFilter, make_persistent_transition_matrix,
    )
    transition = make_persistent_transition_matrix(
        class_ids, float(config["stable_stay"]), device=device)
    prior = torch.full((len(class_ids),), 1.0 / len(class_ids), device=device)
    observation = torch.eye(len(class_ids), device=device)
    template = BayesianTerrainFilter(
        class_ids, prior, transition, observation,
        evidence_power=float(config["evidence_power"]),
        adaptive_evidence=bool(config["adaptive_evidence"]),
        min_evidence_power=float(config["evidence_power"]),
        confidence_gamma=1.0, stay_probability=float(config["stable_stay"]),
        transition_source="persistent", device=device)
    return [copy.deepcopy(template) for _ in range(count)]


def run_eval(args):
    if "genesis" in globals().get("SIMULATOR", ""):
        init_genesis(args, gs)
    else:
        configure_runtime_device(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    override_configs_multiterrain(env_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    device = env.device
    track_ids = torch.arange(env.num_envs, device=device, dtype=torch.long) % 10
    assign_tracks_and_reset(env, track_ids)
    obs_buf, _, obs_history, _, _, depth = unpack_observations(env.get_observations(), 6)

    paper_method = getattr(args, "paper_method", None)
    runtime_classifier = deployment = paper_manifest = None
    filters = ema_filters = None
    bayes_path = None
    approach = args.classifier_approach
    selector_mode = args.selector_mode
    if paper_method in PAPER_METHOD_SPECS:
        architecture, selector_mode = PAPER_METHOD_SPECS[paper_method]
        approach = architecture
        runtime_classifier, deployment, paper_manifest = _load_paper_classifier(
            args.paper_offline_dir, architecture, args.classifier_seed, device)
        if selector_mode == "bayes":
            filters = _make_paper_bayes_filters(
                runtime_classifier.class_ids, paper_manifest["fixed_bayes_configuration"],
                env.num_envs, device)
        elif selector_mode == "ema":
            from legged_gym.scripts.depth_data_pipeline.sequential_terrain_filter_extensions import (
                EMALogitPatienceFilter,
            )
            ema_filters = [EMALogitPatienceFilter(
                runtime_classifier.class_ids,
                **paper_manifest["fixed_ema_configuration"], device=device)
                for _ in range(env.num_envs)]
    elif paper_method in ("oracle", "distilled"):
        selector_mode = paper_method
        approach = None
    else:
        runtime_classifier, deployment = _load_neural(args.classifier_dir, approach, device)
        bayes_path = Path(deployment["temporal_filter_path"])
        if not bayes_path.is_absolute():
            suite_root = Path(args.classifier_dir)
            architecture = METHOD_TO_APPROACH[approach][0]
            run_root = suite_root / architecture if (suite_root / architecture).is_dir() else suite_root
            bayes_path = run_root / bayes_path
        bayes_template = _load_bayes_template(bayes_path, device)
        if runtime_classifier.class_ids != list(bayes_template.labels):
            raise ValueError("classifier and Bayes-filter class order differ")
        filters = [copy.deepcopy(bayes_template) for _ in range(env.num_envs)]

    if filters is not None:
        for filt in filters:
            filt.reset()
    batch1_filters = batch1_ema_filters = None
    if paper_method in PAPER_METHOD_SPECS and selector_mode == "bayes":
        batch1_filters = _make_paper_bayes_filters(
            runtime_classifier.class_ids, paper_manifest["fixed_bayes_configuration"],
            env.num_envs, device)
    elif paper_method in PAPER_METHOD_SPECS and selector_mode == "ema":
        from legged_gym.scripts.depth_data_pipeline.sequential_terrain_filter_extensions import (
            EMALogitPatienceFilter,
        )
        batch1_ema_filters = [EMALogitPatienceFilter(
            runtime_classifier.class_ids,
            **paper_manifest["fixed_ema_configuration"], device=device)
            for _ in range(env.num_envs)]
    policy_set = (None if paper_method == "distilled"
                  else PerTerrainPolicySet(args.jit, device))
    distilled_policy = (torch.jit.load(args.distilled_jit, map_location=device).eval()
                        if paper_method == "distilled" else None)
    # Model reconstruction can consume framework RNG state. Restore the paired
    # environment seed after every method's artifacts are loaded so later
    # episode resets remain identical across methods and classifier seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    terminal_capture = TerminalCapture(env)

    selected_skills = ["rough"] * env.num_envs
    selection_initialized = [False] * env.num_envs

    from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_float, quat_apply, quat_from_euler_xyz
    forward = quat_apply(env.simulator.base_quat, env.forward_vec)
    env.heading = torch.atan2(forward[:, 1], forward[:, 0]) 

    if selector_mode == "oracle":
        _, selected_skills = get_ground_truth_labels(env, args.look_ahead_frac)
    assigned_lora = torch.tensor(
        [label_to_lora(label) for label in selected_skills], device=device, dtype=torch.long
    )

    start_x = env.simulator.base_pos[:, 0].clone()
    max_progress = torch.zeros(env.num_envs, device=device)
    episode_steps = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    inst_correct = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    inst_total = torch.zeros_like(inst_correct)
    bayes_correct = torch.zeros_like(inst_correct)
    bayes_total = torch.zeros_like(inst_correct)
    selected_correct = torch.zeros_like(inst_correct)
    selected_total = torch.zeros_like(inst_correct)
    episode_switches = torch.zeros_like(inst_correct)
    episode_delays = [[] for _ in range(env.num_envs)]
    completed_by_track = [0] * 10
    episode_rows = []
    classification = ClassificationStats()
    trajectory_replay = defaultdict(list)
    previous_truth = [None] * env.num_envs
    pending_transition = [None] * env.num_envs
    finish_x = 5 * env_cfg.terrain.terrain_length - args.finish_margin
    width = env_cfg.terrain.terrain_width
    steps_run = 0
    timing_stage_names = (
        "depth_preprocess_ms", "oracle_lookup_ms", "feature_extraction_ms",
        "standardization_ms", "classifier_ms", "temporal_filter_ms",
        "skill_selection_ms", "selector_total_ms",
    )
    timing_samples = {name: [] for name in timing_stage_names}
    timing_samples["specialist_policy_ms"] = []
    timing_samples["classification_step_latency_ms"] = []
    for name in timing_stage_names:
        timing_samples[f"batch1_{name}"] = []
    timing_samples["batch1_specialist_policy_ms"] = []
    timing_samples["batch1_classification_step_latency_ms"] = []
    classification_updates_seen = 0

    while steps_run < args.num_steps and min(completed_by_track) < args.episodes_per_track:
        _force_commands(env, selected_skills, args.fixed_forward_command)
        policy_started = _timing_start(device)
        with torch.inference_mode():
            actions = (distilled_policy(obs_buf.detach(), obs_history.detach(), depth.detach())
                       if distilled_policy is not None else
                       policy_set.act(obs_buf, obs_history, depth, assigned_lora))
        policy_elapsed_ms = _timing_stop(policy_started, device)
        rng_devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            batch1_policy_started = _timing_start(device)
            with torch.inference_mode():
                if distilled_policy is not None:
                    distilled_policy(
                        obs_buf[:1].detach(), obs_history[:1].detach(), depth[:1].detach())
                else:
                    policy_set.act(
                        obs_buf[:1], obs_history[:1], depth[:1], assigned_lora[:1])
            batch1_policy_elapsed_ms = _timing_stop(batch1_policy_started, device)
        if classification_updates_seen >= args.latency_warmup_updates:
            timing_samples["specialist_policy_ms"].append(policy_elapsed_ms)
            timing_samples["batch1_specialist_policy_ms"].append(batch1_policy_elapsed_ms)
        terminal_capture.begin()
        try:
            step_value = env.step(actions.detach())
        finally:
            terminal_capture.end()
        obs_buf, _, obs_history, _, _, _, dones, infos, depth = unpack_observations(step_value, 9)
        steps_run += 1
        episode_steps += 1
        timeout_flags = infos.get("time_outs") if isinstance(infos, dict) else None

        step_positions = env.simulator.base_pos.clone()
        for env_id, state in terminal_capture.states.items():
            step_positions[env_id] = state["position"]
        finish_distance = (finish_x - start_x).clamp_min(0.0)
        progress = (step_positions[:, 0] - start_x).clamp_min(0.0)
        max_progress = torch.maximum(max_progress, torch.minimum(progress, finish_distance))

        done_ids = set(dones.nonzero(as_tuple=False).flatten().detach().cpu().tolist())
        course_ids = set((step_positions[:, 0] >= finish_x).nonzero(as_tuple=False).flatten().cpu().tolist())
        columns = track_ids.to(step_positions.device)
        lateral_failure = (
            (step_positions[:, 1] < columns * width)
            | (step_positions[:, 1] >= (columns + 1) * width)
        )
        lateral_ids = set(lateral_failure.nonzero(as_tuple=False).flatten().cpu().tolist())

        if (steps_run - 1) % args.classify_every == 0:
            record_latency = classification_updates_seen >= args.latency_warmup_updates
            stage_timing = {name: 0.0 for name in timing_stage_names[:-1]}
            selector_executed = False
            if selector_mode == "oracle" and paper_method == "oracle":
                selector_executed = True
                lookup_started = _timing_start(device)
                raw_truth, canonical_truth = get_ground_truth_labels(env, args.look_ahead_frac)
                stage_timing["oracle_lookup_ms"] = _timing_stop(lookup_started, device)
            else:
                raw_truth, canonical_truth = get_ground_truth_labels(env, args.look_ahead_frac)
            valid = (torch.ones(env.num_envs, dtype=torch.bool, device=device)
                     if selector_mode == "oracle" and paper_method == "oracle"
                     else _depth_valid(depth))
            if done_ids:
                valid[list(done_ids)] = False
            valid_ids = valid.nonzero(as_tuple=False).flatten()
            # Keep reported vectorized-selector latency at the declared fixed
            # batch size; reset ticks with temporarily invalid environments are
            # still evaluated normally but omitted from timing statistics.
            if paper_method != "distilled" and valid_ids.numel() != env.num_envs:
                record_latency = False
            if selector_mode == "oracle" and paper_method == "oracle":
                oracle_outputs = []
                skill_started = _timing_start(device)
                for env_id in valid_ids.detach().cpu().tolist():
                    previous = selected_skills[env_id]
                    selected_skills[env_id] = canonical_truth[env_id]
                    assigned_lora[env_id] = label_to_lora(selected_skills[env_id])
                    changed = selection_initialized[env_id] and selected_skills[env_id] != previous
                    oracle_outputs.append((env_id, changed))
                    selection_initialized[env_id] = True
                stage_timing["skill_selection_ms"] = _timing_stop(skill_started, device)
                for env_id, changed in oracle_outputs:
                    if changed:
                        classification.switch_count += 1
                        episode_switches[env_id] += 1
                    selected_correct[env_id] += 1
                    selected_total[env_id] += 1
                    truth = canonical_truth[env_id]
                    if previous_truth[env_id] is not None and truth != previous_truth[env_id]:
                        classification.delays.append(0)
                        episode_delays[env_id].append(0)
                    previous_truth[env_id] = truth
                    track_id = int(track_ids[env_id].item())
                    if completed_by_track[track_id] < args.episodes_per_track:
                        position = env.simulator.base_pos[env_id].detach()
                        segment = min(max(int(float(position[0]) /
                                              env_cfg.terrain.terrain_length), 0), 4)
                        sequence = args.track_layout[track_id]["sequence"]
                        next_segment = sequence[segment + 1] if segment < 4 else None
                        boundary_x = ((segment + 1) * env_cfg.terrain.terrain_length
                                      if segment < 4 else None)
                        trajectory_replay[track_id].append({
                            "episode_index": completed_by_track[track_id],
                            "environment_id": env_id, "track_id": track_id,
                            "control_step": steps_run,
                            "timestamp_s": steps_run * float(env.dt),
                            "depth": depth[env_id].detach().cpu().clone(),
                            "orientation_rpy":
                                env.simulator._base_euler[env_id].detach().cpu().clone(),
                            "angular_velocity":
                                env.simulator.base_ang_vel[env_id].detach().cpu().clone(),
                            "base_position": position.cpu().clone(),
                            "ground_truth_raw": raw_truth[env_id],
                            "ground_truth": canonical_truth[env_id],
                            "current_segment_index": segment,
                            "current_segment": sequence[segment],
                            "next_segment": next_segment,
                            "boundary_x_m": boundary_x,
                            "terrain_length_m": float(env_cfg.terrain.terrain_length),
                        })
                    record = {
                        "step": steps_run, "env_id": env_id,
                        "track_id": int(track_ids[env_id].item()),
                        "raw_ground_truth": raw_truth[env_id],
                        "canonical_ground_truth": canonical_truth[env_id],
                        "instantaneous_label": canonical_truth[env_id],
                        "bayes_label": canonical_truth[env_id],
                        "selected_skill": canonical_truth[env_id],
                    }
                    classification.update(canonical_truth[env_id], canonical_truth[env_id],
                                          canonical_truth[env_id], canonical_truth[env_id], record)
            elif runtime_classifier is not None and valid_ids.numel():
                selector_executed = True
                sensor_depth = depth[valid_ids].detach()
                euler = env.simulator._base_euler[valid_ids].detach()
                angular_velocity = env.simulator.base_ang_vel[valid_ids].detach()
                if paper_method:
                    logits, probabilities, classifier_timing = runtime_classifier.predict_deterministic_timed(
                        sensor_depth, euler, angular_velocity,
                        temperature=paper_manifest["fixed_bayes_configuration"]["T_filter"])
                    stage_timing.update(classifier_timing)
                    logits, probabilities = logits.to(device), probabilities.to(device)
                else:
                    classifier_started = _timing_start(device)
                    probabilities, event_probabilities, mutual_information = runtime_classifier.predict(
                        sensor_depth, euler, angular_velocity)
                    logits = event_probabilities.clamp_min(1e-8).log().to(device)
                    probabilities = probabilities.to(device)
                    stage_timing["classifier_ms"] = _timing_stop(classifier_started, device)
                if probabilities.shape[0] != valid_ids.numel():
                    raise RuntimeError("Classifier batch size does not match valid environment count")
                base_outputs = []
                postprocess_started = _timing_start(device)
                for batch_index, env_id in enumerate(valid_ids.detach().cpu().tolist()):
                    probability = probabilities[batch_index]
                    instant_id = runtime_classifier.class_ids[
                        int(logits[batch_index].argmax().item())]
                    base_outputs.append({
                        "batch_index": batch_index, "env_id": env_id,
                        "truth": canonical_truth[env_id], "probability": probability,
                        "instant_id": instant_id,
                    })
                stage_timing["classifier_ms"] += _timing_stop(
                    postprocess_started, device)

                native_outputs = []
                selection_started = _timing_start(device)
                for base_output in base_outputs:
                    batch_index, env_id = base_output["batch_index"], base_output["env_id"]
                    probability, instant_id = base_output["probability"], base_output["instant_id"]
                    posterior = probability
                    bayes_id = instant_id
                    if selector_mode == "bayes":
                        if paper_method:
                            bayes_step = filters[env_id].update(probability)
                        else:
                            bayes_step = filters[env_id].update(
                                probability, event_probabilities=event_probabilities[batch_index],
                                mutual_information=float(mutual_information[batch_index]))
                        bayes_id = bayes_step.label
                        posterior = bayes_step.posterior
                        selected_id = bayes_id
                    elif selector_mode == "oracle":
                        selected_id = truth
                    elif selector_mode == "ema":
                        selected_id = ema_filters[env_id].update(logits[batch_index])
                    elif selector_mode == "instantaneous":
                        selected_id = instant_id
                    else:
                        selected_id = "rough"
                    native_outputs.append({
                        **base_output,
                        "posterior": posterior, "bayes_id": bayes_id,
                        "selected_id": selected_id,
                    })
                selection_elapsed = _timing_stop(selection_started, device)
                if selector_mode in ("ema", "bayes"):
                    stage_timing["temporal_filter_ms"] = selection_elapsed
                else:
                    stage_timing["classifier_ms"] += selection_elapsed

                classifier_outputs = []
                skill_started = _timing_start(device)
                for native in native_outputs:
                    env_id = native["env_id"]
                    instant_label = canonicalize_label(native["instant_id"])
                    bayes_label = canonicalize_label(native["bayes_id"])
                    selected = canonicalize_label(native["selected_id"])
                    changed = selection_initialized[env_id] and selected != selected_skills[env_id]
                    selection_initialized[env_id] = True
                    selected_skills[env_id] = selected
                    assigned_lora[env_id] = label_to_lora(selected)
                    classifier_outputs.append({
                        **native, "instant_label": instant_label, "bayes_label": bayes_label,
                        "selected": selected, "changed": changed,
                    })
                stage_timing["skill_selection_ms"] = _timing_stop(skill_started, device)
                for output_record in classifier_outputs:
                    batch_index = output_record["batch_index"]
                    env_id, truth = output_record["env_id"], output_record["truth"]
                    probability = output_record["probability"]
                    instant_label = output_record["instant_label"]
                    posterior, bayes_label = output_record["posterior"], output_record["bayes_label"]
                    selected = output_record["selected"]
                    if output_record["changed"]:
                        classification.switch_count += 1
                        episode_switches[env_id] += 1
                    inst_correct[env_id] += int(instant_label == truth)
                    inst_total[env_id] += 1
                    bayes_correct[env_id] += int(bayes_label == truth)
                    bayes_total[env_id] += 1
                    selected_correct[env_id] += int(selected == truth)
                    selected_total[env_id] += 1

                    if previous_truth[env_id] is not None and truth != previous_truth[env_id]:
                        pending_transition[env_id] = (truth, steps_run)
                    previous_truth[env_id] = truth
                    pending = pending_transition[env_id]
                    if pending is not None and selected == pending[0]:
                        delay = steps_run - pending[1]
                        classification.delays.append(delay)
                        episode_delays[env_id].append(delay)
                        pending_transition[env_id] = None
                    record = {
                        "step": steps_run, "env_id": env_id,
                        "track_id": int(track_ids[env_id].item()),
                        "raw_ground_truth": raw_truth[env_id],
                        "canonical_ground_truth": truth,
                        "class_ids": [str(value) for value in runtime_classifier.class_ids],
                        "instantaneous_probabilities": probability.detach().cpu().tolist(),
                        "instantaneous_label": instant_label,
                        "bayes_posterior": posterior.detach().cpu().tolist(),
                        "bayes_label": bayes_label,
                        "selected_skill": selected,
                    }
                    classification.update(truth, instant_label, bayes_label, selected, record)

            if paper_method != "distilled" and selector_executed and record_latency:
                selector_total = sum(stage_timing.values())
                stage_timing["selector_total_ms"] = selector_total
                if abs(selector_total - sum(
                        stage_timing[name] for name in timing_stage_names[:-1])) > 1e-9:
                    raise AssertionError("selector total does not equal the sum of timed stages")
                for name in timing_stage_names:
                    timing_samples[name].append(stage_timing[name])
                timing_samples["classification_step_latency_ms"].append(
                    policy_elapsed_ms + selector_total)

            # Explicit batch-size-one deployment benchmark.  It uses separate
            # temporal state and discards all outputs, so evaluated actions and
            # locomotion results remain unchanged.
            if paper_method and paper_method != "distilled" and valid_ids.numel():
                benchmark_env = int(valid_ids[0].item())
                batch1_timing = {name: 0.0 for name in timing_stage_names[:-1]}
                if paper_method == "oracle":
                    started = _timing_start(device)
                    _, batch1_selected = get_ground_truth_label_for_env(
                        env, benchmark_env, args.look_ahead_frac)
                    batch1_timing["oracle_lookup_ms"] = _timing_stop(started, device)
                else:
                    batch1_logits, batch1_probabilities, classifier_timing = (
                        runtime_classifier.predict_deterministic_timed(
                            depth[benchmark_env:benchmark_env + 1].detach(),
                            env.simulator._base_euler[benchmark_env:benchmark_env + 1].detach(),
                            env.simulator.base_ang_vel[benchmark_env:benchmark_env + 1].detach(),
                            temperature=paper_manifest["fixed_bayes_configuration"]["T_filter"])
                    )
                    batch1_timing.update(classifier_timing)
                    started = _timing_start(device)
                    batch1_instant = runtime_classifier.class_ids[
                        int(batch1_logits[0].argmax().item())]
                    batch1_timing["classifier_ms"] += _timing_stop(started, device)
                    if selector_mode == "bayes":
                        started = _timing_start(device)
                        batch1_selected = batch1_filters[benchmark_env].update(
                            batch1_probabilities[0]).label
                        batch1_timing["temporal_filter_ms"] = _timing_stop(started, device)
                    elif selector_mode == "ema":
                        started = _timing_start(device)
                        batch1_selected = batch1_ema_filters[benchmark_env].update(
                            batch1_logits[0])
                        batch1_timing["temporal_filter_ms"] = _timing_stop(started, device)
                    else:
                        batch1_selected = batch1_instant
                started = _timing_start(device)
                batch1_selected = canonicalize_label(batch1_selected)
                label_to_lora(batch1_selected)
                batch1_timing["skill_selection_ms"] = _timing_stop(started, device)
                batch1_total = sum(batch1_timing.values())
                batch1_timing["selector_total_ms"] = batch1_total
                if abs(batch1_total - sum(
                        batch1_timing[name] for name in timing_stage_names[:-1])) > 1e-9:
                    raise AssertionError("batch-1 selector total does not equal timed stage sum")
                if record_latency:
                    for name in timing_stage_names:
                        timing_samples[f"batch1_{name}"].append(batch1_timing[name])
                    timing_samples["batch1_classification_step_latency_ms"].append(
                        batch1_policy_elapsed_ms + batch1_total)
            classification_updates_seen += 1

        completed_ids = sorted(done_ids | course_ids | lateral_ids)
        manual_reset_ids = []
        for env_id in completed_ids:
            track_id = int(track_ids[env_id].item())
            timeout = (
                bool(timeout_flags[env_id].item()) if timeout_flags is not None
                else terminal_capture.states.get(env_id, {}).get("timeout", False)
            )
            if env_id in course_ids:
                success, reason = True, "course_complete"
            elif timeout:
                success, reason = True, "timeout"
            elif env_id in lateral_ids:
                success, reason = False, "left_column"
            else:
                success, reason = False, "termination"
            if completed_by_track[track_id] < args.episodes_per_track:
                episode_rows.append({
                    "paper_method": paper_method, "selector_mode": selector_mode,
                    "classifier_approach": approach,
                    "bayes_filter_approach": "fixed_persistent" if selector_mode == "bayes" else None,
                    "bayes_filter_path": str(bayes_path) if bayes_path else None,
                    "seed": args.seed, "classifier_seed": getattr(args, "classifier_seed", None),
                    "difficulty": args.resolved_difficulty["normalized"],
                    "difficulty_level": getattr(args, "difficulty_level", None),
                    "track_id": track_id,
                    "terrain_sequence": args.track_layout[track_id]["sequence"],
                    "success": success, "termination_reason": reason,
                    "max_forward_distance_m": float(max_progress[env_id].item()),
                    "episode_steps": terminal_capture.states.get(env_id, {}).get(
                        "episode_steps", int(episode_steps[env_id].item())
                    ),
                    "instantaneous_correct": int(inst_correct[env_id].item()),
                    "instantaneous_total": int(inst_total[env_id].item()),
                    "bayes_correct": int(bayes_correct[env_id].item()),
                    "bayes_total": int(bayes_total[env_id].item()),
                    "wrong_skill_fraction": (None if paper_method == "distilled" else
                        0.0 if selector_mode == "oracle" else
                        1.0 - float(selected_correct[env_id].item()) /
                        max(int(selected_total[env_id].item()), 1)),
                    "skill_switch_count": int(episode_switches[env_id].item()),
                    "skill_switch_rate": float(episode_switches[env_id].item()) /
                        max(int(selected_total[env_id].item()), 1),
                    "terrain_transition_detection_delay_steps": (
                        float(np.mean(episode_delays[env_id])) if episode_delays[env_id] else None),
                })
                completed_by_track[track_id] += 1
            if env_id not in done_ids:
                manual_reset_ids.append(env_id)

        if completed_ids:
            for env_id in completed_ids:
                if filters is not None:
                    filters[env_id].reset()
                if ema_filters is not None:
                    ema_filters[env_id].reset()
                if batch1_filters is not None:
                    batch1_filters[env_id].reset()
                if batch1_ema_filters is not None:
                    batch1_ema_filters[env_id].reset()
                selected_skills[env_id] = "rough"
                selection_initialized[env_id] = False
                previous_truth[env_id] = None
                pending_transition[env_id] = None
                episode_delays[env_id] = []
            if manual_reset_ids:
                reset_tensor = torch.tensor(manual_reset_ids, device=device, dtype=torch.long)
                env.reset_idx(reset_tensor)
                env.compute_observations()
                obs_buf, _, obs_history, _, _, depth = unpack_observations(env.get_observations(), 6)
            if selector_mode == "oracle":
                _, oracle_labels = get_ground_truth_labels(env, args.look_ahead_frac)
                for env_id in completed_ids:
                    selected_skills[env_id] = oracle_labels[env_id]
            assigned_lora = torch.tensor(
                [label_to_lora(label) for label in selected_skills], device=device, dtype=torch.long
            )
            reset_tensor = torch.tensor(completed_ids, device=device, dtype=torch.long)
            start_x[reset_tensor] = env.simulator.base_pos[reset_tensor, 0]
            max_progress[reset_tensor] = 0.0
            episode_steps[reset_tensor] = 0
            inst_correct[reset_tensor] = 0
            inst_total[reset_tensor] = 0
            bayes_correct[reset_tensor] = 0
            bayes_total[reset_tensor] = 0
            selected_correct[reset_tensor] = 0
            selected_total[reset_tensor] = 0
            episode_switches[reset_tensor] = 0

    classification_summary = classification.summary()
    overall = _summary_rows(episode_rows)
    overall.update({
        "quotas_complete": all(value == args.episodes_per_track for value in completed_by_track),
        "completed_by_track": completed_by_track,
        "simulation_steps": steps_run,
        "instantaneous_classification_accuracy": classification_summary["instantaneous"]["accuracy"],
        "bayes_classification_accuracy": classification_summary["bayes"]["accuracy"],
    })
    selector_values = timing_samples["selector_total_ms"]
    policy_values = timing_samples["specialist_policy_ms"]
    selector_overhead_values = [value / args.classify_every for value in selector_values]
    selector_overhead_mean = (
        float(np.mean(selector_overhead_values)) if selector_overhead_values else None
    )
    if paper_method == "distilled":
        total_values = list(policy_values)
        additional_selector_values = []
        timing_samples["classification_step_latency_ms"] = list(policy_values)
    else:
        # Policy and selector samples have different cadence.  Adding the mean
        # amortized selector cost to every policy sample preserves all measured
        # policy jitter without inventing a one-to-one pairing.
        total_values = [value + (selector_overhead_mean or 0.0) for value in policy_values]
        additional_selector_values = list(selector_overhead_values)
    batch1_selector_values = timing_samples["batch1_selector_total_ms"]
    batch1_policy_values = timing_samples["batch1_specialist_policy_ms"]
    batch1_selector_overhead_values = [
        value / args.classify_every for value in batch1_selector_values]
    batch1_selector_overhead_mean = (
        float(np.mean(batch1_selector_overhead_values))
        if batch1_selector_overhead_values else 0.0)
    batch1_total_values = [
        value + batch1_selector_overhead_mean for value in batch1_policy_values]
    if paper_method == "distilled":
        timing_samples["batch1_classification_step_latency_ms"] = list(batch1_policy_values)
    batch_size = int(env.num_envs)
    timing_samples.update({
        "selector_ms_per_update": list(selector_values),
        "selector_overhead_ms_per_control_step": selector_overhead_values,
        "specialist_policy_ms_per_step": list(policy_values),
        "total_inference_ms_per_control_step": total_values,
        "total_routed_inference_ms_per_step": list(total_values),
        "effective_inference_hz": [1000.0 / value for value in total_values if value > 0.0],
        "additional_selector_only_ms_per_step": additional_selector_values,
        "batch_latency_ms": list(total_values),
        "per_env_amortized_ms": [value / batch_size for value in total_values],
        "batch1_selector_overhead_ms_per_control_step": batch1_selector_overhead_values,
        "batch1_selector_ms_per_update": list(batch1_selector_values),
        "batch1_specialist_policy_ms_per_step": list(batch1_policy_values),
        "batch1_total_inference_ms_per_control_step": batch1_total_values,
        "batch1_deployment_latency_ms": list(batch1_total_values),
        "batch1_effective_inference_hz": [
            1000.0 / value for value in batch1_total_values if value > 0.0],
    })
    for name in timing_stage_names:
        timing_samples[f"per_env_amortized_{name}"] = [
            value / batch_size for value in timing_samples[name]]
    timing_samples["per_env_amortized_specialist_policy_ms"] = [
        value / batch_size for value in policy_values]
    timing_samples["per_env_amortized_classification_step_latency_ms"] = [
        value / batch_size for value in timing_samples["classification_step_latency_ms"]]
    timing_statistics = {
        name: _timing_statistics(values) for name, values in timing_samples.items()
    }
    selector_ms = timing_statistics["selector_total_ms"]["mean"]
    policy_ms = timing_statistics["specialist_policy_ms_per_step"]["mean"]
    total_ms = timing_statistics["total_inference_ms_per_control_step"]["mean"]
    classification_step_ms = timing_statistics["classification_step_latency_ms"]["mean"]
    per_env_ms = timing_statistics["per_env_amortized_ms"]["mean"]
    batch1_deployment_ms = timing_statistics["batch1_deployment_latency_ms"]["mean"]
    batch1_classification_step_ms = timing_statistics[
        "batch1_classification_step_latency_ms"]["mean"]
    effective_hz = 1000.0 / total_ms if total_ms and total_ms > 0.0 else None
    if paper_method in ("feature_instantaneous", "raw_depth_instantaneous") and any(
            value != 0.0 for name in ("temporal_filter_ms", "batch1_temporal_filter_ms")
            for value in timing_samples[name]):
        raise AssertionError("instantaneous selector unexpectedly incurred temporal-filter timing")
    if paper_method in ("raw_depth_instantaneous", "raw_depth_ema", "raw_depth_bayes") and any(
            value != 0.0 for name in ("feature_extraction_ms", "batch1_feature_extraction_ms")
            for value in timing_samples[name]):
        raise AssertionError("raw-depth timing unexpectedly included engineered feature extraction")
    if paper_method in ("oracle", "distilled") and any(
            value != 0.0 for name in (
                "feature_extraction_ms", "classifier_ms",
                "batch1_feature_extraction_ms", "batch1_classifier_ms")
            for value in timing_samples[name]):
        raise AssertionError("oracle/distilled unexpectedly inherited classifier timing")
    stage_fractions = {}
    if paper_method == "feature_bayes" and selector_ms:
        stage_fractions = {
            "feature_extraction_fraction":
                timing_statistics["feature_extraction_ms"]["mean"] / selector_ms,
            "classifier_fraction": timing_statistics["classifier_ms"]["mean"] / selector_ms,
            "bayes_fraction": timing_statistics["temporal_filter_ms"]["mean"] / selector_ms,
            "skill_selection_fraction":
                timing_statistics["skill_selection_ms"]["mean"] / selector_ms,
        }
    wrong_skill = (None if paper_method == "distilled" else
                   0.0 if selector_mode == "oracle" else
                   1.0 - classification_summary["selected_skill_accuracy"])
    for row in episode_rows:
        row.update({
            "selector_ms_per_update": selector_ms,
            "selector_overhead_ms_per_control_step": selector_overhead_mean,
            "specialist_policy_ms_per_step": policy_ms,
            "classification_step_latency_ms": classification_step_ms,
            "total_inference_ms_per_control_step": total_ms,
            "total_routed_inference_ms_per_step": total_ms,
            "effective_inference_hz": effective_hz,
            "batch_size": batch_size, "batch_latency_ms": total_ms,
            "per_env_amortized_ms": per_env_ms,
            "batch1_deployment_latency_ms": batch1_deployment_ms,
            "batch1_classification_step_latency_ms": batch1_classification_step_ms,
            "selector_latency_ms_per_update": selector_ms,
            "policy_latency_ms_per_step": policy_ms,
            "amortized_total_inference_ms_per_step": total_ms,
            "effective_hz": effective_hz,
        })
    device_name = str(device)
    gpu_name = (torch.cuda.get_device_name(device)
                if torch.device(device).type == "cuda" else None)
    return {
        "metadata": {
            "task": args.task, "policy_jit": args.jit,
            "distilled_jit": getattr(args, "distilled_jit", None), "paper_method": paper_method,
            "selector_mode": selector_mode,
            "classifier_seed": getattr(args, "classifier_seed", None),
            "classifier_approach": approach, "classifier_dir": args.classifier_dir,
            "paper_offline_dir": getattr(args, "paper_offline_dir", None),
            "bayes_filter_approach": "fixed_persistent" if selector_mode == "bayes" else None,
            "bayes_filter_path": str(bayes_path) if bayes_path else None, "seed": args.seed,
            "inference_mode": deployment.get("inference_mode") if deployment else None,
            "mc_samples": deployment.get("mc_samples") if deployment else None,
            "dropout_p": deployment.get("dropout_p") if deployment else None,
            "model_path": deployment.get("model_path") if deployment else None,
            "filter_parameters": (paper_manifest.get("fixed_bayes_configuration")
                                  if paper_manifest and selector_mode == "bayes" else None),
            "ema_parameters": (paper_manifest.get("fixed_ema_configuration")
                                if paper_manifest and selector_mode == "ema" else None),
            "difficulty": args.resolved_difficulty["normalized"],
            "difficulty_level": getattr(args, "difficulty_level", None),
            "resolved_difficulty_parameters": args.resolved_difficulty,
            "fixed_forward_command": args.fixed_forward_command,
            "classify_every": args.classify_every,
            "latency_warmup_updates": args.latency_warmup_updates,
            "timing_device": device_name, "gpu_name": gpu_name,
            "timing_batch_size": env.num_envs, "num_envs": env.num_envs,
            "latency_units": "synchronized wall-clock milliseconds",
            "batch_latency_is_control_loop_deadline": True,
            "effective_hz_uses_batch_latency_without_per_env_division": True,
            "policy_control_frequency_hz": 1.0 / float(env.dt),
            "control_period_s": float(env.dt),
            "timing_clock": "time.perf_counter",
            "cuda_synchronized_timing": torch.device(device).type == "cuda",
            "simulator": globals().get("SIMULATOR"),
            "look_ahead_frac": args.look_ahead_frac,
            "disabled_randomizations": sorted(
                name for name, enabled in EVAL_DOMAIN_RANDOMIZATION_ENABLED.items()
                if not enabled
            ),
            "track_layout": args.track_layout,
        },
        "headline": {
            "episodic_success_rate": overall["success_rate"],
            "episodic_forward_distance_m": overall["mean_forward_distance_m"],
            "wrong_skill_fraction": wrong_skill,
            "selector_ms_per_update": selector_ms,
            "selector_overhead_ms_per_control_step": selector_overhead_mean,
            "specialist_policy_ms_per_step": policy_ms,
            "classification_step_latency_ms": classification_step_ms,
            "total_inference_ms_per_control_step": total_ms,
            "total_routed_inference_ms_per_step": total_ms,
            "effective_inference_hz": effective_hz,
            "batch_size": batch_size, "batch_latency_ms": total_ms,
            "per_env_amortized_ms": per_env_ms,
            "batch1_deployment_latency_ms": batch1_deployment_ms,
            "batch1_classification_step_latency_ms": batch1_classification_step_ms,
            "additional_selector_only_ms_per_step": (
                selector_overhead_mean if paper_method != "distilled" else None),
            "selector_latency_ms_per_update": selector_ms,
            "policy_latency_ms_per_step": policy_ms,
            "amortized_total_inference_ms_per_step": total_ms,
            "effective_hz": effective_hz,
            "skill_switch_count": classification_summary["switch_count"],
            "skill_switch_rate": classification_summary["switch_count"] /
                max(classification_summary["selected_skill_total"], 1),
            "terrain_transition_detection_delay_steps":
                classification_summary["terrain_transition_detection_delay_steps"],
            "instantaneous_classification_accuracy": overall["instantaneous_classification_accuracy"],
            "bayes_classification_accuracy": overall["bayes_classification_accuracy"],
        },
        "overall": overall,
        "per_column": [
            {
                **_summary_rows(episode_rows, track),
                "terrain_sequence": args.track_layout[track]["sequence"],
                "seed": args.seed,
                "difficulty": args.resolved_difficulty["normalized"],
                "difficulty_level": getattr(args, "difficulty_level", None),
                "selector_mode": selector_mode, "paper_method": paper_method,
                "classifier_approach": approach,
                "bayes_filter_approach": "suite_selected",
            }
            for track in range(10)
        ],
        "classification": classification_summary,
        "classification_ticks": classification.ticks,
        "latency": {
            "warmup_updates_excluded": args.latency_warmup_updates,
            "samples": timing_samples,
            "statistics": timing_statistics,
            "stage_fractions": stage_fractions,
        },
        "_trajectory_replay": dict(trajectory_replay) if paper_method == "oracle" else None,
        "episodes": episode_rows,
    }


def save_results(payload, out_dir, task, result_name=None):
    root = Path(out_dir or Path(LEGGED_GYM_ROOT_DIR) / "exp_logs" / "classifier_eval")
    root.mkdir(parents=True, exist_ok=True)
    stem = result_name or f"eval_{task}_{datetime.now():%Y%m%d_%H%M%S}"
    json_path = root / f"{stem}.json"
    csv_path = root / f"{stem}_episodes.csv"
    replay_records = payload.pop("_trajectory_replay", None)
    if replay_records:
        replay_root = root / f"{stem}_trajectory_replay"
        replay_root.mkdir(parents=True, exist_ok=True)
        replay_files = []
        tensor_keys = ("depth", "orientation_rpy", "angular_velocity", "base_position")
        for track_id, records in sorted(replay_records.items()):
            if not records:
                continue
            packed = {
                key: torch.stack([record[key] for record in records]) for key in tensor_keys
            }
            for key in records[0]:
                if key not in tensor_keys:
                    packed[key] = [record[key] for record in records]
            path = replay_root / f"track_{int(track_id):02d}.pt"
            torch.save(packed, path)
            replay_files.append({
                "track_id": int(track_id), "path": str(path), "num_ticks": len(records)})
        payload["trajectory_replay"] = {
            "format": "torch_pt", "contains_depth_images": True,
            "files": replay_files,
        }
    else:
        payload["trajectory_replay"] = None
    with json_path.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=True)
    rows = payload["episodes"]
    with csv_path.open("w", newline="") as stream:
        if rows:
            csv_rows = [{**row, "terrain_sequence": "|".join(row["terrain_sequence"])} for row in rows]
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    return json_path, csv_path


def main():
    args = get_args()
    payload = run_eval(args)
    headline = payload["headline"]
    print("\n=== Headline results ===")
    print(f"Episodic success rate:             {headline['episodic_success_rate']:.3f}")
    print(f"Mean episodic forward distance:    {headline['episodic_forward_distance_m']:.3f} m")
    print(f"Instantaneous classification acc.: {headline['instantaneous_classification_accuracy']:.3f}")
    print(f"Bayes classification accuracy:     {headline['bayes_classification_accuracy']:.3f}")
    if not payload["overall"]["quotas_complete"]:
        print(f"WARNING: incomplete track quotas: {payload['overall']['completed_by_track']}")
    for track in payload["metadata"]["track_layout"]:
        print(f"Track {track['track_id']}: {' -> '.join(track['sequence'])}")
        for cell in track["cells"]:
            print(f"  row {cell['row']}: {cell['raw_label']} {cell['parameters']}")
    json_path, csv_path = save_results(payload, args.out_dir, args.task, args.result_name)
    print(f"Saved JSON: {json_path}")
    print(f"Saved episodes CSV: {csv_path}")


if __name__ == "__main__":
    main()
