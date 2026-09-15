"""Shared helpers for terrain-classifier training scripts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    FeatureStandardizer, collect_classifier_scores_batched, evaluate_predictions,
    fit_nn, make_persistent_transition_matrix, run_filter_sequences,
)
from .sequential_terrain_filter_extensions import (
    search_uncertainty_aware_temporal, staged_sequential_search,
)


def make_terrain_extractor(calibration_file: str | Path) -> SobelDepthTerrainFeatureExtractor:
    calibration = torch.load(calibration_file, map_location="cpu", weights_only=False)
    images = calibration["depth_images"].float()
    extractor = SobelDepthTerrainFeatureExtractor(
        output_size=tuple(images.shape[-2:]), min_depth=0.02, max_depth=1.0,
        far_depth=0.6, close_depth=0.15, close_residual_threshold=0.05,
        sobel_edge_threshold=0.007, depth_scale=None,
    )
    extractor.fit_reference_model(images, calibration["orientation_rpy"])
    return extractor


def extract_in_chunks(extractor: Any, depth_images: torch.Tensor, orientation_rpy: torch.Tensor,
                      angular_velocity: torch.Tensor, chunk_size: int = 256,
                      device: str | torch.device = "cpu") -> torch.Tensor:
    outputs = []
    for start in range(0, depth_images.shape[0], chunk_size):
        stop = min(start + chunk_size, depth_images.shape[0])
        with torch.inference_mode():
            features = extractor.extract_batch(
                depth_images[start:stop].float(), orientation_rpy[start:stop], angular_velocity[start:stop]
            )
        outputs.append(features.detach().to(device))
    if not outputs:
        raise ValueError("cannot extract features from an empty dataset")
    return torch.cat(outputs)


def extract_dataset_features(extractor: Any, data: Mapping[str, Any], chunk_size: int = 256) -> torch.Tensor:
    return extract_in_chunks(extractor, data["depth_images"], data["orientation_rpy"],
                             data["angular_velocity"], chunk_size=chunk_size)


def fit_standardizer(features: torch.Tensor, labels: Sequence[Any], batch_size: int = 512) -> FeatureStandardizer:
    """Fit strictly from supplied structural-training feature rows."""
    loader = DataLoader(TensorDataset(features.cpu(), torch.arange(len(labels))), batch_size=batch_size)
    return FeatureStandardizer().fit(loader)


def collect_engineered_scores(
    classifier: Any, extractor: Any, standardizer: FeatureStandardizer,
    data: Mapping[str, Any], chunk_size: int = 256,
) -> torch.Tensor:
    """Extract, standardize, and score chunks without materializing all features."""
    def batches():
        n = data["depth_images"].shape[0]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            with torch.inference_mode():
                features = extractor.extract_batch(
                    data["depth_images"][start:stop].float(),
                    data["orientation_rpy"][start:stop],
                    data["angular_velocity"][start:stop],
                )
            yield standardizer.transform(features)

    return collect_classifier_scores_batched(classifier, batches(), cache_device="cpu")


def collect_raw_depth_scores(
    classifier: Any, depth_images: torch.Tensor, chunk_size: int = 256,
    orientation_rpy: Optional[torch.Tensor] = None,
    angular_velocity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score raw depth chunks without creating a full float/channel-expanded copy."""
    def batches():
        for start in range(0, depth_images.shape[0], chunk_size):
            stop = min(start + chunk_size, depth_images.shape[0])
            if getattr(classifier.model, "robot_state_dim", 0):
                if orientation_rpy is None or angular_velocity is None:
                    raise ValueError("state-aware raw-depth models require orientation and angular velocity")
                yield pack_raw_depth_state_inputs(
                    depth_images[start:stop], orientation_rpy[start:stop],
                    angular_velocity[start:stop])
            else:
                yield depth_images[start:stop].unsqueeze(1).float()

    return collect_classifier_scores_batched(classifier, batches(), cache_device="cpu")


def pack_raw_depth_state_inputs(
    depth_images: torch.Tensor, orientation_rpy: torch.Tensor,
    angular_velocity: torch.Tensor,
) -> torch.Tensor:
    """Pack depth with roll, pitch, and roll/pitch/yaw angular velocities."""
    depth = torch.as_tensor(depth_images).float()
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4:
        raise ValueError("depth_images must have shape [N,H,W] or [N,C,H,W]")
    orientation = torch.as_tensor(orientation_rpy, dtype=depth.dtype, device=depth.device)
    angular = torch.as_tensor(angular_velocity, dtype=depth.dtype, device=depth.device)
    if orientation.ndim != 2 or orientation.shape[1] < 2:
        raise ValueError("orientation_rpy must provide roll and pitch")
    if angular.ndim != 2 or angular.shape[1] < 3:
        raise ValueError("angular_velocity must provide roll, pitch, and yaw rates")
    if orientation.shape[0] != depth.shape[0] or angular.shape[0] != depth.shape[0]:
        raise ValueError("depth and robot-state batch sizes must match")
    robot_state = torch.cat((orientation[:, :2], angular[:, :3]), dim=1)
    return torch.cat((depth.flatten(1), robot_state), dim=1)


@torch.inference_mode()
def collect_neural_logits_batched(
    classifier: Any, inputs: Any, *, batch_size: int = 256, mc_samples: int = 1,
    mc_dropout: bool = False, expanded_batch_size: int = 4096,
    cache_device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, float]:
    """Collect ``[K,N,C]`` logits with one replicated/chunked forward stream per batch."""
    if batch_size <= 0 or mc_samples <= 0 or expanded_batch_size <= 0:
        raise ValueError("batch sizes and mc_samples must be positive")
    model = classifier.model
    model.eval()
    if mc_dropout:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def batch_input(batch: Any) -> Any:
        if isinstance(batch, Mapping):
            return batch.get("inputs", batch.get("features"))
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    if torch.is_tensor(inputs) or hasattr(inputs, "shape"):
        batches = (inputs[i:i + batch_size] for i in range(0, int(inputs.shape[0]), batch_size))
    else:
        batches = (batch_input(batch) for batch in inputs)
    pieces, runtime = [], 0.0
    for raw in batches:
        x = classifier._prepare(raw)
        b = x.shape[0]
        expanded = x.unsqueeze(0).expand(mc_samples, *x.shape).reshape(
            mc_samples * b, *x.shape[1:]
        )
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        logits = torch.cat([
            model(expanded[i:i + expanded_batch_size])
            for i in range(0, expanded.shape[0], expanded_batch_size)
        ], dim=0)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        runtime += time.perf_counter() - started
        pieces.append(logits.reshape(mc_samples, b, -1).detach().to(cache_device))
    model.eval()
    if not pieces:
        raise ValueError("classifier input yielded no samples")
    return torch.cat(pieces, dim=1), runtime


def collect_engineered_logits(
    classifier: Any, extractor: Any, standardizer: FeatureStandardizer,
    data: Mapping[str, Any], *, chunk_size: int, mc_samples: int = 1,
    mc_dropout: bool = False, expanded_batch_size: int = 4096,
) -> tuple[torch.Tensor, float]:
    def batches():
        for start in range(0, len(data["labels"]), chunk_size):
            stop = min(start + chunk_size, len(data["labels"]))
            features = extractor.extract_batch(
                data["depth_images"][start:stop].float(),
                data["orientation_rpy"][start:stop], data["angular_velocity"][start:stop],
            )
            yield standardizer.transform(features)
    return collect_neural_logits_batched(
        classifier, batches(), batch_size=chunk_size, mc_samples=mc_samples,
        mc_dropout=mc_dropout, expanded_batch_size=expanded_batch_size,
    )


def collect_raw_depth_logits(
    classifier: Any, data: Mapping[str, Any], *, chunk_size: int,
    mc_samples: int = 1, mc_dropout: bool = False,
    expanded_batch_size: int = 4096,
) -> tuple[torch.Tensor, float]:
    def batches():
        for start in range(0, len(data["labels"]), chunk_size):
            stop = min(start + chunk_size, len(data["labels"]))
            if getattr(classifier.model, "robot_state_dim", 0):
                yield pack_raw_depth_state_inputs(
                    data["depth_images"][start:stop],
                    data["orientation_rpy"][start:stop],
                    data["angular_velocity"][start:stop])
            else:
                yield data["depth_images"][start:stop].unsqueeze(1).float()
    return collect_neural_logits_batched(
        classifier, batches(), batch_size=chunk_size, mc_samples=mc_samples,
        mc_dropout=mc_dropout, expanded_batch_size=expanded_batch_size,
    )


def probabilities_and_uncertainty(logits: torch.Tensor, temperature: float = 1.0):
    """Return MC-mean probabilities, predictive/expected entropy, and MI."""
    values = torch.as_tensor(logits).float()
    if values.ndim == 2:
        values = values.unsqueeze(0)
    q_samples = F.softmax(values / float(temperature), dim=-1)
    q_mean = q_samples.mean(0)
    predictive_entropy = -(q_mean * q_mean.clamp_min(1e-8).log()).sum(-1)
    expected_entropy = -(q_samples * q_samples.clamp_min(1e-8).log()).sum(-1).mean(0)
    return q_mean, predictive_entropy, expected_entropy, predictive_entropy - expected_entropy


def probability_metrics(classifier: Any, probabilities: torch.Tensor, labels: Sequence[Any]) -> dict[str, Any]:
    truth = classifier._normalize_labels(labels)
    encoded = torch.tensor([classifier.class_to_index[v] for v in truth], dtype=torch.long)
    probabilities = torch.as_tensor(probabilities).cpu().clamp_min(1e-8)
    predicted = [classifier.class_ids[i] for i in probabilities.argmax(1).tolist()]
    metrics = evaluate_predictions(truth, predicted, classifier.class_ids).as_dict()
    targets = F.one_hot(encoded, len(classifier.class_ids)).to(probabilities.dtype)
    metrics["nll"] = float(F.nll_loss(probabilities.log(), encoded))
    metrics["brier"] = float((probabilities - targets).square().sum(1).mean())
    confidence, prediction = probabilities.max(1)
    correct = prediction.eq(encoded)
    ece = 0.0
    for lo in torch.linspace(0, 0.9, 10):
        mask = (confidence > lo) & (confidence <= lo + 0.1)
        if mask.any():
            ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs())
    metrics["expected_calibration_error"] = ece
    return metrics


def classifier_metrics(classifier: Any, inputs: Any, labels: Sequence[Any], temperature: float = 1.0) -> dict[str, Any]:
    scores = classifier.decision_function(inputs).detach().cpu()
    return classifier_metrics_from_scores(classifier, scores, labels, temperature)


def classifier_metrics_from_scores(
    classifier: Any, scores: torch.Tensor, labels: Sequence[Any], temperature: float = 1.0,
) -> dict[str, Any]:
    ordered_labels = list(classifier.class_ids)
    probabilities = F.softmax(torch.as_tensor(scores).cpu() / temperature, dim=1)
    truth = list(labels.detach().cpu().tolist() if torch.is_tensor(labels) else labels)
    predicted = [ordered_labels[i] for i in probabilities.argmax(1).tolist()]
    metrics = evaluate_predictions(truth, predicted, ordered_labels).as_dict()
    encoded = torch.tensor([classifier.class_to_index[label] for label in truth], dtype=torch.long)
    targets = F.one_hot(encoded, len(ordered_labels)).to(probabilities.dtype)
    metrics["nll"] = float(F.nll_loss(probabilities.clamp_min(classifier.eps).log(), encoded))
    metrics["brier"] = float((probabilities - targets).square().sum(1).mean())
    return metrics


def evaluate_bayes(classifier: Any, bayes_filter: Any, inputs: Any, labels: Sequence[Any],
                   sequence_ids: Sequence[Any], temperature: float) -> dict[str, Any]:
    scores = classifier.decision_function(inputs).detach().cpu()
    return evaluate_bayes_from_scores(
        classifier, bayes_filter, scores, labels, sequence_ids, temperature
    )


def evaluate_bayes_from_scores(
    classifier: Any, bayes_filter: Any, scores: torch.Tensor, labels: Sequence[Any],
    sequence_ids: Sequence[Any], temperature: float,
) -> dict[str, Any]:
    probabilities = F.softmax(torch.as_tensor(scores).cpu() / temperature, dim=1).clamp_min(1e-8)
    probabilities = probabilities / probabilities.sum(1, keepdim=True)
    ordered_labels = list(classifier.class_ids)
    predictions, _, _ = run_filter_sequences(
        bayes_filter, probabilities, sequence_ids=sequence_ids, return_traces=False
    )
    truth = classifier._normalize_labels(labels)
    return evaluate_predictions(truth, predictions, ordered_labels, sequence_ids=sequence_ids).as_dict()


def sequence_ids_for(data: Mapping[str, Any], *, allow_legacy: bool = False) -> torch.Tensor:
    from legged_gym.utils.dataset_provenance import validate_dataset_provenance
    validate_dataset_provenance(data, allow_legacy)
    if "sequence_ids" in data:
        # Stable integer coding preserves globally-qualified identities for
        # runners expecting tensors, including merged unequal-length episodes.
        values = data["sequence_ids"]
        if torch.is_tensor(values):
            values = values.tolist()
        mapping = {value: i for i, value in enumerate(dict.fromkeys(values))}
        return torch.tensor([mapping[value] for value in values], dtype=torch.long)
    import warnings
    warnings.warn("Legacy sequence fallback: per_eps boundaries are unverified; never use as verified provenance")
    per_episode = int(data.get("per_eps", 0))
    n = len(data["labels"])
    if per_episode <= 0:
        return torch.zeros(n, dtype=torch.long)
    return torch.arange(math.ceil(n / per_episode)).repeat_interleave(per_episode)[:n]


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_results(path: str | Path, results: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(json_safe(results), stream, indent=2, sort_keys=True)


def load_training_files(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, dict[str, Path]]:
    parser = argparse.ArgumentParser(description="Train a terrain classifier")
    parser.add_argument("--classifier_folder", required=True, type=Path)
    parser.add_argument("--bayesian_folder", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch/chunk size for training, extraction, and classifier scoring")
    parser.add_argument("--no_batch_processing", action="store_true",
                        help="use each complete split as one batch (may require substantial RAM/VRAM)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    files = {
        "train": args.classifier_folder / "train.pt",
        "validation": args.classifier_folder / "val.pt",
        "calibration": args.classifier_folder / "calibration.pt",
        "structural_test": args.classifier_folder / "test.pt",
        "bayes_validation": args.bayesian_folder / "val.pt",
        "ordered_test": args.bayesian_folder / "test.pt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required data files: " + ", ".join(missing))
    # Optional ordered transition-training data. Prefer train over calibration;
    # when neither exists the search explicitly warns before reusing validation.
    for name in ("train", "calibration"):
        candidate = args.bayesian_folder / f"{name}.pt"
        if candidate.is_file():
            files["transition_training"] = candidate
            break
    return args, files


def processing_batch_size(args: argparse.Namespace, sample_count: int) -> int:
    """Resolve the shared bounded-batch/full-split processing choice."""
    if sample_count <= 0:
        raise ValueError("dataset split is empty")
    return sample_count if args.no_batch_processing else args.batch_size


def transition_training_kwargs(files: Mapping[str, Path]) -> dict[str, Any]:
    """Load optional classifier-independent ordered transition-training truth."""
    path = files.get("transition_training")
    if path is None:
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "transition_training_labels": data["labels"],
        "transition_training_sequence_ids": sequence_ids_for(data),
    }


def run_staged_sequential_pipeline(
    classifier: Any, calibration_scores: torch.Tensor, calibration_labels: Sequence[Any],
    validation_scores: torch.Tensor, validation_labels: Sequence[Any],
    validation_sequence_ids: Sequence[Any], test_scores: torch.Tensor,
    test_labels: Sequence[Any], test_sequence_ids: Sequence[Any], output: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the common validation-only temporal search and return new + legacy views."""
    sequential, trials = staged_sequential_search(
        classifier, calibration_scores, calibration_labels,
        validation_scores, validation_labels, validation_sequence_ids,
        test_scores, test_labels, test_sequence_ids, output,
    )
    torch.save(trials, output / "sequential_search.pt")
    torch.save(trials, output / "bayes_search.pt")
    best = sequential["best_bayes"]
    best_stage = sequential["best_bayes_stage"]
    stage = sequential["stages"][best_stage]
    filter_parameters = dict(stage["parameters"])
    stable_stay = filter_parameters["stable_stay"]
    filter_parameters.update({
        # Historical aliases retained for readers of schema_version=1.
        "stay_probability": stable_stay,
        "evidence_power": 1.0,
        "min_evidence_power": 1.0,
        "confidence_gamma": 1.0,
        "transition_alpha": 0.0,
        "transition_matrix": make_persistent_transition_matrix(
            classifier.class_ids, stable_stay, device="cpu"
        ),
        "transition_source": "persistent",
        "observation_mode": "mixed_soft",
        "observation_pseudocount": 0.5,
    })
    legacy_bayes = {
        "selected_temperature": best["temperature"],
        "filter_parameters": filter_parameters,
        "validation_score": best["selection_score"],
        "unfiltered_metrics": classifier_metrics_from_scores(
            classifier, test_scores, test_labels, best["temperature"]
        ),
        "metrics": stage["test_metrics"],
    }
    return sequential, trials, legacy_bayes


def train_uncertainty_aware_nn_suite(
    *, architecture: str, model_factory: Any, class_ids: Sequence[Any],
    train_inputs: torch.Tensor, train_labels: Sequence[Any],
    validation_inputs: torch.Tensor, validation_labels: Sequence[Any],
    split_data: Mapping[str, Mapping[str, Any]], collect_logits: Any,
    validation_sequence_ids: Sequence[Any], test_sequence_ids: Sequence[Any],
    output: Path, device: str, train_batch: int, validation_batch: int,
) -> dict[str, Any]:
    """Train the frozen paper configurations, cache exact-K logits, and tune filters."""
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        NeuralClassifierAdapter,
    )
    mc_values = (10, 25, 50)
    candidates, histories, model_records = [], {}, {}
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    fixed_models = (
        (("deterministic", 0.0, 1e-5), ("mc", 0.10, 1e-4))
        if architecture == "feature_nn"
        else (("deterministic_mc", 0.20, 1e-5),)
    )

    def collect_repeated(classifier, data, samples, mc):
        collect_logits(classifier, data, samples, mc)  # warm-up
        measured = [collect_logits(classifier, data, samples, mc) for _ in range(3)]
        runtimes = sorted(float(value[1]) for value in measured)
        return measured[-1][0], runtimes[len(runtimes) // 2]

    for role, dropout_p, weight_decay in fixed_models:
        tag = (f"dropout_{str(dropout_p).replace('.', 'p')}_"
               f"wd_{weight_decay:.0e}".replace("-", "m"))
        model = model_factory(dropout_p)
        classifier = NeuralClassifierAdapter(model, class_ids, fit_callback=fit_nn, device=device)
        classifier.require_feature = architecture == "feature_nn"
        started = time.perf_counter()
        classifier.fit(
            train_inputs, train_labels, val=(validation_inputs, validation_labels),
            epochs=50, batch_size=train_batch, validation_batch_size=validation_batch,
            optimizer_kwargs={"weight_decay": weight_decay},
            early_stopping_patience=10, restore_best_weights=True)
        training_runtime = time.perf_counter() - started
        model_dir = models_dir / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path, args_path = model_dir / "classifier.pt", model_dir / "nn_model_args.pt"
        classifier.save(model_path)
        torch.save(model.get_args(), args_path)
        histories[tag] = classifier.training_history
        model_records[tag] = {"model_path": model_path, "model_args_path": args_path,
                              "role": role, "dropout_p": dropout_p,
                              "weight_decay": weight_decay,
                              "training_runtime_seconds": training_runtime}

        if role in {"deterministic", "deterministic_mc"}:
            logits, runtimes = {}, {}
            for split, data in split_data.items():
                logits[split], runtimes[split] = collect_repeated(classifier, data, 1, False)
            candidates.append(_nn_candidate_record(
                f"{tag}_deterministic", architecture, dropout_p, "deterministic", 1,
                model_path, args_path, logits, runtimes, classifier, split_data,
                training_runtime, weight_decay, False, None))
        if role in {"mc", "deterministic_mc"}:
            for samples in mc_values:
                logits, runtimes = {}, {}
                for split, data in split_data.items():
                    logits[split], runtimes[split] = collect_repeated(
                        classifier, data, samples, True)
                candidates.append(_nn_candidate_record(
                    f"{tag}_mc{samples}", architecture, dropout_p, "mc", samples,
                    model_path, args_path, logits, runtimes, classifier, split_data,
                    training_runtime, weight_decay, False, None))
        classifier.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    search, trials = search_uncertainty_aware_temporal(
        candidates, classifier._normalize_labels(split_data["ordered_validation"]["labels"]),
        validation_sequence_ids, classifier._normalize_labels(split_data["ordered_test"]["labels"]),
        test_sequence_ids, class_ids, output,
    )
    torch.save(trials, output / "uncertainty_temporal_search.pt")
    search["all_trials"] = trials

    def write_table(name, rows):
        save_results(output / f"{name}.json", rows)
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([{key: json.dumps(json_safe(value)) if isinstance(value, (dict, list))
                                   else value for key, value in row.items()} for row in rows])

    experiment_a_rows = []
    for candidate_id, record in search["stage0"]["candidates"].items():
        validation_metrics = record["validation_metrics"]
        test_metrics = record["ordered_test_metrics"]
        experiment_a_rows.append({
            "architecture": architecture, "candidate_id": candidate_id,
            "inference_mode": record["inference_mode"], "K": record["mc_samples"],
            "selected_for_temporal_search": candidate_id in {
                search["stage0"]["deterministic_selected"], search["stage0"]["mc_selected"]},
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_nll": validation_metrics["nll"],
            "validation_uncertainty_error_auroc": validation_metrics["uncertainty_error_auroc"],
            "validation_latency_seconds": validation_metrics["inference_latency_seconds"],
            "validation_effective_hz": validation_metrics["effective_hz"],
            "ordered_test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "ordered_test_nll": test_metrics["nll"],
            "ordered_test_uncertainty_error_auroc": test_metrics["uncertainty_error_auroc"],
            "ordered_test_latency_seconds": test_metrics["inference_latency_seconds"],
            "ordered_test_effective_hz": test_metrics["effective_hz"],
        })
    write_table("experiment_A_search_results", experiment_a_rows)
    write_table("experiment_A", experiment_a_rows)
    save_results(output / "experiment_A_search_selection.json", {
        "architecture": architecture,
        "deterministic_candidate_id": search["stage0"]["deterministic_selected"],
        "mc_candidate_id": search["stage0"]["mc_selected"],
        "selection_order": ["balanced_accuracy", "lower_nll", "lower_latency"],
        "selection_split": "ordered_validation",
    })

    frontier_rows, pareto_rows = [], []
    for candidate_id, stages in trials.items():
        for stage, values in stages.items():
            for value in values:
                row = {"architecture": architecture, "candidate_id": candidate_id,
                       "stage": stage, "trial_id": value.get("trial_id"),
                       "parent_trial_id": value.get("parent_trial_id"),
                       "parent_stage": value.get("parent_stage"),
                       "lineage": value.get("lineage"), "score_v2": value.get("score_v2"),
                       "is_noop_baseline": value.get("is_noop_baseline", False),
                       "selected_at_lower_boundary": value.get("selected_at_lower_boundary", False),
                       "selected_at_upper_boundary": value.get("selected_at_upper_boundary", False),
                       "balanced_accuracy": value.get("balanced_accuracy"),
                       "transition_window_accuracy": value.get("transition_window_accuracy"),
                       "mean_transition_delay": value.get("mean_transition_delay"),
                       "false_transition_rate": value.get("false_transition_rate"),
                       "parameters": _trial_parameters(value)}
                if value.get("on_stage_frontier"):
                    frontier_rows.append(row)
                if value.get("on_pareto_frontier"):
                    pareto_rows.append(row)
    write_table("stage_frontiers", frontier_rows)
    write_table("pareto_frontiers", pareto_rows)
    trial_rows = [{"candidate_id": candidate_id, "stage": "stage0",
        "family": metadata["inference_mode"],
        "trial_id": metadata.get("stage0_trial_id"), "parent_trial_id": None,
        "parent_stage": None, "lineage": metadata["inference_mode"],
        "score_v2": metadata["validation_metrics"].get("score_v2"),
        "on_stage_frontier": False, "on_pareto_frontier": False,
        "stage_best": False, "selected": False,
        "selection_score": metadata["validation_metrics"].get("selection_score"),
        "balanced_accuracy": metadata["validation_metrics"].get("balanced_accuracy"),
        "transition_window_accuracy": metadata["validation_metrics"].get("transition_window_accuracy"),
        "mean_transition_delay": metadata["validation_metrics"].get("mean_transition_delay"),
        "false_transition_rate": metadata["validation_metrics"].get("false_transition_rate"),
        "nll": metadata["validation_metrics"].get("nll"),
        "brier": metadata["validation_metrics"].get("brier"),
        "predictive_entropy_mean": metadata["validation_metrics"].get("predictive_entropy_mean"),
        "mutual_information_mean": metadata["validation_metrics"].get("mutual_information_mean"),
        "parameters": json.dumps({"dropout_p": metadata["dropout_p"],
            "weight_decay": metadata["weight_decay"], "mc_samples": metadata["mc_samples"]},
            sort_keys=True)} for candidate_id, metadata in search["stage0"]["candidates"].items()]
    for candidate_id, stages in trials.items():
        for stage, values in stages.items():
            for value in values:
                trial_rows.append({"candidate_id": candidate_id, "stage": stage,
                    "family": value.get("family"), "trial_id": value.get("trial_id"),
                    "parent_trial_id": value.get("parent_trial_id"),
                    "parent_stage": value.get("parent_stage"), "lineage": value.get("lineage"),
                    "score_v2": value.get("score_v2"),
                    "on_stage_frontier": value.get("on_stage_frontier", False),
                    "on_pareto_frontier": value.get("on_pareto_frontier", False),
                    "is_noop_baseline": value.get("is_noop_baseline", False),
                    "noop_verified": value.get("noop_verified"),
                    "selected_at_lower_boundary": value.get("selected_at_lower_boundary", False),
                    "selected_at_upper_boundary": value.get("selected_at_upper_boundary", False),
                    "selection_score": value.get("selection_score"),
                    "stage_best": value.get("stage_best", False),
                    "selected": value.get("selected", False),
                    "balanced_accuracy": value.get("balanced_accuracy"),
                    "transition_window_accuracy": value.get("transition_window_accuracy"),
                    "mean_transition_delay": value.get("mean_transition_delay"),
                    "false_transition_rate": value.get("false_transition_rate"),
                    "nll": value.get("nll"), "brier": value.get("brier"),
                    "predictive_entropy_mean": value.get("predictive_entropy_mean"),
                    "mutual_information_mean": value.get("mutual_information_mean"),
                    "parameters": json.dumps(json_safe({k: v for k, v in value.items()
                        if k not in {"confusion_matrix", "per_class_recall", "per_class_precision"}
                        and not k.endswith("accuracy") and k not in {"selection_score", "macro_f1"}}), sort_keys=True)})
    if trial_rows:
        with (output / "search_trials.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = list(dict.fromkeys(key for row in trial_rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(trial_rows)
    deployments = {}
    for mode, key in (("deterministic", "best_deterministic"), ("mc", "best_mc")):
        winner = search[key]
        candidate = winner["candidate"]
        ema = winner["ema"]

        def relative_record(record):
            if isinstance(record, list):
                return [relative_record(value) for value in record]
            if not isinstance(record, dict):
                return record
            return {
                key: ({split: _relative_artifact(path, output)
                       for split, path in value.items()}
                      if key == "trace_paths" and isinstance(value, dict)
                      else _relative_artifact(value, output)
                      if key.endswith("_path") and isinstance(value, (str, Path))
                      else relative_record(value))
                for key, value in record.items()}

        uncertainty_search = winner.get("uncertainty_search")
        if uncertainty_search is not None:
            uncertainty_search = {key: relative_record(value)
                                  if isinstance(value, dict) else value
                                  for key, value in uncertainty_search.items()}
        deployment = {
            "architecture": architecture,
            "model_path": _relative_artifact(candidate["model_path"], output),
            "model_args_path": _relative_artifact(candidate["model_args_path"], output),
            "dropout_p": candidate["dropout_p"],
            "weight_decay": candidate["weight_decay"],
            "inference_mode": mode, "mc_samples": candidate["mc_samples"],
            "selected_temporal_filter_family": winner["winner"]["parameters"]["family"],
            "selected_at_lower_boundary": winner["winner"].get(
                "selected_at_lower_boundary", False),
            "selected_at_upper_boundary": winner["winner"].get(
                "selected_at_upper_boundary", False),
            "temporal_filter_path": _relative_artifact(winner["winner"]["filter_path"], output),
            "temporal_trace_paths": {
                split: _relative_artifact(path, output)
                for split, path in winner["winner"].get("trace_paths", {}).items()},
            "structural_metrics": candidate["structural_metrics"],
            "sequential_instantaneous_validation": candidate["stage0_validation"],
            "sequential_instantaneous_test": candidate["stage0_ordered_test"],
            "uncertainty_stats": candidate["uncertainty_stats"],
            "inference_runtime_seconds": candidate["inference_runtime_seconds"],
            "inference_runtime_estimated_from_mc50": candidate["inference_runtime_estimated_from_mc50"],
            "mc50_cache_runtime_seconds": candidate["mc50_cache_runtime_seconds"],
            "training_runtime_seconds": candidate["training_runtime_seconds"],
            **winner["winner"]["parameters"],
            "validation_metrics": winner["winner"]["validation_metrics"],
            "cv_metrics": winner["winner"]["cv"],
            "ordered_test_metrics": winner["winner"]["ordered_test_metrics"],
            "best_bayes": relative_record(winner["best_bayes"]),
            "best_ema": relative_record(ema),
            "best_low_delay": relative_record(winner["best_low_delay"]),
            "fixed_bayes_winner": relative_record(winner["stage1"]["best"]),
            "candidate_release_winner": relative_record(winner["stage2"]["best"]),
            "mi_gated_winner": relative_record(winner["stage3"]["best"])
            if winner.get("stage3") else None,
            "ambiguity_aware_winner": relative_record(winner["stage4"]["best"]),
            "adaptive_beta_winner": relative_record(
                uncertainty_search.get("adaptive_beta", {}).get("best"))
            if uncertainty_search else None,
            "accumulated_evidence_winner": relative_record(
                uncertainty_search.get("accumulated_evidence", {}).get(
                    "selected", uncertainty_search.get("accumulated_evidence", {}).get("best")))
            if uncertainty_search else None,
            "uncertainty_search": uncertainty_search,
            "controlled_C_chain": relative_record(winner.get("controlled_C_chain")),
        }
        if architecture == "feature_nn":
            deployment.update(extractor_path="extractor.pt", standardizer_path="standardizer.pt")
        else:
            deployment["robot_state_inputs"] = [
                "roll", "pitch", "angular_velocity_roll",
                "angular_velocity_pitch", "angular_velocity_yaw"]
        deployments[mode] = deployment
        save_results(output / f"deployment_{mode}.json", deployment)
        suffix = "" if mode == "deterministic" else "_mc"
        shutil.copyfile(output / deployment["model_path"], output / f"classifier{suffix}.pt")
        shutil.copyfile(output / deployment["model_args_path"], output / f"nn_model_args{suffix}.pt")
        shutil.copyfile(output / deployment["temporal_filter_path"], output / f"bayes_filter{suffix}.pt")
        shutil.copyfile(output / deployment["best_ema"]["filter_path"],
                        output / f"ema_filter{suffix}.pt")
        if mode == "mc":
            shutil.copyfile(output / deployment["best_low_delay"]["filter_path"],
                            output / "bayes_filter_mc_low_delay.pt")

    experiment_a_selected = {
        "architecture": architecture, "selection_split": "ordered_validation",
        "deterministic": {key: deployments["deterministic"].get(key) for key in (
            "model_path", "model_args_path", "dropout_p", "weight_decay",
            "inference_mode", "mc_samples", "extractor_path", "standardizer_path",
            "robot_state_inputs") if key in deployments["deterministic"]},
        "mc": {key: deployments["mc"].get(key) for key in (
            "model_path", "model_args_path", "dropout_p", "weight_decay",
            "inference_mode", "mc_samples", "extractor_path", "standardizer_path",
            "robot_state_inputs") if key in deployments["mc"]},
    }
    save_results(output / "experiment_A_selected_configs.json", experiment_a_selected)

    experiment_b = {mode: search[key]["experiment_B"]
                    for mode, key in (("deterministic", "best_deterministic"),
                                      ("mc", "best_mc"))}
    experiment_b_payload = {
        "architecture": architecture,
        "selection_split": "ordered_validation",
        "classifiers": {mode: {key: deployment.get(key) for key in (
            "model_path", "model_args_path", "dropout_p", "weight_decay",
            "inference_mode", "mc_samples", "extractor_path", "standardizer_path",
            "robot_state_inputs") if key in deployment}
            for mode, deployment in deployments.items()},
        "configurations": relative_record(experiment_b),
    }
    save_results(output / "experiment_B_selected_configs.json", experiment_b_payload)
    b_rows = []
    for mode, values in experiment_b.items():
        for method, record in values.items():
            metrics = record["validation_metrics"]
            b_rows.append({"architecture": architecture, "inference_mode": mode,
                           "method": method, "balanced_accuracy": metrics.get("balanced_accuracy"),
                           "transition_window_accuracy": metrics.get("transition_window_accuracy"),
                           "mean_transition_delay": metrics.get("mean_transition_delay"),
                           "false_transition_rate": metrics.get("false_transition_rate")})
    write_table("experiment_B_search_results", b_rows)

    mc_winner = search["best_mc"]
    experiment_c = {
        "architecture": architecture, "inference_mode": "mc",
        "classifier": experiment_b_payload["classifiers"]["mc"],
        "controlled_C_chain": relative_record(mc_winner["controlled_C_chain"]),
        "best_unrestricted_uncertainty_pipeline": relative_record(mc_winner["winner"]),
        "best_unrestricted_low_delay_pipeline": relative_record(mc_winner["best_low_delay"]),
    }
    save_results(output / "experiment_C_selected_configs.json", experiment_c)
    c_rows = []
    for stage, record in mc_winner["controlled_C_chain"].items():
        if not isinstance(record, dict) or "validation_metrics" not in record:
            continue
        metrics = record["validation_metrics"]
        c_rows.append({"architecture": architecture, "stage": stage,
                       "parent_id": record["parent_id"], "score_v2": metrics["score_v2"],
                       "balanced_accuracy": metrics["balanced_accuracy"],
                       "transition_window_accuracy": metrics["transition_window_accuracy"],
                       "mean_transition_delay": metrics["mean_transition_delay"],
                       "false_transition_rate": metrics["false_transition_rate"]})
    write_table("experiment_C_search_results", c_rows)

    return {"schema_version": 3, "architecture": architecture,
            "fixed_nn_configurations": [
                {"role": role, "dropout_p": dropout, "weight_decay": decay}
                for role, dropout, decay in fixed_models],
            "dropout_search": [], "mc_samples_search": list(mc_values),
            "weight_decay_search": [], "nn_hyperparameter_search": False,
            "max_epochs": 50, "early_stopping_patience": 10,
            "models": model_records, "training_history": histories,
            "search": search, "deployments": deployments,
            "experiment_A_search_results": experiment_a_rows,
            "experiment_A_selected_configs": experiment_a_selected,
            "experiment_B_selected_configs": experiment_b_payload,
            "experiment_C_selected_configs": experiment_c}


def _relative_artifact(path, root):
    path, root = Path(path), Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _trial_parameters(value):
    excluded = {
        "trial_id", "parent_trial_id", "parent_stage", "lineage", "stage", "family",
        "on_stage_frontier", "on_pareto_frontier", "stage_best", "selected", "cv",
        "is_noop_baseline", "noop_expected_exact", "noop_verified",
        "noop_max_posterior_abs_error", "metric_deltas_vs_parent",
        "selected_at_lower_boundary", "selected_at_upper_boundary",
        "lower_boundary_parameters", "upper_boundary_parameters", "trace_paths",
        "selection_score", "score_v2", "legacy_selection_score", "accuracy",
        "balanced_accuracy", "macro_f1", "transition_window_accuracy",
        "steady_state_accuracy", "mean_transition_delay", "false_transition_rate",
        "per_class_recall", "per_class_precision", "minimum_class_recall",
        "missing_predicted_classes", "confusion_matrix",
    }
    return json_safe({key: item for key, item in value.items() if key not in excluded})


def _nn_candidate_record(cid, architecture, dropout_p, mode, samples, model_path,
                         args_path, logits, runtimes, classifier, split_data,
                         training_runtime, weight_decay, runtime_estimated,
                         mc50_cache_runtimes):
    structural = {}
    uncertainty = {}
    for split in ("structural_validation", "structural_test"):
        probabilities, pe, ee, mi = probabilities_and_uncertainty(logits[split], 1.0)
        structural[split] = probability_metrics(classifier, probabilities, split_data[split]["labels"])
        uncertainty[split] = {"predictive_entropy_mean": float(pe.mean()),
                              "expected_entropy_mean": float(ee.mean()),
                              "mutual_information_mean": float(mi.mean()),
                              "mutual_information_p90": float(torch.quantile(mi, 0.9))}
    return {"id": cid, "architecture": architecture, "dropout_p": dropout_p,
            "weight_decay": weight_decay,
            "inference_mode": mode, "mc_samples": samples,
            "model_path": str(model_path), "model_args_path": str(args_path),
            "validation_logits": logits["ordered_validation"],
            "test_logits": logits["ordered_test"],
            "structural_metrics": structural, "uncertainty_stats": uncertainty,
            "inference_runtime_seconds": runtimes,
            "inference_runtime_estimated_from_mc50": runtime_estimated,
            "mc50_cache_runtime_seconds": mc50_cache_runtimes,
            "training_runtime_seconds": training_runtime}


def get_files_for_training() -> tuple[Path, ...]:
    """Compatibility wrapper returning paths in the historical order."""
    _, files = load_training_files()
    return tuple(files[key] for key in (
        "train", "validation", "calibration", "structural_test", "bayes_validation", "ordered_test"
    ))

def get_activation_fn(activation):
    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, str):
        activations = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
        }

        name = activation.lower()
        if name not in activations:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available: {list(activations.keys())}"
            )

        return activations[name]()

    raise TypeError(
        f"activation must be a string or nn.Module, got {type(activation)}"
    )

__all__ = [
    "classifier_metrics", "classifier_metrics_from_scores", "collect_engineered_scores",
    "collect_raw_depth_scores", "collect_neural_logits_batched", "collect_engineered_logits",
    "collect_raw_depth_logits", "pack_raw_depth_state_inputs",
    "probabilities_and_uncertainty", "probability_metrics",
    "evaluate_bayes", "evaluate_bayes_from_scores",
    "extract_dataset_features", "extract_in_chunks",
    "FeatureStandardizer", "fit_nn", "fit_standardizer", "json_safe", "load_training_files",
    "get_activation_fn", "make_terrain_extractor", "save_results", "sequence_ids_for",
    "transition_training_kwargs", "processing_batch_size",
    "run_staged_sequential_pipeline",
    "train_uncertainty_aware_nn_suite",
]
