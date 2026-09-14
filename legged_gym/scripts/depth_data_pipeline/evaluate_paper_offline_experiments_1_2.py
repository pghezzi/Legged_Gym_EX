"""Train and evaluate the frozen offline terrain-classification Experiments 1--2.

This runner deliberately contains no model or temporal-filter search.  Every
seed uses the same compiled files and the fixed configurations below.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.dataset_provenance import validate_partitions
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    BayesianTerrainFilter,
    TRANSITION_ACCOUNTING_VERSION,
    NeuralClassifierAdapter,
    fit_nn,
    make_persistent_transition_matrix,
    run_filter_sequences,
)
from .sequential_terrain_filter_extensions import (
    EMALogitPatienceFilter,
    evaluate_sequential_predictions,
    run_ema_logit_patience_sequences,
)
from .train_feature_nn import TerrainDepthFeatureClassifierNN
from .train_raw_depth_nn import TerrainDepthClassifierNN
from .util_func import (
    collect_engineered_logits,
    collect_raw_depth_logits,
    extract_dataset_features,
    fit_standardizer,
    json_safe,
    make_terrain_extractor,
    pack_raw_depth_state_inputs,
    probability_metrics,
    save_results,
    sequence_ids_for,
)
from rsl_rl.utils.training_cost import (
    WallTimer,
    stage_accounting,
    artifact_size_mb,
    cuda_device_info,
    peak_memory_mb,
    provenance_map,
    reset_peak_memory,
    write_cost_record,
)


MODEL_SEEDS = (0, 1, 2)
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
FIXED_MODEL_CONFIGS = {
    "feature_nn": {"dropout_p": 0.0, "weight_decay": 1e-5},
    "raw_depth_nn": {"dropout_p": 0.20, "weight_decay": 1e-5},
}
FIXED_EMA_CONFIG = {"ema_alpha": 0.6, "change_patience": 1}
FIXED_BAYES_CONFIG = {
    "T_filter": 1.0,
    "stable_stay": 0.90,
    "evidence_power": 1.0,
    "observation_mix": 0.0,
    "adaptive_evidence": False,
}
SCALAR_EXPERIMENT_1_METRICS = (
    "accuracy", "balanced_accuracy", "macro_f1", "nll", "brier",
    "inference_latency_seconds", "inference_latency_ms_per_sample",
)
SCALAR_EXPERIMENT_2_METRICS = (
    "accuracy", "balanced_accuracy", "macro_f1", "transition_window_accuracy",
    "steady_state_accuracy", "mean_transition_delay", "false_transition_rate",
    "minimum_class_recall",
)
TRANSITION_V2_METRICS = (
    "total_transitions_v2", "matched_transitions_v2", "missed_transitions_v2",
    "transition_miss_rate_v2", "mean_matched_transition_delay_frames_v2",
)
TRANSITION_RECORD_FIELDS = (
    "architecture", "temporal_method", "seed", "transition_metric_version",
    "sequence_id", "sequence_start_frame", "transition_frame", "sequence_transition_frame",
    "segment_end_frame_exclusive", "from_label", "target_label", "matched", "missed",
    "matched_frame", "delay_classification_frames",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _contains(folder: Path, names: Sequence[str]) -> bool:
    return folder.is_dir() and all((folder / f"{name}.pt").is_file() for name in names)


def _resolve_data_folder(explicit: Path | None, root: Path, candidates: Sequence[str],
                         required: Sequence[str], kind: str) -> Path:
    if explicit is not None:
        folder = explicit.expanduser().resolve()
        if not _contains(folder, required):
            raise FileNotFoundError(f"{kind} folder {folder} must contain {tuple(required)}")
        return folder
    for folder in [*(root / name for name in candidates), root]:
        if _contains(folder, required):
            return folder.resolve()
    raise FileNotFoundError(f"could not locate {kind} data below {root}")


def _labels(values: Sequence[Any] | torch.Tensor) -> list[Any]:
    return values.detach().cpu().tolist() if torch.is_tensor(values) else list(values)


def _metric_details(metrics: Mapping[str, Any], class_ids: Sequence[Any]) -> dict[str, Any]:
    result = dict(metrics)
    matrix = torch.as_tensor(result["confusion_matrix"])
    recalls = {}
    for index, label in enumerate(class_ids):
        recalls[str(label)] = float(matrix[index, index]) / max(float(matrix[index].sum()), 1.0)
    result["per_class_recall"] = recalls
    result["minimum_class_recall"] = min(recalls.values()) if recalls else float("nan")
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str],
               scalar_metrics: Sequence[str], class_ids: Sequence[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    summaries = []
    for group, members in grouped.items():
        summary = dict(zip(group_keys, group))
        summary["num_seeds"] = len(members)
        summary["seeds"] = [int(row["seed"]) for row in members]
        for metric in scalar_metrics:
            values = np.asarray([float(row[metric]) for row in members], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(finite.mean()) if finite.size else float("nan")
            summary[f"{metric}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            if metric in TRANSITION_V2_METRICS and finite.size < 2:
                summary[f"{metric}_std"] = float("nan")
            summary[f"{metric}_raw"] = values.tolist()
        recall_values = {
            str(label): np.asarray([row["per_class_recall"][str(label)] for row in members])
            for label in class_ids
        }
        summary["per_class_recall_mean"] = {
            label: float(values.mean()) for label, values in recall_values.items()
        }
        summary["per_class_recall_std"] = {
            label: float(values.std(ddof=1)) if len(values) > 1 else 0.0
            for label, values in recall_values.items()
        }
        matrices = np.asarray([row["confusion_matrix"] for row in members], dtype=np.float64)
        summary["confusion_matrix_mean"] = matrices.mean(axis=0).tolist()
        summary["confusion_matrix_std"] = matrices.std(axis=0, ddof=1).tolist()
        summaries.append(summary)
        if "total_transitions_v2" in scalar_metrics:
            summary["transition_metric_version"] = TRANSITION_ACCOUNTING_VERSION
    return summaries


def _csv_value(value: Any) -> Any:
    return json.dumps(json_safe(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else value


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]],
                fields: Sequence[str] | None = None) -> None:
    fields = list(fields) if fields is not None else list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _csv_value(row.get(key)) for key in fields} for row in rows])


def _instantaneous_metrics(classifier: NeuralClassifierAdapter, logits: torch.Tensor,
                           labels: Sequence[Any], runtime: float) -> dict[str, Any]:
    scores = torch.as_tensor(logits).squeeze(0).cpu()
    probabilities = F.softmax(scores, dim=1)
    result = probability_metrics(classifier, probabilities, labels)
    result = _metric_details(result, classifier.class_ids)
    result["inference_latency_seconds"] = float(runtime)
    result["inference_latency_ms_per_sample"] = 1000.0 * float(runtime) / max(len(labels), 1)
    return result


def _sequential_metrics(classifier: NeuralClassifierAdapter, logits: torch.Tensor,
                        labels: Sequence[Any], sequence_ids: Sequence[Any],
                        trace_sink: dict | None = None) -> dict[str, dict[str, Any]]:
    scores = torch.as_tensor(logits).squeeze(0).cpu()
    probabilities = F.softmax(scores / FIXED_BAYES_CONFIG["T_filter"], dim=1)
    truth = _labels(labels)
    ids = _labels(sequence_ids)
    instantaneous = [classifier.class_ids[index] for index in scores.argmax(1).tolist()]

    ema_filter = EMALogitPatienceFilter(classifier.class_ids, **FIXED_EMA_CONFIG)
    ema = run_ema_logit_patience_sequences(ema_filter, scores, sequence_ids=ids)

    prior = torch.full((len(classifier.class_ids),), 1.0 / len(classifier.class_ids))
    transition = make_persistent_transition_matrix(
        classifier.class_ids, FIXED_BAYES_CONFIG["stable_stay"], device="cpu")
    observation = torch.eye(len(classifier.class_ids))
    bayes_filter = BayesianTerrainFilter(
        classifier.class_ids, prior, transition, observation,
        evidence_power=FIXED_BAYES_CONFIG["evidence_power"], adaptive_evidence=False,
        min_evidence_power=1.0, confidence_gamma=1.0,
        stay_probability=FIXED_BAYES_CONFIG["stable_stay"],
        transition_source="persistent", device="cpu")
    bayes, beliefs, powers = run_filter_sequences(
        bayes_filter, probabilities, sequence_ids=ids, return_traces=trace_sink is not None)

    if trace_sink is not None:
        # Reuse the deployed EMA update; verify tracing reproduces the runner.
        ema_scores, replay = [], []
        for i, score in enumerate(scores):
            if i == 0 or ids[i] != ids[i-1]:
                ema_filter.reset()
            replay.append(ema_filter.update(score))
            ema_scores.append(ema_filter.ema_scores.detach().cpu().clone())
        if replay != ema:
            raise AssertionError("EMA trace differs from evaluated predictions")
        smoothed = torch.stack(ema_scores)
        trace_sink.update(logits=scores.clone(), probabilities=scores.softmax(-1),
                          ema_scores=smoothed, ema_probabilities=smoothed.softmax(-1),
                          bayes_beliefs=beliefs, bayes_evidence_powers=powers)
        for name, values in (("instantaneous", instantaneous), ("ema", ema), ("bayes", bayes)):
            trace_sink[name + "_selected"] = torch.tensor([classifier.class_ids.index(v) for v in values])

    return {
        name: evaluate_sequential_predictions(truth, predictions, classifier.class_ids, ids,
                                              transition_accounting_v2=True)
        for name, predictions in (("instantaneous", instantaneous), ("ema", ema), ("bayes", bayes))
    }


def _write_figure_bundle(output, structural_test, ordered_test, ordered_ids, extractor,
                         runs, e1_rows, e2_rows, e1_summary, e2_summary, classes, settings):
    from .paper_figure_bundle import (PLOT_CONFIG, geometric_examples, provenance,
                                     sample_transitions, transition_thumbnails, save_bundle, plot_bundle)
    examples, unavailable = geometric_examples(extractor, structural_test, classes,
        PLOT_CONFIG["images_per_class"], PLOT_CONFIG["sampling_seed"])
    ordered_meta = provenance(ordered_test, _labels(ordered_ids))
    selections, transition_sampling = sample_transitions(ordered_test, ordered_meta["sequence_ids"], runs,
        PLOT_CONFIG["transition_samples"], PLOT_CONFIG["sampling_seed"], PLOT_CONFIG["timeline_padding"],
        PLOT_CONFIG["disagreement_fraction"])
    timeline = selections[0] if selections else None
    if len(selections) < PLOT_CONFIG["transition_samples"]:
        unavailable.append(f"Only {len(selections)}/{PLOT_CONFIG['transition_samples']} distinct GT transitions available")
    thumbnails = transition_thumbnails(ordered_test, selections, PLOT_CONFIG["thumbnail_count"])
    if ordered_meta["unavailable"]:
        unavailable.append("Ordered provenance unavailable: " + ", ".join(ordered_meta["unavailable"]))
    bundle = dict(schema_version=1, class_ordering=list(classes), model_seeds=list(MODEL_SEEDS),
                  runs=runs, structural_truth=_labels(structural_test["labels"]), ordered_truth=_labels(ordered_test["labels"]),
                  structural_provenance=provenance(structural_test), ordered_provenance=ordered_meta,
                  experiment_1_rows=e1_rows, experiment_2_rows=e2_rows,
                  experiment_1_summary=e1_summary, experiment_2_summary=e2_summary,
                  examples=examples, timeline_selection=timeline, timeline_selections=selections,
                  transition_thumbnails=thumbnails, unavailable=unavailable,
                  sampling_metadata={"seed": PLOT_CONFIG["sampling_seed"], "images_per_class": PLOT_CONFIG["images_per_class"],
                    "example_dataset": "structural_test", "transitions": transition_sampling,
                    "thumbnail_rule": "Boundary frames, preceding frames, endpoints, focal frame, then evenly-spaced context; 48x64 bilinear thumbnails"},
                  example_selection_rule="Seeded random source/env round-robin with sequence/time strata; up to 100 distinct valid structural-test images per GT class; retry invalid inputs",
                  timeline_selection_rule="Up to 100 unique focal GT transitions: 50% disagreement-enriched quota and remaining unrestricted coverage; include adjacent GT boundary where available; no episode crossing",
                  aggregation_rules={"confusion": "sum seed counts then normalize each GT row; absent class=N/A",
                                     "uncertainty": "sample standard deviation across seeds, not standard error; scatter uses finite paired observations",
                                     "legacy_figures": "unchanged equal-weight seed mean/std"},
                  filter_settings={"ema": dict(FIXED_EMA_CONFIG), "bayes": dict(FIXED_BAYES_CONFIG)},
                  model_settings=settings, plot_config=dict(PLOT_CONFIG),
                  skill_semantics="Offline emitted terrain label requests its corresponding skill; no locomotion policy is executed")
    info = save_bundle(bundle, output / "figure_data.pt")
    info.update(plot_bundle(bundle, output))
    save_results(output / "figure_manifest.json", info)
    print("Figure inputs saved to", info["bundle"])
    for reason in info["unavailable"]:
        print("Figure input unavailable:", reason)
    return info


def _bundle_from_existing(args):
    """Reporting-only entry point; never enters the training/cost-accounting path."""
    from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import FeatureStandardizer
    source = args.bundle_from_existing.expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["fixed_ema_configuration"] != FIXED_EMA_CONFIG or manifest["fixed_bayes_configuration"] != FIXED_BAYES_CONFIG:
        raise ValueError("Existing run's frozen filter settings differ from this evaluator")
    output = args.output.expanduser().resolve()
    if output == source:
        raise ValueError("Use a separate --output directory to preserve existing offline results/cost records")
    output.mkdir(parents=True, exist_ok=True)
    structural_dir = (args.classifier_data or Path(manifest["structural_dataset"])).expanduser().resolve()
    ordered_dir = (args.ordered_data or Path(manifest["ordered_dataset"])).expanduser().resolve()
    load = lambda path: torch.load(path, map_location="cpu", weights_only=False)
    structural, ordered = load(structural_dir / "test.pt"), load(ordered_dir / "test.pt")
    train, val = load(structural_dir / "train.pt"), load(structural_dir / "val.pt")
    checks = validate_partitions({"train": train, "val": val, "test": structural}, args.allow_legacy_provenance)
    ordered_checks = validate_partitions({"train": train, "val": val, "test": ordered}, args.allow_legacy_provenance)
    del train, val
    ids = sequence_ids_for(ordered, allow_legacy=args.allow_legacy_provenance)
    classes = manifest["class_ordering"]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    feature_dir = source / "artifacts" / "feature_nn"
    extractor = SobelDepthTerrainFeatureExtractor.load(feature_dir / "extractor.pt")
    standardizer = FeatureStandardizer.load(feature_dir / "standardizer.pt")
    e1, e2, runs = [], [], []
    for architecture, constructor in (("feature_nn", TerrainDepthFeatureClassifierNN), ("raw_depth_nn", TerrainDepthClassifierNN)):
        for seed in MODEL_SEEDS:
            seed_dir = source / "artifacts" / architecture / f"seed_{seed}"
            model_args = dict(load(seed_dir / "nn_model_args.pt"))
            model_args.pop("cls", None)
            model = constructor(**model_args)
            classifier = NeuralClassifierAdapter.load(seed_dir / "classifier.pt", model, device=device)
            classifier.require_feature = architecture == "feature_nn"
            if list(classifier.class_ids) != list(classes):
                raise ValueError("Checkpoint class ordering differs from offline manifest")
            if architecture == "feature_nn":
                collect = lambda data: collect_engineered_logits(classifier, extractor, standardizer, data,
                    chunk_size=args.batch_size, mc_samples=1, mc_dropout=False)
            else:
                collect = lambda data: collect_raw_depth_logits(classifier, data, chunk_size=args.batch_size, mc_samples=1, mc_dropout=False)
            structural_logits, runtime = collect(structural)
            ordered_logits, _ = collect(ordered)
            instant = _instantaneous_metrics(classifier, structural_logits, structural["labels"], runtime)
            common = {"architecture": architecture, "seed": seed, "model_path": str(seed_dir / "classifier.pt")}
            e1.append({**common, **{key: instant[key] for key in SCALAR_EXPERIMENT_1_METRICS},
                       "per_class_recall": instant["per_class_recall"], "confusion_matrix": json_safe(instant["confusion_matrix"])})
            trace = {}
            metrics = _sequential_metrics(classifier, ordered_logits, ordered["labels"], ids, trace_sink=trace)
            for method, values in metrics.items():
                e2.append({**common, "temporal_method": method,
                           **{key: values[key] for key in SCALAR_EXPERIMENT_2_METRICS + TRANSITION_V2_METRICS},
                           "per_class_recall": values["per_class_recall"], "confusion_matrix": json_safe(values["confusion_matrix"])})
            runs.append({**common, "model_args": model.get_args(), "trace": trace,
                         "structural_logits": structural_logits.squeeze(0).cpu(),
                         "structural_probabilities": structural_logits.squeeze(0).cpu().softmax(-1)})
            classifier.to("cpu")
    s1 = _aggregate(e1, ("architecture",), SCALAR_EXPERIMENT_1_METRICS, classes)
    s2 = _aggregate(e2, ("architecture", "temporal_method"), SCALAR_EXPERIMENT_2_METRICS + TRANSITION_V2_METRICS, classes)
    _make_plots(output, s1, s2, classes)
    info = _write_figure_bundle(output, structural, ordered, ids, extractor, runs, e1, e2, s1, s2, classes,
        {"source_manifest": manifest, "extractor": load(feature_dir / "extractor.pt"),
         "standardizer": load(feature_dir / "standardizer.pt")})
    save_results(output / "manifest.json", {"source_offline_dir": str(source), "training_performed": False,
        "provenance_checks": checks, "ordered_provenance_checks": ordered_checks, "figures": info})


def _make_plots(output: Path, experiment_1: Sequence[Mapping[str, Any]],
                experiment_2: Sequence[Mapping[str, Any]], class_ids: Sequence[Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    architectures = ("feature_nn", "raw_depth_nn")
    display = {"feature_nn": "Feature NN", "raw_depth_nn": "Raw-depth NN"}
    colors = {"feature_nn": "#377eb8", "raw_depth_nn": "#e6550d"}
    e1 = {row["architecture"]: row for row in experiment_1}

    def simple_bar(metric: str, ylabel: str, filename: str) -> None:
        means = [e1[name][f"{metric}_mean"] for name in architectures]
        errors = [e1[name][f"{metric}_std"] for name in architectures]
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.bar([display[name] for name in architectures], means, yerr=errors,
               capsize=4, color=[colors[name] for name in architectures])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(output / filename, dpi=300)
        fig.savefig((output / filename).with_suffix(".pdf")); plt.close(fig)

    simple_bar("balanced_accuracy", "Balanced accuracy", "experiment_1_balanced_accuracy.png")
    simple_bar("nll", "Negative log-likelihood", "experiment_1_nll.png")

    x = np.arange(len(class_ids)); width = 0.36
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(class_ids)), 4.0))
    for offset, architecture in zip((-width / 2, width / 2), architectures):
        means = e1[architecture]["per_class_recall_mean"]
        ax.bar(x + offset, [means[str(label)] for label in class_ids], width,
               label=display[architecture], color=colors[architecture])
    ax.set_xticks(x, [str(label) for label in class_ids], rotation=25, ha="right")
    ax.set_ylabel("Recall"); ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "experiment_1_per_class_recall.png", dpi=300)
    fig.savefig(output / "experiment_1_per_class_recall.pdf"); plt.close(fig)

    methods = ("instantaneous", "ema", "bayes")
    e2 = {(row["architecture"], row["temporal_method"]): row for row in experiment_2}

    def grouped_bar(metric: str, ylabel: str, filename: str) -> None:
        x = np.arange(len(methods)); width = 0.36
        fig, ax = plt.subplots(figsize=(6.2, 3.9))
        for offset, architecture in zip((-width / 2, width / 2), architectures):
            rows = [e2[(architecture, method)] for method in methods]
            ax.bar(x + offset, [row[f"{metric}_mean"] for row in rows], width,
                   yerr=[row[f"{metric}_std"] for row in rows], capsize=3,
                   label=display[architecture], color=colors[architecture])
        ax.set_xticks(x, ("Instantaneous", "EMA", "Bayes"))
        ax.set_ylabel(ylabel); ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(output / filename, dpi=300)
        fig.savefig((output / filename).with_suffix(".pdf")); plt.close(fig)

    grouped_bar("balanced_accuracy", "Balanced accuracy", "experiment_2_balanced_accuracy.png")
    grouped_bar("transition_window_accuracy", "Transition-window accuracy (±5)",
                "experiment_2_transition_accuracy.png")
    grouped_bar("false_transition_rate", "False-transition rate",
                "experiment_2_false_transition_rate.png")
    grouped_bar("mean_transition_delay", "Mean transition delay (frames)",
                "experiment_2_transition_delay.png")

    abbreviations = {"feature_nn": "feat", "raw_depth_nn": "raw"}
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    markers = {"instantaneous": "o", "ema": "s", "bayes": "^"}
    for architecture in architectures:
        for method in methods:
            row = e2[(architecture, method)]
            x_value = row["mean_transition_delay_mean"]
            y_value = row["balanced_accuracy_mean"]
            ax.scatter(x_value, y_value, color=colors[architecture], marker=markers[method], s=55)
            ax.annotate(f"{abbreviations[architecture]}-{method}", (x_value, y_value),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean transition delay (frames)"); ax.set_ylabel("Balanced accuracy")
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "experiment_2_accuracy_delay_scatter.png", dpi=300)
    fig.savefig(output / "experiment_2_accuracy_delay_scatter.pdf"); plt.close(fig)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(os.environ.get(
        "TERRAIN_CLASSIFIER_DATASET",
        str(Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "processed_data")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_root)
    parser.add_argument("--classifier-data", type=Path,
                        help="compiled structural train/val/calibration/test folder")
    parser.add_argument("--ordered-data", "--bayesian-data", dest="ordered_data", type=Path,
                        help="compiled ordered test folder (defaults below --dataset)")
    parser.add_argument("--output", type=Path, default=Path("paper_offline_eval"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet-training", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plot-only", type=Path, metavar="BUNDLE", help="Regenerate all figures from figure_data.pt only")
    modes.add_argument("--bundle-from-existing", type=Path, metavar="OFFLINE_DIR", help="Evaluate saved checkpoints into a new bundle without training")
    parser.add_argument("--allow-legacy-provenance", action="store_true",
                        help="Explicit unverified per_eps fallback for historical compiled datasets")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.plot_only:
        from .paper_figure_bundle import load_bundle, plot_bundle
        bundle = load_bundle(args.plot_only)
        args.output.mkdir(parents=True, exist_ok=True)
        _make_plots(args.output, bundle["experiment_1_summary"], bundle["experiment_2_summary"], bundle["class_ordering"])
        report = plot_bundle(bundle, args.output)
        report["bundle"] = str(args.plot_only.resolve())
        save_results(args.output / "figure_manifest.json", report)
        for reason in report["unavailable"]:
            print("Figure input unavailable:", reason)
        print("Figures regenerated from bundle only:", args.output)
        return
    if args.bundle_from_existing:
        _bundle_from_existing(args)
        return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    setup_timer = WallTimer(device).start()
    dataset_root = args.dataset.expanduser().resolve()
    structural_dir = _resolve_data_folder(
        args.classifier_data, dataset_root, ("structural", "classifier"),
        ("train", "val", "calibration", "test"), "structural")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    loading_timer = WallTimer(device).start()
    train = torch.load(structural_dir / "train.pt", map_location="cpu", weights_only=False)
    validation = torch.load(structural_dir / "val.pt", map_location="cpu", weights_only=False)
    data_loading_s = loading_timer.stop()
    class_ids = list(dict.fromkeys(_labels(train["labels"])))
    unknown = set(_labels(validation["labels"])) - set(class_ids)
    if unknown:
        raise ValueError(f"validation contains labels absent from training: {sorted(unknown, key=str)}")
    shared_setup_s = setup_timer.stop()
    # Evaluation-only datasets/sequence preparation are outside integration costs.
    ordered_dir = _resolve_data_folder(
        args.ordered_data, dataset_root, ("bayesian", "bayes", "sequences"),
        ("test",), "ordered")
    structural_test = torch.load(structural_dir / "test.pt", map_location="cpu", weights_only=False)
    ordered_test = torch.load(ordered_dir / "test.pt", map_location="cpu", weights_only=False)
    for name, data in (("structural test", structural_test),
                       ("ordered test", ordered_test)):
        unknown = set(_labels(data["labels"])) - set(class_ids)
        if unknown:
            raise ValueError(f"{name} contains labels absent from training: {sorted(unknown, key=str)}")
    provenance_checks = validate_partitions(
        {"train": train, "validation": validation, "structural_test": structural_test},
        args.allow_legacy_provenance)
    # Ordered test may legitimately equal structural test, but never train/val.
    ordered_provenance_checks = validate_partitions(
        {"train": train, "validation": validation, "ordered_test": ordered_test},
        args.allow_legacy_provenance)
    ordered_ids = sequence_ids_for(ordered_test, allow_legacy=args.allow_legacy_provenance)

    reset_peak_memory(device)
    feature_total_timer = WallTimer(device).start()
    preprocessing_timer = WallTimer(device).start()
    extractor = make_terrain_extractor(structural_dir / "calibration.pt")
    train_features = extract_dataset_features(extractor, train, chunk_size=args.batch_size)
    validation_features = extract_dataset_features(extractor, validation, chunk_size=args.batch_size)
    standardizer = fit_standardizer(train_features, train["labels"], batch_size=args.batch_size)
    train_features = standardizer.transform(train_features)
    validation_features = standardizer.transform(validation_features)
    feature_preprocessing_s = preprocessing_timer.stop()
    feature_serialization_timer = WallTimer(device).start()
    feature_artifacts = output / "artifacts" / "feature_nn"
    feature_artifacts.mkdir(parents=True, exist_ok=True)
    extractor.save(feature_artifacts / "extractor.pt")
    standardizer.save(feature_artifacts / "standardizer.pt")
    feature_serialization_s = feature_serialization_timer.stop()
    feature_total_s = feature_total_timer.stop()
    feature_preprocessing_peak_mb = peak_memory_mb(device)

    experiment_1_rows, experiment_2_rows, training_records = [], [], []
    figure_runs = []
    transition_records = []
    raw_train = raw_validation = None
    for architecture in ("feature_nn", "raw_depth_nn"):
        config = FIXED_MODEL_CONFIGS[architecture]
        if architecture == "raw_depth_nn":
            # Match the existing raw-depth trainer while avoiding a second copy
            # of these large packed inputs during all feature-NN runs.
            reset_peak_memory(device)
            raw_total_timer = WallTimer(device).start()
            preprocessing_timer = WallTimer(device).start()
            raw_train = pack_raw_depth_state_inputs(
                train["depth_images"], train["orientation_rpy"], train["angular_velocity"])
            raw_validation = pack_raw_depth_state_inputs(
                validation["depth_images"], validation["orientation_rpy"],
                validation["angular_velocity"])
            raw_preprocessing_s = preprocessing_timer.stop()
            raw_total_s = raw_total_timer.stop()
            raw_preprocessing_peak_mb = peak_memory_mb(device)
        for seed in MODEL_SEEDS:
            seed_total_timer = WallTimer(device).start()
            model_setup_timer = WallTimer(device).start()
            _set_seed(seed)
            if architecture == "feature_nn":
                model = TerrainDepthFeatureClassifierNN(
                    train_features.shape[1], [512, 256, 128, len(class_ids)],
                    nn.ELU(), config["dropout_p"])
                train_inputs, validation_inputs = train_features, validation_features
            else:
                model = TerrainDepthClassifierNN(
                    train["depth_images"].shape[-2:], 1, [8, 16], [1, 1],
                    [128, len(class_ids)], [5, 3], nn.ELU(), config["dropout_p"],
                    robot_state_dim=5)
                train_inputs, validation_inputs = raw_train, raw_validation
            classifier = NeuralClassifierAdapter(model, class_ids, fit_callback=fit_nn, device=device)
            classifier.require_feature = architecture == "feature_nn"
            reset_peak_memory(device)
            model_setup_s = model_setup_timer.stop()
            optimization_timer = WallTimer(device).start()
            classifier.fit(
                train_inputs, train["labels"], val=(validation_inputs, validation["labels"]),
                epochs=MAX_EPOCHS, batch_size=args.batch_size,
                validation_batch_size=args.batch_size,
                optimizer_kwargs={"weight_decay": config["weight_decay"]},
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, verbose=not args.quiet_training)
            training_seconds = optimization_timer.stop()

            seed_dir = output / "artifacts" / architecture / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            model_path = seed_dir / "classifier.pt"
            args_path = seed_dir / "nn_model_args.pt"
            history_path = seed_dir / "training_history.json"
            serialization_timer = WallTimer(device).start()
            classifier.save(model_path)
            torch.save(model.get_args(), args_path)
            save_results(history_path, classifier.training_history)
            serialization_s = serialization_timer.stop()
            preprocessing_s = (
                feature_preprocessing_s if architecture == "feature_nn" else raw_preprocessing_s
            )
            epochs_completed = int(classifier.training_history["epochs_completed"])
            optimization_peak_mb = peak_memory_mb(device)
            preprocessing_peak_mb = (
                feature_preprocessing_peak_mb if architecture == "feature_nn"
                else raw_preprocessing_peak_mb
            )
            seed_total_s = seed_total_timer.stop()
            shared_preparation_s = feature_total_s if architecture == "feature_nn" else raw_total_s
            total_s = shared_setup_s + shared_preparation_s + seed_total_s
            all_serialization_s = serialization_s + (feature_serialization_s if architecture == "feature_nn" else 0.)
            cost_record = {
                "schema_version": 1,
                "run_type": "classifier_training",
                "method": "Feature Router" if architecture == "feature_nn" else "Raw-Depth Router",
                "architecture": architecture, "seed": seed, "model_path": str(model_path),
                "structural_dataset": str(structural_dir),
                "data_loading_s": data_loading_s,
                "shared_preparation_s": shared_setup_s + shared_preparation_s,
                "model_args_path": str(args_path), "training_history_path": str(history_path),
                "dropout_p": config["dropout_p"], "weight_decay": config["weight_decay"],
                "training_runtime_seconds": training_seconds,
                "training_samples": len(train["labels"]),
                "validation_samples": len(validation["labels"]),
                "test_samples": len(structural_test["labels"]),
                "batch_size": args.batch_size,
                "epochs_completed": epochs_completed,
                "optimizer_updates": int(classifier.training_history.get(
                    "optimizer_updates",
                    epochs_completed * math.ceil(len(train["labels"]) / args.batch_size))),
                "best_epoch": classifier.training_history["best_epoch"],
                "best_epoch_one_based": None if classifier.training_history["best_epoch"] is None
                else int(classifier.training_history["best_epoch"]) + 1,
                "early_stopping_epoch": epochs_completed if classifier.training_history["stopped_early"] else None,
                "stopped_early": classifier.training_history["stopped_early"],
                "best_weights_restored": classifier.training_history["best_weights_restored"],
                "preprocessing_s": preprocessing_s,
                "optimization_s": training_seconds,
                "artifact_serialization_s": serialization_s,
                "total_classifier_training_s": preprocessing_s + training_seconds + serialization_s,
                "total_wallclock_s": preprocessing_s + training_seconds + serialization_s,
                "gpu_active_time_s": None,
                "gpu_hours": (preprocessing_s + training_seconds + serialization_s) / 3600.0
                if torch.device(device).type == "cuda" else 0.0,
                "peak_gpu_memory_mb": max(
                    value for value in (preprocessing_peak_mb, optimization_peak_mb)
                    if value is not None
                ) if torch.device(device).type == "cuda" else None,
                "trainable_params": sum(parameter.numel() for parameter in model.parameters()
                                        if parameter.requires_grad),
                "checkpoint_size_mb": artifact_size_mb(model_path),
                "artifact_size_mb": artifact_size_mb(model_path) + artifact_size_mb(args_path) + (
                    artifact_size_mb(feature_artifacts / "extractor.pt")
                    + artifact_size_mb(feature_artifacts / "standardizer.pt")
                    if architecture == "feature_nn" else 0.0),
                "data_env_steps": 0,
                "data_generation_s": 0.0,
                "additional_locomotion_policy_training": False,
                "additional_locomotion_policy_env_steps": 0,
                "total_post_specialist_env_steps": 0,
                "ema_training_s": 0.0,
                "bayes_training_s": 0.0,
                **cuda_device_info(device),
                "notes": [
                    "Specialist locomotion policies are pre-existing and excluded.",
                    "Feature generation/prepacking was measured once and is reported for each independent seed.",
                    "GPU-hours are synchronized allocated-device wall-clock; kernel-active time requires an external profiler.",
                    "Optimizer updates are counted at each optimizer.step call.",
                ],
            }
            cost_record.update(stage_accounting(total_s, cost_record["gpu_count"],
                setup_s=shared_setup_s + model_setup_s, preprocessing_s=preprocessing_s,
                optimization_s=training_seconds, artifact_serialization_s=all_serialization_s))
            cost_record["total_classifier_training_s"] = total_s
            cost_record["notes"].append(
                "Setup includes train/validation loading and model construction; calibration loading/fitting is preprocessing. "
                "Shared preparation is charged once per independent seed/architecture alternative; evaluation-only loading, sequence preparation and inference are excluded.")
            cost_record["metric_status"] = provenance_map(cost_record)
            cost_record["metric_status"]["optimizer_updates"] = (
                "measured" if "optimizer_updates" in classifier.training_history
                else "reconstructed"
            )
            cost_record["metric_status"]["gpu_active_time_s"] = "unavailable"
            cost_record["metric_status"]["gpu_hours"] = (
                "reconstructed" if torch.device(device).type == "cuda" else "measured"
            )
            write_cost_record(seed_dir / "training_cost.json", cost_record)
            training_records.append(cost_record)

            if architecture == "feature_nn":
                structural_logits, structural_runtime = collect_engineered_logits(
                    classifier, extractor, standardizer, structural_test,
                    chunk_size=args.batch_size, mc_samples=1, mc_dropout=False)
                ordered_logits, _ = collect_engineered_logits(
                    classifier, extractor, standardizer, ordered_test,
                    chunk_size=args.batch_size, mc_samples=1, mc_dropout=False)
            else:
                structural_logits, structural_runtime = collect_raw_depth_logits(
                    classifier, structural_test, chunk_size=args.batch_size,
                    mc_samples=1, mc_dropout=False)
                ordered_logits, _ = collect_raw_depth_logits(
                    classifier, ordered_test, chunk_size=args.batch_size,
                    mc_samples=1, mc_dropout=False)

            instant = _instantaneous_metrics(
                classifier, structural_logits, structural_test["labels"], structural_runtime)
            experiment_1_rows.append({
                "architecture": architecture, "seed": seed, "model_path": str(model_path),
                **{key: instant[key] for key in SCALAR_EXPERIMENT_1_METRICS},
                "per_class_recall": instant["per_class_recall"],
                "confusion_matrix": json_safe(instant["confusion_matrix"]),
            })
            trace = {}
            sequential_metrics = _sequential_metrics(
                classifier, ordered_logits, ordered_test["labels"], ordered_ids, trace_sink=trace)
            figure_runs.append({"architecture": architecture, "seed": seed, "trace": trace,
                                "model_path": str(model_path), "model_args": model.get_args(),
                                "structural_logits": structural_logits.squeeze(0).cpu(),
                                "structural_probabilities": structural_logits.squeeze(0).cpu().softmax(-1)})
            for method, metrics in sequential_metrics.items():
                experiment_2_rows.append({
                    "architecture": architecture, "temporal_method": method,
                    "seed": seed, "model_path": str(model_path),
                    **{key: metrics[key] for key in SCALAR_EXPERIMENT_2_METRICS + TRANSITION_V2_METRICS},
                    "transition_metric_version": TRANSITION_ACCOUNTING_VERSION,
                    "per_class_recall": metrics["per_class_recall"],
                    "confusion_matrix": json_safe(metrics["confusion_matrix"]),
                })
                transition_records.extend({
                    "architecture": architecture, "temporal_method": method, "seed": seed,
                    "transition_metric_version": TRANSITION_ACCOUNTING_VERSION, **record,
                } for record in metrics["transition_records_v2"])
            classifier.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    experiment_1_summary = _aggregate(
        experiment_1_rows, ("architecture",), SCALAR_EXPERIMENT_1_METRICS, class_ids)
    experiment_2_summary = _aggregate(
        experiment_2_rows, ("architecture", "temporal_method"),
        SCALAR_EXPERIMENT_2_METRICS + TRANSITION_V2_METRICS, class_ids)
    expected = 2 * len(MODEL_SEEDS)
    if len(experiment_1_rows) != expected or len(experiment_2_rows) != 3 * expected:
        raise AssertionError("unexpected number of architecture/seed/method results")

    _write_rows(output / "experiment_1_instantaneous_per_seed.csv", experiment_1_rows)
    _write_rows(output / "experiment_1_instantaneous_summary.csv", experiment_1_summary)
    _write_rows(output / "experiment_2_sequential_per_seed.csv", experiment_2_rows)
    _write_rows(output / "experiment_2_sequential_summary.csv", experiment_2_summary)
    save_results(output / "experiment_1_instantaneous_per_seed.json", experiment_1_rows)
    save_results(output / "experiment_1_instantaneous_summary.json", experiment_1_summary)
    save_results(output / "experiment_2_sequential_per_seed.json", experiment_2_rows)
    save_results(output / "experiment_2_sequential_summary.json", experiment_2_summary)
    _write_rows(output / "experiment_2_transitions_v2.csv", transition_records, TRANSITION_RECORD_FIELDS)
    save_results(output / "experiment_2_transitions_v2.json", transition_records)
    _make_plots(output, experiment_1_summary, experiment_2_summary, class_ids)
    figure_info = _write_figure_bundle(output, structural_test, ordered_test, ordered_ids, extractor,
        figure_runs, experiment_1_rows, experiment_2_rows, experiment_1_summary, experiment_2_summary,
        class_ids, {"fixed_models": FIXED_MODEL_CONFIGS,
                    "extractor": torch.load(feature_artifacts / "extractor.pt", map_location="cpu", weights_only=False),
                    "standardizer": torch.load(feature_artifacts / "standardizer.pt", map_location="cpu", weights_only=False)})

    manifest = {
        "figures": figure_info,
        "provenance_checks": provenance_checks,
        "ordered_provenance_checks": ordered_provenance_checks,
        "legacy_provenance_fallback": args.allow_legacy_provenance,
        "transition_accounting": {
            "version": TRANSITION_ACCOUNTING_VERSION,
            "match_window": "[transition frame, target segment end), within sequence",
            "delay_units": "classification frames", "unmatched_delay": "unavailable",
            "legacy_mean_transition_delay_and_search_scoring": "unchanged",
            "false_transition_definition": "unchanged",
            "aggregation": "equal-weight seed mean/std; unavailable values excluded",
        },
        "dataset_root": str(dataset_root), "structural_dataset": str(structural_dir),
        "ordered_dataset": str(ordered_dir),
        "structural_split_manifest": str(structural_dir / "split_manifest.json")
        if (structural_dir / "split_manifest.json").is_file() else None,
        "ordered_split_manifest": str(ordered_dir / "split_manifest.json")
        if (ordered_dir / "split_manifest.json").is_file() else None,
        "model_seeds": list(MODEL_SEEDS), "class_ordering": class_ids,
        "fixed_model_configurations": FIXED_MODEL_CONFIGS,
        "preprocessing_artifacts": {
            "feature_extractor": str(feature_artifacts / "extractor.pt"),
            "feature_standardizer": str(feature_artifacts / "standardizer.pt"),
            "raw_depth_robot_state_inputs": [
                "roll", "pitch", "angular_velocity_roll",
                "angular_velocity_pitch", "angular_velocity_yaw"],
        },
        "training": {"max_epochs": MAX_EPOCHS,
                     "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                     "restore_best_validation_weights": True},
        "fixed_ema_configuration": FIXED_EMA_CONFIG,
        "fixed_bayes_configuration": FIXED_BAYES_CONFIG,
        "transition_window_radius": 5, "selection_or_search_performed": False,
        "test_aggregation_only": True, "training_runs": training_records,
    }
    save_results(output / "manifest.json", manifest)
    print(f"Saved frozen Experiments 1--2 to {output}")


if __name__ == "__main__":
    main()
