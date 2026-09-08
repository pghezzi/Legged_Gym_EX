"""Audit post-specialist integration costs for routers and distillation.

New runs provide measured training-cost sidecars. Historical runs are
reconstructed only from persisted counters/configuration and are marked as such;
missing values remain explicitly unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from rsl_rl.utils.training_cost import artifact_size_mb, write_cost_record


METHODS = ("Feature Router", "Raw-Depth Router", "Distilled Policy")
CORE_FIELDS = (
    "data_env_steps", "additional_locomotion_policy_env_steps",
    "total_post_specialist_env_steps", "data_generation_s", "preprocessing_s",
    "optimization_s", "total_wallclock_s", "gpu_hours", "peak_gpu_memory_mb",
    "trainable_params", "training_samples", "artifact_size_mb",
)
TABLE_FIELDS = (
    "data_env_steps", "additional_locomotion_policy_env_steps",
    "total_post_specialist_env_steps", "data_generation_s", "optimization_s",
    "total_wallclock_s", "gpu_hours", "peak_gpu_memory_mb", "trainable_params",
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _status(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    return record.get("metric_status", {}).get(
        key, "unavailable" if value is None else "reconstructed"
    )


def _combine_status(statuses: Iterable[str]) -> str:
    statuses = list(statuses)
    if not statuses or "unavailable" in statuses:
        return "unavailable"
    return "reconstructed" if "reconstructed" in statuses else "measured"


def _sum(records: Sequence[Mapping[str, Any]], key: str) -> tuple[float | int | None, str]:
    if not records:
        return None, "unavailable"
    values = [record.get(key) for record in records]
    if any(value is None for value in values):
        return None, "unavailable"
    return sum(values), _combine_status(_status(record, key) for record in records)


def _find_sidecars(paths: Sequence[Path], suffix: str) -> list[Path]:
    found = set()
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file() and path.name.endswith(suffix):
            found.add(path)
        elif path.is_dir():
            found.update(path.rglob(f"*{suffix}"))
    return sorted(found)


def _collection_paths(offline_dir: Path, explicit: Sequence[Path]) -> list[Path]:
    candidates = list(explicit)
    manifest_path = offline_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _load(manifest_path)
        # Only classifier structural-data collection belongs to the integration
        # boundary. Ordered-test collection is evaluation cost, not training.
        if manifest.get("structural_dataset"):
            candidates.append(Path(manifest["structural_dataset"]))
        split_path = manifest.get("structural_split_manifest")
        if split_path and Path(split_path).is_file():
            split = _load(Path(split_path))
            for source in split.get("source_files", split.get("sources", [])):
                source_path = Path(source)
                candidates.extend(
                    (source_path, Path(str(source_path) + ".training_cost.json"))
                )
    return _find_sidecars(candidates, ".training_cost.json")


def _legacy_classifier_record(
    record: Mapping[str, Any], architecture: str
) -> dict[str, Any]:
    model_path = Path(record.get("model_path", ""))
    epochs = record.get("epochs_completed")
    batch = record.get("batch_size")
    samples = record.get("training_samples")
    updates = (
        math.ceil(samples / batch) * epochs
        if all(value is not None for value in (samples, batch, epochs))
        else None
    )
    optimization = record.get(
        "optimization_s", record.get("training_runtime_seconds")
    )
    result = {
        **record,
        "schema_version": 1,
        "run_type": "classifier_training",
        "method": (
            "Feature Router" if architecture == "feature_nn"
            else "Raw-Depth Router"
        ),
        "architecture": architecture,
        "optimizer_updates": record.get("optimizer_updates", updates),
        "data_env_steps": 0,
        "additional_locomotion_policy_training": False,
        "additional_locomotion_policy_env_steps": 0,
        "total_post_specialist_env_steps": 0,
        "data_generation_s": 0.0,
        "preprocessing_s": record.get("preprocessing_s"),
        "optimization_s": optimization,
        "artifact_serialization_s": record.get("artifact_serialization_s"),
        "total_wallclock_s": record.get("total_wallclock_s"),
        "gpu_hours": record.get("gpu_hours"),
        "peak_gpu_memory_mb": record.get("peak_gpu_memory_mb"),
        "trainable_params": record.get("trainable_params"),
        "training_samples": samples,
        "artifact_size_mb": record.get(
            "artifact_size_mb", artifact_size_mb(model_path)
        ),
    }
    result["metric_status"] = {
        key: ("unavailable" if result.get(key) is None else "reconstructed")
        for key in result
        if not isinstance(result.get(key), (dict, list))
    }
    result["notes"] = [
        "Historical classifier cost reconstructed from offline manifest/artifacts."
    ]
    return result


def _classifier_records(offline_dir: Path) -> list[dict[str, Any]]:
    manifest_path = offline_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing offline manifest: {manifest_path}")
    manifest = _load(manifest_path)
    split_counts = {}
    split_path = manifest.get("structural_split_manifest")
    if split_path and Path(split_path).is_file():
        split_manifest = _load(Path(split_path))
        split_counts = {
            name: values.get("num_frames")
            for name, values in split_manifest.get("splits", {}).items()
        }
    results = []
    for source in manifest.get("training_runs", []):
        source = dict(source)
        architecture = source["architecture"]
        sidecar = (
            offline_dir / "artifacts" / architecture
            / f"seed_{source['seed']}" / "training_cost.json"
        )
        if sidecar.is_file():
            results.append(_load(sidecar))
            continue
        source.setdefault("training_samples", split_counts.get("train"))
        source.setdefault("validation_samples", split_counts.get("val"))
        source.setdefault("test_samples", split_counts.get("test"))
        model_path = Path(source.get("model_path", ""))
        if model_path.is_file() and source.get("trainable_params") is None:
            try:
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                state = checkpoint.get("model_state_dict", {})
                source["trainable_params"] = sum(
                    value.numel() for value in state.values()
                )
            except Exception:
                pass
        results.append(_legacy_classifier_record(source, architecture))
    return results


def _config_integer(text: str, name: str) -> int | None:
    matches = re.findall(rf"\b{re.escape(name)}\s*=\s*(\d+)\b", text)
    return int(matches[-1]) if matches else None


def _event_times(run_dir: Path) -> tuple[float | None, float | None]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        collection, optimization = [], []
        for event_file in run_dir.glob("events.out.tfevents*"):
            accumulator = EventAccumulator(
                str(event_file), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
            if "Perf/collection time" in tags:
                collection.extend(
                    value.value
                    for value in accumulator.Scalars("Perf/collection time")
                )
            if "Perf/learning_time" in tags:
                optimization.extend(
                    value.value
                    for value in accumulator.Scalars("Perf/learning_time")
                )
        return (
            sum(collection) if collection else None,
            sum(optimization) if optimization else None,
        )
    except Exception:
        return None, None


def _historical_distillation(run_dir: Path) -> dict[str, Any]:
    checkpoints = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)),
    )
    final = checkpoints[-1] if checkpoints else None
    iteration = (
        int(re.search(r"(\d+)$", final.stem).group(1)) if final else None
    )
    config_text = "\n".join(
        path.read_text(errors="replace")
        for path in run_dir.glob("*config*.py")
    )
    num_envs = _config_integer(config_text, "num_envs")
    steps_per_env = _config_integer(config_text, "num_steps_per_env")
    mini_batches = _config_integer(config_text, "num_mini_batches")
    learning_epochs = _config_integer(config_text, "num_learning_epochs")
    seed = _config_integer(config_text, "seed")
    samples = (
        iteration * num_envs * steps_per_env
        if all(v is not None for v in (iteration, num_envs, steps_per_env))
        else None
    )
    updates = (
        iteration * mini_batches * learning_epochs
        if all(v is not None for v in (iteration, mini_batches, learning_epochs))
        else None
    )
    data_s, optimization_s = _event_times(run_dir)
    trainable_params = None
    if final:
        try:
            checkpoint = torch.load(final, map_location="cpu", weights_only=False)
            state = checkpoint.get("model_state_dict", {})
            prefixes = ("actor.", "vae.", "visual_encoder.")
            trainable_params = sum(
                value.numel()
                for key, value in state.items()
                if key.startswith(prefixes)
            )
        except Exception:
            pass
    recorded_loop_s = (
        data_s + optimization_s
        if data_s is not None and optimization_s is not None
        else None
    )
    record = {
        "schema_version": 1,
        "run_type": "distillation_training",
        "method": "Distilled Policy",
        "seed": seed,
        "log_dir": str(run_dir),
        "final_checkpoint": str(final) if final else None,
        "distillation_iterations": iteration,
        "optimizer_updates": updates,
        "batch_size": (
            samples // max(iteration * mini_batches, 1)
            if samples is not None and mini_batches
            else None
        ),
        "num_parallel_envs": num_envs,
        "training_samples": samples,
        "teacher_labelled_samples": samples,
        "data_env_steps": samples,
        "additional_locomotion_policy_training": True,
        "additional_locomotion_policy_env_steps": samples,
        "total_post_specialist_env_steps": samples,
        "data_generation_s": data_s,
        "preprocessing_s": 0.0,
        "optimization_s": optimization_s,
        "recorded_training_loop_s": recorded_loop_s,
        "total_wallclock_s": None,
        "gpu_active_time_s": None,
        "gpu_hours": None,
        "peak_gpu_memory_mb": None,
        "gpu_model": None,
        "gpu_count": None,
        "trainable_params": trainable_params,
        "artifact_size_mb": artifact_size_mb(final) if final else None,
        "notes": [
            "Historical distillation record; only persisted config, events, "
            "and checkpoint values were reconstructed. Full wall-clock remains "
            "unavailable because setup/loading/serialization time was not logged."
        ],
    }
    record["metric_status"] = {
        key: ("unavailable" if value is None else "reconstructed")
        for key, value in record.items()
        if not isinstance(value, (dict, list, bool))
    }
    return record


def _distillation_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    results, seen = [], set()
    for path in paths:
        path = path.expanduser().resolve()
        candidates = (
            [path] if path.is_file() else list(path.rglob("training_cost.json"))
        )
        if candidates:
            for candidate in candidates:
                record = _load(candidate)
                if (
                    record.get("run_type") == "distillation_training"
                    and candidate not in seen
                ):
                    results.append(record)
                    seen.add(candidate)
        elif path.is_dir():
            results.append(_historical_distillation(path))
        if path.is_dir():
            for checkpoint in path.rglob("model_*.pt"):
                run_dir = checkpoint.parent
                if "distill" not in str(run_dir).lower():
                    continue
                sidecar = run_dir / "training_cost.json"
                if sidecar in seen or any(
                    record.get("log_dir") == str(run_dir) for record in results
                ):
                    continue
                results.append(
                    _load(sidecar) if sidecar.is_file()
                    else _historical_distillation(run_dir)
                )
                seen.add(sidecar)
    return results


def _compose_router_run(
    classifier: Mapping[str, Any],
    collection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = dict(classifier)
    statuses = dict(classifier.get("metric_status", {}))
    for key in ("data_env_steps", "data_generation_s"):
        value, state = _sum(collection, key)
        result[key], statuses[key] = value, state
    result["additional_locomotion_policy_env_steps"] = 0
    statuses["additional_locomotion_policy_env_steps"] = "measured"
    result["total_post_specialist_env_steps"] = result["data_env_steps"]
    statuses["total_post_specialist_env_steps"] = statuses["data_env_steps"]
    collection_total, collection_total_status = _sum(
        collection, "total_wallclock_s"
    )
    preprocessing = result.get("preprocessing_s")
    optimization = result.get("optimization_s")
    serialization = result.get("artifact_serialization_s")
    result["total_wallclock_s"] = (
        collection_total + preprocessing + optimization + serialization
        if all(
            value is not None
            for value in (
                collection_total, preprocessing, optimization, serialization
            )
        )
        else None
    )
    statuses["total_wallclock_s"] = _combine_status(
        (
            collection_total_status,
            statuses.get(
                "preprocessing_s", _status(classifier, "preprocessing_s")
            ),
            statuses.get(
                "optimization_s", _status(classifier, "optimization_s")
            ),
            statuses.get(
                "artifact_serialization_s",
                _status(classifier, "artifact_serialization_s"),
            ),
        )
    )
    collection_gpu, collection_gpu_status = _sum(collection, "gpu_hours")
    classifier_gpu = classifier.get("gpu_hours")
    result["gpu_hours"] = (
        collection_gpu + classifier_gpu
        if collection_gpu is not None and classifier_gpu is not None
        else None
    )
    statuses["gpu_hours"] = _combine_status(
        (collection_gpu_status, _status(classifier, "gpu_hours"))
    )
    collection_peaks = [
        record.get("peak_gpu_memory_mb") for record in collection
    ]
    peaks = [
        value
        for value in [*collection_peaks, classifier.get("peak_gpu_memory_mb")]
        if value is not None
    ]
    result["peak_gpu_memory_mb"] = max(peaks) if peaks else None
    statuses["peak_gpu_memory_mb"] = _combine_status(
        [_status(record, "peak_gpu_memory_mb") for record in collection]
        + [_status(classifier, "peak_gpu_memory_mb")]
    )
    result["metric_status"] = statuses
    result["collection_run_ids"] = [
        record.get("dataset_path") for record in collection
    ]
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened = []
    for source in rows:
        row = dict(source)
        for key, state in source.get("metric_status", {}).items():
            row[f"{key}_status"] = state
        flattened.append(row)
    fields = list(
        dict.fromkeys(
            key
            for row in flattened
            for key in row
            if key not in {"metric_status", "notes"}
        )
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in flattened:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key))
                        if isinstance(row.get(key), (dict, list))
                        else row.get(key)
                    )
                    for key in fields
                }
            )


def _aggregate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        members = [
            record for record in records if record.get("method") == method
        ]
        row: dict[str, Any] = {
            "method": method,
            "num_runs": len(members),
            "seeds": [record.get("seed") for record in members],
        }
        for key in CORE_FIELDS:
            values = np.asarray(
                [
                    float(record[key])
                    for record in members
                    if record.get(key) is not None
                ],
                dtype=float,
            )
            row[f"{key}_mean"] = float(values.mean()) if values.size else None
            row[f"{key}_std"] = (
                float(values.std(ddof=1))
                if values.size > 1
                else (0.0 if values.size else None)
            )
            row[f"{key}_status"] = _combine_status(
                _status(member, key) for member in members
            )
        rows.append(row)
    distilled = next(
        row for row in rows if row["method"] == "Distilled Policy"
    )
    for row in rows:
        for key, label in (
            ("total_post_specialist_env_steps", "simulator_interactions"),
            ("gpu_hours", "gpu_hours"),
            ("total_wallclock_s", "wallclock"),
        ):
            router = row.get(f"{key}_mean")
            distill = distilled.get(f"{key}_mean")
            row[f"relative_reduction_vs_distill_{label}"] = (
                1.0 - router / distill
                if router is not None and distill not in (None, 0)
                else None
            )
            row[f"relative_reduction_vs_distill_{label}_status"] = (
                _combine_status((
                    row.get(f"{key}_status", "unavailable"),
                    distilled.get(f"{key}_status", "unavailable"),
                ))
                if row[f"relative_reduction_vs_distill_{label}"] is not None
                else "unavailable"
            )
    return rows


def _format(value: Any, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "--"
    value = float(value)
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.{digits}f}"


def _write_table(output: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for source in summary:
        row = {"method": source["method"]}
        for key in TABLE_FIELDS:
            row[key] = source.get(f"{key}_mean")
            row[f"{key}_std"] = source.get(f"{key}_std")
        for key in (
            "relative_reduction_vs_distill_simulator_interactions",
            "relative_reduction_vs_distill_gpu_hours",
            "relative_reduction_vs_distill_wallclock",
        ):
            row[key] = source.get(key)
            row[f"{key}_status"] = source.get(f"{key}_status")
        rows.append(row)
    _write_csv(output / "training_cost_comparison.csv", rows)
    labels = {
        "data_env_steps": "Data steps",
        "additional_locomotion_policy_env_steps": "Additional policy steps",
        "total_post_specialist_env_steps": "Total steps",
        "data_generation_s": "Data time (s)",
        "optimization_s": "Optimization (s)",
        "total_wallclock_s": "Wall-clock (s)",
        "gpu_hours": "GPU-hours",
        "peak_gpu_memory_mb": "Peak GPU MB",
        "trainable_params": "Parameters",
    }
    columns = ["Method", *(labels[key] for key in TABLE_FIELDS)]
    lines = [
        "\\begin{tabular}{l" + "r" * len(TABLE_FIELDS) + "}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = []
        for key in TABLE_FIELDS:
            mean, std = row[key], row[f"{key}_std"]
            values.append(
                "--"
                if mean is None
                else f"{_format(mean)} $\\pm$ {_format(std)}"
            )
        lines.append(" & ".join([row["method"], *values]) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "training_cost_comparison.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plots(
    output: Path,
    summary: Sequence[Mapping[str, Any]],
    locomotion: Path | None,
) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["method"] for row in summary]
    colors = ["#377eb8", "#e6550d", "#4daf4a"]
    generated = []

    def plot_value(row, key):
        value = row.get(key)
        return np.nan if value is None else value

    def save(fig, stem):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300)
            generated.append(str(path))
        plt.close(fig)

    values = [
        plot_value(row, "total_post_specialist_env_steps_mean")
        for row in summary
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Post-specialist simulator interactions")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=.25)
    save(fig, "training_cost_simulator_interactions")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ylabel in (
        (axes[0], "total_wallclock_s_mean", "Total wall-clock (s)"),
        (axes[1], "gpu_hours_mean", "GPU-hours"),
    ):
        ax.bar(labels, [plot_value(row, key) for row in summary], color=colors)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.25)
    save(fig, "training_cost_wallclock_gpu_hours")

    fig, ax = plt.subplots(figsize=(7, 4))
    generation = np.asarray(
        [plot_value(row, "data_generation_s_mean") for row in summary]
    )
    optimization = np.asarray(
        [plot_value(row, "optimization_s_mean") for row in summary]
    )
    ax.bar(labels, generation, label="Data generation", color="#9ecae1")
    ax.bar(
        labels, optimization, bottom=generation,
        label="Optimization", color="#3182bd",
    )
    ax.set_ylabel("Time (s)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.25)
    save(fig, "training_cost_data_vs_optimization")

    if locomotion and locomotion.is_file():
        with locomotion.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        success_by_method = {
            row.get("method"): row.get("episodic_success_rate_mean")
            for row in rows
        }
        points = [
            (
                row["total_post_specialist_env_steps_mean"],
                success_by_method.get(row["method"]),
                row["method"],
            )
            for row in summary
        ]
        points = [
            (float(x), float(y), label)
            for x, y, label in points
            if x is not None and y not in (None, "")
        ]
        if points:
            fig, ax = plt.subplots(figsize=(6, 4))
            for x, y, label in points:
                ax.scatter(x, y)
                ax.annotate(
                    label, (x, y), xytext=(4, 4),
                    textcoords="offset points",
                )
            ax.set_xlabel("Post-specialist simulator interactions")
            ax.set_ylabel("Locomotion success")
            ax.grid(alpha=.25)
            save(fig, "locomotion_success_vs_training_cost")
    return generated


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-offline-dir", type=Path, required=True)
    parser.add_argument("--collection-cost", type=Path, nargs="*", default=[])
    parser.add_argument("--distillation-run", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--output", type=Path, default=Path("training_cost_audit")
    )
    parser.add_argument("--locomotion-summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    offline_dir = args.paper_offline_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    collection_paths = _collection_paths(offline_dir, args.collection_cost)
    collection_records = [
        record for record in (_load(path) for path in collection_paths)
        if record.get("run_type") == "classifier_data_collection"
    ]
    classifier_components = _classifier_records(offline_dir)
    router_records = [
        _compose_router_run(record, collection_records)
        for record in classifier_components
    ]
    distillation_inputs = list(args.distillation_run)
    if not distillation_inputs and (Path.cwd() / "logs").is_dir():
        distillation_inputs = [Path.cwd() / "logs"]
    distillation_records = _distillation_records(distillation_inputs)
    per_run = [*router_records, *distillation_records]
    summary = _aggregate(per_run)
    _write_csv(output / "training_cost_per_run.csv", per_run)
    _write_csv(output / "training_cost_summary.csv", summary)
    _write_table(output, summary)
    figures = _plots(output, summary, args.locomotion_summary)

    unavailable = sorted(
        {
            f"{record.get('method')}[{record.get('seed')}].{key}"
            for record in per_run
            for key in CORE_FIELDS
            if record.get(key) is None
        }
    )
    counts = {
        method: sum(record.get("method") == method for record in per_run)
        for method in METHODS
    }
    audit_warnings = []
    for method in METHODS[:2]:
        if counts[method] != 3:
            audit_warnings.append(
                f"Expected 3 {method} seeds, found {counts[method]}."
            )
    if counts["Distilled Policy"] not in (0, 3):
        audit_warnings.append(
            f"Expected 3 distillation seeds when supplied, found {counts['Distilled Policy']}."
        )
    if not collection_records:
        audit_warnings.append(
            "No collection sidecar was found; router collection costs remain unavailable."
        )
    manifest = {
        "schema_version": 1,
        "accounting_boundary": {
            "start": "specialist locomotion policies already trained and frozen",
            "end": (
                "deployable router/classifier or distilled unified policy produced"
            ),
            "excluded": "original specialist-policy training",
        },
        "counted_stages": {
            "Feature Router": [
                "shared classifier rollout collection",
                "feature generation/standardization",
                "feature-NN optimization",
            ],
            "Raw-Depth Router": [
                "shared classifier rollout collection",
                "raw-depth/state packing",
                "raw-depth-NN optimization",
            ],
            "Distilled Policy": [
                "student-controlled teacher-labelled rollout collection",
                "online mini-batch imitation optimization",
                "student/teacher setup and final serialization",
            ],
        },
        "run_counts": counts,
        "collection_cost_records": [str(path) for path in collection_paths],
        "classifier_offline_dir": str(offline_dir),
        "distillation_inputs": [
            str(path.expanduser().resolve())
            for path in distillation_inputs
        ],
        "metric_provenance_values": [
            "measured", "reconstructed", "unavailable"
        ],
        "unavailable_metrics": unavailable,
        "audit_warnings": audit_warnings,
        "fairness_notes": [
            "The same shared router dataset-collection cost is charged to "
            "Feature and Raw-Depth routes, but never summed across classifier seeds.",
            "Distillation teacher-labelled rollouts are simultaneously data "
            "generation and locomotion-policy training; total interactions count "
            "their union once.",
            "EMA and fixed Bayes have zero training cost because their frozen "
            "parameters are not learned here.",
            "GPU-hours use synchronized allocated-device wall-clock; true "
            "kernel-active time is unavailable without profiler telemetry.",
            "Historical quantities are never silently estimated: persisted-value "
            "reconstruction is explicitly marked.",
        ],
        "figures": figures,
        "summary": summary,
        "per_run_records": per_run,
        "collection_components": collection_records,
    }
    write_cost_record(output / "training_cost_manifest.json", manifest)
    print("\nPost-specialist training-cost audit")
    for method in METHODS:
        print(
            f"- {method}: "
            f"{', '.join(manifest['counted_stages'][method])} "
            f"({counts[method]} runs)"
        )
    print(
        f"- Unavailable metrics: {len(unavailable)}"
        + (
            f"; see {output / 'training_cost_manifest.json'}"
            if unavailable else ""
        )
    )
    for warning in audit_warnings:
        print(f"- Warning: {warning}")
    print(
        "- Fairness: shared router collection is charged once per method; "
        "distillation rollout/training overlap is counted once in total steps."
    )
    print(f"Saved audit outputs to {output}")


if __name__ == "__main__":
    main()
