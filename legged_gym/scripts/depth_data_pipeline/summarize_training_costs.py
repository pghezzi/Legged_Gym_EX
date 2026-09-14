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
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from rsl_rl.utils.training_cost import STAGE_FIELDS, artifact_size_mb, write_cost_record


METHODS = ("Feature Router", "Raw-Depth Router", "Distilled Policy")
CORE_FIELDS = (
    "data_env_steps", "additional_locomotion_policy_env_steps",
    "total_post_specialist_env_steps", "data_generation_s", "preprocessing_s",
    "optimization_s", "setup_s", "artifact_serialization_s", "overhead_s", "compilation_s",
    "total_wallclock_s", "gpu_hours", "peak_gpu_memory_mb",
    "trainable_params", "training_samples", "artifact_size_mb", "deployment_size_mb",
)
TABLE_FIELDS = (
    "data_env_steps", "additional_locomotion_policy_env_steps",
    "total_post_specialist_env_steps", "setup_s", "data_generation_s", "preprocessing_s",
    "optimization_s", "artifact_serialization_s", "overhead_s", "compilation_s",
    "total_wallclock_s", "gpu_hours", "peak_gpu_memory_mb", "trainable_params", "deployment_size_mb",
)


class PathResolver:
    """Explicit longest-prefix relocation; never rewrite an input record/file."""

    def __init__(self, mappings=()):
        self.mappings = {}
        for source, destination in mappings:
            source, destination = Path(source).expanduser(), Path(destination).expanduser()
            if not source.is_absolute() or not destination.is_absolute():
                raise ValueError("Path mappings must use absolute recorded and local paths")
            self.mappings[Path(os.path.normpath(source))] = Path(os.path.normpath(destination))

    @staticmethod
    def _map(path, mappings):
        for source, destination in sorted(mappings, key=lambda pair: len(pair[0].parts), reverse=True):
            try:
                return destination / path.relative_to(source)
            except ValueError:
                continue
        return path

    def resolve(self, value, owner=None):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (owner or Path.cwd()) / path
        path = Path(os.path.normpath(path))
        return self._map(path, self.mappings.items()).resolve()

    def record_dir(self, local_file):
        # Single sidecars can be mounted separately from the data they describe.
        # Relative references belong to their original directory, not /inputs/0.
        return self._map(Path(local_file), [(dst, src) for src, dst in self.mappings.items()]).parent

    def metadata(self):
        return [{"recorded_prefix": str(src), "resolved_prefix": str(dst)}
                for src, dst in self.mappings.items()]


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
    status = _combine_status(_status(record, key) for record in records)
    return (None, status) if status == "unavailable" else (sum(values), status)


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


def _classifier_records(offline_dir: Path, resolver=None) -> list[dict[str, Any]]:
    resolver = resolver or PathResolver()
    manifest_path = offline_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing offline manifest: {manifest_path}")
    manifest = _load(manifest_path)
    split_counts = {}
    split_path = manifest.get("structural_split_manifest")
    split_path = resolver.resolve(split_path, resolver.record_dir(manifest_path)) if split_path else None
    if split_path and split_path.is_file():
        split_manifest = _load(split_path)
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
        model_path = resolver.resolve(source.get("model_path", ""), resolver.record_dir(manifest_path))
        if not model_path.is_file():
            model_path = sidecar.parent / "classifier.pt"
        source["recorded_model_path"] = source.get("model_path")
        source["model_path"] = str(model_path)
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


def _preparation_records(offline_dir: Path, explicit: Sequence[Path], dataset=None, resolver=None):
    """Walk only training-dataset provenance; deduplicate artifacts, not filenames.

    Missing known inputs get placeholder records so partial discovery cannot be
    mistaken for a complete (or free) integration pipeline.
    """
    resolver = resolver or PathResolver()
    resolve = resolver.resolve
    manifest = _load(offline_dir / "manifest.json")
    manifest_owner = resolver.record_dir(offline_dir / "manifest.json")
    indexed = {}
    for sidecar in _find_sidecars(explicit, "training_cost.json"):
        record = _load(sidecar)
        if record.get("run_type") in ("classifier_data_collection", "dataset_compilation"):
            if record.get("dataset_path"):
                indexed[resolve(record["dataset_path"], resolver.record_dir(sidecar))] = (record, sidecar)
    root = dataset or manifest.get("structural_dataset")
    split_ref = manifest.get("structural_split_manifest")
    if root is None and split_ref:
        root = str(resolve(split_ref, manifest_owner).parent)
    results, seen = [], set()
    def visit(path, compiled=False):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        sidecar = path / "training_cost.json" if compiled else Path(str(path) + ".training_cost.json")
        record, sidecar = indexed.get(path, (None, sidecar))
        expected = "dataset_compilation" if compiled else "classifier_data_collection"
        if record is None and sidecar.is_file():
            candidate = _load(sidecar)
            if candidate.get("run_type") == expected:
                record = candidate
        result = dict(record) if record else {"run_type": expected, "dataset_path": str(path)}
        result["recorded_dataset_path"] = result.get("dataset_path")
        result["dataset_path"] = str(path)  # Canonical identity for shared-source deduplication.
        result["cost_sidecar"] = str(sidecar)
        results.append(result)
        if not compiled:
            return
        split_path = path / "split_manifest.json"
        if split_ref and path == resolve(root, manifest_owner):
            split_path = resolve(split_ref, manifest_owner)
        split = _load(split_path) if split_path.is_file() else {}
        sources = (record or {}).get("source_files") or split.get("source_files") or split.get("sources", [])
        sources = [item.get("source_file") if isinstance(item, dict) else item for item in sources]
        calibration = split.get("calibration_source_file")
        if calibration:
            sources = [*sources, calibration]
        if not sources:
            results.append({"run_type": "classifier_data_collection", "dataset_path": None})
        for source in sources:
            if source:
                owner = resolver.record_dir(sidecar if (record or {}).get("source_files") else split_path)
                source_path = resolve(source, owner)
                # Also support compiled inputs in chained dataset pipelines.
                is_compiled = source_path.is_dir() or indexed.get(source_path, ({},))[0].get("run_type") == "dataset_compilation"
                visit(source_path, compiled=is_compiled)
    if root:
        visit(resolve(root, manifest_owner), compiled=True)
    else:
        # Explicit historical records can recover costs, but compilation remains unknown.
        results = [dict(record, recorded_dataset_path=record.get("dataset_path"), dataset_path=str(path),
                        cost_sidecar=str(sidecar)) for path, (record, sidecar) in indexed.items()]
        if not any(r.get("run_type") == "dataset_compilation" for r in results):
            results.append({"run_type": "dataset_compilation", "dataset_path": None})
        if not any(r.get("run_type") == "classifier_data_collection" for r in results):
            results.append({"run_type": "classifier_data_collection", "dataset_path": None})
    return results


def _compose_router_run(classifier: Mapping[str, Any],
                        preparation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = dict(classifier)
    statuses = dict(classifier.get("metric_status", {}))
    # A caller may discover the same source through calibration and training.
    unique = {}
    for record in preparation:
        identity = (record.get("run_type"), record.get("dataset_path"))
        unique.setdefault(identity, record)
    preparation = list(unique.values())
    collection = [r for r in preparation if r.get("run_type") == "classifier_data_collection"]
    compilation = [r for r in preparation if r.get("run_type") == "dataset_compilation"]
    components = [*preparation, classifier]
    for key in STAGE_FIELDS:
        if key == "preprocessing_s" and any(
                r.get("timing_schema_version", 1) < 2 for r in collection):
            # Legacy collection's hard-coded zero was not a measurement.
            result[key], statuses[key] = None, "unavailable"
        else:
            result[key], statuses[key] = _sum(components, key)
    for key in ("total_wallclock_s", "gpu_hours"):
        # Old sidecars had shorter, incompatible boundaries. Keep their source
        # values intact, but do not present them as complete integration totals.
        if not collection or not compilation or any(r.get("timing_schema_version", 1) < 2 for r in components):
            result[key], statuses[key] = None, "unavailable"
        else:
            result[key], statuses[key] = _sum(components, key)
    result["compilation_s"], statuses["compilation_s"] = _sum(compilation, "total_wallclock_s")
    result["data_env_steps"], statuses["data_env_steps"] = _sum(collection, "data_env_steps")
    result["additional_locomotion_policy_env_steps"] = 0
    statuses["additional_locomotion_policy_env_steps"] = "measured"
    result["total_post_specialist_env_steps"] = result["data_env_steps"]
    statuses["total_post_specialist_env_steps"] = statuses["data_env_steps"]
    peaks = [r.get("peak_gpu_memory_mb") for r in components if r.get("gpu_count") != 0]
    result["peak_gpu_memory_mb"] = max(peaks) if peaks and all(v is not None for v in peaks) else None
    statuses["peak_gpu_memory_mb"] = _combine_status(
        _status(r, "peak_gpu_memory_mb") for r in components if r.get("gpu_count") != 0)
    result["metric_status"] = statuses
    result["collection_run_ids"] = [r.get("dataset_path") for r in collection]
    result["compilation_run_ids"] = [r.get("dataset_path") for r in compilation]
    return result


def _deployment_sizes(records, offline_dir, artifacts=(), resolver=None):
    """On-disk deployable files, separate from legacy training-checkpoint size.

    Router files match the existing frozen loader. Distillation exports must be
    supplied explicitly: never call a critic/optimizer checkpoint a deployment.
    A method-wide argument is an explicitly shared file; :SEED scopes an export.
    """
    resolver = resolver or PathResolver()
    names = {"feature_nn": METHODS[0], "raw_depth_nn": METHODS[1], "distilled": METHODS[2]}
    supplied = []
    for target, value in artifacts:
        architecture, separator, seed = target.partition(":")
        if architecture not in names or (separator and not seed.isdigit()):
            raise ValueError(f"Invalid deployment target: {target}; use feature_nn/raw_depth_nn/distilled[:SEED]")
        path = resolver.resolve(value)
        supplied.append({"target": target, "method": names[architecture],
                         "seed": int(seed) if separator else None,
                         "recorded_path": str(value), "resolved_path": str(path),
                         "size_mb": artifact_size_mb(path)})
    for record in records:
        paths = []
        if record.get("method") in METHODS[:2]:
            architecture = "feature_nn" if record["method"] == METHODS[0] else "raw_depth_nn"
            root = offline_dir / "artifacts" / architecture
            seeded = root / f"seed_{record.get('seed')}"
            paths = [offline_dir / "manifest.json", seeded / "classifier.pt", seeded / "nn_model_args.pt"]
            if architecture == "feature_nn":
                paths += [root / "extractor.pt", root / "standardizer.pt"]
        paths += [Path(item["resolved_path"]) for item in supplied
                  if item["method"] == record.get("method") and item["seed"] in (None, record.get("seed"))]
        components, seen_files = [], set()
        for path in dict.fromkeys(p.resolve() for p in paths):
            # Repeated Docker bind mounts can expose the same file at different
            # absolute paths; do not charge that deployment component twice.
            info = path.stat() if path.is_file() else None
            identity = (info.st_dev, info.st_ino) if info else str(path)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            size = artifact_size_mb(path)
            components.append({"path": str(path), "size_mb": size,
                               "status": "measured" if size is not None else "unavailable"})
        available = bool(components) and all(item["size_mb"] is not None for item in components)
        record["deployment_size_mb"] = sum(item["size_mb"] for item in components) if available else None
        record["deployment_artifacts"] = components
        record.setdefault("metric_status", {})["deployment_size_mb"] = "measured" if available else "unavailable"
    return supplied


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
            if values.size != len(members):
                values = np.asarray([], dtype=float)
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
        "setup_s": "Setup/load (s)",
        "preprocessing_s": "Preprocessing (s)",
        "artifact_serialization_s": "Serialization (s)",
        "overhead_s": "Overhead (s)",
        "compilation_s": "Compilation subtotal (s)",
        "optimization_s": "Optimization (s)",
        "total_wallclock_s": "Wall-clock (s)",
        "gpu_hours": "GPU-hours",
        "peak_gpu_memory_mb": "Peak GPU MB",
        "trainable_params": "Parameters",
        "deployment_size_mb": "Deployment (MiB)",
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
    output = output / "figures" / "results"
    output.mkdir(parents=True, exist_ok=True)
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
    bottom = np.zeros(len(summary))
    complete = np.asarray([all(row.get(f"{key}_mean") is not None for key in STAGE_FIELDS)
                           for row in summary])
    for key, label in zip(STAGE_FIELDS, ("Setup/loading", "Data generation", "Preprocessing",
                                        "Optimization", "Serialization", "Other overhead")):
        values = np.asarray([plot_value(row, f"{key}_mean") for row in summary])
        values = np.where(complete, values, np.nan)
        ax.bar(labels, values, bottom=bottom, label=label)
        bottom += values
    for index, available in enumerate(complete):
        if not available:
            ax.text(index, 0, "unavailable", rotation=90, ha="center", va="bottom")
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
    parser.add_argument("--compilation-cost", type=Path, nargs="*", default=[])
    parser.add_argument("--distillation-run", type=Path, nargs="*", default=[])
    parser.add_argument("--path-map", nargs=2, action="append", default=[], metavar=("RECORDED", "LOCAL"),
                        help="Read-only path relocation, longest absolute prefix wins; repeat as needed")
    parser.add_argument("--deployment-artifact", nargs=2, action="append", default=[], metavar=("METHOD[:SEED]", "FILE"),
                        help="Additional deployable file: feature_nn, raw_depth_nn or distilled, optionally :SEED")
    parser.add_argument("--no-auto-distillation", action="store_true",
                        help="Use only explicitly supplied distillation records (no cwd/logs discovery)")
    parser.add_argument(
        "--output", type=Path, default=Path("training_cost_audit")
    )
    parser.add_argument("--locomotion-summary", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    resolver = PathResolver(args.path_map)
    offline_dir = resolver.resolve(args.paper_offline_dir)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    classifier_components = _classifier_records(offline_dir, resolver)
    router_records, preparation_records = [], []
    for record in classifier_components:
        preparation = _preparation_records(offline_dir,
            [resolver.resolve(p) for p in [*args.collection_cost, *args.compilation_cost]],
            record.get("structural_dataset"), resolver)
        preparation_records.extend(preparation)
        router_records.append(_compose_router_run(record, preparation))
    collection_paths = sorted({Path(r["cost_sidecar"]) for r in preparation_records
                               if r.get("run_type") == "classifier_data_collection" and r.get("cost_sidecar")})
    compilation_paths = sorted({Path(r["cost_sidecar"]) for r in preparation_records
                                if r.get("run_type") == "dataset_compilation" and r.get("cost_sidecar")})
    collection_records = [r for r in preparation_records if r.get("run_type") == "classifier_data_collection"]
    distillation_inputs = [resolver.resolve(path) for path in args.distillation_run]
    if not distillation_inputs and not args.no_auto_distillation and (Path.cwd() / "logs").is_dir():
        distillation_inputs = [Path.cwd() / "logs"]
    distillation_records = _distillation_records(distillation_inputs)
    for record in distillation_records:
        record["compilation_s"] = 0.0  # Online rollout storage; no offline dataset compilation.
        record.setdefault("metric_status", {})["compilation_s"] = "measured"
    per_run = [*router_records, *distillation_records]
    deployment_inputs = _deployment_sizes(per_run, offline_dir, args.deployment_artifact, resolver)
    summary = _aggregate(per_run)
    _write_csv(output / "training_cost_per_run.csv", per_run)
    _write_csv(output / "training_cost_summary.csv", summary)
    _write_table(output, summary)
    figures = _plots(output, summary, resolver.resolve(args.locomotion_summary) if args.locomotion_summary else None)

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
    if not any(record.get("total_wallclock_s") is not None for record in collection_records):
        audit_warnings.append(
            "No collection sidecar was found; router collection costs remain unavailable."
        )
    if any(record.get("total_wallclock_s") is None for record in preparation_records
           if record.get("run_type") == "dataset_compilation"):
        audit_warnings.append("Compilation timing is missing for a referenced dataset; integration totals remain unavailable.")
    manifest = {
        "schema_version": 1,
        "path_mappings": resolver.metadata(),
        "deployment_inputs": deployment_inputs,
        "deployment_size_definition": "Unique required on-disk loader files in MiB (2**20 bytes); router manifest/model/args plus feature preprocessing; explicitly supplied distilled exports. Existing specialist policies excluded. Training artifact_size_mb unchanged. Missing files make the deployment total unavailable.",
        "additive_timing_fields": list(STAGE_FIELDS),
        "compilation_s_is_subtotal": True,
        "accounting_boundary": {
            "start": "specialist locomotion policies already trained and frozen",
            "end": (
                "deployable router/classifier or distilled unified policy produced"
            ),
            "excluded": "original specialist-policy training",
        },
        "counted_stages": {
            "Feature Router": [
                "shared classifier rollout collection and dataset compilation",
                "feature generation/standardization",
                "feature-NN optimization",
            ],
            "Raw-Depth Router": [
                "shared classifier rollout collection and dataset compilation",
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
        "compilation_cost_records": [str(path) for path in compilation_paths],
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
            "each independent architecture/seed alternative, together with provenance-linked compilation; repeated sources are counted once.",
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
