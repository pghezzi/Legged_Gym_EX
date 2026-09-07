"""Run and aggregate the frozen closed-loop paper locomotion evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METHODS = (
    "oracle", "feature_instantaneous", "feature_ema", "feature_bayes",
    "raw_depth_instantaneous", "raw_depth_ema", "raw_depth_bayes", "distilled",
)
METHOD_LABELS = {
    "oracle": "Oracle",
    "feature_instantaneous": "Feature + Instantaneous",
    "feature_ema": "Feature + EMA",
    "feature_bayes": "Feature + Bayes",
    "raw_depth_instantaneous": "Raw Depth + Instantaneous",
    "raw_depth_ema": "Raw Depth + EMA",
    "raw_depth_bayes": "Raw Depth + Bayes",
    "distilled": "Distilled",
}
DIFFICULTIES = ("easy", "nominal", "hard")
EVAL_SEEDS = (101, 202, 303)
MODEL_SEEDS = (0, 1, 2)
LEARNED_METHODS = set(METHODS) - {"oracle", "distilled"}
SUMMARY_METRICS = (
    "episodic_success_rate", "forward_distance_m", "wrong_skill_fraction",
    "selector_ms_per_update", "selector_overhead_ms_per_control_step",
    "specialist_policy_ms_per_step", "total_inference_ms_per_control_step",
    "classification_step_latency_ms", "batch_latency_ms", "per_env_amortized_ms",
    "batch1_deployment_latency_ms", "batch1_classification_step_latency_ms",
    "effective_inference_hz", "additional_router_latency_ms_per_step",
    "additional_selector_only_ms_per_step",
    "selector_latency_ms_per_update", "policy_latency_ms_per_step",
    "amortized_total_inference_ms_per_step", "effective_hz",
    "skill_switch_count", "skill_switch_rate",
    "terrain_transition_detection_delay_steps",
)


def _json_value(value):
    return json.dumps(value, sort_keys=True) if isinstance(value, (list, dict, tuple)) else value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _json_value(row.get(key)) for key in fields} for row in rows])


def _read_csv(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                    continue
                if value in ("True", "False"):
                    parsed[key] = value == "True"
                    continue
                try:
                    parsed[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _timing_statistics(values):
    array = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "std": None, "median": None, "p95": None}
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)), "p95": float(np.percentile(array, 95)),
    }


def _select_classifier_seeds(offline_dir: Path):
    """Select one seed per architecture from the saved offline classifier results."""
    results_path = offline_dir / "experiment_1_instantaneous_per_seed.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"missing offline per-seed results: {results_path}")
    with results_path.open(encoding="utf-8") as stream:
        offline_results = json.load(stream)
    records = defaultdict(list)
    for result in offline_results:
        architecture = result.get("architecture")
        seed = result.get("seed")
        if architecture not in ("feature_nn", "raw_depth_nn") or seed is None:
            continue
        records[architecture].append({
            "seed": int(seed), "balanced_accuracy": float(result["balanced_accuracy"]),
            "nll": float(result["nll"]), "brier": float(result["brier"]),
            "model_path": result.get("model_path"),
        })
    selected, details = {}, {}
    for architecture in ("feature_nn", "raw_depth_nn"):
        if not records[architecture]:
            raise FileNotFoundError(
                f"no per-seed offline classifier results found for {architecture} in {results_path}")
        winner = sorted(records[architecture], key=lambda row: (
            -row["balanced_accuracy"], row["nll"], row["brier"], row["seed"]))[0]
        selected[architecture] = winner["seed"]
        details[architecture] = {
            "source": str(results_path),
            "selection_split": "offline_held_out_structural_test",
            "criterion": "balanced_accuracy_then_lower_nll_then_lower_brier",
            "selected": winner, "candidates": records[architecture],
        }
    return selected, details


def _finite_values(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)


def _aggregate(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, members in groups.items():
        record = dict(zip(group_keys, group))
        record["num_runs"] = len(members)
        for metric in SUMMARY_METRICS:
            values = _finite_values(members, metric)
            record[f"{metric}_mean"] = float(values.mean()) if values.size else None
            record[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else None
            record[f"{metric}_raw"] = values.tolist()
        output.append(record)
    order = {method: index for index, method in enumerate(METHODS)}
    return sorted(output, key=lambda row: (
        order.get(row.get("method"), 99), DIFFICULTIES.index(row["difficulty_level"])
        if "difficulty_level" in row else -1))


def _run_conditions(args):
    results = []
    for method in args.methods:
        architecture = ("raw_depth_nn" if method.startswith("raw_depth_") else
                        "feature_nn" if method.startswith("feature_") else None)
        classifier_seeds = ((args.selected_classifier_seeds[architecture],)
                            if architecture else (None,))
        for difficulty in args.difficulties:
            for evaluation_seed in args.eval_seeds:
                for classifier_seed in classifier_seeds:
                    seed_name = "none" if classifier_seed is None else str(classifier_seed)
                    run_dir = (args.output / "runs" / method / difficulty /
                               f"eval_seed_{evaluation_seed}" / f"classifier_seed_{seed_name}")
                    result_path = run_dir / "result.json"
                    if result_path.is_file() and not args.force:
                        with result_path.open(encoding="utf-8") as stream:
                            existing = json.load(stream)
                        latency_samples = existing.get("latency", {}).get("samples", {})
                        replay_ready = (method != "oracle" or
                                        bool((existing.get("trajectory_replay") or {}).get("files")))
                        if (existing.get("overall", {}).get("quotas_complete")
                                and latency_samples.get("total_inference_ms_per_control_step")
                                and latency_samples.get("batch1_deployment_latency_ms")
                                and replay_ready):
                            results.append(result_path)
                            continue
                    if args.aggregate_only:
                        continue
                    run_dir.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable, "-m", "legged_gym.scripts.evaluation.high_level_evaluation",
                        "--paper_method", method, "--difficulty_level", difficulty,
                        "--seed", str(evaluation_seed), "--num_envs", "10",
                        "--episodes_per_track", "1", "--num_steps", str(args.num_steps),
                        "--classify_every", str(args.classify_every),
                        "--latency_warmup_updates", str(args.latency_warmup_updates),
                        "--fixed_forward_command", str(args.fixed_forward_command),
                        "--task", args.task, "--out_dir", str(run_dir),
                        "--result_name", "result",
                    ]
                    if args.cpu:
                        command.append("--cpu")
                    else:
                        command.extend(("--gpu", args.gpu))
                    if args.headless:
                        command.append("--headless")
                    if method in LEARNED_METHODS:
                        command.extend(("--paper_offline_dir", str(args.paper_offline_dir),
                                        "--classifier_seed", str(classifier_seed),
                                        "--jit", str(args.jit)))
                    elif method == "oracle":
                        command.extend(("--jit", str(args.jit)))
                    else:
                        command.extend(("--distilled_jit", str(args.distilled_jit)))
                    print("Running:", " ".join(command), flush=True)
                    completed = subprocess.run(command, check=False, env=os.environ.copy())
                    if completed.returncode or not result_path.is_file():
                        if args.continue_on_error:
                            continue
                        raise SystemExit(completed.returncode or 1)
                    results.append(result_path)
    return results


def _load_results(paths):
    payloads = []
    layout_by_condition = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not payload.get("overall", {}).get("quotas_complete"):
            continue
        if not payload.get("latency", {}).get("samples", {}).get(
                "total_inference_ms_per_control_step"):
            continue
        if not payload.get("latency", {}).get("samples", {}).get(
                "batch1_deployment_latency_ms"):
            continue
        if (payload.get("metadata", {}).get("paper_method") == "oracle"
                and not (payload.get("trajectory_replay") or {}).get("files")):
            continue
        metadata = payload["metadata"]
        key = (metadata["difficulty_level"], int(metadata["seed"]))
        layout = json.dumps(metadata["track_layout"], sort_keys=True)
        if key in layout_by_condition and layout_by_condition[key] != layout:
            raise AssertionError(f"track layout mismatch for difficulty/evaluation seed {key}")
        layout_by_condition[key] = layout
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads, layout_by_condition


def _rows(payloads):
    per_episode, per_run = [], []
    for payload in payloads:
        metadata, headline = payload["metadata"], payload["headline"]
        method = metadata["paper_method"]
        common = {
            "method": method, "method_label": METHOD_LABELS[method],
            "difficulty_level": metadata["difficulty_level"],
            "evaluation_seed": metadata["seed"],
            "classifier_seed": metadata.get("classifier_seed"),
            "batch_size": metadata.get("timing_batch_size", metadata.get("num_envs")),
            "result_path": payload["_path"],
        }
        run = {
            **common, "completed_episodes": payload["overall"]["completed_episodes"],
            "episodic_success_rate": headline["episodic_success_rate"],
            "forward_distance_m": headline["episodic_forward_distance_m"],
            **{key: headline.get(key) for key in SUMMARY_METRICS if key not in {
                "episodic_success_rate", "forward_distance_m"}},
        }
        per_run.append(run)
        for episode in payload["episodes"]:
            per_episode.append({**common, **episode})
    return per_episode, per_run


def _add_paired_latency(payloads, per_run, per_episode):
    oracle_policy = {}
    for payload in payloads:
        metadata = payload["metadata"]
        if metadata["paper_method"] != "oracle":
            continue
        key = (metadata["difficulty_level"], int(metadata["seed"]))
        oracle_policy[key] = payload["headline"].get("specialist_policy_ms_per_step")
    run_by_path = {row["result_path"]: row for row in per_run}
    for payload in payloads:
        metadata, headline = payload["metadata"], payload["headline"]
        key = (metadata["difficulty_level"], int(metadata["seed"]))
        baseline = oracle_policy.get(key)
        total = headline.get("total_inference_ms_per_control_step")
        additional = None if baseline is None or total is None else float(total) - float(baseline)
        selector_only = (None if metadata["paper_method"] == "distilled" else
                         headline.get("selector_overhead_ms_per_control_step"))
        run = run_by_path[payload["_path"]]
        run["additional_router_latency_ms_per_step"] = additional
        run["additional_selector_only_ms_per_step"] = selector_only
        headline["additional_router_latency_ms_per_step"] = additional
        headline["additional_selector_only_ms_per_step"] = selector_only
    additions = {row["result_path"]: (
        row.get("additional_router_latency_ms_per_step"),
        row.get("additional_selector_only_ms_per_step")) for row in per_run}
    for row in per_episode:
        router, selector = additions[row["result_path"]]
        row["additional_router_latency_ms_per_step"] = router
        row["additional_selector_only_ms_per_step"] = selector
    return oracle_policy


def _latency_tables(payloads, oracle_policy):
    sample_names = set()
    for payload in payloads:
        sample_names.update(payload.get("latency", {}).get("samples", {}))
    sample_names.update(("additional_router_latency_ms_per_step",
                         "additional_selector_only_ms_per_step"))
    per_run, samples_by_method = [], defaultdict(lambda: defaultdict(list))
    for payload in payloads:
        metadata = payload["metadata"]
        method = metadata["paper_method"]
        samples = {key: list(values) for key, values in
                   payload.get("latency", {}).get("samples", {}).items()}
        key = (metadata["difficulty_level"], int(metadata["seed"]))
        baseline = oracle_policy.get(key)
        totals = samples.get("total_inference_ms_per_control_step", [])
        samples["additional_router_latency_ms_per_step"] = (
            [] if baseline is None else [float(value) - float(baseline) for value in totals]
        )
        samples.setdefault("additional_selector_only_ms_per_step",
                           samples.get("selector_overhead_ms_per_control_step", []))
        if method == "distilled":
            samples["additional_selector_only_ms_per_step"] = []
        record = {
            "method": method, "method_label": METHOD_LABELS[method],
            "difficulty_level": metadata["difficulty_level"],
            "evaluation_seed": metadata["seed"],
            "classifier_seed": metadata.get("classifier_seed"),
            "batch_size": metadata.get("timing_batch_size", metadata.get("num_envs")),
            "result_path": payload["_path"],
        }
        for name in sorted(sample_names):
            values = samples.get(name, [])
            stats = _timing_statistics(values)
            for statistic, value in stats.items():
                record[f"{name}_{statistic}"] = value
            samples_by_method[method][name].extend(values)
        selector_mean = record.get("selector_total_ms_mean")
        if method == "feature_bayes" and selector_mean:
            record.update({
                "feature_extraction_fraction_of_selector":
                    record.get("feature_extraction_ms_mean", 0.0) / selector_mean,
                "classifier_fraction_of_selector":
                    record.get("classifier_ms_mean", 0.0) / selector_mean,
                "bayes_fraction_of_selector":
                    record.get("temporal_filter_ms_mean", 0.0) / selector_mean,
                "skill_selection_fraction_of_selector":
                    record.get("skill_selection_ms_mean", 0.0) / selector_mean,
            })
        per_run.append(record)
    summary = []
    for method in METHODS:
        if method not in samples_by_method:
            continue
        method_payloads = [payload for payload in payloads
                           if payload["metadata"]["paper_method"] == method]
        batch_sizes = {payload["metadata"].get(
            "timing_batch_size", payload["metadata"].get("num_envs"))
            for payload in method_payloads}
        record = {
            "method": method, "method_label": METHOD_LABELS[method],
            "batch_size": next(iter(batch_sizes)) if len(batch_sizes) == 1 else None,
        }
        for name in sorted(sample_names):
            stats = _timing_statistics(samples_by_method[method][name])
            for statistic, value in stats.items():
                record[f"{name}_{statistic}"] = value
        selector_mean = record.get("selector_total_ms_mean")
        if selector_mean:
            record.update({
                "feature_extraction_fraction_of_selector":
                    record.get("feature_extraction_ms_mean", 0.0) / selector_mean,
                "classifier_fraction_of_selector":
                    record.get("classifier_ms_mean", 0.0) / selector_mean,
                "temporal_filter_fraction_of_selector":
                    record.get("temporal_filter_ms_mean", 0.0) / selector_mean,
                "skill_selection_fraction_of_selector":
                    record.get("skill_selection_ms_mean", 0.0) / selector_mean,
            })
            if method == "feature_bayes":
                record["bayes_fraction_of_selector"] = record[
                    "temporal_filter_fraction_of_selector"]
        summary.append(record)
    return per_run, summary


REPLAY_METHODS = ("instantaneous", "ema", "bayes")
REPLAY_PERSISTENCE_TICKS = 2
REPLAY_TRANSITION_RADIUS = 5


def _canonical_probabilities(probabilities, class_ids, canonicalize_label):
    result = {label: 0.0 for label in ("rough", "gap", "pit", "stairs")}
    for index, label in enumerate(class_ids):
        result[canonicalize_label(label)] += float(probabilities[index])
    return result


def _persistent_switch_index(predictions, target, start, stop,
                             persistence=REPLAY_PERSISTENCE_TICKS):
    stop = min(stop, len(predictions))
    for index in range(max(start, 0), max(start, stop - persistence + 1)):
        if all(predictions[offset] == target
               for offset in range(index, index + persistence)):
            return index
    return None


def _balanced_accuracy(truth, predictions):
    recalls = []
    for label in ("rough", "gap", "pit", "stairs"):
        indices = [index for index, value in enumerate(truth) if value == label]
        if indices:
            recalls.append(np.mean([predictions[index] == label for index in indices]))
    return float(np.mean(recalls)) if recalls else None


def _offline_replay_baselines(offline_dir, selected_seeds):
    path = offline_dir / "experiment_2_sequential_per_seed.json"
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    return {
        (row["architecture"], row["temporal_method"]): row
        for row in rows
        if int(row["seed"]) == int(selected_seeds[row["architecture"]])
    }


def _run_trajectory_replay(args, payloads):
    """Replay oracle observations through both frozen perception pipelines."""
    import torch
    from legged_gym.scripts.evaluation.high_level_evaluation import (
        _load_paper_classifier, _make_paper_bayes_filters, canonicalize_label,
    )
    from legged_gym.scripts.depth_data_pipeline.sequential_terrain_filter_extensions import (
        EMALogitPatienceFilter,
    )

    oracle_payloads = [payload for payload in payloads
                       if payload["metadata"]["paper_method"] == "oracle"]
    if not oracle_payloads:
        return [], [], [], []
    gpu = str(args.gpu)
    if gpu.isdigit():
        gpu = f"cuda:{gpu}"
    device = torch.device("cpu" if args.cpu else gpu)
    runtimes = {}
    for architecture in ("feature_nn", "raw_depth_nn"):
        runtime, _, manifest = _load_paper_classifier(
            args.paper_offline_dir, architecture,
            args.selected_classifier_seeds[architecture], device)
        runtimes[architecture] = (runtime, manifest)

    prediction_rows, transition_rows = [], []
    group_rows = {}
    for payload in sorted(oracle_payloads, key=lambda item: item["_path"]):
        metadata = payload["metadata"]
        layout = {int(track["track_id"]): track for track in metadata["track_layout"]}
        for replay_file in (payload.get("trajectory_replay") or {}).get("files", []):
            path = Path(replay_file["path"])
            if not path.is_file():
                raise FileNotFoundError(f"missing oracle trajectory replay file: {path}")
            data = torch.load(path, map_location="cpu", weights_only=False)
            count = int(data["depth"].shape[0])
            if not count:
                continue
            track_id = int(replay_file["track_id"])
            sequence = layout[track_id]["sequence"]
            terrain_length = float(data.get("terrain_length_m", [0.0])[0] or 0.0)
            if terrain_length <= 0.0:
                # Boundaries were recorded directly, so recover the common cell
                # length without depending on simulator configuration here.
                boundaries = sorted({float(value) for value in data["boundary_x_m"]
                                     if value is not None})
                terrain_length = boundaries[0] if boundaries else 1.0
            episode_ids = data.get("episode_index", [0] * count)
            for architecture, (runtime, manifest) in runtimes.items():
                logits_parts, probability_parts = [], []
                for start in range(0, count, 256):
                    stop = min(start + 256, count)
                    logits, probabilities = runtime.predict_deterministic(
                        data["depth"][start:stop].to(device),
                        data["orientation_rpy"][start:stop].to(device),
                        data["angular_velocity"][start:stop].to(device),
                        temperature=manifest["fixed_bayes_configuration"]["T_filter"])
                    logits_parts.append(logits.cpu()); probability_parts.append(probabilities.cpu())
                logits = torch.cat(logits_parts); probabilities = torch.cat(probability_parts)
                for episode_id in sorted(set(int(value) for value in episode_ids)):
                    indices = [index for index, value in enumerate(episode_ids)
                               if int(value) == episode_id]
                    ema = EMALogitPatienceFilter(
                        runtime.class_ids, **manifest["fixed_ema_configuration"], device=device)
                    bayes = _make_paper_bayes_filters(
                        runtime.class_ids, manifest["fixed_bayes_configuration"], 1, device)[0]
                    predictions = {method: [] for method in REPLAY_METHODS}
                    canonical_probabilities = []
                    for index in indices:
                        instant_native = runtime.class_ids[int(logits[index].argmax())]
                        predictions["instantaneous"].append(canonicalize_label(instant_native))
                        predictions["ema"].append(canonicalize_label(
                            ema.update(logits[index].to(device))))
                        predictions["bayes"].append(canonicalize_label(
                            bayes.update(probabilities[index].to(device)).label))
                        canonical_probabilities.append(_canonical_probabilities(
                            probabilities[index], runtime.class_ids, canonicalize_label))

                    positions = [float(data["base_position"][index, 0]) for index in indices]
                    timestamps = [float(data["timestamp_s"][index]) for index in indices]
                    truth = [canonicalize_label(data["ground_truth"][index]) for index in indices]
                    boundary_specs = []
                    for segment in range(len(sequence) - 1):
                        boundary_x = (segment + 1) * terrain_length
                        nearest = min(range(len(positions)), key=lambda idx: abs(positions[idx] - boundary_x))
                        if max(positions) < boundary_x:
                            continue
                        crossing = next((idx for idx, value in enumerate(positions)
                                         if value >= boundary_x), nearest)
                        previous = max(crossing - 1, 0)
                        if crossing != previous and positions[crossing] != positions[previous]:
                            fraction = ((boundary_x - positions[previous]) /
                                        (positions[crossing] - positions[previous]))
                            boundary_time = timestamps[previous] + fraction * (
                                timestamps[crossing] - timestamps[previous])
                        else:
                            boundary_time = timestamps[crossing]
                        boundary_specs.append({
                            "segment": segment, "x": boundary_x, "crossing": crossing,
                            "time": boundary_time,
                            "raw_current": sequence[segment],
                            "raw_next": sequence[segment + 1],
                            "current": canonicalize_label(sequence[segment]),
                            "next": canonicalize_label(sequence[segment + 1]),
                        })

                    transition_mask = np.zeros(len(indices), dtype=bool)
                    for spec in boundary_specs:
                        lo = max(0, spec["crossing"] - REPLAY_TRANSITION_RADIUS)
                        hi = min(len(indices), spec["crossing"] + REPLAY_TRANSITION_RADIUS + 1)
                        transition_mask[lo:hi] = True
                    group_key_base = (architecture, metadata["difficulty_level"],
                                      int(metadata["seed"]), track_id, episode_id)
                    group_rows[group_key_base] = {
                        "truth": truth, "predictions": predictions,
                        "transition_mask": transition_mask,
                    }
                    for local_index, source_index in enumerate(indices):
                        nearest_spec = (min(boundary_specs,
                                            key=lambda spec: abs(positions[local_index] - spec["x"]))
                                        if boundary_specs else None)
                        for method in REPLAY_METHODS:
                            probs = canonical_probabilities[local_index]
                            prediction_rows.append({
                                "architecture": architecture, "temporal_method": method,
                                "classifier_seed": args.selected_classifier_seeds[architecture],
                                "difficulty_level": metadata["difficulty_level"],
                                "evaluation_seed": metadata["seed"], "track_id": track_id,
                                "episode_index": episode_id, "frame_index": local_index,
                                "control_step": data["control_step"][source_index],
                                "timestamp_s": timestamps[local_index],
                                "forward_position_m": positions[local_index],
                                "ground_truth": truth[local_index],
                                "predicted_skill": predictions[method][local_index],
                                "class_probabilities": probs,
                                "nearest_boundary_segment": (
                                    nearest_spec["segment"] if nearest_spec else None),
                                "current_segment": (
                                    nearest_spec["raw_current"] if nearest_spec else None),
                                "next_segment": nearest_spec["raw_next"] if nearest_spec else None,
                                "boundary_position_m": nearest_spec["x"] if nearest_spec else None,
                                "boundary_timestamp_s": nearest_spec["time"] if nearest_spec else None,
                                "transition_pair": (f"{nearest_spec['current']}->{nearest_spec['next']}"
                                                    if nearest_spec else None),
                                "raw_transition_pair": (
                                    f"{nearest_spec['raw_current']}->{nearest_spec['raw_next']}"
                                    if nearest_spec else None),
                                "signed_distance_to_boundary_m": (
                                    positions[local_index] - nearest_spec["x"]
                                    if nearest_spec else None),
                                "time_relative_to_boundary_s": (
                                    timestamps[local_index] - nearest_spec["time"]
                                    if nearest_spec else None),
                                "probability_upcoming_skill": (
                                    probs[nearest_spec["next"]] if nearest_spec else None),
                                "in_transition_window": bool(transition_mask[local_index]),
                            })

                    for spec_index, spec in enumerate(boundary_specs):
                        crossing = spec["crossing"]
                        previous_crossing = (boundary_specs[spec_index - 1]["crossing"]
                                             if spec_index else 0)
                        next_crossing = (boundary_specs[spec_index + 1]["crossing"]
                                         if spec_index + 1 < len(boundary_specs) else len(indices))
                        lo = max(0, crossing - REPLAY_TRANSITION_RADIUS)
                        hi = min(len(indices), crossing + REPLAY_TRANSITION_RADIUS + 1)
                        for method in REPLAY_METHODS:
                            predicted = predictions[method]
                            skill_change = spec["current"] != spec["next"]
                            switch_index = (_persistent_switch_index(
                                predicted, spec["next"], previous_crossing, next_crossing)
                                if skill_change else None)
                            missed = bool(skill_change and switch_index is None)
                            delta_t = (None if switch_index is None else
                                       timestamps[switch_index] - spec["time"])
                            delta_x = (None if switch_index is None else
                                       positions[switch_index] - spec["x"])
                            pre = range(max(0, crossing - REPLAY_TRANSITION_RADIUS), crossing)
                            post = range(crossing, min(len(indices), crossing + REPLAY_TRANSITION_RADIUS))
                            transition_rows.append({
                                "architecture": architecture, "temporal_method": method,
                                "classifier_seed": args.selected_classifier_seeds[architecture],
                                "difficulty_level": metadata["difficulty_level"],
                                "evaluation_seed": metadata["seed"], "track_id": track_id,
                                "episode_index": episode_id, "boundary_segment": spec["segment"],
                                "transition_pair": f"{spec['current']}->{spec['next']}",
                                "raw_transition_pair":
                                    f"{spec['raw_current']}->{spec['raw_next']}",
                                "skill_change_required": skill_change,
                                "boundary_position_m": spec["x"],
                                "boundary_timestamp_s": spec["time"],
                                "switch_frame_index": switch_index,
                                "switch_timestamp_s": (
                                    timestamps[switch_index] if switch_index is not None else None),
                                "switch_position_m": (
                                    positions[switch_index] if switch_index is not None else None),
                                "delta_t_switch_s": delta_t,
                                "switch_distance_offset_m": delta_x,
                                "transition_window_accuracy": float(np.mean([
                                    predicted[index] == truth[index] for index in range(lo, hi)])),
                                "pre_transition_wrong_skill_occupancy": (
                                    float(np.mean([predicted[index] != spec["current"] for index in pre]))
                                    if len(pre) else None),
                                "pre_transition_upcoming_skill_occupancy": (
                                    float(np.mean([predicted[index] == spec["next"] for index in pre]))
                                    if len(pre) else None),
                                "post_transition_wrong_skill_occupancy": (
                                    float(np.mean([predicted[index] != spec["next"] for index in post]))
                                    if len(post) else None),
                                "post_transition_previous_skill_occupancy": (
                                    float(np.mean([predicted[index] == spec["current"] for index in post]))
                                    if len(post) else None),
                                "missed_transition": missed,
                                "late_transition": bool(
                                    switch_index is not None and
                                    switch_index > crossing + REPLAY_TRANSITION_RADIUS),
                                "premature_transition": bool(
                                    switch_index is not None and
                                    switch_index < crossing - REPLAY_TRANSITION_RADIUS),
                                "persistence_ticks": REPLAY_PERSISTENCE_TICKS,
                            })
    baselines = _offline_replay_baselines(
        args.paper_offline_dir, args.selected_classifier_seeds)
    with (args.paper_offline_dir / "manifest.json").open(encoding="utf-8") as stream:
        offline_manifest = json.load(stream)
    offline_classes = list(offline_manifest.get("class_ordering", []))
    canonical_offline_classes = [canonicalize_label(label) for label in offline_classes]
    same_label_space = (len(set(canonical_offline_classes)) == len(offline_classes)
                        and set(canonical_offline_classes) ==
                        {"rough", "gap", "pit", "stairs"})
    accuracy_rows = []
    for architecture in ("feature_nn", "raw_depth_nn"):
        for method in REPLAY_METHODS:
            for difficulty in (None, *args.difficulties):
                selected = [value for key, value in group_rows.items()
                            if key[0] == architecture and
                            (difficulty is None or key[1] == difficulty)]
                if not selected:
                    continue
                truth = [label for value in selected for label in value["truth"]]
                predicted = [label for value in selected
                             for label in value["predictions"][method]]
                masks = np.concatenate([value["transition_mask"] for value in selected])
                correct = np.asarray([a == b for a, b in zip(truth, predicted)])
                offline = baselines.get((architecture, method), {})
                balanced = _balanced_accuracy(truth, predicted)
                accuracy_rows.append({
                    "architecture": architecture, "temporal_method": method,
                    "classifier_seed": args.selected_classifier_seeds[architecture],
                    "difficulty_level": difficulty or "all",
                    "num_frames": len(truth), "accuracy": float(correct.mean()),
                    "balanced_accuracy": balanced,
                    "transition_window_accuracy": float(correct[masks].mean()) if masks.any() else None,
                    "steady_state_accuracy": float(correct[~masks].mean()) if (~masks).any() else None,
                    "offline_balanced_accuracy": offline.get("balanced_accuracy"),
                    "online_minus_offline_balanced_accuracy": (
                        None if balanced is None or offline.get("balanced_accuracy") is None else
                        balanced - float(offline["balanced_accuracy"])),
                    "offline_reference": "experiment_2_sequential_per_seed.json",
                    "online_label_space": ["rough", "gap", "pit", "stairs"],
                    "offline_label_space": offline_classes,
                    "offline_comparison_same_label_space": same_label_space,
                })
    transition_summary = _summarize_transition_replay(transition_rows)
    return prediction_rows, transition_rows, transition_summary, accuracy_rows


def _summarize_transition_replay(rows):
    output = []
    groupings = (("architecture", "temporal_method"),
                 ("architecture", "temporal_method", "difficulty_level"),
                 ("architecture", "temporal_method", "transition_pair"),
                 ("architecture", "temporal_method", "difficulty_level", "transition_pair"))
    for keys in groupings:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        for group, members in groups.items():
            record = dict(zip(keys, group))
            record["scope"] = "+".join(key.replace("_level", "") for key in keys[2:]) or "overall"
            record["num_boundaries"] = len(members)
            valid = [row for row in members if row["skill_change_required"]]
            record["num_skill_change_boundaries"] = len(valid)
            for metric in ("delta_t_switch_s", "switch_distance_offset_m",
                           "transition_window_accuracy", "pre_transition_wrong_skill_occupancy",
                           "pre_transition_upcoming_skill_occupancy",
                           "post_transition_wrong_skill_occupancy",
                           "post_transition_previous_skill_occupancy"):
                stats = _timing_statistics([row[metric] for row in valid
                                            if row.get(metric) is not None])
                for statistic in ("mean", "std", "median", "p95"):
                    record[f"{metric}_{statistic}"] = stats[statistic]
            for metric in ("missed_transition", "late_transition", "premature_transition"):
                record[f"{metric}_rate"] = (float(np.mean([row[metric] for row in valid]))
                                              if valid else None)
            output.append(record)
    return output


def _transition_pair_rows(per_episode):
    expanded = []
    for row in per_episode:
        sequence = row.get("terrain_sequence", [])
        if isinstance(sequence, str):
            sequence = sequence.split("|")
        canonical = ["rough" if value == "random_uniform" else
                     "stairs" if value in ("upwards_stairs", "stairs") else value
                     for value in sequence]
        for source, target in zip(canonical, canonical[1:]):
            expanded.append({"method": row["method"],
                             "difficulty_level": row["difficulty_level"],
                             "transition_pair": f"{source}->{target}",
                             "success": float(row["success"])})
    groups = defaultdict(list)
    for row in expanded:
        groups[(row["method"], row["difficulty_level"], row["transition_pair"])].append(row["success"])
    return [{"method": key[0], "difficulty_level": key[1], "transition_pair": key[2],
             "episode_count": len(values), "success_rate": float(np.mean(values))}
            for key, values in sorted(groups.items())]


def _latex(summary, output):
    by_method = {row["method"]: row for row in summary}
    def value(row, metric, scale=1.0, missing="--"):
        mean, std = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
        return missing if mean is None else f"{scale * mean:.2f} $\\pm$ {scale * std:.2f}"
    lines = [
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Method & Success $\uparrow$ & Distance $\uparrow$ & Wrong Skill $\downarrow$ & Selector Overhead ms/step $\downarrow$ & Total Inference ms/step $\downarrow$ & Hz $\uparrow$ \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = by_method[method]
        wrong = "--" if method == "distilled" else value(row, "wrong_skill_fraction", 100.0)
        selector = "--" if method == "distilled" else value(
            row, "selector_overhead_ms_per_control_step")
        lines.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(
            METHOD_LABELS[method], value(row, "episodic_success_rate", 100.0),
            value(row, "forward_distance_m"), wrong, selector,
            value(row, "total_inference_ms_per_control_step"),
            value(row, "effective_inference_hz")))
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "main_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figures(summary, by_difficulty, latency_summary, payloads, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    present_methods = [method for method in METHODS if any(row["method"] == method for row in summary)]
    present_difficulties = [difficulty for difficulty in DIFFICULTIES
                            if any(row["difficulty_level"] == difficulty for row in by_difficulty)]
    positions = np.arange(len(present_difficulties)); width = 0.72 / max(len(present_methods), 1)
    diff_rows = {(row["method"], row["difficulty_level"]): row for row in by_difficulty}
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    for index, method in enumerate(present_methods):
        rows = [diff_rows.get((method, difficulty)) for difficulty in present_difficulties]
        ax.bar(positions + (index - (len(present_methods) - 1) / 2) * width,
               [row["episodic_success_rate_mean"] if row else np.nan for row in rows], width,
               yerr=[row["episodic_success_rate_std"] if row else 0.0 for row in rows], capsize=2,
               label=METHOD_LABELS[method], color=colors[index])
    ax.set_xticks(positions, [value.title() for value in present_difficulties])
    ax.set_ylabel("Episodic success rate")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"success_by_difficulty.{extension}", dpi=300)
    plt.close(fig)

    summary_by_method = {row["method"]: row for row in summary}
    latency_by_method = {row["method"]: row for row in latency_summary}

    selector_methods = ("oracle", "feature_instantaneous", "feature_ema",
                        "feature_bayes", "raw_depth_instantaneous", "raw_depth_ema",
                        "raw_depth_bayes")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for index, method in enumerate(selector_methods):
        row = summary_by_method.get(method)
        if not row or row.get("selector_overhead_ms_per_control_step_mean") is None:
            continue
        x = row["selector_overhead_ms_per_control_step_mean"]
        ax.scatter(x, row["episodic_success_rate_mean"], color=colors[index], s=55)
        ax.annotate(METHOD_LABELS[method], (x, row["episodic_success_rate_mean"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Selector overhead (ms/control step)")
    ax.set_ylabel("Episodic success rate"); ax.grid(alpha=0.25); fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"success_vs_selector_overhead.{extension}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for index, method in enumerate(METHODS):
        row = summary_by_method.get(method)
        if not row or row.get("total_inference_ms_per_control_step_mean") is None:
            continue
        x = row["total_inference_ms_per_control_step_mean"]
        ax.scatter(x, row["episodic_success_rate_mean"], color=colors[index], s=55)
        ax.annotate(METHOD_LABELS[method], (x, row["episodic_success_rate_mean"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total inference latency (ms/control step)")
    ax.set_ylabel("Episodic success rate"); ax.grid(alpha=0.25); fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"success_vs_total_inference_latency.{extension}", dpi=300)
    plt.close(fig)

    breakdown_methods = ("feature_instantaneous", "feature_ema", "feature_bayes",
                         "raw_depth_instantaneous", "raw_depth_ema", "raw_depth_bayes")
    components = (
        ("Preprocess/features", ("depth_preprocess_ms_mean", "feature_extraction_ms_mean")),
        ("Standardization", ("standardization_ms_mean",)),
        ("Classifier", ("classifier_ms_mean",)),
        ("Temporal filter", ("temporal_filter_ms_mean",)),
        ("Skill selection", ("skill_selection_ms_mean",)),
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    bottoms = np.zeros(len(breakdown_methods))
    for index, (label, keys) in enumerate(components):
        values = np.asarray([
            sum(float(latency_by_method.get(method, {}).get(key) or 0.0) for key in keys)
            for method in breakdown_methods
        ])
        ax.bar(np.arange(len(breakdown_methods)), values, bottom=bottoms,
               label=label, color=colors[index])
        bottoms += values
    ax.set_xticks(np.arange(len(breakdown_methods)),
                  [METHOD_LABELS[method] for method in breakdown_methods], rotation=12)
    ax.set_ylabel("Selector latency (ms/update)"); ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"selector_latency_breakdown.{extension}", dpi=300)
    plt.close(fig)

    feature_bayes = latency_by_method.get("feature_bayes", {})
    stage_labels = [item[0] for item in components]
    stage_values = [sum(float(feature_bayes.get(key) or 0.0) for key in keys)
                    for _, keys in components]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(np.arange(len(stage_labels)), stage_values, color=colors[:len(stage_labels)])
    ax.set_xticks(np.arange(len(stage_labels)), stage_labels, rotation=18, ha="right")
    ax.set_ylabel("Mean latency (ms/update)"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"feature_bayes_latency_breakdown.{extension}", dpi=300)
    plt.close(fig)

    chosen = None
    for payload in sorted(payloads, key=lambda value: value["_path"]):
        if payload["metadata"]["paper_method"] != "feature_bayes":
            continue
        groups = defaultdict(list)
        for tick in payload.get("classification_ticks", []):
            groups[(tick["env_id"], tick["track_id"])].append(tick)
        for key, ticks in sorted(groups.items()):
            if any(tick["instantaneous_label"] != tick["canonical_ground_truth"]
                   and tick["bayes_label"] == tick["canonical_ground_truth"] for tick in ticks):
                chosen = payload, key, ticks, "first_transient_error_suppressed_by_bayes"
                break
        if chosen:
            break
    if chosen is None:
        candidates = [payload for payload in payloads
                      if payload["metadata"]["paper_method"] == "feature_bayes"]
        if candidates:
            payload = sorted(candidates, key=lambda value: value["_path"])[0]
            groups = defaultdict(list)
            for tick in payload.get("classification_ticks", []):
                groups[(tick["env_id"], tick["track_id"])].append(tick)
            if groups:
                key = sorted(groups)[0]
                chosen = payload, key, groups[key], "first_available_feature_bayes_rollout"
    timeline_metadata = None
    if chosen:
        payload, (env_id, track_id), ticks, rule = chosen
        labels = ["rough", "gap", "pit", "stairs"]
        index = {label: value for value, label in enumerate(labels)}
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        series = (("canonical_ground_truth", "GT terrain"),
                  ("instantaneous_label", "Instantaneous terrain"),
                  ("bayes_label", "Bayes-filtered terrain"),
                  ("selected_skill", "Selected skill"))
        for offset, (key, label) in enumerate(series):
            ax.step([tick["step"] for tick in ticks],
                    [index[tick[key]] + 0.06 * offset for tick in ticks], where="post", label=label)
        ax.set_yticks(range(len(labels)), labels); ax.set_xlabel("Simulation step")
        ax.set_ylabel("Canonical terrain / skill"); ax.grid(alpha=0.2); ax.legend(frameon=False)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"closed_loop_timeline.{extension}", dpi=300)
        plt.close(fig)
        timeline_metadata = {"selection_rule": rule, "result_path": payload["_path"],
                             "environment_id": env_id, "track_id": track_id}
    return timeline_metadata


def _replay_figures(predictions, transitions, transition_summary, accuracy, output):
    if not predictions:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"feature_nn": "#377eb8", "raw_depth_nn": "#e6550d"}
    display = {"feature_nn": "Feature", "raw_depth_nn": "Raw depth"}
    instant = [row for row in predictions if row["temporal_method"] == "instantaneous"
               and row.get("signed_distance_to_boundary_m") is not None
               and abs(float(row["signed_distance_to_boundary_m"])) <= 2.0]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = np.arange(-2.0, 2.01, 0.2)
    for architecture in ("feature_nn", "raw_depth_nn"):
        members = [row for row in instant if row["architecture"] == architecture]
        x = np.asarray([float(row["signed_distance_to_boundary_m"]) for row in members])
        y = np.asarray([float(row["probability_upcoming_skill"]) for row in members])
        centers, means = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            selected = y[(x >= lo) & (x < hi)]
            if selected.size:
                centers.append((lo + hi) / 2); means.append(float(selected.mean()))
        ax.plot(centers, means, marker="o", markersize=3, color=colors[architecture],
                label=display[architecture])
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Signed forward distance to nearest terrain boundary (m)")
    ax.set_ylabel("Mean probability of upcoming skill")
    ax.grid(alpha=0.25); ax.legend(frameon=False); fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"replay_probability_vs_signed_distance.{extension}", dpi=300)
    plt.close(fig)

    valid = [row for row in transitions if row.get("delta_t_switch_s") is not None
             and row.get("skill_change_required")]
    labels, time_values, distance_values = [], [], []
    for architecture in ("feature_nn", "raw_depth_nn"):
        for method in REPLAY_METHODS:
            members = [row for row in valid if row["architecture"] == architecture
                       and row["temporal_method"] == method]
            labels.append(f"{display[architecture]}\n{method.title()}")
            time_values.append([float(row["delta_t_switch_s"]) for row in members])
            distance_values.append([float(row["switch_distance_offset_m"]) for row in members])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for ax, values, ylabel in zip(
            axes, (time_values, distance_values),
            (r"$\Delta t_{switch}$ (s)", "Switch distance offset (m)")):
        nonempty = [(index + 1, value) for index, value in enumerate(values) if value]
        if nonempty:
            ax.boxplot([value for _, value in nonempty],
                       positions=[index for index, _ in nonempty], showfliers=False)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"transition_switch_lead_lag.{extension}", dpi=300)
    plt.close(fig)

    overall = [row for row in accuracy if row["difficulty_level"] == "all"]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    markers = {"instantaneous": "o", "ema": "s", "bayes": "^"}
    for row in overall:
        ax.scatter(row["steady_state_accuracy"], row["transition_window_accuracy"],
                   color=colors[row["architecture"]], marker=markers[row["temporal_method"]], s=55)
        ax.annotate(f"{display[row['architecture']]}-{row['temporal_method']}",
                    (row["steady_state_accuracy"], row["transition_window_accuracy"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Steady-state accuracy"); ax.set_ylabel("Transition-window accuracy")
    ax.grid(alpha=0.25); fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"replay_transition_vs_steady_accuracy.{extension}", dpi=300)
    plt.close(fig)

    detailed = [row for row in transition_summary
                if row.get("scope") == "difficulty+transition_pair"
                and row["temporal_method"] == "bayes"]
    pairs = sorted({row["transition_pair"] for row in detailed})
    fig, axes = plt.subplots(1, len(DIFFICULTIES), figsize=(max(10.0, 4.0 * len(DIFFICULTIES)), 4.4),
                             sharey=True, squeeze=False)
    width = 0.36
    for axis, difficulty in zip(axes[0], DIFFICULTIES):
        for offset, architecture in ((-width / 2, "feature_nn"), (width / 2, "raw_depth_nn")):
            values = []
            for pair in pairs:
                row = next((item for item in detailed if item["difficulty_level"] == difficulty
                            and item["transition_pair"] == pair
                            and item["architecture"] == architecture), None)
                values.append(row.get("transition_window_accuracy_mean") if row else np.nan)
            axis.bar(np.arange(len(pairs)) + offset, values, width,
                     color=colors[architecture], label=display[architecture])
        axis.set_title(difficulty.title()); axis.set_xticks(np.arange(len(pairs)), pairs,
                                                             rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Bayes transition-window accuracy")
    axes[0, -1].legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"replay_results_by_transition_pair_difficulty.{extension}", dpi=300)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper_offline_dir", type=Path, required=True)
    parser.add_argument("--jit", type=Path, required=True, help="specialist JIT/LoRA bundle")
    parser.add_argument("--distilled_jit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper_locomotion_eval"))
    parser.add_argument("--task", default="go2_depth_waq")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES))
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(EVAL_SEEDS))
    parser.add_argument("--classify-every", type=int, default=5)
    parser.add_argument("--latency-warmup-updates", type=int, default=20)
    parser.add_argument("--fixed-forward-command", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=100000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    args.paper_offline_dir = args.paper_offline_dir.expanduser().resolve()
    args.jit = args.jit.expanduser().resolve()
    args.distilled_jit = args.distilled_jit.expanduser().resolve()
    for path in (args.paper_offline_dir / "manifest.json", args.jit, args.distilled_jit):
        if not path.exists():
            parser.error(f"required artifact does not exist: {path}")
    if args.classify_every < 1 or args.num_steps < 1 or args.latency_warmup_updates < 0:
        parser.error("classification interval/step cap must be positive and warm-up nonnegative")
    if len(args.eval_seeds) != 3 or len(set(args.eval_seeds)) != 3:
        parser.error("--eval-seeds must contain exactly three distinct held-out seeds")
    try:
        args.selected_classifier_seeds, args.classifier_seed_selection = (
            _select_classifier_seeds(args.paper_offline_dir))
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = _run_conditions(args)
    payloads, layouts = _load_results(paths)
    if not payloads:
        raise RuntimeError("no complete runs are available to aggregate")
    timing_shapes = {
        (payload["metadata"].get("num_envs"), payload["metadata"].get("classify_every"),
         payload["metadata"].get("control_period_s"))
        for payload in payloads if payload["metadata"].get("num_envs") is not None
    }
    if len(timing_shapes) > 1:
        raise AssertionError(f"timing batch/cadence mismatch across methods: {timing_shapes}")
    expected_runs = len(args.difficulties) * len(args.eval_seeds) * len(args.methods)
    if not args.aggregate_only and not args.continue_on_error and len(payloads) != expected_runs:
        raise AssertionError(f"expected {expected_runs} complete paired runs, found {len(payloads)}")
    per_episode, per_run = _rows(payloads)
    oracle_policy = _add_paired_latency(payloads, per_run, per_episode)
    if not args.aggregate_only and not args.continue_on_error:
        expected_episodes = len(args.difficulties) * len(args.eval_seeds) * 10
        for method in args.methods:
            architecture = ("raw_depth_nn" if method.startswith("raw_depth_") else
                            "feature_nn" if method.startswith("feature_") else None)
            seeds = ((args.selected_classifier_seeds[architecture],)
                     if architecture else (None,))
            for seed in seeds:
                count = sum(row["method"] == method and row["classifier_seed"] == seed
                            for row in per_episode)
                if count != expected_episodes:
                    raise AssertionError(
                        f"{method}/classifier_seed={seed} has {count} episodes; "
                        f"expected {expected_episodes}")
    summary = _aggregate(per_run, ("method",))
    by_difficulty = _aggregate(per_run, ("method", "difficulty_level"))
    latency_per_run, latency_summary = _latency_tables(payloads, oracle_policy)
    _write_csv(args.output / "locomotion_per_episode.csv", per_episode)
    _write_csv(args.output / "locomotion_per_run.csv", per_run)
    _write_csv(args.output / "locomotion_summary.csv", summary)
    _write_csv(args.output / "locomotion_by_difficulty.csv", by_difficulty)
    _write_csv(args.output / "locomotion_by_transition_pair.csv", _transition_pair_rows(per_episode))
    _write_csv(args.output / "latency_per_run.csv", latency_per_run)
    _write_csv(args.output / "latency_summary.csv", latency_summary)
    replay_predictions, transition_lead_lag, transition_lead_lag_summary, replay_accuracy = (
        _run_trajectory_replay(args, payloads))
    _write_csv(args.output / "trajectory_replay_predictions.csv", replay_predictions)
    _write_csv(args.output / "transition_lead_lag.csv", transition_lead_lag)
    _write_csv(args.output / "transition_lead_lag_summary.csv", transition_lead_lag_summary)
    _write_csv(args.output / "replay_accuracy_summary.csv", replay_accuracy)
    # Tables and figures intentionally reload their saved CSV inputs so the
    # reporting path is reproducible without rerunning simulation or inference.
    saved_summary = _read_csv(args.output / "locomotion_summary.csv")
    saved_by_difficulty = _read_csv(args.output / "locomotion_by_difficulty.csv")
    saved_latency_summary = _read_csv(args.output / "latency_summary.csv")
    if {row["method"] for row in saved_summary} == set(METHODS):
        _latex(saved_summary, args.output)
    timeline = _figures(
        saved_summary, saved_by_difficulty, saved_latency_summary, payloads, args.output)
    _replay_figures(
        _read_csv(args.output / "trajectory_replay_predictions.csv"),
        _read_csv(args.output / "transition_lead_lag.csv"),
        _read_csv(args.output / "transition_lead_lag_summary.csv"),
        _read_csv(args.output / "replay_accuracy_summary.csv"), args.output)
    with (args.paper_offline_dir / "manifest.json").open(encoding="utf-8") as stream:
        offline_manifest = json.load(stream)
    resolved_difficulties = {}
    resolved_models = {}
    for payload in payloads:
        metadata = payload["metadata"]
        resolved_difficulties.setdefault(
            metadata["difficulty_level"], metadata["resolved_difficulty_parameters"])
        if metadata.get("model_path"):
            resolved_models[
                f"{metadata['paper_method']}:seed_{metadata['classifier_seed']}"] = metadata["model_path"]
    manifest = {
        "paper_offline_dir": str(args.paper_offline_dir),
        "specialist_jit": str(args.jit), "distilled_checkpoint": str(args.distilled_jit),
        "task": args.task, "simulator": os.environ.get("SIMULATOR"),
        "methods": args.methods,
        "classifier_seeds": sorted(set(args.selected_classifier_seeds.values())),
        "available_classifier_seeds": list(MODEL_SEEDS),
        "selected_classifier_seeds": args.selected_classifier_seeds,
        "classifier_seed_selection": args.classifier_seed_selection,
        "classifier_models_per_architecture": 1,
        "factorial_design": {
            "perception": ["feature", "raw_depth"],
            "temporal": ["instantaneous", "ema", "bayes"],
            "methods": [
                "feature_instantaneous", "feature_ema", "feature_bayes",
                "raw_depth_instantaneous", "raw_depth_ema", "raw_depth_bayes",
            ],
        },
        "evaluation_seeds": args.eval_seeds, "difficulties": args.difficulties,
        "resolved_difficulty_parameters": resolved_difficulties,
        "fixed_forward_command": args.fixed_forward_command,
        "classify_every": args.classify_every, "episodes_per_track": 1,
        "latency_warmup_updates": args.latency_warmup_updates,
        "latency_statistics": ["mean", "std", "median", "p95"],
        "latency_raw_samples_saved_in_per_run_json": True,
        "latency_definitions": {
            "batch_latency_ms": "full synchronized vectorized-batch total inference latency per control step",
            "classification_step_latency_ms": "full-batch policy plus selector latency on routing-update steps",
            "selector_overhead_ms_per_control_step": "full-batch selector latency divided by classify_every",
            "per_env_amortized_ms": "batch_latency_ms divided by batch_size; throughput metric only",
            "batch1_deployment_latency_ms": "explicit batch-size-one policy plus amortized selector latency",
            "effective_inference_hz": "1000 / batch_latency_ms; never divided by environment count",
        },
        "timing_configuration": {
            key: payloads[0]["metadata"].get(key) for key in (
                "timing_device", "gpu_name", "timing_batch_size", "num_envs",
                "classify_every", "simulator", "policy_control_frequency_hz",
                "control_period_s", "timing_clock", "cuda_synchronized_timing")
        },
        "tracks_per_seed": 10,
        "episodes_per_learned_method_classifier_seed":
            len(args.difficulties) * len(args.eval_seeds) * 10,
        "episodes_per_method": len(args.difficulties) * len(args.eval_seeds) * 10,
        "fixed_model_configurations":
            offline_manifest["fixed_model_configurations"],
        "fixed_ema_configuration": offline_manifest["fixed_ema_configuration"],
        "fixed_bayes_configuration": offline_manifest["fixed_bayes_configuration"],
        "resolved_classifier_model_paths": resolved_models,
        "track_layouts": {f"{key[0]}:{key[1]}": json.loads(value)
                          for key, value in layouts.items()},
        "representative_timeline": timeline,
        "trajectory_replay_diagnostics": {
            "source_method": "oracle",
            "classifier_seeds": args.selected_classifier_seeds,
            "temporal_methods": list(REPLAY_METHODS),
            "persistence_criterion": (
                f"first {REPLAY_PERSISTENCE_TICKS} consecutive classification ticks "
                "emitting the upcoming canonical skill"),
            "transition_window_radius_classification_ticks": REPLAY_TRANSITION_RADIUS,
            "trained_models_or_filter_parameters_changed": False,
            "outputs": [
                "trajectory_replay_predictions.csv", "transition_lead_lag.csv",
                "transition_lead_lag_summary.csv", "replay_accuracy_summary.csv",
            ],
            "source_files": [entry for payload in payloads
                             if payload["metadata"]["paper_method"] == "oracle"
                             for entry in (payload.get("trajectory_replay") or {}).get("files", [])],
        },
        "completed_result_files": [payload["_path"] for payload in payloads],
    }
    (args.output / "locomotion_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Aggregated {len(payloads)} completed runs in {args.output}")


if __name__ == "__main__":
    main()
