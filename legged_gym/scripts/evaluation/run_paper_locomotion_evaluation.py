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
    "raw_depth_bayes", "distilled",
)
METHOD_LABELS = {
    "oracle": "Oracle",
    "feature_instantaneous": "Feature + Instantaneous",
    "feature_ema": "Feature + EMA",
    "feature_bayes": "Feature + Bayes",
    "raw_depth_bayes": "Raw Depth + Bayes",
    "distilled": "Distilled",
}
DIFFICULTIES = ("easy", "nominal", "hard")
EVAL_SEEDS = (101, 202, 303, 404, 505)
MODEL_SEEDS = (0, 1, 2)
LEARNED_METHODS = set(METHODS) - {"oracle", "distilled"}
SUMMARY_METRICS = (
    "episodic_success_rate", "forward_distance_m", "wrong_skill_fraction",
    "selector_ms_per_update", "selector_overhead_ms_per_control_step",
    "specialist_policy_ms_per_step", "total_inference_ms_per_control_step",
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
        classifier_seeds = MODEL_SEEDS if method in LEARNED_METHODS else (None,)
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
                        if (existing.get("overall", {}).get("quotas_complete")
                                and latency_samples.get("total_inference_ms_per_control_step")):
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
    if args.aggregate_only:
        results = sorted((args.output / "runs").glob("**/result.json"))
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
        record = {"method": method, "method_label": METHOD_LABELS[method]}
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
                        "feature_bayes", "raw_depth_bayes")
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
                         "raw_depth_bayes")
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper_offline_dir", type=Path, required=True)
    parser.add_argument("--jit", type=Path, required=True, help="specialist JIT/LoRA bundle")
    parser.add_argument("--distilled_jit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper_locomotion_eval"))
    parser.add_argument("--task", default="go2_depth_waq")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
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
    expected_runs = len(args.difficulties) * len(args.eval_seeds) * sum(
        len(MODEL_SEEDS) if method in LEARNED_METHODS else 1 for method in args.methods)
    if not args.aggregate_only and not args.continue_on_error and len(payloads) != expected_runs:
        raise AssertionError(f"expected {expected_runs} complete paired runs, found {len(payloads)}")
    per_episode, per_run = _rows(payloads)
    oracle_policy = _add_paired_latency(payloads, per_run, per_episode)
    if not args.aggregate_only and not args.continue_on_error:
        expected_episodes = len(args.difficulties) * len(args.eval_seeds) * 10
        for method in args.methods:
            seeds = MODEL_SEEDS if method in LEARNED_METHODS else (None,)
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
    # Tables and figures intentionally reload their saved CSV inputs so the
    # reporting path is reproducible without rerunning simulation or inference.
    saved_summary = _read_csv(args.output / "locomotion_summary.csv")
    saved_by_difficulty = _read_csv(args.output / "locomotion_by_difficulty.csv")
    saved_latency_summary = _read_csv(args.output / "latency_summary.csv")
    if {row["method"] for row in saved_summary} == set(METHODS):
        _latex(saved_summary, args.output)
    timeline = _figures(
        saved_summary, saved_by_difficulty, saved_latency_summary, payloads, args.output)
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
        "methods": args.methods, "classifier_seeds": list(MODEL_SEEDS),
        "evaluation_seeds": args.eval_seeds, "difficulties": args.difficulties,
        "resolved_difficulty_parameters": resolved_difficulties,
        "fixed_forward_command": args.fixed_forward_command,
        "classify_every": args.classify_every, "episodes_per_track": 1,
        "latency_warmup_updates": args.latency_warmup_updates,
        "latency_statistics": ["mean", "std", "median", "p95"],
        "latency_raw_samples_saved_in_per_run_json": True,
        "timing_configuration": {
            key: payloads[0]["metadata"].get(key) for key in (
                "timing_device", "gpu_name", "timing_batch_size", "num_envs",
                "classify_every", "simulator", "policy_control_frequency_hz",
                "control_period_s", "timing_clock", "cuda_synchronized_timing")
        },
        "tracks_per_seed": 10,
        "episodes_per_learned_method_classifier_seed":
            len(args.difficulties) * len(args.eval_seeds) * 10,
        "fixed_model_configurations":
            offline_manifest["fixed_model_configurations"],
        "fixed_ema_configuration": offline_manifest["fixed_ema_configuration"],
        "fixed_bayes_configuration": offline_manifest["fixed_bayes_configuration"],
        "resolved_classifier_model_paths": resolved_models,
        "track_layouts": {f"{key[0]}:{key[1]}": json.loads(value)
                          for key, value in layouts.items()},
        "representative_timeline": timeline,
        "completed_result_files": [payload["_path"] for payload in payloads],
    }
    (args.output / "locomotion_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    print(f"Aggregated {len(payloads)} completed runs in {args.output}")


if __name__ == "__main__":
    main()
