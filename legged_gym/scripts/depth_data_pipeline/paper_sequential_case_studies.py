"""Post-hoc case studies: saved full-sequence selections, never new inference."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ARCHES = ("feature_nn", "raw_depth_nn")
METHODS = ("instantaneous", "ema", "bayes")
KEYS = ("probabilities", "ema_probabilities", "bayes_beliefs")
# Display labels preserve the evaluator's four semantic labels (Pit, not Climb).
COLORS = {"stairs": "#0072B2", "gap": "#E69F00", "pit": "#009E73", "random_uniform": "#CC79A7"}
NAMES = {"stairs": "Stairs", "gap": "Gap", "pit": "Pit", "random_uniform": "Rough"}


def native(v):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def segments_and_mask(truth, ids, radius=5):
    """Equivalent to evaluator _transition_window_mask, clipped per sequence."""
    truth, ids = np.asarray(truth), np.asarray(ids)
    seq_edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1, len(truth)]
    assert len(set(ids[seq_edges[:-1]])) == len(seq_edges)-1
    mask, segments = np.zeros(len(truth), bool), []
    for q0, q1 in zip(seq_edges[:-1], seq_edges[1:]):
        edges = np.r_[q0, np.flatnonzero(truth[q0+1:q1] != truth[q0:q1-1])+q0+1, q1]
        for t in edges[1:-1]:
            mask[max(q0, t-radius):min(q1, t+radius+1)] = True
        for j, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
            segments.append(dict(start=int(s), end=int(e), sequence_start=int(q0), sequence_end=int(q1),
                                 previous_start=int(edges[max(0, j-1)]), label=int(truth[s]),
                                 sequence_id=str(ids[s]), terminal=bool(e == q1)))
    return segments, mask


def first_match(pred, segment):
    hits = np.flatnonzero(pred[segment["start"]:segment["end"]] == segment["label"])
    return int(hits[0]) if len(hits) else None


def runs_of(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def metrics(pred, truth, window, lo, hi):
    p, y, w = pred[lo:hi], truth[lo:hi], window[lo:hi]
    mean = lambda a: float(a.mean()) if len(a) else None
    return dict(steady_state_accuracy=mean((p == y)[~w]), transition_window_accuracy=mean((p == y)[w]),
                frame_accuracy=mean(p == y),
                # Exact existing false-transition definition: denominator all adjacent frames.
                erroneous_switch_rate=float(((p[1:] != p[:-1]) & (y[1:] == y[:-1])).sum()/(hi-lo-1))
                if hi-lo > 1 else None)


def select_cases(bundle, old_example):
    classes = bundle["class_ordering"]
    truth = np.array([classes.index(t) for t in bundle["ordered_truth"]])
    segments, window = segments_and_mask(truth, bundle["ordered_provenance"]["sequence_ids"])
    runs = {(r["architecture"], r["seed"]): r for r in bundle["runs"]}
    candidates = {"A": [], "B": []}
    for seg in segments:
        s, e = seg["start"], seg["end"]
        if s == seg["sequence_start"]:
            continue
        for seed in bundle["model_seeds"]:
            for arch in ARCHES:
                predictions = [runs[(arch, seed)]["trace"][m+"_selected"].numpy() for m in METHODS]
                delays = [first_match(p, seg) for p in predictions]
                length = e-s
                if 1 <= length <= 10 and not seg["terminal"] and delays[2] is None and any(d is not None for d in delays[:2]):
                    retains = bool(np.all(predictions[2][s:e] == truth[s-1]))
                    candidates["B"].append(dict(case="B", seed=int(seed), architecture=arch, segment=seg,
                        lo=max(seg["sequence_start"], s-15), hi=min(seg["sequence_end"], e+15),
                        delays=delays, retains_preceding=retains, length=length,
                        rank=[int(retains), int(delays[1] is not None), length]))
                lo, hi = max(seg["previous_start"], s-40), min(e, s+20)
                scores = [metrics(p, truth, window, lo, hi) for p in predictions]
                # Brief incorrect stable-state emitted runs suppressed by Bayes.
                suppressed = []
                for p in predictions[:2]:
                    start = seg["previous_start"]
                    bad = p[start:s] != truth[start:s]
                    suppressed.append(sum(b-a for a, b in runs_of(bad) if b-a <= 4
                                          and start+a >= lo and not window[start+a:start+b].any()
                                          and np.all(predictions[2][start+a:start+b] == truth[start+a:start+b])))
                if not max(suppressed):
                    continue
                gain = (scores[2]["steady_state_accuracy"] or 0)-(scores[1]["steady_state_accuracy"] or 0)
                responsive_cost = ((delays[1] is not None and (delays[2] is None or delays[2] > delays[1])) or
                    ((scores[2]["transition_window_accuracy"] or 0) < (scores[1]["transition_window_accuracy"] or 0)))
                if gain > 0 and responsive_cost and length >= 20 and delays[1] is not None and delays[2] is not None:
                    candidates["A"].append(dict(case="A", seed=int(seed), architecture=arch, segment=seg,
                        lo=lo, hi=hi, delays=delays, metrics=scores, suppressed_frames=suppressed,
                        rank=[int(suppressed[1] > 0),
                              int(delays[2] > delays[1] and scores[2]["transition_window_accuracy"] < scores[1]["transition_window_accuracy"]),
                              int(not seg["terminal"]), gain, suppressed[1], max(suppressed)]))
    chosen = []
    for name in ("A", "B"):
        candidates[name].sort(key=lambda c: (tuple(-v for v in c["rank"]), c["segment"]["start"], c["seed"], c["architecture"]))
        if candidates[name]:
            chosen.append(candidates[name][0])
    if not Path(old_example).exists():
        return chosen, candidates, segments, window, truth, runs
    old = json.loads(Path(old_example).read_text())
    sel = old["selection"]
    seg = next(s for s in segments if s["sequence_id"] == sel["sequence_id"] and
               s["start"] == sel["focus_frame"])
    assert classes[seg["label"]] == "gap"
    p = runs[("raw_depth_nn", old["seed"])]["trace"]["bayes_selected"].numpy()
    intervals = runs_of(p[seg["start"]:seg["end"]] == classes.index("random_uniform"))
    if not intervals:
        return chosen, candidates, segments, window, truth, runs
    a, b = max(intervals, key=lambda ab: (ab[1]-ab[0], -ab[0]))
    chosen.append(dict(case="C", seed=old["seed"], architecture="raw_depth_nn", segment=seg,
        lo=sel["start"], hi=max(sel["end"], seg["start"]+b), old_example=str(old_example),
        longest_incorrect_interval=[seg["start"]+int(a), seg["start"]+int(b)],
        recognized_gap_before_error=bool(np.any(p[seg["start"]:seg["start"]+a] == seg["label"]))))
    return chosen, candidates, segments, window, truth, runs


def load_images(dataset, bundle, rows):
    """Validate the entire ordered alignment before resolving requested source rows."""
    data = torch.load(dataset, map_location="cpu", weights_only=False, mmap=True)
    assert data["labels"] == bundle["ordered_truth"], "Dataset label order differs"
    for k in ("source_ids", "sequence_ids", "original_env_ids", "episode_ids", "frame_indices", "control_step_indices"):
        a, b = data[k], bundle["ordered_provenance"][k]
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"Dataset provenance mismatch: {k}"
    images = data["depth_images"][rows].detach().cpu().clone()
    assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
    return images


def make_bundle(offline_dir, dataset):
    offline = Path(offline_dir)
    output = offline / "figures/depth_and_trajectories/sequential_case_studies"
    output.mkdir(parents=True, exist_ok=True)
    b = torch.load(offline / "figure_data.pt", map_location="cpu", weights_only=False)
    old = offline / "figures/depth_and_trajectories/sequential_comparison_examples/sequential_comparison_01.json"
    chosen, candidates, segments, window, truth, runs = select_cases(b, old)
    radius = json.loads((offline / "manifest.json").read_text()).get("transition_window_radius", 5)
    assert radius == 5
    csv_path = offline / "experiment_2_transitions_v2.csv"
    transitions = {}
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                transitions[(row["architecture"], int(row["seed"]), row["temporal_method"], int(row["transition_frame"]))] = row
    result = dict(schema_version=1, classes=b["class_ordering"], filter_settings=b["filter_settings"],
        model_settings=b["model_settings"], candidates=candidates, cases=[],
        plot_config=dict(width_inches=7.16, case_height_inches=5.25, extra_height_inches=.35,
                         font="Times New Roman", minimum_font_pt=12, title_font_pt=13,
                         dpi=300, pdf_fonttype=42, palette=COLORS, display_labels=NAMES,
                         method_order=METHODS, line_styles=[":", "--", "-"], depth_limits=[0,1]),
        selection_rule="Full-sequence emitted-label eligibility; A: target segment >=20 frames, EMA and Bayes match; Bayes improves steady accuracy over EMA, suppresses complete 1–4-frame errors, and loses boundary responsiveness. Rank: EMA error suppression, both longer delay and lower transition accuracy, nonterminal target, steady accuracy gain, suppressed EMA/any frames. B: complete nonterminal 1–10-frame segment, Bayes misses, another selector matches. Rank: preceding-label retention, EMA match, segment length. Ties: start row, seed, architecture. No probability-smoothness ranking. C: verified original example 01 identity.",
        interpretation="Illustrative post-hoc cases, not aggregate estimates. First match is not sustained switching. Short label duration is not physical obstacle length. No causal/control claims.",
        sources={"bundle": str((offline / "figure_data.pt").resolve()), "dataset": str(Path(dataset).resolve()),
                 "transitions_csv": str(csv_path) if csv_path.exists() else None}, unavailable=[])
    for case in chosen:
        lo, hi, s = case["lo"], case["hi"], case["segment"]
        assert hi <= s["sequence_end"] and lo >= s["sequence_start"]
        rows = sorted(set([lo, s["start"], min(hi-1, s["end"]-1), hi-1]))
        if case["case"] == "C":
            rows = sorted(set([lo, s["start"], case["longest_incorrect_interval"][0], hi-1]))
        scores, traces = {}, {}
        for arch in ARCHES:
            trace = runs[(arch, case["seed"])]["trace"]
            traces[arch] = {k: trace[k][lo:hi].clone() for k in ("logits", *KEYS, *(m+"_selected" for m in METHODS))}
            scores[arch] = {}
            for method in METHODS:
                pred = trace[method+"_selected"].numpy()
                delay = first_match(pred, s)
                scores[arch][method] = dict(metrics(pred, truth, window, lo, hi),
                    first_match_delay=delay, whole_segment_missed=delay is None)
                if transitions and s["start"] != s["sequence_start"]:
                    record = transitions[(arch, int(case["seed"]), method, s["start"])]
                    assert int(record["segment_end_frame_exclusive"]) == s["end"]
                    assert int(record["sequence_start_frame"]) == s["sequence_start"]
                    assert record["target_label"] == b["class_ordering"][s["label"]]
                    expected = None if record["missed"] == "True" else int(record["delay_classification_frames"])
                    assert expected == delay, "Mismatch with evaluator's segment-bounded matching"
            for key in KEYS:
                np.testing.assert_allclose(traces[arch][key].sum(-1), 1, atol=1e-5)
            for method, key in zip(METHODS, KEYS):
                assert torch.equal(traces[arch][method+"_selected"], traces[arch][key].argmax(-1))
        provenance = {k: v[lo:hi] for k, v in b["ordered_provenance"].items()
                      if v is not None and k not in ("unavailable", "provenance_verified")}
        assert len(set(provenance["sequence_ids"])) == 1
        item = dict(selection=case, rows=np.arange(lo, hi), truth=truth[lo:hi].copy(),
                    window_mask=window[lo:hi].copy(), traces=traces, metrics=scores, provenance=provenance,
                    boundaries=[g["start"] for g in segments if g["sequence_id"] == s["sequence_id"]
                                and lo <= g["start"] < hi and g["start"] != g["sequence_start"]],
                    thumbnail_rows=rows, thumbnail_labels=[b["ordered_truth"][i] for i in rows])
        try:
            item["images"] = load_images(dataset, b, rows)
        except (FileNotFoundError, KeyError, AssertionError) as exc:
            item["images"] = None
            result["unavailable"].append(f"Case {case['case']} images: {exc}")
        if case["case"] == "C":
            raw = traces["raw_depth_nn"]["bayes_selected"].numpy()
            item["raw_bayes_gt_disagreement_fraction"] = float((raw != item["truth"]).mean())
            feat = traces["feature_nn"]["bayes_selected"].numpy()
            item["bayes_representation_disagreement_fraction"] = float((raw != feat).mean())
        result["cases"].append(item)
        print(f"Case {case['case']}: {case['architecture']} seed {case['seed']}, segment {s['start']}:{s['end']}", flush=True)
    for name in ("A", "B", "C"):
        if not any(c["selection"]["case"] == name for c in result["cases"]):
            result["unavailable"].append(f"No eligible Case {name}")
    result["checks"] = dict(full_sequence_segments=True, no_filter_reset_or_inference=True,
        probability_normalization=True, emitted_selections_preserved=True,
        segment_matches_crosschecked_csv=bool(transitions), transition_radius=5)
    path = output / "case_study_figure_data.pt"
    torch.save(result, path)
    (output / "selection_metadata.json").write_text(json.dumps({k:v for k,v in result.items() if k not in ("cases", "candidates")}, indent=2, default=native))
    (output / "candidate_metrics.json").write_text(json.dumps(candidates, indent=2, default=native))
    (output / "chosen_metrics.json").write_text(json.dumps([dict(selection=c["selection"], metrics=c["metrics"],
        thumbnail_rows=c["thumbnail_rows"]) for c in result["cases"]], indent=2, default=native))
    return path


def render_case_studies(path):
    """Self-contained regeneration: no source dataset or predictions recomputed."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    output = Path(path).parent
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    colors = [COLORS[c] for c in data["classes"]]
    style = dict(zip(METHODS, (":", "--", "-")))
    for filename, cases in (("main_stability_and_short_segment", [c for c in data["cases"] if c["selection"]["case"] in ("A", "B")]),
                            ("appendix_sustained_misclassification", [c for c in data["cases"] if c["selection"]["case"] == "C"])):
        if not cases:
            continue
        with plt.rc_context({"font.family":"Times New Roman", "font.size":12, "axes.labelsize":12,
                             "axes.titlesize":13, "xtick.labelsize":12, "ytick.labelsize":12,
                             "legend.fontsize":12, "pdf.fonttype":42, "ps.fonttype":42}):
            fig = plt.figure(figsize=(7.16, 5.25*len(cases)+.35))
            grid = fig.add_gridspec(4*len(cases)+1, 2, height_ratios=[.85, 1.1, 2.4, .8]*len(cases)+[.5])
            for ci, c in enumerate(cases):
                sel, s = c["selection"], c["selection"]["segment"]
                origin, row0 = s["start"], ci*4
                x = c["rows"]-origin
                case_id = sel["case"]
                title = {"A":"A: Stability Vs. Responsiveness", "B":"B: Missed Short Segment", "C":"C: Sustained Misclassification"}[case_id]
                thumbs = grid[row0, :].subgridspec(1, len(c["thumbnail_rows"]))
                for j, row in enumerate(c["thumbnail_rows"]):
                    ax = fig.add_subplot(thumbs[0, j])
                    if c["images"] is not None:
                        image = c["images"][j].squeeze().numpy()
                        assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
                        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
                    else:
                        ax.text(.5, .5, "Unavailable", ha="center")
                    ax.set_title(f"{chr(65+j)}: {row-origin:+d}\nGT: {NAMES[c['thumbnail_labels'][j]]}", fontsize=12)
                    ax.set_xticks([]); ax.set_yticks([])
                for col, arch in enumerate(ARCHES):
                    ax = fig.add_subplot(grid[row0+1, col])
                    targets = [s["label"]] if case_id != "C" else [data["classes"].index(k) for k in ("gap", "random_uniform")]
                    for target in targets:
                        for method, key in zip(METHODS, KEYS):
                            ax.plot(x, c["traces"][arch][key][:, target], ls=style[method], color=colors[target], lw=1.4)
                    label = NAMES[data["classes"][s["label"]]] if case_id != "C" else "Gap / Rough"
                    ax.set_title(f"{'Features' if col == 0 else 'Raw Depth'}: {label}")
                    ax.set_ylim(-.03, 1.15); ax.set_yticks([0, 1]); ax.set_ylabel("Prob. / Belief")
                    ax.set_xlim(x[0]-.5, x[-1]+.5)
                    ax.set_xticks([t for t in ax.get_xticks() if x[0] <= t <= x[-1]])
                    for j, row in enumerate(c["thumbnail_rows"]):
                        ax.axvline(row-origin, color=".7", lw=.5)
                        ax.text(row-origin, 1.04, chr(65+j), ha="center", fontsize=12)
                    for a, b in runs_of(c["window_mask"]):
                        ax.axvspan(x[a]-.5, x[b-1]+.5, color=".6", alpha=.18, zorder=-1)
                timeline = fig.add_subplot(grid[row0+2, :])
                selections = [c["truth"]]+[c["traces"][a][m+"_selected"].numpy() for a in ARCHES for m in METHODS]
                timeline.imshow(np.stack(selections), cmap=ListedColormap(colors), vmin=-.5, vmax=3.5,
                    aspect="auto", interpolation="nearest", extent=[x[0]-.5, x[-1]+.5, 6.5, -.5])
                timeline.set_yticks(range(7), ["Ground Truth", "Features Inst.", "Features EMA", "Features Bayes", "Raw Inst.", "Raw EMA", "Raw Bayes"])
                for t in c["boundaries"]:
                    timeline.axvline(t-origin, color="black", ls="--", lw=1)
                for a, b in runs_of(c["window_mask"]):
                    timeline.axvspan(x[a]-.5, x[b-1]+.5, color=".6", alpha=.15)
                timeline.set_xlabel("Classification Frames Relative To Target Boundary")
                timeline.set_xticks([t for t in timeline.get_xticks() if x[0] <= t <= x[-1]])
                note = fig.add_subplot(grid[row0+3, :]); note.axis("off")
                lines = [title+f" — Seed {sel['seed']}"]
                def delays(arch):
                    return "/".join("Miss" if c["metrics"][arch][m]["first_match_delay"] is None else str(c["metrics"][arch][m]["first_match_delay"]) for m in METHODS)
                lines.append(f"First Match (Inst./EMA/Bayes): Features {delays('feature_nn')}; Raw {delays('raw_depth_nn')}")
                if case_id == "B":
                    lines.append(f"Complete Segment: {s['end']-s['start']} Frames; Both Boundaries Shown")
                elif case_id == "A":
                    arch = sel["architecture"]; met = c["metrics"][arch]
                    lines.append(f"{'Features' if arch == 'feature_nn' else 'Raw'} Steady Accuracy: EMA {met['ema']['steady_state_accuracy']:.0%} → Bayes {met['bayes']['steady_state_accuracy']:.0%}; {sel['suppressed_frames'][1]} Error Frames Suppressed")
                else:
                    a, b = sel["longest_incorrect_interval"]
                    lines.append(f"Raw Bayes Error: {a-origin:+d} To {b-origin-1:+d} ({b-a} Frames); Window Error {c['raw_bayes_gt_disagreement_fraction']:.0%}")
                    timeline.plot([a-origin-.5,b-origin-.5], [6,6], color="black", lw=2)
                note.text(0, 1, "\n".join(lines), va="top", fontsize=12)
            legend = fig.add_subplot(grid[-1, :]); legend.axis("off")
            handles = [Line2D([], [], ls=style[m], color="black", label=l) for m,l in zip(METHODS,("Inst. Prob.","EMA Prob.","Bayes Belief"))]
            handles += [Patch(facecolor=colors[i], label=NAMES[k]) for i,k in enumerate(data["classes"])]
            handles += [Patch(facecolor=".8", alpha=.5, label="±5 Frames")]
            legend.legend(handles=handles, loc="center", ncol=4, frameon=False, handlelength=1.2, columnspacing=.6)
            fig.tight_layout(pad=.5, h_pad=.3, w_pad=.5)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for text in fig.findobj(matplotlib.text.Text):
                if text.get_text() and text.get_visible():
                    assert text.get_fontsize() >= 12
                    box = text.get_window_extent(renderer)
                    assert box.x0 >= -1 and box.x1 <= fig.bbox.width+1, f"Clipped text: {text.get_text()}"
                    assert box.y0 >= -1 and box.y1 <= fig.bbox.height+1, f"Clipped text: {text.get_text()}"
            for ext in ("pdf", "png"):
                fig.savefig(output / f"{filename}.{ext}", dpi=300)
            plt.close(fig)
            print(f"Saved {filename}", flush=True)
    return output
