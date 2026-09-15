"""Illustrative paired sequential comparisons using saved paper figure inputs only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


METHODS = ("instantaneous", "ema", "bayes")
KEYS = ("probabilities", "ema_probabilities", "bayes_beliefs")
LABELS = {"stairs": "Stairs", "gap": "Gap", "pap": "Gap", "pit": "Climb",
          "random_uniform": "Rough"}


def json_value(value):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def diagnostics(trace, selection, truth):
    lo, hi = selection["start"], selection["end"]
    # Exclude +/-5 frames around true transitions when assessing noise.
    steady = np.ones(hi-lo, dtype=bool)
    for boundary in selection["boundaries"]:
        steady[max(0, boundary-lo-5):min(hi-lo, boundary-lo+6)] = False
    pairs = steady[1:] & steady[:-1]
    result = []
    for method, key in zip(METHODS, KEYS):
        p = trace[key][lo:hi].numpy()
        y = trace[method+"_selected"][lo:hi].numpy()
        result.append({
            "steady_probability_variation": float(np.abs(np.diff(p, axis=0))[pairs].sum(-1).mean()) if pairs.any() else 0.,
            "steady_switches": int(((y[1:] != y[:-1]) & pairs).sum()),
            "window_accuracy": float((y == truth[lo:hi]).mean()),
        })
    return result


def generate(offline_dir, count=10):
    offline_dir = Path(offline_dir)
    bundle = torch.load(offline_dir / "figure_data.pt", map_location="cpu", weights_only=False)
    classes = bundle["class_ordering"]
    truth = np.array([classes.index(x) for x in bundle["ordered_truth"]])
    runs = {(r["architecture"], r["seed"]): r for r in bundle["runs"]}
    thumbnails = {t["sample_id"]: t for t in bundle["transition_thumbnails"]}
    candidates = []
    for selection in bundle["timeline_selections"]:
        if selection["sample_id"] not in thumbnails:
            continue
        for seed in bundle["model_seeds"]:
            scores = [diagnostics(runs[(a, seed)]["trace"], selection, truth)
                      for a in ("feature_nn", "raw_depth_nn")]
            monotonic = sum(s[0]["steady_probability_variation"] > s[1]["steady_probability_variation"]
                            > s[2]["steady_probability_variation"] for s in scores)
            reduction = sum(s[0]["steady_probability_variation"]-s[2]["steady_probability_variation"] for s in scores)
            switches_removed = sum(s[0]["steady_switches"]-s[2]["steady_switches"] for s in scores)
            candidates.append((monotonic, switches_removed, reduction, selection, seed, scores))
    candidates.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]["sample_id"], x[4]))
    chosen, used = [], set()
    for candidate in candidates:
        selection = candidate[3]
        # Distinct, nonoverlapping excerpts; same seed for both architectures.
        if any(selection["sequence_id"] == old[3]["sequence_id"] and
               max(selection["start"], old[3]["start"]) < min(selection["end"], old[3]["end"])
               for old in chosen):
            continue
        if selection["sample_id"] not in used:
            chosen.append(candidate)
            used.add(selection["sample_id"])
        if len(chosen) == count:
            break
    output = offline_dir / "figures/depth_and_trajectories/sequential_comparison_examples"
    output.mkdir(parents=True, exist_ok=True)
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    colors = plt.get_cmap("tab10")(np.arange(len(classes)))
    records = []
    for number, (monotonic, removed, reduction, sel, seed, scores) in enumerate(chosen, 1):
        lo, hi = sel["start"], sel["end"]
        ids = bundle["ordered_provenance"]["sequence_ids"][lo:hi]
        assert len(set(ids)) == 1, "Timeline crosses a sequence reset"
        origin = sel["boundaries"][0]
        x = np.arange(lo, hi)-origin
        thumb = thumbnails[sel["sample_id"]]
        traces = {}
        with plt.rc_context({"font.family": "Times New Roman", "font.size": 12,
                             "axes.titlesize": 16, "axes.labelsize": 13,
                             "xtick.labelsize": 12, "ytick.labelsize": 12, "pdf.fonttype": 42}):
            fig = plt.figure(figsize=(18, 11))
            grid = fig.add_gridspec(5, 2, height_ratios=[1.05, .24, 1, 1, 1])
            strip = grid[0, :].subgridspec(1, len(thumb["row_indices"]))
            for j, row in enumerate(thumb["row_indices"]):
                ax = fig.add_subplot(strip[0, j])
                ax.imshow(thumb["depth_images"][j], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f'Frame {row-origin:+d}\n{LABELS.get(thumb["labels"][j], thumb["labels"][j].title())}', fontsize=12)
                ax.set_xticks([]); ax.set_yticks([])
            legend_ax = fig.add_subplot(grid[1, :]); legend_ax.axis("off")
            handles = [Line2D([0], [0], color=colors[c], label=LABELS.get(label, label.title()), lw=2)
                       for c, label in enumerate(classes)]
            legend_ax.legend(handles=handles, loc="center left", ncol=len(classes), frameon=False)
            legend_ax.text(.55, .5, "Lower Color Bands: Ground Truth (Upper) / Selected Skill (Lower)", va="center", fontsize=12)
            for col, (arch, title) in enumerate((("feature_nn", "Features"), ("raw_depth_nn", "Raw Depth"))):
                trace = runs[(arch, seed)]["trace"]
                traces[arch] = {key: trace[key][lo:hi].clone() for key in
                               ("logits", *KEYS, *(m+"_selected" for m in METHODS))}
                for row, (method, key, name) in enumerate(zip(METHODS, KEYS, ("Instantaneous", "EMA", "Bayes"))):
                    ax = fig.add_subplot(grid[row+2, col])
                    p = traces[arch][key].numpy()
                    assert p.shape == (hi-lo, len(classes))
                    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-5)
                    for c in range(len(classes)):
                        ax.plot(x, p[:, c], color=colors[c], lw=1.5)
                    emitted = traces[arch][method+"_selected"].numpy()
                    bands = np.stack([colors[truth[lo:hi]], colors[emitted]])
                    ax.imshow(bands, aspect="auto", interpolation="nearest", extent=[x[0]-.5, x[-1]+.5, -.27, -.08])
                    for boundary in sel["boundaries"]:
                        ax.axvline(boundary-origin, color="black", ls=":", lw=1)
                    ax.set(xlim=(x[0]-.5, x[-1]+.5), ylim=(-.29, 1.03), yticks=[0, .5, 1])
                    ax.set_ylabel(name+"\n"+("Belief" if method == "bayes" else "Probability"))
                    if row == 0:
                        ax.set_title(f"{title} — Seed {seed}")
                    if row == 2:
                        ax.set_xlabel("Classification Frames Relative To First Terrain Boundary")
                    else:
                        ax.tick_params(labelbottom=False)
                    ax.grid(alpha=.15)
            fig.tight_layout(pad=.65, h_pad=.65, w_pad=1.3)
            assert all(t.get_fontsize() >= 12 for t in fig.findobj(matplotlib.text.Text) if t.get_text())
            stem = output / f"sequential_comparison_{number:02d}"
            for ext in ("png", "pdf"):
                fig.savefig(stem.with_suffix("."+ext), dpi=220, bbox_inches="tight", pad_inches=.03)
            plt.close(fig)
        metadata = {"example": number, "selection": sel, "seed": seed,
                    "class_ordering": classes, "filter_settings": bundle["filter_settings"],
                    "model_settings": bundle["model_settings"],
                    "monotonic_smoothing_architectures": monotonic,
                    "steady_switches_removed": removed, "steady_variation_reduction": reduction,
                    "diagnostics": dict(zip(("feature_nn", "raw_depth_nn"), scores)),
                    "diagnostic_method_order": METHODS,
                    "thumbnail_note": "Saved depth thumbnails; original bundle resized these to 48x64. No additional calibration.",
                    "skill_note": bundle.get("skill_semantics"),
                    "selection_rule": "Rank saved windows by number of architectures with strictly decreasing steady-state probability total variation (instantaneous > EMA > Bayes), then steady switches removed, then variation reduction. Exclude +/-5 boundary frames for noise diagnostics. Same seed for paired classifiers; nonoverlapping windows. Illustrative post-hoc selection, not performance estimation; smoothing does not imply better accuracy or delay."}
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=json_value))
        provenance = {k: v[lo:hi] for k, v in bundle["ordered_provenance"].items()
                      if v is not None and k not in ("unavailable", "provenance_verified")}
        torch.save({"metadata": metadata, "traces": traces, "truth": truth[lo:hi],
                    "provenance": provenance, "thumbnails": thumb}, stem.with_suffix(".pt"))
        records.append(metadata)
        print(f"Saved {stem.name}: Monotonic Smoothing In {monotonic}/2 Architectures", flush=True)
    (output / "selection_manifest.json").write_text(json.dumps({"requested": count, "generated": len(records),
        "source_bundle": str((offline_dir / "figure_data.pt").resolve()), "examples": records}, indent=2, default=json_value))
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Sequential Comparisons</title>'
        '<h1>Sequential Comparison Examples</h1>' + ''.join(
        f'<h2>Example {i:02d}</h2><a href="sequential_comparison_{i:02d}.pdf">PDF</a> | '
        f'<a href="sequential_comparison_{i:02d}.png">PNG</a><br>'
        f'<img src="sequential_comparison_{i:02d}.png" width="1200" alt="Example {i}">'
        for i in range(1, len(records)+1)))
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline_dir", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--case-studies", action="store_true", help="Select full-sequence main/appendix cases without inference")
    parser.add_argument("--ordered-test", type=Path, help="Existing compiled ordered test.pt for provenance-aligned images")
    parser.add_argument("--plot-case-bundle", type=Path, help="Regenerate exclusively from a saved case-study bundle")
    args = parser.parse_args()
    if args.case_studies or args.plot_case_bundle:
        if __package__:
            from .paper_sequential_case_studies import make_bundle, render_case_studies
        else:
            from paper_sequential_case_studies import make_bundle, render_case_studies
        if not args.plot_case_bundle and not args.ordered_test:
            parser.error("--case-studies requires --ordered-test")
        path = args.plot_case_bundle or make_bundle(args.offline_dir, args.ordered_test)
        render_case_studies(path)
    else:
        generate(args.offline_dir, args.count)
