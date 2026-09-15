"""Illustrative PIT/GAP/stairs feature comparison from saved figure inputs only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot(offline_dir, indices, *, bundle=None, example_number=None):
    offline_dir = Path(offline_dir)
    bundle_path = offline_dir / "figure_data.pt"
    if bundle is None:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    examples = [bundle["examples"][index] for index in indices]
    assert [e["label"] for e in examples] == ["pit", "gap", "stairs"]
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    output = offline_dir / "figures" / "depth_and_trajectories"
    if example_number is not None:
        output = output / "terrain_feature_comparison_examples"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "terrain_feature_comparison_pit_gap_stairs"
    if example_number is not None:
        stem = stem.with_name(stem.name + f"_{example_number:02d}")
    raw = [e["arrays"]["raw_depth"].squeeze().numpy() for e in examples]
    edges = [np.ma.masked_where(~e["arrays"]["valid_neighborhood"].numpy(),
                               e["arrays"]["sobel_y"].numpy()) for e in examples]
    scale = max(float(np.abs(edge).max()) for edge in edges)
    scale = max(scale, np.finfo(float).eps)
    occupancy = [(e["arrays"]["close_mask"].numpy().astype(int)
                  + 2*e["arrays"]["far_or_invalid_mask"].numpy().astype(int)) for e in examples]
    # Keep overlapping masks explicit; do not silently collapse category 3.
    colors = ListedColormap(["#e6e6e6", "#df7d28", "#377eb8", "#984ea3"])
    norm = BoundaryNorm(np.arange(-.5, 4.5), colors.N)
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 12,
                         "axes.titlesize": 14, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(10.2, 8.5))
        grid = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, .06])
        axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(3)])
        titles = ("Extracted Sobel Edges", "Occupancy Masks", "Raw Depth")
        for r, (example, label) in enumerate(zip(examples, ("Pit (Climb)", "Gap", "Stairs"))):
            edge_plot = axes[r, 0].imshow(edges[r], cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="nearest")
            occ_plot = axes[r, 1].imshow(occupancy[r], cmap=colors, norm=norm, interpolation="nearest")
            # Direct saved raw observation; no reference subtraction, calibration,
            # crop/fill/resize, edge overlay, or per-image contrast normalization.
            raw_plot = axes[r, 2].imshow(raw[r], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            assert np.array_equal(np.asarray(raw_plot.get_array()), raw[r], equal_nan=True)
            axes[r, 0].set_ylabel(label)
            for c in range(3):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(titles[c])
        fig.colorbar(edge_plot, cax=fig.add_subplot(grid[3, 0]), orientation="horizontal").set_label("Signed Residual Sobel-Y", fontsize=12)
        bar = fig.colorbar(occ_plot, cax=fig.add_subplot(grid[3, 1]), orientation="horizontal", ticks=range(4))
        bar.ax.set_xticklabels(["Other", "Near", "Far", "Both"])
        fig.colorbar(raw_plot, cax=fig.add_subplot(grid[3, 2]), orientation="horizontal", ticks=[0, .5, 1]).set_label("Recorded Depth (Normalized)", fontsize=12)
        fig.tight_layout(pad=.55, w_pad=.7, h_pad=.5)
        assert all(text.get_fontsize() >= 12 for text in fig.findobj(matplotlib.text.Text)
                   if text.get_text())
        for suffix in ("png", "pdf"):
            fig.savefig(stem.with_suffix("."+suffix), dpi=300, bbox_inches="tight", pad_inches=.03)
        plt.close(fig)
    metadata = {
        "bundle": str(bundle_path.resolve()),
        "selection_rule": "Visually selected from a shortlist of similar processed-depth PIT/GAP pairs with strong interior horizontal Sobel responses; stairs selected for distinct visible step boundaries. Illustrative selection, not a representative accuracy sample or proof these frames were misclassified.",
        "columns": ["Saved residual Sobel-y, masked by extractor valid-neighborhood mask (spatial evidence underlying aggregated engineered features)",
                    "Saved close_mask + 2*far_or_invalid_mask: 0=other, 1=near/obstacle, 2=far/invalid, 3=both; these masks underpin scalar occupancy features, not GT occupancy",
                    "Saved raw_depth observation, before feature-extractor preprocessing/calibration; existing simulator acquisition/normalization is retained"],
        "shared_scales": {"sobel": [-scale, scale], "raw_depth": [0, 1]},
        "style": {"font": "Times New Roman", "minimum_font_size": 12,
                  "text_case": "Title Case", "layout": "tight_layout"},
        "examples": [{"bundle_example_index": index, "label": e["label"], "row_index": e["row_index"],
                      "provenance": e["provenance"]} for index, e in zip(indices, examples)],
    }
    if example_number is not None:
        metadata["selection_rule"] = (
            "Ten visually shortlisted PIT views with foreground lip/drop-edge cues; "
            "each paired greedily with its closest unused GAP raw observation by pixel MSE. "
            "Distinct stairs views selected for visible step boundaries. "
            "Illustrative selection, not an unbiased accuracy sample.")
        metadata["pit_descent_status"] = "Descent-like appearance only; traversal direction is not verified by bundle metadata"
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    # Compact self-contained inputs make the illustrative selection inspectable.
    torch.save({"metadata": metadata, "examples": examples}, stem.with_suffix(".pt"))
    print(stem.with_suffix(".png").resolve())
    print(stem.with_suffix(".pdf").resolve())
    return stem


def plot_examples(offline_dir):
    """Reproducible, unique illustrative triplets from the inspected saved bundle."""
    bundle = torch.load(Path(offline_dir) / "figure_data.pt", map_location="cpu", weights_only=False)
    pit_indices = [200, 204, 207, 237, 241, 263, 269, 272, 285, 299]
    stair_indices = [66, 19, 52, 53, 51, 89, 29, 95, 69, 98]
    gaps = {i for i, e in enumerate(bundle["examples"]) if e["label"] == "gap"}
    stems = []
    for number, (pit, stairs) in enumerate(zip(pit_indices, stair_indices), 1):
        assert bundle["examples"][pit]["label"] == "pit"
        raw = bundle["examples"][pit]["arrays"]["raw_depth"]
        gap = min(sorted(gaps), key=lambda i: float(
            (raw - bundle["examples"][i]["arrays"]["raw_depth"]).square().mean()))
        gaps.remove(gap)
        stems.append(plot(offline_dir, [pit, gap, stairs], bundle=bundle, example_number=number))
    index = stems[0].parent / "index.html"
    index.write_text('<!doctype html><meta charset="utf-8"><title>Terrain Feature Examples</title>'
                     '<h1>Terrain Feature Examples</h1>' + ''.join(
                         f'<h2>Example {i}</h2><a href="{s.name}.pdf">PDF</a> | '
                         f'<a href="{s.name}.png">PNG</a><br>'
                         f'<img src="{s.name}.png" width="800" alt="Example {i}">'
                         for i, s in enumerate(stems, 1)))
    print(index.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline_dir", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--indices", type=int, nargs=3, metavar=("PIT", "GAP", "STAIRS"))
    group.add_argument("--ten-examples", action="store_true", help="Render the ten curated triplets from the inspected paper bundle")
    args = parser.parse_args()
    if args.ten_examples:
        plot_examples(args.offline_dir)
    else:
        plot(args.offline_dir, args.indices)
