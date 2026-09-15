"""Restyle only the saved aggregate structural confusion pair; no inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot(offline_dir):
    offline_dir = Path(offline_dir)
    classes = json.loads((offline_dir / "manifest.json").read_text())["class_ordering"]
    rows = json.loads((offline_dir / "experiment_1_instantaneous_per_seed.json").read_text())
    labels = {"stairs": "Stairs", "gap": "Gap", "pap": "Gap", "pit": "Climb", "random_uniform": "Rough"}
    # Require the actual requested font instead of silently substituting a serif.
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    destination = offline_dir / "figures" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 12,
                         "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(7.2, 3.4))
        grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, .045])
        first = fig.add_subplot(grid[0, 0])
        second = fig.add_subplot(grid[0, 1], sharey=first)
        color_axis = fig.add_subplot(grid[0, 2])
        for architecture, title, ax in (("feature_nn", "Features", first), ("raw_depth_nn", "Raw Depth", second)):
            selected = [row for row in rows if row["architecture"] == architecture]
            if not selected:
                raise ValueError(f"No saved confusion counts for {architecture}")
            counts = np.asarray([row["confusion_matrix"] for row in selected], dtype=float).sum(0)
            if counts.shape != (len(classes), len(classes)):
                raise ValueError("Confusion dimensions differ from saved class ordering")
            totals = counts.sum(1, keepdims=True)
            matrix = np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
                   xticklabels=[labels.get(label, label) for label in classes],
                   yticklabels=[labels.get(label, label) for label in classes],
                   xlabel="Predicted", title=title)
            for y in range(len(classes)):
                for x in range(len(classes)):
                    value = matrix[y, x]
                    ax.text(x, y, f"{value:.2f}" if np.isfinite(value) else "N/A",
                            ha="center", va="center", fontsize=12,
                            color="white" if value > .5 else "black")
        first.set_ylabel("Ground truth")
        second.tick_params(axis="y", labelleft=False)
        fig.colorbar(im, cax=color_axis, label="Fraction within GT class")
        fig.tight_layout(pad=.5, w_pad=.9)
        fig.canvas.draw()
        assert first.get_shared_y_axes().joined(first, second)
        assert all(text.get_fontsize() >= 12 for text in fig.findobj(matplotlib.text.Text))
        for suffix in ("png", "pdf"):
            path = destination / f"confusion_structural_aggregate.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.03)
            print(path.resolve())
        plt.close(fig)
    print(f"Font: {font_path}; native ordering preserved: {classes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline_dir", type=Path)
    plot(parser.parse_args().offline_dir)
