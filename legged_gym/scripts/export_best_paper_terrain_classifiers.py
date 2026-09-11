"""Export the best held-out paper terrain classifiers for DepthWaQ deployment.

One model seed is selected per architecture from Experiment 1, exactly as in
``run_paper_locomotion_evaluation.py``: highest balanced accuracy, then lower
NLL, lower Brier score, and finally lower seed.  The resulting two TorchScript
models support all paper deployment modes (instantaneous, EMA, Bayes); the
generated JSON records ready-to-copy ``terrain_selector`` configurations.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from legged_gym.scripts.export_depth_terrain_classifier import export_classifier


MODES = ("instantaneous", "ema", "bayes")


def select_best(results_path: Path):
    records = defaultdict(list)
    for record in json.loads(results_path.read_text(encoding="utf-8")):
        architecture = record.get("architecture")
        if architecture in ("feature_nn", "raw_depth_nn"):
            records[architecture].append(record)
    winners = {}
    for architecture in ("feature_nn", "raw_depth_nn"):
        if not records[architecture]:
            raise ValueError(f"No {architecture} records in {results_path}")
        winners[architecture] = min(records[architecture], key=lambda row: (
            -float(row["balanced_accuracy"]), float(row["nll"]),
            float(row["brier"]), int(row["seed"])))
    return winners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-offline-dir", type=Path, default=Path("paper_offline_eval"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination models directory in go2_deploy_python")
    args = parser.parse_args()

    root = args.paper_offline_dir.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    winners = select_best(root / "experiment_1_instantaneous_per_seed.json")
    exported = {}
    for architecture, winner in winners.items():
        seed = int(winner["seed"])
        artifact = root / "artifacts" / architecture
        output = args.output_dir / f"terrain_selector_{architecture}_best_seed_{seed}.pt"
        kwargs = {
            "architecture": architecture, "checkpoint": artifact / f"seed_{seed}" / "classifier.pt",
            "model_args_path": artifact / f"seed_{seed}" / "nn_model_args.pt",
            "output": output,
        }
        if architecture == "feature_nn":
            kwargs.update(extractor_path=artifact / "extractor.pt",
                          standardizer_path=artifact / "standardizer.pt")
        exported[architecture] = {
            "seed": seed,
            "selection_metrics": {key: winner[key] for key in ("balanced_accuracy", "nll", "brier")},
            "model_path": str(output), "model_manifest": export_classifier(**kwargs),
            "deployment_modes": [
                {"enabled": True, "model_path": str(output), "mode": mode,
                 **(manifest["fixed_ema_configuration"] if mode == "ema" else {}),
                 **({"stable_stay": manifest["fixed_bayes_configuration"]["stable_stay"]}
                    if mode == "bayes" else {})}
                for mode in MODES
            ],
        }
        print(f"Selected {architecture} seed {seed}: {output}")
    output_manifest = args.output_dir / "best_terrain_selectors.json"
    output_manifest.write_text(json.dumps({
        "selection": "highest_balanced_accuracy_then_lower_nll_then_lower_brier_then_seed",
        "source": str(root), "class_ordering": manifest["class_ordering"],
        "models": exported,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote deployment choices to {output_manifest}")


if __name__ == "__main__":
    main()
