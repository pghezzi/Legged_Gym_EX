"""Synthetic figure tests; no checkpoint training or simulator runs."""
from types import SimpleNamespace
from pathlib import Path
import copy
import json

import pytest
import torch

from legged_gym.scripts.depth_data_pipeline import evaluate_paper_offline_experiments_1_2 as paper
from legged_gym.scripts.depth_data_pipeline import paper_figure_bundle as figures
from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor


def sample_bundle():
    classes = ["flat", "gap", "stairs"]
    truth = ["flat"]*4 + ["gap"]*4 + ["stairs"]*4
    ids = ["capture/env0/episode0"]*12
    target = torch.tensor([classes.index(label) for label in truth])
    runs, rows1, rows2 = [], [], []
    for architecture in figures.ARCHITECTURES:
        for seed in (0, 1, 2):
            logits = torch.randn(12, 3, generator=torch.Generator().manual_seed(seed)) + torch.nn.functional.one_hot(target, 3)*3
            trace = {}
            values = paper._sequential_metrics(SimpleNamespace(class_ids=classes), logits[None], truth, ids, trace_sink=trace)
            counts = values["instantaneous"]["confusion_matrix"].tolist()
            row = {"architecture": architecture, "seed": seed, "confusion_matrix": counts,
                   "per_class_recall": values["instantaneous"]["per_class_recall"]}
            row.update({key: values["instantaneous"].get(key, .1) for key in paper.SCALAR_EXPERIMENT_1_METRICS})
            rows1.append(row)
            for method, metrics in values.items():
                rows2.append({"architecture": architecture, "seed": seed, "temporal_method": method,
                              **{key: metrics[key] for key in paper.SCALAR_EXPERIMENT_2_METRICS + paper.TRANSITION_V2_METRICS},
                              "confusion_matrix": metrics["confusion_matrix"].tolist(), "per_class_recall": metrics["per_class_recall"]})
            runs.append({"architecture": architecture, "seed": seed, "trace": trace,
                         "structural_logits": logits, "structural_probabilities": logits.softmax(-1)})
    data = {"labels": truth, "sequence_ids": ids, "source_ids": ["capture"]*12,
            "frame_indices": torch.arange(12), "depth_images": torch.rand(12, 24, 32)*.7+.1,
            "orientation_rpy": torch.zeros(12, 3), "angular_velocity": torch.zeros(12, 3)}
    extractor = SobelDepthTerrainFeatureExtractor(output_size=(24, 32), min_depth=.02, max_depth=1., sobel_edge_threshold=.007)
    examples, unavailable = figures.geometric_examples(extractor, data, classes)
    bundle = dict(schema_version=1, class_ordering=classes, model_seeds=[0, 1, 2], runs=runs,
                  ordered_truth=truth, structural_truth=truth, ordered_provenance=figures.provenance(data),
                  structural_provenance=figures.provenance(data), examples=examples, unavailable=unavailable,
                  experiment_1_rows=rows1, experiment_2_rows=rows2,
                  experiment_1_summary=paper._aggregate(rows1, ("architecture",), paper.SCALAR_EXPERIMENT_1_METRICS, classes),
                  experiment_2_summary=paper._aggregate(rows2, ("architecture", "temporal_method"), paper.SCALAR_EXPERIMENT_2_METRICS + paper.TRANSITION_V2_METRICS, classes),
                  timeline_selection=figures.select_timeline(truth, ids), filter_settings={"ema": paper.FIXED_EMA_CONFIG, "bayes": paper.FIXED_BAYES_CONFIG},
                  plot_config={**figures.PLOT_CONFIG, "dpi": 45})
    return bundle, extractor


def test_cues_are_actual_extractor_features():
    bundle, extractor = sample_bundle()
    for example in bundle["examples"]:
        a = example["arrays"]
        energy = (a["sobel_y"].abs()*a["valid_neighborhood"]).sum() / a["valid_neighborhood"].sum().clamp_min(1) / extractor.depth_scale
        index = example["feature_names"].index("horizontal_edge_energy")
        torch.testing.assert_close(energy, a["features"][index])
        torch.testing.assert_close(a["processed_depth"]-a["reference"], a["residual"])


def test_validation_and_roundtrip(tmp_path):
    bundle, _ = sample_bundle()
    path = tmp_path / "figure_data.pt"
    info = figures.save_bundle(bundle, path)
    assert len(info["sha256"]) == 64
    restored = figures.load_bundle(path)
    for original, loaded in zip(bundle["runs"], restored["runs"]):
        for key in original["trace"]:
            torch.testing.assert_close(original["trace"][key], loaded["trace"][key])
    for original, loaded in zip(bundle["examples"], restored["examples"]):
        assert original["provenance"] == loaded["provenance"]
        for key in original["arrays"]:
            torch.testing.assert_close(original["arrays"][key], loaded["arrays"][key])
    bad = copy.deepcopy(bundle)
    bad["runs"][0]["trace"]["bayes_beliefs"][0] *= .5
    with pytest.raises(AssertionError):
        figures.validate_bundle(bad)
    bad = copy.deepcopy(bundle)
    bad["plot_config"]["confusion_vmax"] = .9
    with pytest.raises(ValueError):
        figures.validate_bundle(bad)
    matrix = figures.normalized_confusion([[2, 2], [0, 0]])
    assert matrix[0].tolist() == [.5, .5]
    assert bool(torch.isnan(torch.as_tensor(matrix[1])).all())


def test_plot_only_no_checkpoints_or_datasets(tmp_path, monkeypatch):
    bundle, _ = sample_bundle()
    path = tmp_path / "figure_data.pt"
    figures.save_bundle(bundle, path)
    original_load = torch.load
    def guarded_load(filename, *args, **kwargs):
        assert Path(filename).resolve() == path.resolve(), "Plot-only tried to load external input"
        return original_load(filename, *args, **kwargs)
    monkeypatch.setattr(torch, "load", guarded_load)
    def forbidden(*args, **kwargs):
        raise AssertionError("Plot-only invoked evaluation/inference/training")
    for name in ("_sequential_metrics", "_instantaneous_metrics", "collect_engineered_logits", "collect_raw_depth_logits", "fit_nn", "_resolve_data_folder"):
        monkeypatch.setattr(paper, name, forbidden)
    import matplotlib.axes
    original_imshow = matplotlib.axes.Axes.imshow
    scales = []
    def checked_imshow(self, x, *args, **kwargs):
        if getattr(x, "shape", None) == (3, 3):
            scales.append((kwargs["vmin"], kwargs["vmax"]))
        return original_imshow(self, x, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", checked_imshow)
    output = tmp_path / "plots"
    paper.main(["--plot-only", str(path), "--output", str(output), "--dataset", "/missing/dataset"])
    assert len(scales) == 32 and set(scales) == {(0, 1)}
    assert len(list(output.glob("figures/depth_and_trajectories/timeline_*.png"))) == 18
    assert len(list(output.glob("figures/results/confusion_*.png"))) == 16
    assert len(list(output.rglob("*.pdf"))) == len(list(output.rglob("*.png")))
    assert not list(output.glob("*.png"))
    manifest = json.loads((output / "figure_manifest.json").read_text())
    assert manifest["bundle"] == str(path.resolve())
    assert all(Path(filename).is_file() for filename in manifest["outputs"])


def test_unavailable_timeline_and_examples():
    assert figures.select_timeline([0, 1, 0], ["a", "a", "b"]) is None
    assert figures.geometric_examples(None, {"labels": [0]}, [0])[1]


def test_image_coverage_sampling_and_invalid_replacement():
    import numpy as np
    data = dict(labels=["flat"]*120, source_ids=["a"]*60+["b"]*60,
                original_env_ids=torch.tensor([0]*30+[1]*30+[0]*30+[1]*30),
                sequence_ids=[f"seq{i//30}" for i in range(120)], frame_indices=torch.arange(120),
                depth_images=torch.full((120, 24, 32), .4),
                orientation_rpy=torch.zeros(120, 3), angular_velocity=torch.zeros(120, 3))
    a = list(figures.coverage_order(range(120), data, np.random.default_rng(42)))
    b = list(figures.coverage_order(range(120), data, np.random.default_rng(42)))
    c = list(figures.coverage_order(range(120), data, np.random.default_rng(43)))
    assert a == b and a != c and len(set(a)) == 120
    assert len({data["sequence_ids"][i] for i in a[:4]}) == 4
    data["depth_images"][a[:5]] = 0.  # retry invalid draws rather than stopping
    extractor = SobelDepthTerrainFeatureExtractor(output_size=(24, 32), min_depth=.02, max_depth=1.)
    examples, unavailable = figures.geometric_examples(extractor, data, ["flat"], count_per_class=100)
    assert len(examples) == 100 and not unavailable
    assert len({e["row_index"] for e in examples}) == 100
    assert not set(a[:5]) & {e["row_index"] for e in examples}


def test_transition_coverage_disagreement_sampling_and_thumbnails():
    n = 1000
    labels = [i//5 % 3 for i in range(n)]
    ids = [f"seq{i//100}" for i in range(n)]
    data = dict(labels=labels, sequence_ids=ids, source_ids=["capture"]*n,
                original_env_ids=torch.arange(n)//100, frame_indices=torch.arange(n),
                depth_images=torch.arange(n, dtype=torch.float32)[:, None, None].expand(n, 8, 8))
    trace = {method+"_selected": torch.tensor(labels) for method in figures.METHODS}
    trace["bayes_selected"] = (trace["bayes_selected"] + 1) % 3
    runs = [{"trace": trace}]
    selected, report = figures.sample_transitions(data, ids, runs)
    again, _ = figures.sample_transitions(data, ids, runs)
    assert selected == again and len(selected) == 100
    assert len({x["focus_frame"] for x in selected}) == 100
    assert len({x["sequence_id"] for x in selected}) == 10
    assert sum(x["selection_pool"] == "disagreement" for x in selected) == 50
    thumbs = figures.transition_thumbnails(data, selected)
    for selection, thumbnail in zip(selected, thumbs):
        assert all(ids[i] == selection["sequence_id"] for i in range(selection["start"], selection["end"]))
        assert 1 <= len(thumbnail["row_indices"]) <= 8
        for row, image in zip(thumbnail["row_indices"], thumbnail["depth_images"]):
            assert torch.allclose(image, torch.full_like(image, row))
        assert thumbnail["provenance"]["frame_indices"] == thumbnail["row_indices"]


def test_sampled_timeline_png_pdf_with_thumbnails(tmp_path, capsys):
    bundle, _ = sample_bundle()
    data = {"labels": bundle["ordered_truth"], "sequence_ids": bundle["ordered_provenance"]["sequence_ids"],
            "source_ids": bundle["ordered_provenance"]["source_ids"], "frame_indices": torch.arange(12),
            "depth_images": torch.rand(12, 24, 32)}
    selections, _ = figures.sample_transitions(data, data["sequence_ids"], bundle["runs"], count=1)
    bundle["timeline_selections"] = selections
    bundle["timeline_selection"] = selections[0]
    bundle["transition_thumbnails"] = figures.transition_thumbnails(data, selections)
    bundle["examples"] = []
    path = tmp_path / "figure_data.pt"
    figures.save_bundle(bundle, path)
    restored = figures.load_bundle(path)
    report = figures.plot_bundle(restored, tmp_path / "plots")
    assert (tmp_path / "plots/figures/depth_and_trajectories/transition_depth_sample_000.png").is_file()
    assert (tmp_path / "plots/figures/depth_and_trajectories/transition_depth_sample_000.pdf").is_file()
    assert len(list((tmp_path / "plots").glob("figures/depth_and_trajectories/timeline_sample_000_*.png"))) == 18
    assert Path(report["browse_index"]).is_file()
    output = capsys.readouterr().out
    count = len(list((tmp_path / "plots").rglob("*.png")))
    assert f"{count}/{count} complete" in output


def test_existing_checkpoint_mode_never_fits_or_writes_costs(tmp_path, monkeypatch):
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import NeuralClassifierAdapter, FeatureStandardizer
    source, data_dir = tmp_path / "existing", tmp_path / "data"
    source.mkdir(); data_dir.mkdir()
    classes = ["flat", "gap", "stairs"]
    data = dict(labels=classes*4, depth_images=torch.rand(12, 24, 32),
                orientation_rpy=torch.zeros(12, 3), angular_velocity=torch.zeros(12, 3), per_eps=12)
    for split in ("train", "val", "test"):
        torch.save(data, data_dir / f"{split}.pt")
    manifest = dict(class_ordering=classes, structural_dataset=str(data_dir), ordered_dataset=str(data_dir),
                    fixed_ema_configuration=paper.FIXED_EMA_CONFIG, fixed_bayes_configuration=paper.FIXED_BAYES_CONFIG)
    (source / "manifest.json").write_text(json.dumps(manifest))
    cost_path = source / "training_cost.json"
    cost_path.write_text("unchanged")
    feature_dir = source / "artifacts" / "feature_nn"
    feature_dir.mkdir(parents=True)
    SobelDepthTerrainFeatureExtractor(output_size=(24, 32)).save(feature_dir / "extractor.pt")
    standardizer = FeatureStandardizer()
    standardizer.mean, standardizer.std = torch.zeros(4), torch.ones(4)
    standardizer.save(feature_dir / "standardizer.pt")
    for architecture in figures.ARCHITECTURES:
        for seed in (0, 1, 2):
            model = (paper.TerrainDepthFeatureClassifierNN(4, [3], "elu") if architecture == "feature_nn" else
                     paper.TerrainDepthClassifierNN((24, 32), 1, [8, 16], [1, 1], [128, 3], [5, 3], "elu", .2, robot_state_dim=5))
            directory = source / "artifacts" / architecture / f"seed_{seed}"
            directory.mkdir(parents=True)
            NeuralClassifierAdapter(model, classes).save(directory / "classifier.pt")
            torch.save(model.get_args(), directory / "nn_model_args.pt")
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing-checkpoint mode invoked training or cost writer")
    monkeypatch.setattr(NeuralClassifierAdapter, "fit", forbidden)
    monkeypatch.setattr(paper, "write_cost_record", forbidden)
    calls = []
    def fake_collect(*args, **kwargs):
        calls.append(1)
        return torch.arange(36, dtype=torch.float32).reshape(1, 12, 3), .01
    monkeypatch.setattr(paper, "collect_engineered_logits", fake_collect)
    monkeypatch.setattr(paper, "collect_raw_depth_logits", fake_collect)
    monkeypatch.setattr(paper, "_make_plots", lambda *args: None)
    monkeypatch.setattr(figures, "plot_bundle", lambda *args: {"outputs": [], "unavailable": []})
    output = tmp_path / "new_bundle"
    paper.main(["--bundle-from-existing", str(source), "--output", str(output), "--allow-legacy-provenance", "--device", "cpu"])
    assert len(calls) == 12
    assert cost_path.read_text() == "unchanged"
    assert (output / "figure_data.pt").is_file()
    assert len(figures.load_bundle(output / "figure_data.pt")["runs"]) == 6
