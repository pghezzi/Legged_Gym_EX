"""Relocation is lossless, collision-safe, and independent of datasets/models."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("figure_paths", Path(__file__).resolve().parents[1] /
    "legged_gym/scripts/depth_data_pipeline/paper_figure_paths.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


def test_relocation_and_manifest_gallery_links(tmp_path):
    names = ["experiment_1_nll.png", "experiment_1_nll.pdf", "confusion_structural_0.png",
             "depth_geometry_0.png", "timeline_sample_000_feature_nn_seed_0_bayes.pdf"]
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "figure_data.pt").write_bytes(b"untouched bundle")
    (tmp_path / "metrics.csv").write_bytes(b"untouched metrics")
    report = {"outputs": ["/paper/offline/"+name for name in names],
              "browse_index": "/paper/offline/figure_index.html",
              "depth_examples": [{"figure_stem": "depth_geometry_0"}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"figures": report}))
    original = manifest.read_bytes()
    result = layout.relocate_existing_figures(tmp_path)
    assert result["moved_files"] == len(names)
    for name in names:
        destination = layout.figure_path(tmp_path, name)
        assert destination.read_bytes() == name.encode()
        assert not (tmp_path / name).exists()
    saved = json.loads(manifest.read_text())["figures"]
    assert all(Path(path).exists() for path in saved["outputs"])
    assert saved["depth_examples"][0]["figure_stem"] == "figures/depth_and_trajectories/depth_geometry_0"
    assert manifest.with_name("manifest.json.before_figure_layout.bak").read_bytes() == original
    gallery = (tmp_path / "figure_index.html").read_text()
    assert "figures/results/experiment_1_nll.png" in gallery
    assert "figures/depth_and_trajectories/depth_geometry_0.png" in gallery
    assert (tmp_path / "figure_data.pt").read_bytes() == b"untouched bundle"
    assert (tmp_path / "metrics.csv").read_bytes() == b"untouched metrics"
    assert layout.relocate_existing_figures(tmp_path)["moved_files"] == 0


def test_no_overwrite_on_collision(tmp_path):
    source = tmp_path / "depth_geometry_0.png"
    source.write_bytes(b"old")
    destination = layout.figure_path(tmp_path, source.name)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        layout.relocate_existing_figures(tmp_path)
    assert source.read_bytes() == b"old"
    assert destination.read_bytes() == b"existing"
