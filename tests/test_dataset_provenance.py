"""Synthetic capture/compiler/replay checks; no simulation or training."""
import ast
from pathlib import Path

import pytest
import torch

from legged_gym.utils.dataset_provenance import (
    CaptureProvenance, validate_dataset_provenance, validate_partitions,
)
from legged_gym.scripts.depth_data_pipeline.compile_depth_data import compile_sources, get_data_raw
from legged_gym.scripts.depth_data_pipeline.util_func import sequence_ids_for
from legged_gym.scripts.depth_data_pipeline.sequential_terrain_filter_extensions import (
    EMALogitPatienceFilter, run_ema_logit_patience_sequences,
)
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    BayesianTerrainFilter, make_persistent_transition_matrix, run_filter_sequences,
)


def raw_file(path, frames=3, capture="capture-A", episode=0, offset=0, terrain=None):
    tracker = CaptureProvenance(10, "cpu")
    tracker.capture_id = capture
    tracker.episodes[:] = episode
    for i in range(frames):
        tracker.capture(offset + i*5 + 1)
    p = tracker.export()
    p["frame_indices"] += offset
    raw = dict(depth_images=torch.zeros(frames, 10, 2, 2), base_rpy=torch.zeros(frames, 10, 3),
               base_ang_vel=torch.zeros(frames, 10, 3), terrain_name=[["flat"]*10 for _ in range(frames)],
               provenance=p)
    if terrain is not None:
        raw["terrain_seed_ids"] = terrain
    torch.save(raw, path)
    return raw


def test_between_capture_resets_and_snapshot():
    tracker = CaptureProvenance(2, "cpu")
    tracker.capture(1)
    tracker.reset(torch.tensor([0]))  # control tick 2: no camera observation
    tracker.reset(torch.tensor([0]))  # control tick 4: another short episode
    tracker.capture(6)
    tracker.reset(torch.tensor([1]))  # manual reset AFTER capture belongs to future observation
    tracker.capture(11)
    p = tracker.export()
    assert p["episode_ids"].tolist() == [[0, 0], [2, 0], [2, 1]]
    assert p["control_step_indices"][:, 0].tolist() == [1, 6, 11]
    assert p["frame_indices"][:, 0].tolist() == [0, 1, 2]
    assert p["original_env_ids"].tolist() == [[0, 1]]*3


def test_merge_unequal_lengths_and_all_episodes_grouped(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    raw_file(a, frames=3)
    raw_file(b, frames=5, episode=1, offset=30)
    datasets, manifests, checks = compile_sources([a, b], frac=1.)
    assert checks["environment_disjoint_verified"]
    assert not checks["terrain_disjoint_checked"]
    groups = []
    for data in datasets:
        assert data["per_eps"] == 0  # no borrowed length from first file
        validate_dataset_provenance(data)
        ids = sequence_ids_for(data)
        for env in set(data["original_env_ids"].tolist()):
            mask = data["original_env_ids"] == env
            assert mask.sum() == 8
            assert set(data["episode_ids"][mask].tolist()) == {0, 1}
            assert len(set(ids[mask].tolist())) == 2
        groups.append(set(data["source_environment_group_ids"]))
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])
    again, _, _ = compile_sources([a, b], frac=1.)
    assert [d["sequence_ids"] for d in again] == [d["sequence_ids"] for d in datasets]
    changed, _, _ = compile_sources([a, b], frac=1., seed=13)
    assert changed[0]["source_environment_group_ids"] != datasets[0]["source_environment_group_ids"]


def test_filter_truncation_and_original_environment_subsampling(tmp_path):
    # Use the collector's exact filtering operation for all modalities/metadata.
    path = Path(__file__).parents[1] / "legged_gym/scripts/play_exp_DO_NOT_TOUCH.py"
    node = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == "reset_split")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    raw = raw_file(tmp_path / "raw.pt", frames=8)
    p = raw["provenance"]
    p["episode_ids"][4:] = 1
    resets = torch.zeros(8, 10, dtype=torch.bool)
    resets[4] = True
    for key in ("depth_images", "base_rpy", "base_ang_vel"):
        raw[key] = ns["reset_split"](raw[key], resets)
    for key in ("original_env_ids", "episode_ids", "control_step_indices", "frame_indices"):
        p[key] = ns["reset_split"](p[key], resets)
    raw["terrain_name"] = [["flat"]*20 for _ in range(4)]
    torch.save(raw, tmp_path / "filtered.pt")
    *_, source = get_data_raw(tmp_path / "filtered.pt", frac=.5, return_metadata=True)
    assert len(source["selected_env_ids"]) == 10  # 5 original envs, 2 episodes each
    assert len(set(source["provenance"]["original_env_ids"][0].tolist())) == 5
    datasets, _, _ = compile_sources([tmp_path / "filtered.pt"], frac=1.)
    for data in datasets:
        for env in set(data["original_env_ids"].tolist()):
            assert (data["original_env_ids"] == env).sum() == 8


def test_terrain_realization_grouping_across_files(tmp_path):
    raw_file(tmp_path / "a.pt", capture="a", terrain=torch.arange(10))
    raw_file(tmp_path / "b.pt", capture="b", terrain=torch.arange(10))
    datasets, _, checks = compile_sources([tmp_path / "a.pt", tmp_path / "b.pt"], frac=1.)
    assert checks["terrain_disjoint_checked"]
    for data in datasets:
        for env in set(data["original_env_ids"].tolist()):
            assert (data["original_env_ids"] == env).sum() == 6


def test_explicit_legacy_and_leakage_errors(tmp_path):
    raw = raw_file(tmp_path / "raw.pt")
    del raw["provenance"]
    torch.save(raw, tmp_path / "raw.pt")
    with pytest.raises(ValueError, match="legacy"):
        compile_sources([tmp_path / "raw.pt"], frac=1.)
    with pytest.warns(UserWarning, match="Legacy"):
        datasets, _, checks = compile_sources([tmp_path / "raw.pt"], frac=1., allow_legacy=True)
    assert not checks["environment_disjoint_verified"]
    with pytest.raises(ValueError, match="unverified"):
        sequence_ids_for(datasets[0])
    assert len(sequence_ids_for(datasets[0], allow_legacy=True)) == len(datasets[0]["labels"])
    with pytest.raises(ValueError, match="leakage"):
        validate_partitions({"train": datasets[0], "test": datasets[0]}, allow_legacy=True)


def test_episode_reset_in_ema_and_bayes(tmp_path):
    raw_file(tmp_path / "a.pt", frames=3)
    raw_file(tmp_path / "b.pt", frames=5, episode=1, offset=30)
    datasets, _, _ = compile_sources([tmp_path / "a.pt", tmp_path / "b.pt"], frac=1.)
    data = datasets[0]
    ids = sequence_ids_for(data)
    scores = torch.randn(len(ids), 2, generator=torch.Generator().manual_seed(4))
    def ema(x, seq):
        return run_ema_logit_patience_sequences(EMALogitPatienceFilter([0, 1], ema_alpha=.6, change_patience=1), x, sequence_ids=seq)
    def bayes(x, seq):
        filt = BayesianTerrainFilter([0, 1], torch.ones(2)/2, make_persistent_transition_matrix([0, 1], .9), torch.eye(2))
        return run_filter_sequences(filt, x.softmax(-1), sequence_ids=seq)[0]
    for run in (ema, bayes):
        together = run(scores, ids.tolist())
        separate = []
        for seq in torch.unique_consecutive(ids):
            subset = scores[ids == seq]
            separate.extend(run(subset, [0]*len(subset)))
        assert together == separate


def test_malformed_chronology_and_duplicate_sources(tmp_path):
    raw_file(tmp_path / "a.pt")
    datasets, _, _ = compile_sources([tmp_path / "a.pt"], frac=1.)
    bad = dict(datasets[0], control_step_indices=datasets[0]["control_step_indices"].clone())
    bad["control_step_indices"][1] = 0
    with pytest.raises(ValueError, match="chronological"):
        validate_dataset_provenance(bad)
    with pytest.raises(ValueError):
        compile_sources([tmp_path / "a.pt", tmp_path / "a.pt"], frac=1.)
