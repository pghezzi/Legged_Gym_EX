"""Leakage and reproducibility checks for the depth-dataset compiler."""

import torch

from legged_gym.scripts.depth_data_pipeline.compile_depth_data import train_val_test_data
from legged_gym.utils.dataset_provenance import CaptureProvenance


def _write_raw(path, *, frames=4, envs=20, terrain_seeds=None):
    tracker = CaptureProvenance(envs, "cpu")
    for frame in range(frames):
        tracker.capture(frame + 1)
    env_id = torch.arange(envs).view(1, envs, 1).expand(frames, envs, 1).clone()
    data = {
        "depth_images": env_id.view(frames, envs, 1, 1),
        "base_rpy": env_id.expand(frames, envs, 3).float(),
        "base_ang_vel": env_id.expand(frames, envs, 3).float(),
        "terrain_name": [[f"env_{env}" for env in range(envs)] for _ in range(frames)],
        "provenance": tracker.export(),
    }
    if terrain_seeds is not None:
        data["terrain_seed"] = torch.as_tensor(terrain_seeds)
    torch.save(data, path)


def _assert_partition(data, manifest, frames):
    group_sets = [set(manifest["splits"][name]["group_ids"])
                  for name in ("train", "val", "test")]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    split_envs = []
    for split, name in zip(data, ("train", "val", "test")):
        env_ids = set(manifest["splits"][name]["source_env_ids"])
        observed = set(int(value) for value in split["depth_images"][:, 0, 0].tolist())
        assert observed == env_ids
        assert len(split["labels"]) == split["depth_images"].shape[0]
        assert len(split["labels"]) == split["orientation_rpy"].shape[0]
        assert len(split["labels"]) == split["angular_velocity"].shape[0]
        assert len(split["labels"]) == frames * len(env_ids)
        split_envs.append(env_ids)
    assert split_envs[0].isdisjoint(split_envs[1])
    assert split_envs[0].isdisjoint(split_envs[2])
    assert split_envs[1].isdisjoint(split_envs[2])


def test_environment_split_after_fraction_is_disjoint_and_reproducible(tmp_path):
    raw = tmp_path / "raw.pt"
    _write_raw(raw)
    first = train_val_test_data(raw, frac=0.5, seed=42, return_manifest=True)
    repeat = train_val_test_data(raw, frac=0.5, seed=42, return_manifest=True)
    changed = train_val_test_data(raw, frac=0.5, seed=43, return_manifest=True)
    _assert_partition(first[:3], first[3], frames=4)
    assert first[3]["selected_source_env_ids"] == repeat[3]["selected_source_env_ids"]
    assert first[3]["splits"] == repeat[3]["splits"]
    assert (first[3]["selected_source_env_ids"], first[3]["splits"]) != (
        changed[3]["selected_source_env_ids"], changed[3]["splits"])
    full_first = train_val_test_data(raw, frac=1.0, seed=42, return_manifest=True)
    full_changed = train_val_test_data(raw, frac=1.0, seed=43, return_manifest=True)
    assert full_first[3]["selected_source_env_ids"] == full_changed[3]["selected_source_env_ids"]
    assert full_first[3]["splits"] != full_changed[3]["splits"]


def test_terrain_seed_groups_never_cross_splits(tmp_path):
    raw = tmp_path / "seeded_raw.pt"
    terrain_seeds = [seed for seed in range(10) for _ in range(2)]
    _write_raw(raw, terrain_seeds=terrain_seeds)
    compiled = train_val_test_data(raw, frac=1.0, seed=42, return_manifest=True)
    manifest = compiled[3]
    _assert_partition(compiled[:3], manifest, frames=4)
    assert manifest["group_type"] == "terrain_seed"
    seed_sets = [set(manifest["splits"][name]["terrain_seed_ids"])
                 for name in ("train", "val", "test")]
    assert seed_sets[0].isdisjoint(seed_sets[1])
    assert seed_sets[0].isdisjoint(seed_sets[2])
    assert seed_sets[1].isdisjoint(seed_sets[2])
    for name in ("train", "val", "test"):
        env_ids = manifest["splits"][name]["source_env_ids"]
        assigned_seeds = {terrain_seeds[env_id] for env_id in env_ids}
        assert assigned_seeds == seed_sets[
            ("train", "val", "test").index(name)]
