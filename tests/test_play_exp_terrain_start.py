"""Check play terrain placement without importing the simulator stack."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def helpers():
    path = Path(__file__).parents[1] / "legged_gym/scripts/play_exp.py"
    tree = ast.parse(path.read_text())
    names = {"configure_start_terrain_level", "reset_start_terrain_level"}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("curriculum", [True, False])
def test_exact_row_and_origins_are_set_before_reset(curriculum):
    reset = helpers()["reset_start_terrain_level"]
    terrain = SimpleNamespace(mesh_type="trimesh", num_rows=5, curriculum=curriculum)
    origins = torch.arange(30).reshape(5, 2, 3).float()
    sim = SimpleNamespace(terrain_levels=torch.tensor([0, 1, 2]),
                          terrain_types=torch.tensor([0, 1, 0]),
                          env_origins=torch.zeros(3, 3), _terrain_origins=origins)
    env = SimpleNamespace(cfg=SimpleNamespace(terrain=terrain), simulator=sim)
    calls = []
    def reset_env():
        assert not terrain.curriculum
        assert (sim.terrain_levels == 4).all()
        torch.testing.assert_close(sim.env_origins, origins[4, [0, 1, 0]])
        calls.append(True)
    env.reset = reset_env
    reset(env, 4)
    assert calls == [True]
    assert terrain.curriculum == curriculum


@pytest.mark.parametrize("level", [-1, 5, 9])
def test_rejects_out_of_range_rows(level):
    terrain = SimpleNamespace(mesh_type="trimesh", num_rows=5)
    with pytest.raises(ValueError, match="between 0 and 4"):
        helpers()["configure_start_terrain_level"](terrain, level)


def test_plane_does_not_access_missing_terrain_tensors():
    terrain = SimpleNamespace(mesh_type="plane", curriculum=False)
    env = SimpleNamespace(cfg=SimpleNamespace(terrain=terrain), reset=lambda: None)
    reset = helpers()["reset_start_terrain_level"]
    reset(env, 0)
    with pytest.raises(ValueError, match="heightfield or trimesh"):
        reset(env, 1)


def test_reset_failure_restores_curriculum():
    terrain = SimpleNamespace(mesh_type="trimesh", num_rows=1, curriculum=True)
    sim = SimpleNamespace(terrain_levels=torch.zeros(1, dtype=torch.long),
                          terrain_types=torch.zeros(1, dtype=torch.long),
                          env_origins=torch.zeros(1, 3), _terrain_origins=torch.zeros(1, 1, 3))
    def fail_reset():
        raise RuntimeError("reset failed")
    env = SimpleNamespace(cfg=SimpleNamespace(terrain=terrain), simulator=sim, reset=fail_reset)
    with pytest.raises(RuntimeError, match="reset failed"):
        helpers()["reset_start_terrain_level"](env, 0)
    assert terrain.curriculum
