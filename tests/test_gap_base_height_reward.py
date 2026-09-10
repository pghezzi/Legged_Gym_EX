"""Exercise actual reward methods without importing a simulator."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def reward_class():
    root = Path(__file__).parents[1]
    namespace = {"torch": torch, "Reward": torch.Tensor}
    for path, name, parent in (
        ("legged_gym/envs/base/legged_robot.py", "LeggedRobot", "object"),
        ("legged_gym/envs/go2/go2_depth_waq/go2_depth_waq.py", "Go2DepthWaq", "LeggedRobot"),
    ):
        tree = ast.parse((root / path).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        node.bases = [ast.Name(id=parent, ctx=ast.Load())]
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "_reward_base_height"]
        tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(tree, path, "exec"), namespace)
    return namespace["Go2DepthWaq"]


def environment(enabled=True, gap_fraction=1.):
    env = reward_class()()
    env.cfg = SimpleNamespace(
        rewards=SimpleNamespace(base_height_target=.38, base_height_relative_to_gap_landing=enabled),
        terrain=SimpleNamespace(terrain_proportions=[1-gap_fraction, 0, 0, 0, 0, 0, gap_fraction]),
    )
    env.simulator = SimpleNamespace(
        base_pos=torch.tensor([[0., 0., .38], [0., 0., 2.58]]),
        env_origins=torch.tensor([[0., 0., 0.], [0., 0., 2.]]),
        measured_heights=torch.tensor([[0., -5., -5.], [2., -3., -3.]]),
    )
    return env


def test_gap_reward_uses_each_landing_height_not_hole_samples():
    env = environment()
    expected = torch.tensor([0., .2**2])
    torch.testing.assert_close(env._reward_base_height(), expected)
    env.simulator.measured_heights.fill_(-100.)
    torch.testing.assert_close(env._reward_base_height(), expected)


def test_gap_reference_tracks_environment_origin_changes():
    env = environment()
    before = env._reward_base_height()
    # Simulate changing the first environment's terrain elevation on reset.
    env.simulator.env_origins[0, 2] += 1.
    env.simulator.base_pos[0, 2] += 1.
    torch.testing.assert_close(env._reward_base_height(), before)


def test_other_terrains_and_legacy_configs_keep_original_reward():
    for enabled, fraction in ((False, 1.), (True, .5), (True, 0.)):
        env = environment(enabled, fraction)
        expected = ((env.simulator.base_pos[:, 2, None] - env.simulator.measured_heights).mean(1) - .38).square()
        torch.testing.assert_close(env._reward_base_height(), expected)
    del env.cfg.rewards.base_height_relative_to_gap_landing
    torch.testing.assert_close(env._reward_base_height(), expected)
