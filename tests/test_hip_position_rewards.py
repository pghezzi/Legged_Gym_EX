"""Check configured-order hip groups without loading a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_front_and_rear_hip_rewards():
    path = Path(__file__).parents[1] / "legged_gym/envs/go2/go2_depth_waq/go2_depth_waq.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2DepthWaq")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and
                n.name in ("_reward_hip_pos", "_reward_front_hip_pos", "_reward_rear_hip_pos")]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    env = namespace["Go2DepthWaq"]()
    default = torch.full((1, 12), .2)
    positions = default.repeat(3, 1)
    positions[0, [0, 3]] += torch.tensor([1., 2.])
    positions[1, [6, 9]] += torch.tensor([3., 4.])
    positions[2, [1, 2, 4, 5, 7, 8, 10, 11]] += 10.
    env.simulator = SimpleNamespace(dof_pos=positions, default_dof_pos=default)
    front, rear = env._reward_front_hip_pos(), env._reward_rear_hip_pos()
    torch.testing.assert_close(front, torch.tensor([5., 0., 0.]))
    torch.testing.assert_close(rear, torch.tensor([0., 25., 0.]))
    torch.testing.assert_close(front + rear, env._reward_hip_pos())
