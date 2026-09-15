"""Test swing-only clearance rewards without loading a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("patch_shape", [(7, 4, 9), (7, 4, 3, 3)])
def test_clearance_requires_moving_noncontact_feet(patch_shape):
    path = Path(__file__).parents[1] / "legged_gym/envs/go2/go2_depth_waq/go2_depth_waq.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "Go2DepthWaq")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "_reward_foot_clearance_terrain_aware"]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), "exec"), namespace)
    env = namespace["Go2DepthWaq"]()
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(
        foot_clearance_target=.08, foot_height_offset=.022,
        foot_clearance_tracking_sigma=.01, foot_clearance_min_swing_speed=.05))
    env.feet_max_force_z = torch.full((7, 4), 10.)
    env.simulator = SimpleNamespace(feet_pos=torch.zeros(7, 4, 3),
                                    feet_vel=torch.zeros(7, 4, 3),
                                    _height_around_feet=torch.full(patch_shape, .3))
    # 0: stationary stance; 1: sliding contact; 2: foot held motionless aloft;
    # 3: airborne jitter; 4: good moving swing; 5: low swing; 6: excessive height.
    env.simulator.feet_vel[1, :, 0] = 1.
    env.feet_max_force_z[2:, 0] = 0.
    env.simulator.feet_pos[2:, 0, 2] = .402
    env.simulator.feet_vel[3, 0, 0] = .04
    env.simulator.feet_vel[4:, 0, 0] = 1.
    env.simulator.feet_pos[5, 0, 2] -= .2
    env.simulator.feet_pos[6, 0, 2] += .2
    # Moving contact feet with poor clearance must not dilute the good swing.
    env.simulator.feet_vel[4:, 1:, 0] = 1.
    reward = env._reward_foot_clearance_terrain_aware()
    torch.testing.assert_close(reward[:4], torch.zeros(4))
    assert reward[4].item() == pytest.approx(1.)
    assert 0. < reward[5] < reward[4]
    assert 0. < reward[6] < reward[5]  # extra over-swing penalty is retained
    assert torch.isfinite(reward).all()
