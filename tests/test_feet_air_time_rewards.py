"""Exercise airtime transitions and reward clipping without a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]


def load_subject():
    namespace = {"torch": torch}
    base_path = ROOT / "legged_gym/envs/base/legged_robot.py"
    base_tree = ast.parse(base_path.read_text())
    reward = next(n for n in ast.walk(base_tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "compute_reward")
    base = ast.ClassDef(name="Base", bases=[], keywords=[], body=[reward], decorator_list=[])
    path = ROOT / "legged_gym/envs/go2/go2_depth_waq/go2_depth_waq.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
               and n.name == "Go2DepthWaq")
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    methods = {"compute_reward", "_update_feet_air_time", "_reward_feet_air_time",
               "_reward_feet_prolonged_air_time", "reset_idx"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[base, cls], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace["Go2DepthWaq"], namespace["Base"]


def make_env(dt=.02, touchdown_reward=True):
    subject, base = load_subject()
    env = subject()
    env.dt = dt
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(
        feet_air_time_target=.25, feet_air_time_max=.5,
        feet_prolonged_air_time_penalty_rate=1., only_positive_rewards=True))
    env.feet_air_time = torch.zeros(2, 4)
    env.feet_touchdown_air_time = torch.zeros(2, 4)
    env.last_contacts = torch.zeros(2, 4, dtype=torch.int)
    env.feet_max_force_z = torch.full((2, 4), 10.)
    env.commands = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    env.obstacle_progress = None
    env.rew_buf = torch.zeros(2)
    env.reward_names = ["feet_air_time"] if touchdown_reward else []
    env.reward_functions = [env._reward_feet_air_time] if touchdown_reward else []
    env.reward_scales = {"feet_air_time": .6 * dt} if touchdown_reward else {}
    env.episode_sums = {key: torch.zeros(2) for key in
                        ("feet_air_time", "feet_prolonged_air_time")}
    return env, base


def swing(env, duration, feet=(0,)):
    env.feet_max_force_z[:, list(feet)] = 0.
    for _ in range(round(duration / env.dt)):
        env.compute_reward()


def test_normal_swing_rewards_only_once_on_touchdown():
    env, _ = make_env()
    swing(env, .4)
    assert not env.episode_sums["feet_air_time"].any()
    assert not env.episode_sums["feet_prolonged_air_time"].any()
    env.feet_max_force_z.fill_(10.)
    env.compute_reward()
    torch.testing.assert_close(env.rew_buf, torch.tensor([.15 * .6 * env.dt, 0.]))
    env.compute_reward()
    assert not env.rew_buf.any()


@pytest.mark.parametrize("dt", [.01, .02, .04])
@pytest.mark.parametrize("touchdown_reward", [True, False])
def test_held_feet_cost_per_second_even_at_zero_command_and_after_clipping(dt, touchdown_reward):
    env, _ = make_env(dt, touchdown_reward)
    swing(env, 2., feet=(0, 2))
    # Two feet, 1.5 seconds each after the grace period, at 1 reward/s.
    torch.testing.assert_close(env.episode_sums["feet_prolonged_air_time"],
                               torch.full((2,), -3.))
    torch.testing.assert_close(env.rew_buf, torch.full((2,), -2. * dt))
    env.feet_max_force_z.fill_(10.)
    env.compute_reward()
    assert not env.rew_buf.any()  # No banked bonus on eventual touchdown.
    assert not env.feet_air_time.any()


def test_limit_touchdown_is_bounded_and_short_steps_keep_negative_term():
    env, _ = make_env()
    swing(env, .5)
    env.feet_max_force_z.fill_(10.)
    env.compute_reward()
    assert env._reward_feet_air_time()[0].item() == pytest.approx(.25)
    env, _ = make_env()
    swing(env, .1)
    env.feet_max_force_z.fill_(10.)
    env.compute_reward()
    assert env._reward_feet_air_time()[0].item() == pytest.approx(-.15)


def test_one_frame_contact_dropout_does_not_create_swing():
    env, _ = make_env()
    env.compute_reward()
    env.feet_max_force_z[:, 0] = 0.
    env.compute_reward()
    env.feet_max_force_z.fill_(10.)
    env.compute_reward()
    assert not env.feet_air_time.any()
    assert not env.feet_touchdown_air_time.any()
    assert not env.episode_sums["feet_air_time"].any()


def test_disabling_penalty_still_forfeits_prolonged_touchdown_bonus():
    env, _ = make_env()
    env.cfg.rewards.feet_prolonged_air_time_penalty_rate = 0.
    swing(env, 2.)
    env.feet_max_force_z.fill_(10.)
    env.compute_reward()
    assert not env.rew_buf.any()
    assert not env.episode_sums["feet_prolonged_air_time"].any()


def test_partial_reset_clears_airtime_and_contact_history():
    env, base = make_env()
    # Stub only simulator/depth reset work, preserving the parent's timer reset.
    base.reset_idx = lambda self, ids: self.feet_air_time.__setitem__(ids, 0.)
    base._reset_depth_buffers = lambda self, ids: None
    env.gap_fall_counter = torch.zeros(2)
    env.pit_depth_eval = None
    swing(env, 1.)
    env.last_contacts[:, 1] = 1
    env.feet_touchdown_air_time[:, 1] = .4
    env.reset_idx(torch.tensor([0]))
    assert not env.feet_air_time[0].any()
    assert not env.last_contacts[0].any()
    assert not env.feet_touchdown_air_time[0].any()
    assert env.feet_air_time[1, 0] > .9
    assert env.last_contacts[1, 1] == 1
    assert env.feet_touchdown_air_time[1, 1] == pytest.approx(.4)
