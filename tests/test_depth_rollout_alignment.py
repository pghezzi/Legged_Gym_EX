"""Regression checks for action-time depth storage and camera refreshes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from rsl_rl.algorithms.ppo_dreamwaq_depth import PPO_DreamWaQ_Depth
from rsl_rl.storage import RolloutStorageDreamWaQDepth


class _ActorCriticStub:
    is_recurrent = False

    def act(self, obs, obs_history, depth):
        self.action_depth = depth
        self.action_mean = torch.zeros(obs.shape[0], 2)
        self.action_std = torch.ones_like(self.action_mean)
        return torch.ones_like(self.action_mean)

    def evaluate(self, privileged_obs):
        return torch.zeros(privileged_obs.shape[0], 1)

    def get_actions_log_prob(self, actions):
        return torch.zeros(actions.shape[0])

    def reset(self, dones):
        self.reset_dones = dones


def _make_algorithm():
    algorithm = PPO_DreamWaQ_Depth.__new__(PPO_DreamWaQ_Depth)
    algorithm.actor_critic = _ActorCriticStub()
    algorithm.transition = RolloutStorageDreamWaQDepth.Transition()
    algorithm.storage = RolloutStorageDreamWaQDepth(
        2, 2, 1, [3], [4], [5], [1], [2], [2], [1, 2, 2], "cpu"
    )
    algorithm.gamma = 0.998
    algorithm.device = "cpu"
    return algorithm


def test_transition_keeps_action_time_depth_after_environment_mutation():
    algorithm = _make_algorithm()
    depth = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
    expected = depth.clone()
    obs = torch.zeros(2, 3)
    privileged_obs = torch.zeros(2, 4)
    obs_history = torch.zeros(2, 5)
    labels = torch.zeros(2, 1)

    algorithm.act(obs, privileged_obs, obs_history, labels, depth)
    assert algorithm.actor_critic.action_depth.data_ptr() != depth.data_ptr()
    depth.fill_(99.0)  # camera refresh mutates the environment-owned buffer
    depth.zero_()      # an environment reset clears that same buffer
    algorithm.process_env_step(
        torch.ones(2), torch.tensor([True, False]), {}, torch.zeros(2, 2)
    )

    assert torch.equal(algorithm.storage.depth_images[0], expected)


def _load_depth_mixin():
    path = (
        Path(__file__).parents[1]
        / "legged_gym/envs/go2/go2_depth_waq/depth_mixin.py"
    )
    spec = importlib.util.spec_from_file_location("depth_mixin_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DepthMixin


DepthMixin = _load_depth_mixin()


class _TestDepthMixin(DepthMixin):
    def _process_depth_images(self, depth):
        return depth.clone()


def _make_depth_env(num_history=2):
    camera = SimpleNamespace(
        resized_resolution=[1, 1],
        num_history=num_history,
        latency_range=[0.14, 0.14],
        refresh_duration=0.1,
        sky_artifacts_values=[0.6],
        stereo_full_block_values=[0.0],
    )
    env = _TestDepthMixin()
    env.cfg = SimpleNamespace(
        sensor=SimpleNamespace(add_depth=True, depth_camera_config=camera)
    )
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.dt = 0.02
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.simulator = SimpleNamespace(depth_images=torch.zeros(2, 1, 1, 1))
    env._init_depth_processing()
    env.depth_sensor_latency.fill_(0.14)
    return env


def _step_depth(env, step, first_value=None):
    env.episode_length_buf.fill_(step)
    values = torch.tensor(
        [step if first_value is None else first_value, 100 + step],
        dtype=torch.float32,
    )
    env.simulator.depth_images[:, 0, 0, 0] = values
    env._pre_depth_step()
    env._update_depth_observations()


def test_camera_holds_selected_frame_between_refreshes_while_age_advances():
    env = _make_depth_env()
    for step in range(10):
        _step_depth(env, step)

    selected_values = []
    ages_ms = []
    for step in range(10, 15):
        _step_depth(env, step)
        selected_values.append(env.depth_sensor_output[0, 0, 0, 0].item())
        ages_ms.append(
            round(env.depth_sensor_delayed_frames[0].item() * env.dt * 1000)
        )

    assert selected_values == [3.0] * 5
    assert ages_ms == [140, 160, 180, 200, 220]


def test_reset_clears_one_environment_depth_history_only():
    env = _make_depth_env()
    for step in range(11):
        _step_depth(env, step)

    other_buffer = env.depth_sensor_obs_buffer[:, 1].clone()
    other_output = env.depth_sensor_output[1].clone()
    env._reset_depth_buffers(torch.tensor([0]))

    assert torch.count_nonzero(env.depth_sensor_obs_buffer[:, 0]) == 0
    assert torch.count_nonzero(env.depth_sensor_output[0]) == 0
    assert torch.equal(env.depth_sensor_obs_buffer[:, 1], other_buffer)
    assert torch.equal(env.depth_sensor_output[1], other_output)

    env.episode_length_buf[:] = torch.tensor([0, 11])
    env.simulator.depth_images[:, 0, 0, 0] = torch.tensor([999.0, 111.0])
    env._pre_depth_step()
    env._update_depth_observations()
    assert torch.count_nonzero(env.depth_sensor_output[0]) == 0
    assert torch.equal(env.depth_sensor_output[1], other_output)
