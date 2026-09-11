"""CPU regression checks against actual simplified terrain geometry; no training."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_parkour_auxiliary import geometry, make_terrain

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reward_test.obstacle_progress", ROOT / "legged_gym/utils/obstacle_progress.py")
module = importlib.util.module_from_spec(spec)
with patch.dict("sys.modules", {"reward_test.parkour_auxiliary_targets": geometry}):
    spec.loader.exec_module(module)


def config():
    return SimpleNamespace(base_clearance=.35, foot_clearance=.05, support_duration_s=.1,
                           min_support_feet=2, activation_distance=1., support_height_tolerance=.06,
                           contact_force_threshold=10., foot_height_offset=.022,
                           stationary_radius=1., stationary_speed=.05,
                           stairs=[.1, .2, .3], gap=[.1, .2, .3], pit=[.1, .2, .3])


def candidate(n=2):
    return dict(identity=torch.zeros(n, dtype=torch.long), valid=torch.ones(n, dtype=torch.bool),
                direction=torch.tensor([[1., 0.]]).repeat(n, 1), entry=torch.zeros(n, 2),
                destination=torch.tensor([[2., 0.]]).repeat(n, 1),
                lateral_bounds=torch.tensor([[-1., 1.]]).repeat(n, 1), height=torch.zeros(n, 1),
                difficulty=torch.full((n, 1), .5), coefficients=torch.tensor([[.1, .2, .3]]).repeat(n, 1))


def transition(state, c, old, new, support=True, failed=False, dt=.02):
    n = len(state.rows)
    before = torch.tensor([[old, 0.]]).repeat(n, 1)
    after = torch.tensor([[new, 0.]]).repeat(n, 1)
    feet = torch.zeros(n, 4, 3)
    feet[..., 0] = new
    state.begin(c, before)
    return state.advance(after, feet, torch.full((n, 4), support), torch.full((n,), failed), dt)


def test_wait_retreat_retrace_and_fixed_attempt():
    state = module.ObstacleProgressState(2, 4, config(), "cpu")
    c = candidate()
    for before, after, expected in [(0., 0., 0.), (0., .5, .05), (.5, .2, 0.), (.2, .5, 0.), (.5, .8, .03)]:
        progress, bonus = transition(state, c, before, after)
        torch.testing.assert_close(progress, torch.full((2,), expected))
        assert not bonus.any()
    c["destination"] += 10
    c["difficulty"][:] = 1.
    transition(state, c, .8, .8)
    assert state.fixed["destination"][0, 0] == 2
    assert state.fixed["difficulty"][0, 0] == .5


@pytest.mark.parametrize("distance", [.2, 2., 8.])
@pytest.mark.parametrize("dt", [.01, .02])
def test_progress_budget_is_distance_times_weight(distance, dt):
    cfg = config()
    state = module.ObstacleProgressState(2, 4, cfg, "cpu")
    c = candidate()
    c["destination"][:, 0] = distance
    c["coefficients"][:, 0] = 1.
    for a, b in [(0., .25), (.25, .1), (.1, .25), (.25, .5), (.5, 1.), (1., 1.2)]:
        transition(state, c, a*distance, b*distance, support=False, dt=dt)
    torch.testing.assert_close(state.totals["progress_reward"], torch.full((2,), distance))
    # Retreat cannot change the fixed target or earn additional progress.
    c["destination"][:, 0] = distance * 2
    transition(state, c, distance, 0., support=False, dt=dt)
    torch.testing.assert_close(state.fixed["destination"][:, 0], torch.full((2,), distance))
    assert not transition(state, c, 0., distance, support=False, dt=dt)[0].any()
    state.reset(torch.tensor([0]))
    assert state.totals["progress_reward"][0] == 0 and state.totals["progress_reward"][1] > 0


def test_zero_initial_distance_is_finite():
    cfg = config()
    state = module.ObstacleProgressState(2, 4, cfg, "cpu")
    progress, _ = transition(state, candidate(), 2., 2., support=False)
    assert torch.isfinite(progress).all() and not progress.any()


def test_support_failure_once_and_reset_isolation():
    state = module.ObstacleProgressState(2, 4, config(), "cpu")
    c = candidate()
    assert not transition(state, c, 0., 2., support=False)[1].any()
    for _ in range(8):
        assert not transition(state, c, 2., 2., failed=True)[1].any()
    for _ in range(4):
        assert not transition(state, c, 2., 2.)[1].any()
    torch.testing.assert_close(transition(state, c, 2., 2.)[1], torch.full((2,), .35))
    assert not transition(state, c, 0., 2.)[0].any()  # completed ID cannot rearm
    state.reset(torch.tensor([0]))
    p, _ = transition(state, c, 0., .5)
    torch.testing.assert_close(p, torch.tensor([.05, 0.]))
    assert state.completed[1, 0] and not state.completed[0, 0]


def test_all_feet_clearance_and_corridor():
    state = module.ObstacleProgressState(2, 4, config(), "cpu")
    c = candidate()
    state.begin(c, torch.zeros(2, 2))
    feet = torch.zeros(2, 4, 3)
    feet[..., 0] = 2.
    feet[:, 3, 0] = 1.6  # rear foot hasn't cleared edge 1.65
    _, b = state.advance(torch.tensor([[2., 0.], [2., 2.]]), feet, torch.ones(2, 4, dtype=torch.bool),
                         torch.zeros(2, dtype=torch.bool), 1.)
    assert not b.any()
    assert not state.valid_path[1]


@pytest.mark.parametrize("dt", [.01, .02, .05])
def test_displacement_and_event_not_dt_scaled(dt):
    state = module.ObstacleProgressState(2, 4, config(), "cpu")
    c = candidate()
    steps = round(1. / dt)
    for i in range(steps):
        transition(state, c, 2*i/steps, 2*(i+1)/steps, dt=dt)
    for _ in range(round(.1/dt)):
        transition(state, c, 2., 2., dt=dt)
    torch.testing.assert_close(state.totals["progress_reward"], torch.full((2,), .2))
    torch.testing.assert_close(state.totals["completion_reward"], torch.full((2,), .35))
    assert (state.totals["completions"] == 1).all()


def geometry_reward(label):
    terrain, _ = make_terrain(label)
    terrain.env_length = terrain.env_width = 8.
    proportions = [0.] * 11
    proportions[label] = 1.
    env = SimpleNamespace(device="cpu", num_envs=1, dt=.02,
                          cfg=SimpleNamespace(terrain=SimpleNamespace(terrain_proportions=proportions)),
                          simulator=SimpleNamespace(_gym=True, _terrain=terrain))
    return module.ObstacleProgress(env, config())


@pytest.mark.parametrize("label", [2, 3])
@pytest.mark.parametrize("heading", [-1., 1.])
def test_stairs_whole_flight_in_both_directions(label, heading):
    reward = geometry_reward(label)
    start = .1 if heading == 1 else 3.95
    c = reward.candidates(torch.tensor([[start, 3.95]]), torch.tensor([[heading, 0.]]), torch.tensor([0]))
    assert c["valid"].item() or heading == -1  # center may be outside activation radius
    if heading == -1:
        # Approach the central landing's first descending/ascending riser.
        start = 2.25
        c = reward.candidates(torch.tensor([[start, 3.95]]), torch.tensor([[heading, 0.]]), torch.tensor([0]))
    assert c["valid"].item()
    side_edges = reward.edges[0, 0::4, 1]
    last = side_edges.max() if heading == 1 else side_edges.min()
    assert c["destination"][0, 0] == pytest.approx(last.item() + heading*.35, abs=1e-6)
    initial_height = reward.surface_height(torch.tensor([0]), torch.tensor([[[start, 3.95]]])).item()
    height_change = c["height"].item() - initial_height
    assert height_change * heading * (1 if label == 3 else -1) > 0
    reward.state.begin(c, torch.tensor([[start, 3.95]]))
    feet = torch.zeros(1, 4, 3)
    feet[..., :2] = c["destination"][:, None]
    feet[..., 2] = c["height"]
    for _ in range(5):
        reward.state.advance(c["destination"], feet, torch.ones(1, 4, dtype=torch.bool), torch.tensor([False]), .02)
    assert reward.state.totals["completions"].item() == 1


@pytest.mark.parametrize("label,start,heading", [(6, 3.95, 1.), (6, 1., 1.), (7, 5.5, 1.)])
def test_gap_and_pit_destination_is_upper_surface(label, start, heading):
    reward = geometry_reward(label)
    c = reward.candidates(torch.tensor([[start, 3.95]]), torch.tensor([[heading, 0.]]), torch.tensor([0]))
    if label == 6 and start == 3.95:
        c = reward.candidates(torch.tensor([[5.5, 3.95]]), torch.tensor([[heading, 0.]]), torch.tensor([0]))
    assert c["valid"].item()
    assert c["height"].item() == pytest.approx(0.)
    assert c["destination"][0, 0] > start


def test_contact_on_wrong_height_cannot_complete():
    reward = geometry_reward(7)
    sim = reward.env.simulator
    sim.base_pos = torch.tensor([[5.5, 3.95, 0.]])
    sim.base_quat = torch.tensor([[0., 0., 0., 1.]])
    sim.terrain_levels = sim.terrain_types = torch.tensor([0])
    reward.begin()
    sim.base_pos[:, :2] = reward.state.fixed["destination"]
    sim.feet_pos = sim.base_pos[:, None].repeat(1, 4, 1)
    sim.feet_pos[..., 2] = -.35
    sim.feet_contact_indices = torch.arange(4)
    sim.link_contact_forces = torch.zeros(1, 4, 3)
    sim.link_contact_forces[..., 2] = 30.
    sim.projected_gravity = torch.tensor([[0., 0., -1.]])
    for _ in range(10):
        assert not reward.advance(torch.tensor([False]))[1].any()
    sim.feet_pos[..., 2] = .022
    for _ in range(5):
        reward.advance(torch.tensor([False]))
    assert reward.state.totals["completions"].item() == 1


def test_new_identity_allowed_but_backtracking_cannot_rearm():
    state = module.ObstacleProgressState(2, 4, config(), "cpu")
    c = candidate()
    transition(state, c, 0., 2., dt=.1)
    c["identity"][:] = 1
    assert (transition(state, c, 0., .5)[0] > 0).all()
    transition(state, c, .5, 2., dt=.1)
    c["identity"][:] = 0
    assert not transition(state, c, 0., .5)[0].any()


def test_stationary_time_is_seconds_not_step_count():
    for dt in (.01, .02):
        state = module.ObstacleProgressState(2, 4, config(), "cpu")
        for _ in range(round(.2/dt)):
            transition(state, candidate(), 1.5, 1.5, dt=dt)
        torch.testing.assert_close(state.totals["stationary_s"], torch.full((2,), .2))


@pytest.mark.parametrize("dt", [.01, .02, .05])
def test_stationary_penalty_grace_and_timestep(dt):
    cfg = config()
    cfg.stationary_penalty_rate = .1
    cfg.stationary_grace_s = .075  # intentionally not a multiple of dt
    state = module.ObstacleProgressState(2, 4, cfg, "cpu")
    for _ in range(round(.2 / dt)):
        transition(state, candidate(), .1, .1, dt=dt)  # near entrance, not final edge
    torch.testing.assert_close(state.totals["stationary_penalty"], torch.full((2,), -.0125))
    transition(state, candidate(), .1, .5, dt=dt)
    assert not state.stationary_penalty.any()
    assert not state.stationary_time.any()
    transition(state, candidate(), .5, .5, dt=dt)
    assert not state.stationary_penalty.any()  # fresh grace period after moving


def test_stationary_penalty_reset_support_failure_and_disabled():
    cfg = config()
    cfg.stationary_penalty_rate = .1
    cfg.stationary_grace_s = 0.
    state = module.ObstacleProgressState(2, 4, cfg, "cpu")
    transition(state, candidate(), 1.5, 1.5)
    assert (state.stationary_penalty < 0).all()
    state.reset(torch.tensor([0]))
    assert state.stationary_time[0] == 0 and state.stationary_time[1] > 0
    assert state.totals["stationary_penalty"][0] == 0
    transition(state, candidate(), 1.5, 1.5, failed=True)
    assert not state.stationary_penalty.any()
    # Neither sustained landing support nor a completed attempt is penalized.
    for _ in range(8):
        transition(state, candidate(), 2., 2.)
        assert not state.stationary_penalty.any()
    state.reset(torch.arange(2))
    cfg.stationary_penalty_rate = 0.
    transition(state, candidate(), 1.5, 1.5)
    assert not state.stationary_penalty.any()
    cfg.stationary_penalty_rate = .1
    transition(state, candidate(), -2., -2.)  # outside path radius
    assert not state.stationary_penalty.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_batch():
    state = module.ObstacleProgressState(64, 4, config(), "cuda")
    c = {key: value.cuda() for key, value in candidate(64).items()}
    state.begin(c, torch.zeros(64, 2, device="cuda"))
    p, b = state.advance(torch.tensor([[.5, 0.]], device="cuda").repeat(64, 1),
                         torch.zeros(64, 4, 3, device="cuda"), torch.zeros(64, 4, device="cuda", dtype=torch.bool),
                         torch.zeros(64, device="cuda", dtype=torch.bool), .02)
    torch.testing.assert_close(p, torch.full((64,), .05, device="cuda"))
    assert not b.any()


def test_integration_calls_existing_rewards_then_unscaled_bonus_before_reset():
    path = ROOT / "legged_gym/envs/go2/go2_depth_waq/go2_depth_waq.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Go2DepthWaq")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_reward")
    class Base:
        def compute_reward(self):
            self.rew_buf = torch.tensor([.7])
    replacement = ast.ClassDef(name="Subject", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
    namespace = {"Base": Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[replacement], type_ignores=[])), str(path), "exec"), namespace)
    obj = namespace["Subject"]()
    obj.obstacle_progress = SimpleNamespace(advance=lambda failed: (torch.tensor([.2]), torch.tensor([.3])),
                                           state=SimpleNamespace(stationary_penalty=torch.tensor([-.1])))
    obj.gap_reset_buf = torch.tensor([False])
    obj.fail_buf = torch.zeros(1)
    obj.terminated_bodies_force_norm = torch.zeros(1, 1)
    obj.simulator = SimpleNamespace(projected_gravity=torch.tensor([[0., 0., -1.]]))
    obj.cfg = SimpleNamespace(env=SimpleNamespace(fail_to_terminal_time_s=1., max_projected_gravity=-.5))
    obj.dt = .02
    obj.compute_reward()
    assert obj.rew_buf.item() == pytest.approx(1.1)
    failures = []
    obj.obstacle_progress.advance = lambda failed: (failures.append(failed.clone()) or torch.zeros(1), torch.zeros(1))
    obj.gap_reset_buf[:] = True
    obj.time_out_buf = torch.tensor([True])
    obj.compute_reward()
    assert failures[0].item()  # timeout must not hide a simultaneous failure
    obj.obstacle_progress = None
    obj.compute_reward()
    assert obj.rew_buf.item() == pytest.approx(.7)
    base_tree = ast.parse((ROOT / "legged_gym/envs/base/legged_robot.py").read_text())
    post = next(n for n in ast.walk(base_tree) if isinstance(n, ast.FunctionDef) and n.name == "post_physics_step")
    calls = [n.func.attr for n in ast.walk(post) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.index("compute_reward") < calls.index("reset_idx")
