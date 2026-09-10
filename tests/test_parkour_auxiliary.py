"""Small CPU tests; no simulator, rollout training, or CUDA required."""

import importlib.util
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch

from rsl_rl.algorithms.ppo_dreamwaq_depth import PPO_DreamWaQ_Depth
from rsl_rl.modules.actor_critic_dreamwaq_depth import ActorCriticDreamWaQDepth
from rsl_rl.utils.parkour_auxiliary import ParkourAuxiliary, TARGET_NAMES


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load_file("aux_geometry", "legged_gym/utils/parkour_auxiliary_targets.py")
terrain_utils = load_file("aux_terrain_utils", "legged_gym/utils/terrain_utils.py")


def make_terrain(label):
    tile = terrain_utils.SubTerrain(width=80, length=80, horizontal_scale=.1, vertical_scale=.005)
    common = dict(platform_size=4., terrain_type="trimesh", simplify_mesh=True)
    if label in (2, 3):
        terrain_utils.pyramid_stairs_terrain(tile, step_width=.4, step_height=-.2 if label == 2 else .2, **common)
    elif label == 7:
        terrain_utils.pit_terrain(tile, depth=.35, **common)
    else:
        terrain_utils.gap_terrain(tile, gap_size=.6, **common)
    cfg = SimpleNamespace(mesh_type="trimesh", simplify_mesh=True, selected=False,
                          horizontal_scale=.1, vertical_scale=.005, num_rows=1, num_cols=1,
                          platform_size=4.)
    terrain = SimpleNamespace(cfg=cfg, length_per_env_pixels=80, width_per_env_pixels=80,
                              border=0, labels=np.array([[label]]), height_field_raw=tile.height_field_raw)
    return terrain, tile


def top_height(mesh, x, y):
    """Exact top surface of the generated axis-aligned boxes (no ray backend)."""
    tri = mesh.triangles
    flat = np.ptp(tri[:, :, 2], axis=1) < 1e-8
    xy = tri[:, :, :2]
    a, b, c = xy[:, 0], xy[:, 1], xy[:, 2]
    v0, v1, v2 = b-a, c-a, np.array([x, y])-a
    det = v0[:, 0]*v1[:, 1]-v0[:, 1]*v1[:, 0]
    safe = np.where(abs(det) > 1e-8, det, 1.)
    u = (v2[:, 0]*v1[:, 1]-v2[:, 1]*v1[:, 0])/safe
    v = (v0[:, 0]*v2[:, 1]-v0[:, 1]*v2[:, 0])/safe
    hit = flat & (abs(det) > 1e-8) & (u >= -1e-6) & (v >= -1e-6) & (u+v <= 1+1e-6)
    return tri[hit, 0, 2].max() if hit.any() else -np.inf


@pytest.mark.parametrize("label", [2, 3, 6, 7])
def test_edges_match_actual_simplified_mesh(label):
    terrain, tile = make_terrain(label)
    edges = geometry.build_edges(terrain)[0]
    for axis, plane, low, high, delta in edges:
        # Undo IsaacGym's -hs/2 translation to query the local mesh.
        point = np.array([0., 0.])
        axis = int(axis)
        point[axis], point[1-axis] = plane+.05, (low+high)/2+.05
        before, after = point.copy(), point.copy()
        before[axis] -= .01
        after[axis] += .01
        actual = top_height(tile.terrain_mesh, *after) - top_height(tile.terrain_mesh, *before)
        if label == 6:
            assert np.sign(actual) == np.sign(delta)  # no gap-depth label
        else:
            assert actual == pytest.approx(float(delta), abs=1e-6)


def query(label, x, heading=1.):
    terrain, _ = make_terrain(label)
    edges = torch.tensor(geometry.build_edges(terrain))
    points = torch.tensor([[[x, 3.95], [x+.25*heading, 4.10], [x+.35*heading, 3.80]]])
    return geometry.geometry_targets(points, torch.tensor([[heading, 0.]]), edges, torch.tensor([label]), 3.)


@pytest.mark.parametrize("label,offset,height", [(3, 0, .2), (2, 0, -.2), (7, 4, -.35)])
def test_signed_height_and_independent_front_foot_distances(label, offset, height):
    x = .10 if label in (2, 3) else 1.
    targets, masks = query(label, x)
    assert masks[0, offset:offset+4].all()
    assert masks.sum() == 4
    assert targets[0, offset+3] == pytest.approx(height)
    assert targets[0, offset]-targets[0, offset+1] == pytest.approx(.25)
    assert targets[0, offset]-targets[0, offset+2] == pytest.approx(.35)


def test_pit_exit_is_positive_and_heading_reversal_changes_sign():
    target, mask = query(7, 5.)
    assert mask[0, 7] and target[0, 7] == pytest.approx(.35)
    target, mask = query(7, 6.7, -1.)
    assert mask[0, 7] and target[0, 7] == pytest.approx(-.35)


def test_gap_width_matches_quantized_generator_and_no_gap_depth():
    terrain, _ = make_terrain(6)
    target, mask = query(6, .5)
    assert mask[0, 8:].all()
    assert target[0, 11] == pytest.approx(int(.6/.1)*.1)
    assert not any("gap" in name and ("depth" in name or "height" in name) for name in TARGET_NAMES)


def test_crossed_edge_is_retained_until_all_references_pass():
    # Body passed x=1 but a front reference remains behind (e.g. during turning).
    edges = torch.tensor([geometry.rectangle_edges(0., 0., 1., 2., .2)])
    points = torch.tensor([[[1.1, 0.], [.9, .1], [1.3, -.1]]])
    t, m = geometry.geometry_targets(points, torch.tensor([[1., 0.]]), edges, torch.tensor([3]), 3.)
    assert m[0, :4].all()
    torch.testing.assert_close(t[0, :3], torch.tensor([-.1, .1, -.3]))
    points[:, 1, 0] = 1.2
    _, m = geometry.geometry_targets(points, torch.tensor([[1., 0.]]), edges, torch.tensor([3]), 3.)
    assert not m.any()


def test_absent_obstacles_parallel_heading_and_lateral_foot_are_masked():
    edges = torch.tensor([[(0., 1., -.2, .2, -.3)], [(0., 1., -.2, .2, -.3)]])
    points = torch.tensor([[[0., 0.], [.2, .5], [.3, -.1]]] * 2)
    t, m = geometry.geometry_targets(points, torch.tensor([[1., 0.], [0., 1.]]), edges, torch.tensor([7, 7]), 3.)
    assert m[0, 4] and not m[0, 5] and m[0, 6]
    assert not m[1].any() and t.isfinite().all()
    _, m = geometry.geometry_targets(points, torch.tensor([[1., 0.]] * 2), edges, torch.tensor([1, 1]), 3.)
    assert not m.any()


def make_aux():
    model = ActorCriticDreamWaQDepth(6, 2, 8, 12, 3, 2, 6,
        actor_hidden_dims=[8], critic_hidden_dims=[8], encoder_hidden_dims=[8], decoder_hidden_dims=[8],
        depth_image_resolution=[8, 8], cnn_channel_dims=[2], cnn_strides=[1], cnn_kernel_sizes=[3], cnn_fc_layer_dims=[4])
    alg = PPO_DreamWaQ_Depth(model, device="cpu")
    alg.init_storage(2, 2, 1, [6], [8], [12], [2], [6], [2], [1, 8, 8])
    cfg = dict(hidden_dim=8, learning_rate=1e-3, loss_weight=.1, max_distance=3.)
    return model, alg, ParkourAuxiliary(alg, cfg)


def test_auxiliary_snapshot_reset_gradients_and_sidecar(tmp_path):
    torch.manual_seed(1)
    model, alg, aux = make_aux()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    history, depth = torch.randn(2, 12), torch.randn(2, 1, 8, 8)
    alg.storage.observation_histories[0].copy_(history)
    alg.storage.depth_images[0].copy_(depth)
    target, mask = torch.ones(2, 12), torch.ones(2, 12, dtype=torch.bool)
    aux.ready[:] = torch.tensor([True, False])  # second environment just reset
    with torch.inference_mode():
        aux.capture(target, mask)
    target.zero_()
    assert aux.targets.eq(1).all() and not aux.masks[0, 1].any()
    alg.parkour_auxiliary = aux
    alg.storage.current_batch_indices = torch.tensor([1, 0])
    # Same auxiliary backward as the existing VAE update, including shuffled data.
    loss, _, _, _ = alg._compute_vae_loss(history.flip(0), torch.ones(2, 1),
                                          torch.zeros(2, 2), torch.zeros(2, 6), depth.flip(0))
    alg.optimizer.zero_grad(set_to_none=True)
    alg.vae_optimizer.zero_grad(set_to_none=True)
    aux.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.visual_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vae.encoder.parameters())
    alg.optimizer.step()
    alg.vae_optimizer.step()
    aux.optimizer.step()
    assert np.isfinite(aux.last_loss)
    assert any(not torch.equal(v, original[k]) for k, v in model.state_dict().items() if k.startswith("visual_encoder."))
    assert any(not torch.equal(v, original[k]) for k, v in model.state_dict().items() if k.startswith("vae.encoder."))
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items() if k.startswith(("actor.", "critic.")))
    assert set(model.state_dict()) == set(original)
    path = tmp_path / "decoder.pt"
    aux.save(path)
    expected = aux.predict(history, depth).detach()
    with torch.no_grad():
        for p in aux.heads.parameters():
            p.zero_()
    aux.load(path)
    torch.testing.assert_close(aux.predict(history, depth), expected)
    # Legacy strict loading needs no decoder definition or sidecar.
    model.load_state_dict(model.state_dict(), strict=True)


def test_existing_vae_update_includes_geometry_without_extra_pass():
    torch.manual_seed(3)
    model, alg, aux = make_aux()
    alg.parkour_auxiliary = aux
    aux.ready.fill_(True)
    obs, history, depth = torch.randn(2, 6), torch.randn(2, 12), torch.randn(2, 1, 8, 8)
    privileged, explicit = torch.randn(2, 8), torch.randn(2, 2)
    with torch.inference_mode():
        aux.capture(torch.ones(2, 12), torch.ones(2, 12, dtype=torch.bool))
        alg.act(obs, privileged, history, explicit, depth)
        alg.process_env_step(torch.ones(2), torch.zeros(2), {}, torch.randn(2, 6))
        alg.compute_returns(privileged)
    losses = alg.update()
    assert all(np.isfinite(loss) for loss in losses)
    assert aux.last_loss > 0
    assert not aux.masks.any()
    # Exactly one decoder/encoder update: existing encoder epoch, no extra pass.
    assert all(state["step"] == 1 for state in aux.optimizer.state.values())
    assert all(state["step"] == 1 for state in alg.vae_optimizer.state.values())


def test_geometry_loss_matches_shuffled_target_rows_and_terminal_mask():
    _, alg, aux = make_aux()
    alg.storage.current_batch_indices = torch.tensor([1, 0])
    aux.targets[0, 0] = .2
    aux.targets[0, 1] = .8
    aux.masks.fill_(True)
    z, v, depth = torch.randn(2, 3), torch.randn(2, 2), torch.randn(2, 1, 8, 8)
    terminated = torch.tensor([[1.], [0.]])
    prediction = aux.decode(z, v, depth)
    expected = aux.cfg["loss_weight"] * aux.loss(prediction, aux.targets[0].flip(0),
                                                 aux.masks[0] & terminated.bool())
    torch.testing.assert_close(aux.reconstruction_loss(z, v, depth, terminated), expected)


def test_com_targets_use_mass_weighted_rotated_local_com():
    terrain, _ = make_terrain(7)
    terrain.cfg.terrain_length = terrain.cfg.terrain_width = 8.
    props = [SimpleNamespace(mass=1., com=SimpleNamespace(x=.2, y=0., z=0.)),
             SimpleNamespace(mass=3., com=SimpleNamespace(x=0., y=0., z=0.))]
    states = torch.zeros(1, 2, 13)
    states[..., :3] = torch.tensor([[[0., 3.95, .3], [1., 3.95, .3]]])
    states[..., 6] = 1.
    sim = SimpleNamespace(_terrain=terrain, _envs=[0], _actor_handles=[0],
        _gym=SimpleNamespace(get_actor_rigid_body_properties=lambda *_: props),
        _feet_names=["FR_foot", "FL_foot"], _rigid_body_states=states,
        feet_pos=torch.tensor([[[1.3, 3.8, 0.], [1.2, 4.1, 0.]]]),
        base_pos=torch.tensor([[0., 3.95, .3]]), base_quat=torch.tensor([[0., 0., 0., 1.]]),
        _terrain_levels=torch.tensor([0]), _terrain_types=torch.tensor([0]),
        _env_origins=torch.tensor([[4., 4., 0.]]))
    # Keep base inside its tile; CoM is (.2 + 3*1)/4 = .8, not base position.
    sim.base_pos[:, 0] = .1
    provider = geometry.IsaacGymParkourTargets(SimpleNamespace(simulator=sim, device="cpu"), 3.)
    t, m = provider()
    assert m[0, 4:8].all()
    torch.testing.assert_close(t[0, 4:7], torch.tensor([1.95-.8, 1.95-1.2, 1.95-1.3]))
    # Rotate first link by 180 degrees: its local CoM becomes -.2.
    sim._rigid_body_states[0, 0, 3:7] = torch.tensor([0., 0., 1., 0.])
    t, _ = provider()
    assert t[0, 4] == pytest.approx(1.95-.7)


def test_storage_exposes_exact_minibatch_indices_without_changing_tuple():
    _, alg, aux = make_aux()
    storage = alg.storage
    storage.observation_histories[0, 0].fill_(11.)
    storage.observation_histories[0, 1].fill_(22.)
    storage.depth_images[0, 0].fill_(11.)
    storage.depth_images[0, 1].fill_(22.)
    aux.targets[0, 0].fill_(11.)
    aux.targets[0, 1].fill_(22.)
    for batch in storage.mini_batch_generator(2, 2):
        assert len(batch) == 16
        target = aux.targets.flatten(0, 1)[storage.current_batch_indices, 0]
        torch.testing.assert_close(batch[2][:, 0], target)
        torch.testing.assert_close(batch[-1][:, 0, 0, 0], target)


def test_runner_sidecar_preserves_legacy_checkpoint_and_load_selection(tmp_path):
    # Execute the real save/load methods without importing the simulator registry.
    namespace = dict(torch=torch, os=os, Any=Any, Dict=Dict, Optional=Optional)
    for filename, name in (("on_policy_runner.py", "OnPolicyRunner"),
                           ("dreamwaq_depth_runner.py", "DreamWaQDepthRunner")):
        path = Path(__file__).parents[1] / "rsl_rl/runners" / filename
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in ("save", "load")]
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    DreamWaQDepthRunner = namespace["DreamWaQDepthRunner"]
    model, alg, aux = make_aux()
    runner = DreamWaQDepthRunner.__new__(DreamWaQDepthRunner)
    runner.alg, runner.parkour_auxiliary = alg, aux
    runner.current_learning_iteration = 123
    path = tmp_path / "model_123.pt"
    runner.save(str(path))
    assert (tmp_path / "auxiliary" / "model_123.pt").is_file()
    state = torch.load(path, weights_only=False)
    assert set(state) == {"model_state_dict", "optimizer_state_dict", "iter", "infos"}
    # Legacy helper discovers checkpoints by 'model' in top-level filenames.
    assert [p.name for p in tmp_path.iterdir() if "model" in p.name] == ["model_123.pt"]
    expected = {k: v.clone() for k, v in aux.heads.state_dict().items()}
    with torch.no_grad():
        for parameter in aux.heads.parameters():
            parameter.zero_()
    runner.load(str(path))
    for key, value in aux.heads.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    # Disabled auxiliary mode loads the same checkpoint without any heads.
    runner.parkour_auxiliary = None
    runner.load(str(path))
    model.load_state_dict(state["model_state_dict"], strict=True)
