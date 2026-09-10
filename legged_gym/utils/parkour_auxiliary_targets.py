"""Training-only labels for the standard Go2 simplified stair/pit/gap meshes.

Startup work is CPU-side; per-step label generation is batched on the simulator
device. No ray casts, simulator steps, terrain edits, or actor observations change.
"""

import numpy as np
import torch


def rotate(q, v):
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def rectangle_edges(cx, cy, hx, hy, inward_height):
    # axis, plane coordinate, segment low/high on other axis, +axis height change
    return [(0, cx-hx, cy-hy, cy+hy, inward_height),
            (0, cx+hx, cy-hy, cy+hy, -inward_height),
            (1, cy-hy, cx-hx, cx+hx, inward_height),
            (1, cy+hy, cx-hx, cx+hx, -inward_height)]


def build_edges(terrain):
    """Mirror mesh_*_terrain quantization; do not evaluate curriculum expressions."""
    cfg = terrain.cfg
    if cfg.mesh_type != "trimesh" or not cfg.simplify_mesh or cfg.selected:
        raise ValueError("Parkour auxiliary labels currently require standard simplified trimesh terrain")
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    nx, ny = terrain.length_per_env_pixels, terrain.width_per_env_pixels
    length, width = nx * hs, ny * hs
    all_edges = []
    for i in range(cfg.num_rows):
        for j in range(cfg.num_cols):
            label = terrain.labels[i, j]
            tile = terrain.height_field_raw[terrain.border+i*nx:terrain.border+(i+1)*nx,
                                             terrain.border+j*ny:terrain.border+(j+1)*ny]
            # IsaacGym shifts its triangle mesh by -horizontal_scale/2.
            cx, cy = (i+.5)*length-hs/2, (j+.5)*width-hs/2
            profile = tile[:nx//2+1, ny//2].astype(float)
            changes = np.diff(profile) * vs
            nonzero = changes[np.abs(changes) > 1e-6]
            edges = []
            if label in (2, 3) and nonzero.size:
                step_height = float(nonzero[0])
                step_width = int(0.4 / hs) * hs  # pyramid_stairs_terrain
                platform = int(cfg.platform_size / hs) * hs
                count = int(min((length-platform)//(2*step_width)+1,
                                (width-platform)//(2*step_width)+1))
                for k in range(1, count):
                    edges.extend(rectangle_edges(cx, cy, length/2-k*step_width,
                                                  width/2-k*step_width, step_height))
            elif label == 7 and nonzero.size:
                half = int(cfg.platform_size / hs / 2) * hs
                edges = rectangle_edges(cx, cy, half, half, float(nonzero[0]))
            elif label == 6:
                depressed = np.flatnonzero(profile < -1)
                if depressed.size:
                    gap = len(depressed) * hs
                    half = int(cfg.platform_size / hs) * hs / 2
                    edges = rectangle_edges(cx, cy, half+gap, half+gap, -1.)
                    edges += rectangle_edges(cx, cy, half, half, 1.)
            all_edges.append(edges)
    result = np.zeros((len(all_edges), max(1, max(map(len, all_edges))), 5), dtype=np.float32)
    for i, edges in enumerate(all_edges):
        if edges:
            result[i, :len(edges)] = edges
    return result


def geometry_targets(points, direction, edges, labels, max_distance):
    """points=[N,CoM/FL/FR,xy]. All distances refer to the same next edge.

    Retain a crossed edge until all three reference points pass it. Foot targets
    are intersections with that finite edge, not an unrelated nearest obstacle.
    Stairs/pits include entry and exit; gap width is measured along body heading.
    """
    n, e = edges.shape[:2]
    axis = edges[..., 0].long()
    other = 1-axis
    d = direction.gather(1, axis)
    safe = torch.where(d.abs() > 1e-5, d, torch.ones_like(d))
    coord = points[:, :, None, :].expand(-1, -1, e, -1)
    axis_index = axis[:, None, :, None].expand(-1, 3, -1, -1)
    along = coord.gather(-1, axis_index).squeeze(-1)
    lateral = coord.gather(-1, 1-axis_index).squeeze(-1)
    distances = (edges[:, None, :, 1] - along) / safe[:, None, :]
    hit = lateral + distances * direction.gather(1, other)[:, None, :]
    hits = (hit >= edges[:, None, :, 2]) & (hit <= edges[:, None, :, 3])
    delta = edges[..., 4] * d.sign()
    valid = (d.abs() > 1e-5) & (edges[..., 4] != 0) & hits[:, 0]
    valid &= (distances[:, 0] <= max_distance) & (distances[:, 0] >= -max_distance)
    valid &= distances.amax(dim=1) >= 0  # all reference points have not passed
    valid &= (labels[:, None] != 6) | (delta < 0)  # gap entry, not landing edge
    rank = torch.where(valid, distances[:, 0], torch.full_like(d, float("inf")))
    best, idx = rank.min(dim=1)
    rows = torch.arange(n, device=points.device)
    selected = distances[rows, :, idx]
    selected_delta = delta[rows, idx]
    selected_hits = hits[rows, :, idx] & (selected.abs() <= max_distance)
    exit_valid = (d.abs() > 1e-5) & hits[:, 0] & (delta > 0)
    exit_valid &= distances[:, 0] > best[:, None] + 1e-5
    exit_distance = torch.where(exit_valid, distances[:, 0], torch.full_like(d, float("inf"))).amin(dim=1)
    gap_width = exit_distance - best
    targets = torch.zeros(n, 12, device=points.device)
    masks = torch.zeros_like(targets, dtype=torch.bool)
    for offset, active, scalar in (
        (0, (labels == 2) | (labels == 3), selected_delta),
        (4, labels == 7, selected_delta),
        (8, labels == 6, gap_width),
    ):
        active = active & best.isfinite()
        targets[:, offset:offset+3] = selected.nan_to_num()
        targets[:, offset+3] = scalar.nan_to_num(nan=0., posinf=0., neginf=0.)
        masks[:, offset:offset+3] = active[:, None] & selected_hits
        masks[:, offset+3] = active & scalar.isfinite()
    return targets, masks


class IsaacGymParkourTargets:
    def __init__(self, env, max_distance):
        sim = env.simulator
        if not hasattr(sim, "_gym"):
            raise ValueError("Parkour auxiliary CoM labels currently support IsaacGym only")
        self.env, self.sim, self.max_distance = env, sim, max_distance
        terrain = sim._terrain
        self.edges = torch.as_tensor(build_edges(terrain), device=env.device)
        self.labels = torch.as_tensor(terrain.labels, device=env.device).flatten()
        self.cols = terrain.cfg.num_cols
        # Read randomized masses and local CoMs ONCE, outside the training loop.
        # This backend fixes these properties at actor creation.
        props = [sim._gym.get_actor_rigid_body_properties(e, a)
                 for e, a in zip(sim._envs, sim._actor_handles)]
        self.masses = torch.tensor([[p.mass for p in row] for row in props], device=env.device)
        self.coms = torch.tensor([[[p.com.x, p.com.y, p.com.z] for p in row]
                                 for row in props], device=env.device)
        self.front = [sim._feet_names.index(name) for name in ("FL_foot", "FR_foot")]

    @torch.no_grad()
    def __call__(self):
        sim = self.sim
        states = sim._rigid_body_states
        body_coms = states[..., :3] + rotate(states[..., 3:7], self.coms)
        com = (body_coms * self.masses[..., None]).sum(1) / self.masses.sum(1, keepdim=True)
        points = torch.cat((com[:, None, :2], sim.feet_pos[:, self.front, :2]), dim=1)
        forward = torch.zeros_like(sim.base_pos)
        forward[:, 0] = 1.
        forward = rotate(sim.base_quat, forward)[:, :2]
        norm = forward.norm(dim=-1, keepdim=True)
        direction = forward / norm.clamp_min(1e-6)
        tile = sim._terrain_levels * self.cols + sim._terrain_types
        targets, masks = geometry_targets(points, direction, self.edges[tile], self.labels[tile], self.max_distance)
        # No labels outside assigned terrain or with effectively vertical heading.
        cfg = sim._terrain.cfg
        local = sim.base_pos[:, :2] - sim._env_origins[:, :2]
        inside = (local[:, 0].abs() < cfg.terrain_length/2) & (local[:, 1].abs() < cfg.terrain_width/2)
        masks &= (inside & (norm[:, 0] > 0.1))[:, None]
        return targets, masks
