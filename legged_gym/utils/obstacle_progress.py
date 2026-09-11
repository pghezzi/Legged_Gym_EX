"""Optional, undiscounted displacement/event rewards for standard parkour tiles.

Geometry is constructed once on CPU using the existing auxiliary edge helper;
all attempt selection, support checks, and bookkeeping are batched tensors.
"""

import torch

from .parkour_auxiliary_targets import build_edges, rotate


class ObstacleProgressState:
    def __init__(self, num_envs, num_obstacles, cfg, device):
        self.cfg = cfg
        self.rows = torch.arange(num_envs, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.completed = torch.zeros(num_envs, num_obstacles, dtype=torch.bool, device=device)
        self.identity = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.best = torch.zeros(num_envs, device=device)
        self.support_time = torch.zeros_like(self.best)
        self.stationary_time = torch.zeros_like(self.best)
        self.stationary_penalty = torch.zeros_like(self.best)
        self.entered = torch.zeros_like(self.active)
        self.valid_path = torch.ones_like(self.active)
        self.fixed = {key: torch.zeros(num_envs, width, device=device) for key, width in
                      (("direction", 2), ("entry", 2), ("destination", 2),
                       ("lateral_bounds", 2), ("height", 1), ("difficulty", 1), ("coefficients", 3))}
        self.previous = torch.zeros(num_envs, 2, device=device)
        self.totals = {name: torch.zeros_like(self.best) for name in
                       ("progress_reward", "completion_reward", "completions", "stationary_s", "stationary_penalty")}

    def reset(self, ids):
        self.active[ids] = False
        self.completed[ids] = False
        self.best[ids] = 0.
        self.support_time[ids] = 0.
        self.stationary_time[ids] = 0.
        self.stationary_penalty[ids] = 0.
        self.entered[ids] = False
        self.valid_path[ids] = True
        for value in self.totals.values():
            value[ids] = 0.

    def begin(self, candidate, position):
        identity = candidate["identity"]
        start = candidate["valid"] & ~self.active & ~self.completed[self.rows, identity]
        for name, value in self.fixed.items():
            value.copy_(torch.where(start[:, None], candidate[name], value))
        self.identity.copy_(torch.where(start, identity, self.identity))
        direction = self.fixed["direction"]
        initial = ((self.fixed["destination"] - position) * direction).sum(-1).clamp_min(0.)
        self.best.copy_(torch.where(start, initial, self.best))
        self.support_time[start] = 0.
        self.stationary_time[start] = 0.
        self.valid_path[start] = True
        inside = ((position - self.fixed["entry"]) * direction).sum(-1) >= 0
        self.entered.copy_(torch.where(start, inside, self.entered))
        self.active |= start
        self.previous.copy_(position)

    def advance(self, position, feet, support, failed, dt):
        direction = self.fixed["direction"]
        lateral = direction.flip(-1).abs()  # cardinal traversal, unsigned other axis
        bounds = self.fixed["lateral_bounds"]
        p = (position * lateral).sum(-1)
        prev_p = (self.previous * lateral).sum(-1)
        corridor = (p >= bounds[:, 0]) & (p <= bounds[:, 1])
        previous_corridor = (prev_p >= bounds[:, 0]) & (prev_p <= bounds[:, 1])
        entry_now = ((position - self.fixed["entry"]) * direction).sum(-1)
        entry_before = ((self.previous - self.fixed["entry"]) * direction).sum(-1)
        crossed_entry = (entry_before <= 0) & (entry_now >= 0) & corridor & previous_corridor
        self.entered |= self.active & crossed_entry
        # Leaving the flight/crossing corridor cannot be used to walk around it.
        self.valid_path &= ~(self.active & self.entered & ~corridor)
        remaining = ((self.fixed["destination"] - position) * direction).sum(-1).clamp_min(0.)
        progress = (self.best - remaining).clamp_min(0.) * self.fixed["coefficients"][:, 0]
        progress *= self.active & self.valid_path & corridor
        self.best.copy_(torch.where(self.active, torch.minimum(self.best, remaining), self.best))

        # Target lies base_clearance past the edge. All feet must also clear it.
        edge = self.fixed["destination"] - self.cfg.base_clearance * direction
        feet_clear = ((feet[..., :2] - edge[:, None]) * direction[:, None]).sum(-1)
        feet_lateral = (feet[..., :2] * lateral[:, None]).sum(-1)
        feet_in_lane = (feet_lateral >= bounds[:, :1]) & (feet_lateral <= bounds[:, 1:])
        clear = (remaining <= 1e-6) & (feet_clear >= self.cfg.foot_clearance).all(-1)
        supported = (support & feet_in_lane).sum(-1) >= self.cfg.min_support_feet
        eligible = self.active & self.valid_path & self.entered & corridor & clear & supported & ~failed
        self.support_time.copy_(torch.where(eligible, self.support_time + dt, torch.zeros_like(self.support_time)))
        success = eligible & (self.support_time + 1e-6 >= self.cfg.support_duration_s)
        bonus = success * (self.fixed["coefficients"][:, 1] +
                           self.fixed["coefficients"][:, 2] * self.fixed["difficulty"][:, 0])
        self.completed[self.rows, self.identity] |= success
        self.active &= ~success
        speed = (position - self.previous).norm(dim=-1) / dt
        near = (position - edge).norm(dim=-1) <= self.cfg.stationary_radius
        stationary = self.active & near & (speed <= self.cfg.stationary_speed)
        # Distance to the fixed traversal segment covers the approach and the
        # whole flight, not just the final edge. Preserve the original time log.
        entry = self.fixed["entry"]
        length = ((edge - entry) * direction).sum(-1).clamp_min(0.)
        along = ((position - entry) * direction).sum(-1).clamp_min(0.)
        closest = entry + torch.minimum(along, length)[:, None] * direction
        near_path = (position - closest).norm(dim=-1) <= self.cfg.stationary_radius
        stalled = (self.active & near_path & corridor & (speed <= self.cfg.stationary_speed)
                   & ~failed & ~eligible)
        grace = getattr(self.cfg, "stationary_grace_s", .5)
        previous_time = self.stationary_time.clone()
        self.stationary_time.copy_(torch.where(stalled, previous_time + dt, torch.zeros_like(previous_time)))
        # Charge only time past the grace period, including fractional steps.
        charged_s = ((self.stationary_time - grace).clamp_min(0.) -
                     (previous_time - grace).clamp_min(0.)).clamp_min(0.)
        self.stationary_penalty.copy_(-getattr(self.cfg, "stationary_penalty_rate", 0.) * charged_s)
        self.totals["progress_reward"] += progress
        self.totals["completion_reward"] += bonus
        self.totals["completions"] += success
        self.totals["stationary_s"] += stationary * dt
        self.totals["stationary_penalty"] += self.stationary_penalty
        return progress, bonus


class ObstacleProgress:
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        sim = env.simulator
        proportions = env.cfg.terrain.terrain_proportions
        dedicated = (proportions[6] == 1 or proportions[7] == 1 or
                     proportions[2] + proportions[3] == 1)
        if not dedicated or not hasattr(sim, "_gym"):
            raise ValueError("Obstacle progress supports dedicated Go2 GAP/PIT/STAIRS with IsaacGym only")
        terrain = sim._terrain
        if (cfg.base_clearance < 0 or cfg.foot_clearance < 0 or cfg.support_duration_s <= 0
                or not 1 <= cfg.min_support_feet <= 4 or cfg.activation_distance <= 0
                or cfg.support_height_tolerance <= 0 or cfg.contact_force_threshold <= 0
                or cfg.stationary_speed < 0 or cfg.stationary_radius <= 0):
            raise ValueError("Invalid obstacle-progress clearance/support thresholds")
        import math
        for name in ("stationary_penalty_rate", "stationary_grace_s"):
            value = getattr(cfg, name, 0.)
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        self.edges = torch.as_tensor(build_edges(terrain), device=env.device)
        if self.edges.shape[1] % 4:
            self.edges = torch.nn.functional.pad(self.edges, (0, 0, 0, 4-self.edges.shape[1] % 4))
        self.labels = torch.as_tensor(terrain.labels, device=env.device).flatten()
        self.cols, self.num_rows = terrain.cfg.num_cols, terrain.cfg.num_rows
        self.length, self.width = terrain.env_length, terrain.env_width
        self.offset = terrain.cfg.horizontal_scale / 2
        coefficients = torch.tensor([cfg.stairs, cfg.gap, cfg.pit], device=env.device)
        if coefficients.shape != (3, 3) or not torch.isfinite(coefficients).all() or (coefficients < 0).any():
            raise ValueError("Per-terrain coefficients must be nonnegative finite [k_progress, b0, b1]")
        self.coefficients = coefficients
        self.state = ObstacleProgressState(env.num_envs, len(self.labels) * 4, cfg, env.device)
        self.rows = self.state.rows

    def surface_height(self, tile, xy):
        # Four edges per nested rectangle. Signed increments reproduce the mesh
        # top surfaces; GAP voids use -1 only as an invalid support marker.
        edges = self.edges[tile]
        left, right = edges[:, 0::4], edges[:, 1::4]
        inside = ((xy[..., 0, None] >= left[:, None, :, 1]) &
                  (xy[..., 0, None] <= right[:, None, :, 1]) &
                  (xy[..., 1, None] >= left[:, None, :, 2]) &
                  (xy[..., 1, None] <= left[:, None, :, 3]))
        return (inside * left[:, None, :, 4]).sum(-1)

    def candidates(self, position, forward, tile):
        edges, labels = self.edges[tile], self.labels[tile]
        axis = forward.abs().argmax(-1)
        sign = forward.gather(1, axis[:, None]).sign().squeeze(1)
        direction = torch.zeros_like(forward).scatter_(1, axis[:, None], sign[:, None])
        along = position.gather(1, axis[:, None])
        lateral = position.gather(1, (1-axis)[:, None])
        distances = (edges[..., 1] - along) * sign[:, None]
        hits = ((edges[..., 0] == axis[:, None]) & (edges[..., 4] != 0) &
                (lateral >= edges[..., 2]) & (lateral <= edges[..., 3]) & (sign[:, None] != 0))
        delta = edges[..., 4] * sign[:, None]
        infinity = torch.full_like(distances, float("inf"))
        first, idx = torch.where(hits & (distances > 1e-5), distances, infinity).min(-1)
        side = idx % 4
        stair = (labels == 2) | (labels == 3)
        side_mask = (torch.arange(edges.shape[1], device=edges.device) % 4 == side[:, None]) & (edges[..., 4] != 0)
        # Require a lane reaching the central landing, not a corner/partial flight.
        whole_flight = (~side_mask | hits).all(-1)
        flight_start = torch.where(side_mask, distances, infinity).amin(-1)
        flight_end = torch.where(side_mask, distances, -infinity).amax(-1)
        lo = torch.where(side_mask, edges[..., 2], -infinity).amax(-1)
        hi = torch.where(side_mask, edges[..., 3], infinity).amin(-1)

        ground = self.surface_height(tile, position[:, None]).squeeze(1)
        # GAP: find entry and the next landing edge, including starts over a void.
        entry_distance, entry_idx = torch.where(hits & (delta < 0) & (distances >= 0), distances, infinity).min(-1)
        behind, behind_idx = torch.where(hits & (delta < 0) & (distances < 0), distances, -infinity).max(-1)
        in_gap = (labels == 6) & (ground < -.5)
        entry_distance = torch.where(in_gap, behind, entry_distance)
        entry_idx = torch.where(in_gap, behind_idx, entry_idx)
        gap_end, gap_end_idx = torch.where(hits & (delta > 0) & (distances > entry_distance[:, None]), distances, infinity).min(-1)
        gap_lo = torch.maximum(edges[self.rows, entry_idx, 2], edges[self.rows, gap_end_idx, 2])
        gap_hi = torch.minimum(edges[self.rows, entry_idx, 3], edges[self.rows, gap_end_idx, 3])

        pit = labels == 7
        gap = labels == 6
        start = torch.where(stair, flight_start, torch.where(gap, entry_distance, torch.zeros_like(first)))
        end = torch.where(stair, flight_end, torch.where(gap, gap_end, first))
        lo = torch.where(stair, lo, torch.where(gap, gap_lo, edges[self.rows, idx, 2]))
        hi = torch.where(stair, hi, torch.where(gap, gap_hi, edges[self.rows, idx, 3]))
        # PIT means climbing from the floor to the upper surface, not entering it.
        valid = ((stair & whole_flight) | (gap & entry_distance.isfinite()) |
                 (pit & (ground < 0) & (delta[self.rows, idx] > 0)))
        approach = torch.where(gap, entry_distance.clamp_min(0), first)
        valid &= (approach <= self.cfg.activation_distance) & end.isfinite() & (end > 0)
        valid &= (lo <= lateral[:, 0]) & (lateral[:, 0] <= hi)
        # Padded/invalid candidates must remain finite even when masked out.
        end, start = end.nan_to_num(nan=0., posinf=0., neginf=0.), start.nan_to_num(nan=0., posinf=0., neginf=0.)
        destination = position + direction * (end + self.cfg.base_clearance)[:, None]
        height = self.surface_height(tile, destination[:, None])
        # Don't target beyond a tile boundary or through a second obstacle.
        corner = torch.stack((tile // self.cols * self.length, tile % self.cols * self.width), -1) - self.offset
        local = destination - corner
        valid &= (local[:, 0] > 0) & (local[:, 0] < self.length) & (local[:, 1] > 0) & (local[:, 1] < self.width)
        local_start = position - corner
        valid &= (local_start[:, 0] > 0) & (local_start[:, 0] < self.length) & (local_start[:, 1] > 0) & (local_start[:, 1] < self.width)
        landing = position + direction * (end + 1e-4)[:, None]
        valid &= (height[:, 0] - self.surface_height(tile, landing[:, None])[:, 0]).abs() < 1e-5
        valid &= stair | (height[:, 0].abs() < 1e-5)
        return {"identity": tile * 4 + torch.where(stair, side, torch.zeros_like(side)),
                "valid": valid, "direction": direction,
                "entry": position + direction * start[:, None], "destination": destination,
                "lateral_bounds": torch.stack((lo, hi), -1).nan_to_num(), "height": height,
                "difficulty": (tile // self.cols).float().div(self.num_rows).clamp(0, 1)[:, None],
                "coefficients": self.coefficients[torch.where(gap, 1, torch.where(pit, 2, 0))]}

    @torch.no_grad()
    def begin(self):
        sim = self.env.simulator
        forward = torch.zeros_like(sim.base_pos)
        forward[:, 0] = 1.
        forward = rotate(sim.base_quat, forward)[:, :2]
        tile = sim.terrain_levels * self.cols + sim.terrain_types
        candidate = self.candidates(sim.base_pos[:, :2], forward, tile)
        self.state.begin(candidate, sim.base_pos[:, :2])

    @torch.no_grad()
    def advance(self, failed):
        sim = self.env.simulator
        tile = self.state.identity // 4
        surface = self.surface_height(tile, sim.feet_pos[..., :2])
        expected = self.state.fixed["height"]
        forces = sim.link_contact_forces[:, sim.feet_contact_indices, 2]
        support = ((forces >= self.cfg.contact_force_threshold) &
                   ((surface - expected).abs() < 1e-5) &
                   ((sim.feet_pos[..., 2] - expected - self.cfg.foot_height_offset).abs() <= self.cfg.support_height_tolerance))
        # Do not infer another tile's surface using this attempt's geometry.
        corner = torch.stack((tile // self.cols * self.length, tile % self.cols * self.width), -1) - self.offset
        local_feet = sim.feet_pos[..., :2] - corner[:, None]
        support &= ((local_feet[..., 0] >= 0) & (local_feet[..., 0] < self.length) &
                    (local_feet[..., 1] >= 0) & (local_feet[..., 1] < self.width))
        # Also require an upright body, not a fallen robot with loaded feet.
        failed = failed | (sim.projected_gravity[:, 2] > -.5)
        return self.state.advance(sim.base_pos[:, :2], sim.feet_pos, support, failed, self.env.dt)
