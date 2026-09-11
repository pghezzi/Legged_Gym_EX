# Optional obstacle progress reward

Set `Go2DepthWaqCfg.rewards.obstacle_progress.enabled = True` for dedicated
GAP, PIT, or STAIRS training. Default is **off**. No reward-scale entry or
auxiliary geometry decoder is required. Configuration is saved with the existing
training configuration. Currently supports IsaacGym's standard simplified
trimesh tiles (not selected/custom tracks); unsupported opt-in configurations
raise an error rather than inventing geometry.

Each terrain's `gap`, `pit`, or `stairs` list is `[k_progress, b0, b1]`.
Progress is `k_progress * max(0, D_best - D_next)`, without normalization.
`k_progress` is reward/metre; total progress per attempt is bounded by
`k_progress * D_initial`. Completion bonuses and stationary penalties remain
separate. `b0` is reward/crossing and `b1` is reward/unit normalized difficulty.
Difficulty is the terrain generator's `row / num_rows`, frozen at activation.
These displacement/event rewards are added once after existing reward handling,
without multiplying by control `dt`. Other rewards/curricula are unchanged.

The base/root position measures remaining distance along a cardinal traversal
direction chosen from body heading at activation. Geometry, direction, target,
and difficulty then remain fixed. Only reductions of the best remaining distance
earn progress. Retreating never rearms an attempt. GAP/PIT identities are whole
tile obstacles; each side of a stair pyramid is one whole-flight identity,
shared by both traversal directions. Completed identities persist until episode
reset. Stair lanes must intersect the central landing: corner/partial flights
are deliberately excluded. Leaving an entered crossing's lateral corridor
invalidates the attempt, preventing walk-around completion.

GAP targets are beyond the landing edge, PIT targets beyond the upper lip
(climbing out, not entering), and stairs target beyond the last riser in the
chosen direction, for either positive or negative step heights. Completion
requires base clearance, **all feet** past the edge, and at least
`min_support_feet` on the destination-height surface with vertical contact force
for `support_duration_s` continuously. Foot-link height tolerance/offset and
clearance distances are configurable in metres; force is in newtons. Failure
suppresses completion, including simultaneous failure/timeout. Timeout itself
does not count as success. Evaluation occurs before episode reset.

Episode logs (raw mean per episode, not reward/second):

- `obstacle/progress_reward`
- `obstacle/completion_reward`
- `obstacle/completions`
- `obstacle/stationary_s` (time below `stationary_speed` within
  `stationary_radius` of the active destination edge)
- `obstacle/stationary_penalty` (negative accumulated stationary penalty)

The complementary stationary penalty uses `stationary_penalty_rate` (default
0.05 reward/second; set to 0 to disable). It applies after
`stationary_grace_s` (default 0.50 seconds) of continuous motion at or below
`stationary_speed`, within `stationary_radius` of the fixed crossing path and
inside its lateral corridor. This covers the entrance and whole stair flight,
not only the final edge. Moving or leaving this region resets the grace timer.
Failure, destination-support confirmation, and completed attempts are exempt.
Unlike the displacement/bonus terms, this rate is integrated over elapsed time;
only the portion of a step beyond the grace period is charged. For example,
2 seconds stationary at the defaults costs 0.075 reward. The parent
`obstacle_progress.enabled` flag controls this penalty too. Existing stationary
time logging retains its original final-edge definition.

Geometry is cached once using the existing ground-truth edge helper; runtime
operations are batched device tensors. Auxiliary targets, policy inputs, and
auxiliary optimization are unchanged. Tests: `python -m pytest
tests/test_obstacle_progress.py -q`.
