"""CPU-testable accounting for closed-loop diagnostics; no simulator imports."""
import math
import numpy as np

VERSION = "eligible_episode_segment_v2"
PERSISTENCE_TICKS = 2
TRANSITION_RADIUS = 5
APPROACH_DISTANCE_M = 2.0


class EpisodeQuota:
    """Reserve episode slots so parallel environments cannot overrun a track quota."""
    def __init__(self, tracks, quota):
        self.tracks = list(map(int, tracks))
        self.quota = int(quota)
        self.completed = [0] * (max(self.tracks)+1)
        self.active = [False] * len(tracks)
        self.episode = [0] * len(tracks)
        for i in range(len(tracks)):
            self._reserve(i)

    def _reserve(self, i):
        track = self.tracks[i]
        reserved = sum(a and t == track for a,t in zip(self.active, self.tracks))
        self.active[i] = self.completed[track]+reserved < self.quota

    def reset(self, i):
        if self.active[i]:
            self.completed[self.tracks[i]] += 1
        self.active[i] = False
        self.episode[i] += 1
        self._reserve(i)


def heading_feedback(lateral_error, heading, lookahead=5., gain=.4, limit=1.):
    """World +X lookahead bearing minus current world heading, wrapped to [-pi,pi]."""
    import torch
    bearing = torch.atan2(lateral_error, torch.full_like(lateral_error, lookahead))
    error = torch.atan2(torch.sin(bearing-heading), torch.cos(bearing-heading))
    return (gain*error).clamp(-limit, limit)


def input_validity(depth, orientation=None, angular_velocity=None):
    import torch
    x = depth.reshape(depth.shape[0], -1)
    valid = torch.isfinite(x).all(1) & (x.abs().sum(1) > 0)
    for value in (orientation, angular_velocity):
        if value is not None:
            valid &= torch.isfinite(value.reshape(value.shape[0], -1)).all(1)
    return valid


def transition_diagnostic(predictions, target, start, stop, times, positions, boundary_time,
                          boundary_x, previous_start=0, persistence=PERSISTENCE_TICKS,
                          complete=True, reached=True, valid=None):
    """Segment-bounded first match; only pre-boundary runs surviving the boundary anticipate.

    Persistence = consecutive classification ticks, all valid; first match is
    independent. A reverted early burst can never satisfy within-segment matching.
    """
    n = len(predictions); stop = min(stop, n)
    valid = [True]*n if valid is None else list(valid)
    matches = [predictions[i] == target and valid[i] for i in range(n)]
    first = next((i for i in range(start, stop) if matches[i]), None) if reached else None
    runs = []; i = max(0, previous_start)
    while i < stop:
        if not matches[i]:
            i += 1; continue
        a = i
        while i < stop and matches[i]: i += 1
        runs.append((a,i))
    transients = [(a,b) for a,b in runs if b <= start]
    persistent = next((a for a,b in runs if b-a >= persistence and b > start and
                       (a >= start or b >= start+1)), None) if reached else None
    offset = lambda i, values, reference: None if i is None or reference is None else float(values[i]-reference)
    return dict(metric_version=VERSION, first_match_index=first,
        first_match_delay_ticks=None if first is None else first-start,
        persistent_switch_index=persistent,
        delta_t_switch_s=offset(persistent,times,boundary_time),
        switch_distance_offset_m=offset(persistent,positions,boundary_x),
        first_match_time_offset_s=offset(first,times,boundary_time),
        first_match_distance_offset_m=offset(first,positions,boundary_x),
        transient_early_runs=[dict(start=a,end_exclusive=b,reversion_index=b,
            delta_t_s=offset(a,times,boundary_time),distance_offset_m=offset(a,positions,boundary_x)) for a,b in transients],
        reversion_count=sum(b < stop for a,b in runs),
        missed_transition=(first is None) if reached and complete else None,
        observed_no_match=bool(reached and first is None),
        recognition_status=("unreached" if not reached else "matched" if first is not None else
                            "missed" if complete else "censored_no_match"),
        observation_censored=not complete, persistence_ticks=persistence,
        transition_window_radius_ticks=TRANSITION_RADIUS)


def boundary_events(positions, times, sequence, length, outcome, canonicalize=lambda x:x):
    """Observed geometric events, not whole-episode success assigned to planned pairs.

    Traversal success means crossing the boundary then reaching the next cell end
    (or course finish for the final cell). A crossed-but-unfinished cell is censored
    on timeout/step cap, and failed on a recorded failure in that cell.
    """
    if not positions:
        return []
    events=[]
    for j in range(len(sequence)-1):
        x=(j+1)*length
        approach=next((i for i,p in enumerate(positions) if x-APPROACH_DISTANCE_M <= p < x),None)
        crossing=next((i for i in range(1,len(positions)) if positions[i-1] < x <= positions[i]),None)
        left_censored=positions[0] >= x
        end=next((i for i,p in enumerate(positions) if crossing is not None and i >= crossing and p >= x+length),None)
        failed=outcome in ("termination","left_column")
        if crossing is not None:
            i=crossing; f=(x-positions[i-1])/(positions[i]-positions[i-1])
            crossing_time=times[i-1]+f*(times[i]-times[i-1])
            status=("traversed" if end is not None or (j==len(sequence)-2 and outcome=="course_complete") else
                    "failed_after_crossing" if failed and x <= positions[-1] < x+length else
                    "returned_before_boundary" if positions[-1] < x else "crossed_censored")
        else:
            crossing_time=None
            status=("left_censored" if left_censored else "failed_approach" if failed and x-APPROACH_DISTANCE_M <= positions[-1] < x else
                    "approach_censored" if approach is not None else "unreached")
        events.append(dict(metric_version=VERSION,boundary_segment=j,boundary_position_m=x,
            transition_pair=f"{canonicalize(sequence[j])}->{canonicalize(sequence[j+1])}",
            raw_transition_pair=f"{sequence[j]}->{sequence[j+1]}",approach_index=approach,
            approach_timestamp_s=None if approach is None else times[approach],
            crossing_index=crossing,boundary_timestamp_s=crossing_time,
            traversal_outcome=status,traversal_success=(True if status=="traversed" else
                False if status in ("failed_approach","failed_after_crossing") else None),
            terminal_reason=outcome,approach_distance_m=APPROACH_DISTANCE_M))
    return events
