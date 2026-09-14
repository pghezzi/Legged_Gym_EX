"""Observation provenance shared by capture, compilation and offline replay."""
import json
import uuid
import warnings

import torch

FRAME_FIELDS = ("original_env_ids", "episode_ids", "control_step_indices", "frame_indices")


def token(*parts):
    return json.dumps(parts, separators=(",", ":"))


class CaptureProvenance:
    def __init__(self, num_envs, device):
        self.capture_id = uuid.uuid4().hex
        self.env_ids = torch.arange(num_envs, device=device)
        self.episodes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.logs = {key: [] for key in FRAME_FIELDS}

    def reset(self, env_ids):
        self.episodes[env_ids] += 1

    def capture(self, control_step):
        values = (self.env_ids, self.episodes, torch.full_like(self.env_ids, control_step),
                  torch.full_like(self.env_ids, len(self.logs["frame_indices"])))
        for key, value in zip(FRAME_FIELDS, values):
            self.logs[key].append(value.detach().cpu().clone())

    def export(self):
        return {"provenance_version": 1, "capture_id": self.capture_id,
                "source_id": self.capture_id, "episode_boundaries_verified": True,
                **{key: torch.stack(values) for key, values in self.logs.items()}}


def raw_provenance(raw, source_file, allow_legacy=False):
    t, n = raw["depth_images"].shape[:2]
    if not t or not n:
        raise ValueError("Raw collection must contain observations and environments")
    p = raw.get("provenance")
    if p is None:
        if not allow_legacy:
            raise ValueError("Missing observation provenance; explicitly enable legacy fallback: " + str(source_file))
        warnings.warn("Legacy provenance: episode boundaries and original environment groups unverified: " + str(source_file))
        p = {"capture_id": "legacy:" + str(source_file), "source_id": "legacy:" + str(source_file),
             "episode_boundaries_verified": False,
             "original_env_ids": torch.arange(n).repeat(t, 1), "episode_ids": torch.zeros(t, n, dtype=torch.long),
             "control_step_indices": torch.arange(t)[:, None].repeat(1, n),
             "frame_indices": torch.arange(t)[:, None].repeat(1, n)}
    p = dict(p)
    for key in FRAME_FIELDS:
        value = torch.as_tensor(p[key])
        if value.shape != (t, n) or value.dtype not in (torch.int32, torch.int64) or (value < 0).any():
            raise ValueError("Invalid per-observation provenance: " + key)
        p[key] = value
    if not p.get("capture_id") or not p.get("source_id"):
        raise ValueError("Missing capture/source identity")
    if not p.get("episode_boundaries_verified", False) and not allow_legacy:
        raise ValueError("Unverified episode boundaries require explicit legacy fallback")
    for column in range(n):
        if t and not (p["original_env_ids"][:, column] == p["original_env_ids"][0, column]).all():
            raise ValueError("A raw column mixes original environments")
        for key in ("control_step_indices", "frame_indices"):
            if not (p[key][1:, column] > p[key][:-1, column]).all():
                raise ValueError("Non-chronological raw provenance: " + key)
        if not (p["episode_ids"][1:, column] >= p["episode_ids"][:-1, column]).all():
            raise ValueError("Episode IDs run backwards")
    return p


def flattened_provenance(p, positions):
    result = {key: p[key][:, positions].T.flatten().clone() for key in FRAME_FIELDS}
    n = len(result["episode_ids"])
    result["capture_ids"] = [p["capture_id"]] * n
    result["source_ids"] = [p["source_id"]] * n
    result["sequence_ids"] = [token(p["capture_id"], env, ep) for env, ep in
                              zip(result["original_env_ids"].tolist(), result["episode_ids"].tolist())]
    result["provenance_verified"] = bool(p.get("episode_boundaries_verified", False))
    return result


def validate_dataset_provenance(data, allow_legacy=False):
    n = len(data["labels"])
    for key in ("depth_images", "orientation_rpy", "angular_velocity"):
        if key in data and len(data[key]) != n:
            raise ValueError("Observation/label count mismatch: " + key)
    keys = (*FRAME_FIELDS, "capture_ids", "source_ids", "sequence_ids")
    if not all(key in data for key in keys):
        if not allow_legacy:
            raise ValueError("Dataset lacks explicit provenance; use --allow-legacy-provenance only for unverified replay")
        return {"verified": False, "reason": "legacy metadata unavailable", "num_frames": n}
    if any(len(data[key]) != n for key in keys):
        raise ValueError("Provenance lengths differ from labels")
    for key in FRAME_FIELDS:
        values = torch.as_tensor(data[key])
        if values.shape != (n,) or values.dtype not in (torch.int32, torch.int64) or (values < 0).any():
            raise ValueError("Invalid compiled provenance field: " + key)
    verified = bool(data.get("provenance_verified", False))
    if not verified and not allow_legacy:
        raise ValueError("Dataset provenance is explicitly unverified")
    fields = {key: torch.as_tensor(data[key]).tolist() for key in FRAME_FIELDS}
    sequences, last, seen_obs = {}, {}, set()
    previous, closed = None, set()
    for i in range(n):
        env, ep, step, frame = (fields[k][i] for k in FRAME_FIELDS)
        capture, source, seq = data["capture_ids"][i], data["source_ids"][i], data["sequence_ids"][i]
        identity = (capture, env, ep)
        if seq != token(*identity):
            raise ValueError("Sequence ID does not encode capture/environment/episode")
        if seq != previous:
            if seq in closed:
                raise ValueError("Noncontiguous sequence; sort whole episodes before replay")
            if previous is not None:
                closed.add(previous)
            previous = seq
        key = (source, env)
        if key in last and (step <= last[key][0] or frame <= last[key][1] or ep < last[key][2]):
            raise ValueError("Nonchronological environment observations")
        last[key] = (step, frame, ep)
        observation = (capture, env, frame)
        if observation in seen_obs:
            raise ValueError("Duplicate observation")
        seen_obs.add(observation)
        sequences[seq] = identity
    return {"verified": verified, "num_frames": n, "num_sequences": len(sequences),
            "metadata_lengths_aligned": True, "chronological": True, "contiguous_sequences": True,
            "source_environment_groups": [token(*key) for key in sorted(last)]}


def validate_partitions(datasets, allow_legacy=False):
    checks = {name: validate_dataset_provenance(data, allow_legacy) for name, data in datasets.items()}
    for i, (name, data) in enumerate(datasets.items()):
        for other_name, other in list(datasets.items())[i+1:]:
            for key in ("source_environment_group_ids", "terrain_group_ids"):
                if set(data.get(key, ())) & set(other.get(key, ())):
                    raise ValueError("Split leakage in " + key + ": " + name + "/" + other_name)
            if set(checks[name].get("source_environment_groups", ())) & set(checks[other_name].get("source_environment_groups", ())):
                raise ValueError("Original source/environment appears in multiple splits")
    return {"datasets": checks, "environment_disjoint_verified": all(c["verified"] for c in checks.values()),
            "terrain_disjoint_checked": all(d.get("terrain_provenance_available", False) for d in datasets.values())}
