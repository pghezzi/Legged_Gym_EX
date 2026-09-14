from legged_gym import LEGGED_GYM_ROOT_DIR

import json
import torch
import os
import time
from pathlib import Path
from rsl_rl.utils.training_cost import WallTimer, stage_accounting, provenance_map, write_cost_record
from legged_gym.utils.dataset_provenance import (
    FRAME_FIELDS, raw_provenance, flattened_provenance, validate_dataset_provenance,
    validate_partitions, token,
)


GROUP_METADATA_KEYS = (
    "terrain_seed_ids", "terrain_seeds", "terrain_seed",
    "terrain_seed_id", "terrain_realization_seed", "terrain_realization_ids",
    "track_seed_ids", "track_seeds", "track_seed",
    "track_seed_id", "track_ids",
)

def data_flattening(tensor):
    #data is stacked as episodes, envs, shape. This swaps it to ensure 
    return tensor.transpose(0, 1).flatten(start_dim=0, end_dim=1)

def labels_flattening(labels):
    # labels: [episodes][envs] -> [envs * episodes]
    return [label for env in zip(*labels) for label in env]

def sample_envs(depth_images, base_rpy, base_ang_vel, terrain_labels, fraction=0.5,
                seed=None, return_indices=False):
    n_envs = depth_images.shape[1]
    n_keep = int(n_envs * fraction)

    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)

    idx = torch.randperm(n_envs, generator=g)[:n_keep]
    idx = idx.sort().values  # optional: preserve original ordering

    terrain_labels = [
        [row[i] for i in idx.tolist()]
        for row in terrain_labels
    ]

    sampled = (
        depth_images[:, idx],
        base_rpy[:, idx],
        base_ang_vel[:, idx],
        terrain_labels
    )
    return (*sampled, idx) if return_indices else sampled


def _python_scalar(value):
    if torch.is_tensor(value) and value.numel() == 1:
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _seed_sets_by_env(raw_value, n_envs):
    """Normalize scalar/[N]/[T,N] seed metadata to one seed set per env."""
    if raw_value is None:
        return None
    if not torch.is_tensor(raw_value) and hasattr(raw_value, "tolist"):
        raw_value = raw_value.tolist()
    if torch.is_tensor(raw_value):
        value = raw_value.detach().cpu()
        if value.ndim == 0:
            return [{value.item()} for _ in range(n_envs)]
        if value.shape[-1] == n_envs:
            rows = value.reshape(-1, n_envs).tolist()
            return [{_python_scalar(row[env]) for row in rows} for env in range(n_envs)]
        if value.ndim == 1 and value.numel() == n_envs:
            return [{_python_scalar(item)} for item in value.tolist()]
        return None
    if isinstance(raw_value, (list, tuple)):
        if len(raw_value) == n_envs and not any(
                isinstance(item, (list, tuple)) for item in raw_value):
            return [{_python_scalar(item)} for item in raw_value]
        if raw_value and all(isinstance(row, (list, tuple)) and len(row) == n_envs
                             for row in raw_value):
            return [{_python_scalar(row[env]) for row in raw_value}
                    for env in range(n_envs)]
    return None


def _environment_groups(selected_env_ids, seed_sets=None):
    """Return env groups, joining environments that share any terrain seed."""
    selected_env_ids = [int(value) for value in selected_env_ids]
    if seed_sets is None:
        return [{"env_ids": [env_id], "seed_ids": [], "group_id": f"env:{env_id}"}
                for env_id in selected_env_ids], "environment"

    parent = list(range(len(selected_env_ids)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owner = {}
    for position, seeds in enumerate(seed_sets):
        for terrain_seed in seeds:
            token = repr(terrain_seed)
            if token in owner:
                union(position, owner[token])
            else:
                owner[token] = position
    components = {}
    for position in range(len(selected_env_ids)):
        components.setdefault(find(position), []).append(position)
    groups = []
    for positions in components.values():
        env_ids = [selected_env_ids[position] for position in positions]
        seeds_by_token = {repr(seed): seed for position in positions
                          for seed in seed_sets[position]}
        seed_tokens = sorted(seeds_by_token)
        seeds = [seeds_by_token[token] for token in seed_tokens]
        groups.append({"env_ids": env_ids, "seed_ids": seeds,
                       "group_id": "terrain_seed:" + "|".join(seed_tokens)})
    return groups, "terrain_seed"


def split_environment_groups(groups, seed=42):
    """Seeded 60/20/20 partition over indivisible environment/terrain groups."""
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(groups), generator=generator).tolist()
    n_train = int(len(groups) * 0.6)
    n_val = int(len(groups) * 0.2)
    partitions = {
        "train": [groups[index] for index in order[:n_train]],
        "val": [groups[index] for index in order[n_train:n_train + n_val]],
        "test": [groups[index] for index in order[n_train + n_val:]],
    }
    id_sets = [{group["group_id"] for group in partitions[name]}
               for name in ("train", "val", "test")]
    assert id_sets[0].isdisjoint(id_sets[1])
    assert id_sets[0].isdisjoint(id_sets[2])
    assert id_sets[1].isdisjoint(id_sets[2])
    return partitions

def get_data_raw(load_file, frac=0.1, seed=42, return_metadata=False, timings=None, allow_legacy=False):
    loading_timer = WallTimer().start()
    torch_load = torch.load(load_file, map_location="cpu", weights_only=False)
    loading_s = loading_timer.stop()
    if timings is not None:
        timings["data_loading_s"] += loading_s
    preprocessing_started = time.perf_counter()
    depth_images = torch_load["depth_images"]
    base_rpy = torch_load["base_rpy"]
    base_ang_vel = torch_load["base_ang_vel"]
    terrain_labels = torch_load["terrain_name"]
    source_n_envs = depth_images.shape[1]
    if base_rpy.shape[:2] != depth_images.shape[:2] or base_ang_vel.shape[:2] != depth_images.shape[:2]:
        raise ValueError("Raw sensor dimensions do not align")
    if len(terrain_labels) != len(depth_images) or any(len(row) != source_n_envs for row in terrain_labels):
        raise ValueError("Raw labels do not align")
    provenance = raw_provenance(torch_load, os.path.abspath(os.fspath(load_file)), allow_legacy)
    original_envs = provenance["original_env_ids"][0]
    unique_envs = torch.unique(original_envs, sorted=True)
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0,1]")
    keep = unique_envs[torch.randperm(len(unique_envs), generator=torch.Generator().manual_seed(seed))[:int(len(unique_envs)*frac)]]
    selected_env_ids = torch.nonzero(torch.isin(original_envs, keep), as_tuple=False).flatten()
    if not len(selected_env_ids):
        raise ValueError("Environment subsampling selected no environments")
    depth_images, base_rpy, base_ang_vel = (x[:, selected_env_ids] for x in (depth_images, base_rpy, base_ang_vel))
    terrain_labels = [[row[i] for i in selected_env_ids.tolist()] for row in terrain_labels]
    if timings is not None:
        timings["sampling_s"] += time.perf_counter() - preprocessing_started
    if not return_metadata:
        return depth_images, base_rpy, base_ang_vel, terrain_labels
    seed_key, seed_sets = None, None
    containers = [("", torch_load)]
    if isinstance(torch_load.get("metadata"), dict):
        containers.append(("metadata.", torch_load["metadata"]))
    for prefix, container in containers:
        for key in GROUP_METADATA_KEYS:
            normalized = _seed_sets_by_env(container.get(key), source_n_envs)
            if normalized is not None and any(value is None or
                    (isinstance(value, float) and not torch.isfinite(torch.tensor(value)))
                    for values in normalized for value in values):
                normalized = None
            if normalized is not None:
                seed_key = prefix + key
                seed_sets = [normalized[index] for index in selected_env_ids.tolist()]
                break
        if seed_sets is not None:
            break
    return depth_images, base_rpy, base_ang_vel, terrain_labels, {
        "source_file": os.path.abspath(os.fspath(load_file)),
        "selected_env_ids": selected_env_ids.tolist(),
        "seed_metadata_key": seed_key,
        "seed_sets": seed_sets,
        "provenance": {**provenance, **{key: provenance[key][:, selected_env_ids] for key in FRAME_FIELDS}},
    }

def calibration_data(*args, **kwargs):
    calibration_depth, calibration_rpy, calibration_ang_vel, _, source = get_data_raw(*args, return_metadata=True, **kwargs)
    started = time.perf_counter()
    result = {
        "depth_images": data_flattening(calibration_depth),
        "orientation_rpy": data_flattening(calibration_rpy),
        "angular_velocity": data_flattening(calibration_ang_vel),
        **flattened_provenance(source["provenance"], torch.arange(calibration_depth.shape[1])),
    }

    if kwargs.get("timings") is not None:
        kwargs["timings"]["sampling_s"] += time.perf_counter() - started
    return result

def train_val_test_data(load_file, frac=0.1, seed=42, return_manifest=False, timings=None,
                        allow_legacy=False, prepared=None, assigned=None):
    depth_images, base_rpy, base_ang_vel, terrain_labels, source = prepared or get_data_raw(
        load_file, frac=frac, seed=seed, return_metadata=True, timings=timings, allow_legacy=allow_legacy)
    splitting_started = time.perf_counter()
    episode_length = int(depth_images.shape[0])
    provenance = source["provenance"]
    original_groups = [token(provenance["source_id"], int(env)) for env in provenance["original_env_ids"][0]]
    terrain_groups = ([{token("terrain", value) for value in values} for values in source["seed_sets"]]
                      if source["seed_sets"] is not None else [set() for _ in original_groups])
    groups, group_type = _environment_groups(source["selected_env_ids"],
        [{original} | terrain for original, terrain in zip(original_groups, terrain_groups)])
    partitions = assigned if assigned is not None else split_environment_groups(groups, seed=seed)
    selected_position = {env_id: position
                         for position, env_id in enumerate(source["selected_env_ids"])}

    def build_split(split_groups):
        env_ids = sorted(env_id for group in split_groups for env_id in group["env_ids"])
        positions = torch.tensor([selected_position[env_id] for env_id in env_ids], dtype=torch.long)
        split_labels = [[row[position] for position in positions.tolist()]
                        for row in terrain_labels]
        result = {
            "depth_images": data_flattening(depth_images[:, positions]),
            "orientation_rpy": data_flattening(base_rpy[:, positions]),
            "angular_velocity": data_flattening(base_ang_vel[:, positions]),
            "per_eps": episode_length,
            "labels": labels_flattening(split_labels),
            **flattened_provenance(provenance, positions),
            "source_environment_group_ids": sorted({original_groups[i] for i in positions.tolist()}),
            "terrain_group_ids": sorted({value for i in positions.tolist() for value in terrain_groups[i]}),
            "terrain_provenance_available": source["seed_sets"] is not None,
        }
        count = len(result["labels"])
        assert result["depth_images"].shape[0] == count
        assert result["orientation_rpy"].shape[0] == count
        assert result["angular_velocity"].shape[0] == count
        assert count == episode_length * len(env_ids)
        validate_dataset_provenance(result, allow_legacy)
        return result, env_ids

    datasets, split_manifest = {}, {}
    for name in ("train", "val", "test"):
        datasets[name], env_ids = build_split(partitions[name])
        split_manifest[name] = {
            "source_env_ids": sorted({int(provenance["original_env_ids"][0, selected_position[i]]) for i in env_ids}),
            "selected_column_ids": env_ids,
            "group_ids": [group["group_id"] for group in partitions[name]],
            "terrain_seed_ids": sorted({value for env_id in env_ids
                for value in source["seed_sets"][selected_position[env_id]]}, key=repr) if source["seed_sets"] is not None else [],
            "num_frames": len(datasets[name]["labels"]),
        }
    env_sets = [set(split_manifest[name]["source_env_ids"])
                for name in ("train", "val", "test")]
    seed_sets = [set(split_manifest[name]["terrain_seed_ids"])
                 for name in ("train", "val", "test")]
    assert env_sets[0].isdisjoint(env_sets[1])
    assert env_sets[0].isdisjoint(env_sets[2])
    assert env_sets[1].isdisjoint(env_sets[2])
    assert seed_sets[0].isdisjoint(seed_sets[1])
    assert seed_sets[0].isdisjoint(seed_sets[2])
    assert seed_sets[1].isdisjoint(seed_sets[2])
    manifest = {
        "split_seed": int(seed), "fraction": float(frac),
        "source_file": source["source_file"], "group_type": "terrain_seed" if source["seed_sets"] is not None else "source_environment",
        "seed_metadata_key": source["seed_metadata_key"],
        "selected_source_env_ids": sorted(set(provenance["original_env_ids"][0].tolist())),
        "selected_raw_columns": source["selected_env_ids"],
        "splits": split_manifest,
        "capture_id": provenance["capture_id"], "source_id": provenance["source_id"],
        "provenance_checks": validate_partitions(datasets, allow_legacy),
    }
    result = (datasets["train"], datasets["val"], datasets["test"])
    if timings is not None:
        timings["splitting_s"] += time.perf_counter() - splitting_started
    return (*result, manifest) if return_manifest else result

def merge_sets(*datasets):
    merged = {}
    for key in datasets[0]:
        if isinstance(datasets[0][key], list):
            merged[key] = sum((d[key] for d in datasets), [])
        elif key == "per_eps":
            merged[key] = datasets[0][key] if all(d[key] == datasets[0][key] for d in datasets) else 0
        elif key in ("provenance_verified", "terrain_provenance_available"):
            merged[key] = all(d[key] for d in datasets)
        else:
            
            merged[key] = torch.cat([d[key] for d in datasets], dim=0)
    if "sequence_ids" in merged:
        envs, steps = merged["original_env_ids"].tolist(), merged["control_step_indices"].tolist()
        order = sorted(range(len(merged["labels"])), key=lambda i: (
            merged["source_ids"][i], envs[i], steps[i]))
        indices = torch.tensor(order, dtype=torch.long)
        for key in (*FRAME_FIELDS, "capture_ids", "source_ids", "sequence_ids", "labels",
                    "depth_images", "orientation_rpy", "angular_velocity"):
            value = merged[key]
            merged[key] = [value[i] for i in order] if isinstance(value, list) else value[indices]
    return merged


def compile_sources(files, frac=.1, seed=42, timings=None, allow_legacy=False):
    """Partition connected source/env/terrain groups jointly, never file-by-file."""
    prepared = [get_data_raw(file, frac, seed, True, timings, allow_legacy) for file in files]
    owners, constraints = [], []
    for file_index, data in enumerate(prepared):
        source, p = data[4], data[4]["provenance"]
        for column, selected in enumerate(source["selected_env_ids"]):
            owners.append((file_index, selected))
            keys = {token(p["source_id"], int(p["original_env_ids"][0, column]))}
            if source["seed_sets"] is not None:
                keys.update(token("terrain", value) for value in source["seed_sets"][column])
            constraints.append(keys)
    groups, _ = _environment_groups(range(len(owners)), constraints)
    if len(groups) < 5:
        raise ValueError("Need at least five independent source/environment/terrain groups for nonempty 60/20/20 splits")
    partitions = split_environment_groups(groups, seed)
    results = []
    for file_index, (file, data) in enumerate(zip(files, prepared)):
        assigned = {name: [{**group, "env_ids": [owners[i][1] for i in group["env_ids"] if owners[i][0] == file_index]}
                           for group in values if any(owners[i][0] == file_index for i in group["env_ids"])]
                    for name, values in partitions.items()}
        results.append(train_val_test_data(file, frac, seed, True, timings, allow_legacy, data, assigned))
    merge_timer = WallTimer().start()
    merged = [merge_sets(*(result[i] for result in results)) for i in range(3)]
    checks = validate_partitions(dict(zip(("train", "val", "test"), merged)), allow_legacy)
    if timings is not None:
        timings["merging_s"] = merge_timer.stop()
    return merged, [result[3] for result in results], checks

# reshaped removed as this is now handled by the data generator

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Compile depth ang data")
    parser.add_argument("--files", nargs="+", help="files to compile")
    parser.add_argument("--calibration", type=str, default=None,
                        help="calibration file (optional)")
    parser.add_argument("--frac", type=float, default=0.1,
                        help="amount of envs used from total(optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for environment subsampling and the group split")
    parser.add_argument("--allow-legacy-provenance", action="store_true",
                        help="Unverified fallback: treat each raw column as one pseudo-episode/environment")
    parser.add_argument("--calibration_label", type=str, default="random_uniform",
                        help="label for calibration")
    

    args = parser.parse_args()
    total_timer = WallTimer().start()
    timings = dict(data_loading_s=0.0, sampling_s=0.0, splitting_s=0.0)

    get_data_args = {
        "frac": args.frac,
        "seed": args.seed,
        "timings": timings,
        "allow_legacy": args.allow_legacy_provenance,
    }

    if args.calibration:
        calibration = calibration_data(args.calibration, **get_data_args)
        calibration["labels"] = [args.calibration_label] * calibration["depth_images"].shape[0]   
    
    if args.calibration and os.path.abspath(args.calibration) in {
            os.path.abspath(file) for file in args.files}:
        raise ValueError("calibration must use a separate source file to prevent split overlap")

    (train, val, test), source_manifests, provenance_checks = compile_sources(args.files, **get_data_args)
    if args.calibration:
        provenance_checks = validate_partitions(
            {"train": train, "val": val, "test": test, "calibration": calibration}, args.allow_legacy_provenance)

    print(train["depth_images"].shape)
    print(val["depth_images"].shape)
    print(test["depth_images"].shape)

    out_dir = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/{timestamp}_frac_{str(args.frac).replace('.','_')}"
    os.makedirs(out_dir, exist_ok=True)

    serialization_timer = WallTimer().start()
    with open(os.path.join(out_dir,"parsed_arguments.txt"), "w") as f:
        f.write("Parsed Arguments\n")
        f.write("================\n")
        f.write(f"Files: {args.files}\n")
        f.write(f"Calibration: {args.calibration}\n")
        f.write(f"Frac: {args.frac}\n")
        f.write(f"Split seed: {args.seed}\n")
    
    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val,   os.path.join(out_dir, "val.pt"))
    torch.save(test,  os.path.join(out_dir, "test.pt"))
    if args.calibration:
        torch.save(calibration,  os.path.join(out_dir, "calibration.pt"))

    serialization_s = serialization_timer.stop()
    manifest_timer = WallTimer().start()
    split_manifest = {
        "provenance_version": 1, "provenance_checks": provenance_checks,
        "legacy_fallback_enabled": args.allow_legacy_provenance,
        "split_seed": args.seed,
        "fraction": args.frac,
        "source_files": [os.path.abspath(file) for file in args.files],
        "calibration_source_file": (os.path.abspath(args.calibration)
                                    if args.calibration else None),
        "sources": source_manifests,
        "splits": {},
    }
    split_group_sets = {}
    for split_name in ("train", "val", "test"):
        entries = []
        qualified_group_ids = []
        for source in source_manifests:
            source_split = source["splits"][split_name]
            entries.append({"source_file": source["source_file"], **source_split})
            qualified_group_ids.extend(
                f"{source['source_file']}::{group_id}"
                for group_id in source_split["group_ids"])
        split_manifest["splits"][split_name] = {
            "sources": entries, "group_ids": qualified_group_ids,
            "num_frames": len({"train": train, "val": val, "test": test}[split_name]["labels"]),
        }
        split_group_sets[split_name] = set(qualified_group_ids)
    assert split_group_sets["train"].isdisjoint(split_group_sets["val"])
    assert split_group_sets["train"].isdisjoint(split_group_sets["test"])
    assert split_group_sets["val"].isdisjoint(split_group_sets["test"])
    timings["splitting_s"] += manifest_timer.stop()
    serialization_timer = WallTimer().start()
    with open(os.path.join(out_dir, "split_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(split_manifest, stream, indent=2)

    serialization_s += serialization_timer.stop()
    total_s = total_timer.stop()
    record = {
        "schema_version": 2, "run_type": "dataset_compilation",
        "dataset_path": str(Path(out_dir).resolve()),
        "source_files": list(dict.fromkeys(split_manifest["source_files"] +
            ([split_manifest["calibration_source_file"]] if args.calibration else []))),
        "split_manifest": str(Path(out_dir, "split_manifest.json").resolve()),
        "data_env_steps": 0, "additional_locomotion_policy_env_steps": 0,
        "total_post_specialist_env_steps": 0, "gpu_count": 0, "gpu_model": None,
        **timings,
        **stage_accounting(total_s, 0, setup_s=timings["data_loading_s"],
            preprocessing_s=sum(timings[key] for key in ("sampling_s", "splitting_s", "merging_s")),
            artifact_serialization_s=serialization_s),
        "notes": ["CPU compilation from raw source loading through split/artifact serialization; sidecar writing excluded.",
                  "data_loading_s is within setup_s; sampling_s, splitting_s and merging_s are within preprocessing_s.",
                  "The compiled dataset is a shared preparation artifact; charge it once per integration alternative."],
    }
    record["metric_status"] = provenance_map(record)
    write_cost_record(Path(out_dir) / "training_cost.json", record)
    print(out_dir)
    
