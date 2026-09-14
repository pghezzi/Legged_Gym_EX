"""Self-contained frozen-paper figure inputs and plotting (no models/datasets)."""
from pathlib import Path
import hashlib
import json
from .paper_figure_paths import figure_path, write_figure_index

import numpy as np
import torch
import torch.nn.functional as F

ARCHITECTURES = ("feature_nn", "raw_depth_nn")
METHODS = ("instantaneous", "ema", "bayes")
PLOT_CONFIG = {"dpi": 200, "confusion_vmin": 0., "confusion_vmax": 1.,
               "legacy_dpi": 300, "timeline_padding": 10,
               "sampling_seed": 42, "images_per_class": 100, "transition_samples": 100,
               "disagreement_fraction": .5, "thumbnail_count": 8,
               "delay_metric": "mean_matched_transition_delay_frames_v2"}


def coverage_order(indices, data, rng, strata=None):
    """Randomized round-robin source/env groups, then sequence/time strata."""
    groups, tracks = {}, {}
    def value(key, index, default):
        values = data.get(key)
        if values is None:
            return default
        result = values[index]
        return result.item() if torch.is_tensor(result) else result
    for index in indices:
        group = (str(value("source_ids", index, "unknown")),
                 str(value("original_env_ids", index, value("sequence_ids", index, "unknown"))))
        seq = str(value("sequence_ids", index, "unknown"))
        tracks.setdefault((group, seq), []).append(index)
    for (group, seq), values in tracks.items():
        # Time bins are local to each sequence, not offsets in a merged dataset.
        for rank, index in enumerate(sorted(values)):
            sub = (seq, min(9, rank*10//len(values)), str(strata[index]) if strata is not None else "")
            groups.setdefault(group, {}).setdefault(sub, []).append(index)
    queues = {}
    for group, buckets in groups.items():
        keys = list(buckets)
        rng.shuffle(keys)
        for values in buckets.values():
            rng.shuffle(values)
        queues[group] = (keys, buckets)
    keys = list(queues)
    rng.shuffle(keys)
    while keys:
        next_keys = []
        for group in keys:
            subkeys, buckets = queues[group]
            sub = subkeys.pop(0)
            yield buckets[sub].pop()
            if buckets[sub]:
                subkeys.append(sub)
            if subkeys:
                next_keys.append(group)
        keys = next_keys


def sample_transitions(data, ids, runs, count=100, seed=42, padding=10, disagreement_fraction=.5):
    """Half disagreement-enriched, half unrestricted coverage; no replacement."""
    truth = list(data["labels"])
    emitted = torch.stack([run["trace"][method+"_selected"] for run in runs for method in METHODS])
    disagree = (emitted != emitted[:1]).any(0)
    candidates, pairs = {}, {}
    start = 0
    while start < len(truth):
        end = start + 1
        while end < len(truth) and ids[end] == ids[start]:
            end += 1
        changes = [i for i in range(start+1, end) if truth[i] != truth[i-1]]
        for j, frame in enumerate(changes):
            left = j if j+1 < len(changes) else max(0, j-1)
            right = min(left+1, len(changes)-1)
            boundaries = changes[left:right+1]
            # Include a frame before the first selected change, even when
            # consecutive terrain segments contain only one observation.
            lo = max(start, boundaries[0]-padding, changes[left-1] if left else start)
            hi = min(end, boundaries[-1]+padding+1, changes[right+1] if right+1 < len(changes) else end)
            # Enrichment is measured locally around the focal transition.
            disagreement = float(disagree[max(start, frame-padding):min(end, frame+padding+1)].float().mean())
            candidates[frame] = {"start": lo, "end": hi, "boundaries": boundaries,
                                 "sequence_id": ids[start], "focus_frame": frame,
                                 "from_label": truth[frame-1], "target_label": truth[frame],
                                 "disagreement_fraction": disagreement}
            pairs[frame] = (truth[frame-1], truth[frame])
        start = end
    rng = np.random.default_rng(seed)
    coverage_data = {**data, "sequence_ids": ids}
    enriched = [i for i in candidates if candidates[i]["disagreement_fraction"] > 0]
    selected = []
    quota = min(int(round(count*disagreement_fraction)), len(enriched))
    for index in coverage_order(enriched, coverage_data, rng, pairs):
        if len(selected) >= quota:
            break
        selected.append(index)
    enriched_selected = set(selected)
    for index in coverage_order([i for i in candidates if i not in enriched_selected], coverage_data, rng, pairs):
        if len(selected) >= count:
            break
        selected.append(index)
    result = [{**candidates[i], "sample_id": j,
               "selection_pool": "disagreement" if i in enriched_selected else "coverage"}
              for j, i in enumerate(selected)]
    return result, {"seed": seed, "requested": count, "available_transitions": len(candidates),
                    "available_disagreement_transitions": len(enriched), "selected": len(result),
                    "disagreement_quota": quota,
                    "rule": "Seeded source/env round-robin; sequence/time/terrain-pair strata; half disagreement quota, remainder unrestricted coverage; unique focal transitions"}


def transition_thumbnails(data, selections, max_count=8):
    """Cache only thumbnails used in each timeline, aligned to global row IDs."""
    records = []
    for selected in selections:
        lo, hi = selected["start"], selected["end"]
        indices = {lo, hi-1, selected.get("focus_frame", selected["boundaries"][0])}
        for boundary in selected["boundaries"]:
            indices.update((max(lo, boundary-1), boundary))
        for index in np.linspace(lo, hi-1, max_count, dtype=int):
            if len(indices) >= max_count:
                break
            indices.add(int(index))
        rows = sorted(indices)
        images = data["depth_images"][rows].detach().cpu().float()
        if images.ndim == 3:
            images = images[:, None]
        images = F.interpolate(images, size=(48, 64), mode="bilinear", align_corners=False)[:, 0]
        meta = {key: ([data[key][i].item() if torch.is_tensor(data[key]) else data[key][i] for i in rows])
                for key in ("source_ids", "capture_ids", "sequence_ids", "original_env_ids", "episode_ids", "frame_indices", "control_step_indices") if key in data}
        records.append({"sample_id": selected["sample_id"], "row_indices": rows,
                        "depth_images": images.clone(), "provenance": meta,
                        "labels": [data["labels"][i] for i in rows]})
    return records


def provenance(data, sequence_ids=None):
    n = len(data["labels"])
    keys = ("source_ids", "capture_ids", "sequence_ids", "original_env_ids", "episode_ids",
            "frame_indices", "control_step_indices")
    result = {key: data.get(key) for key in keys}
    if result["sequence_ids"] is None and sequence_ids is not None:
        result["sequence_ids"] = list(sequence_ids)
    result["unavailable"] = [key for key in keys if data.get(key) is None]
    result["provenance_verified"] = bool(data.get("provenance_verified", False))
    result["row_indices"] = torch.arange(n)
    return result


def select_timeline(truth, ids, padding=10):
    """First contiguous sequence with two changes, earliest consecutive pair."""
    start = 0
    while start < len(truth):
        end = start + 1
        while end < len(truth) and ids[end] == ids[start]:
            end += 1
        changes = [i for i in range(start + 1, end) if truth[i] != truth[i-1]]
        if len(changes) >= 2:
            return {"start": max(start, changes[0]-padding),
                    "end": min(end, changes[1]+padding+1, changes[2] if len(changes) > 2 else end),
                    "boundaries": changes[:2], "sequence_id": ids[start]}
        start = end
    return None


@torch.inference_mode()
def geometric_examples(extractor, data, classes, count_per_class=100, seed=42):
    """Reproduce the extractor's actual residual/Sobel/occupancy operations."""
    examples, unavailable = [], []
    labels = list(data["labels"])
    if not all(k in data for k in ("depth_images", "orientation_rpy", "angular_velocity")):
        return [], ["Depth/IMU arrays unavailable for geometric examples"]
    rng = np.random.default_rng(seed)
    counts = {label: 0 for label in classes}
    def candidates():
        for label in classes:
            indices = [i for i, value in enumerate(labels) if value == label]
            for index in coverage_order(indices, data, rng):
                if counts[label] >= count_per_class:
                    break
                yield label, index
    for label, index in candidates():
        raw = data["depth_images"][index:index+1]
        rpy, omega = data["orientation_rpy"][index:index+1], data["angular_velocity"][index:index+1]
        depth, valid = extractor._crop_fill_resize(extractor._as_depth_batch(raw))
        if not bool(valid.any()):
            continue
        orientation = extractor._as_imu_batch(rpy, 1, 3, "orientation_rpy")
        reference = extractor._expected_reference(depth, orientation)
        residual = depth - reference
        smooth = F.conv2d(residual[:, None], extractor.gaussian_kernel, padding=1)
        sx = F.conv2d(smooth, extractor.sobel_x_kernel, padding=1)[:, 0]
        sy = F.conv2d(smooth, extractor.sobel_y_kernel, padding=1)[:, 0]
        neighborhood = F.avg_pool2d(valid.float()[:, None], 3, stride=1, padding=1)[:, 0] > .999
        close = valid & ((depth < extractor.close_depth) | (residual < -extractor.close_residual_threshold))
        far = (depth >= extractor.far_depth) | ~valid
        features = extractor.extract_batch(raw, rpy, omega)
        h, w = depth.shape[-2:]
        cw = max(1, int(round(w*extractor.center_fraction)))
        near = max(0, h-max(1, int(round(h*extractor.near_fraction))))
        arrays = dict(raw_depth=raw.squeeze(0), processed_depth=depth[0], valid=valid[0],
                      reference=reference[0], residual=residual[0], sobel_x=sx[0], sobel_y=sy[0],
                      valid_neighborhood=neighborhood[0], strong_horizontal_edges=(sy.abs() > extractor.sobel_edge_threshold)[0] & neighborhood[0],
                      close_mask=close[0], far_or_invalid_mask=far[0], features=features[0], rpy=rpy[0], angular_velocity=omega[0])
        meta = {key: (data[key][index].item() if torch.is_tensor(data[key]) else data[key][index])
                for key in ("source_ids", "capture_ids", "sequence_ids", "original_env_ids", "episode_ids", "frame_indices", "control_step_indices") if key in data}
        examples.append({"label": label, "row_index": index, "provenance": meta,
                         "arrays": {key: value.detach().cpu().clone() for key, value in arrays.items()},
                         "near_row": near, "center_columns": [(w-cw)//2, (w-cw)//2+cw],
                         "feature_names": list(extractor.FEATURE_NAMES)})
        counts[label] += 1
        if counts[label] == 1 or counts[label] % 25 == 0:
            print(f"[paper figures] Depth examples: {label} {counts[label]}/{count_per_class}", flush=True)
    for label, count in counts.items():
        if count < count_per_class:
            unavailable.append(f"Only {count}/{count_per_class} distinct valid depth examples available for {label}")
    return examples, unavailable


def normalized_confusion(counts):
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(1, keepdims=True)
    return np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)


def validate_bundle(bundle):
    classes = bundle["class_ordering"]
    n, c = len(bundle["ordered_truth"]), len(classes)
    if bundle["schema_version"] != 1 or bundle["plot_config"]["confusion_vmin"] != 0 or bundle["plot_config"]["confusion_vmax"] != 1:
        raise ValueError("Unsupported bundle or non-shared confusion scale")
    def probability(x, length):
        x = torch.as_tensor(x)
        if x.shape != (length, c) or not torch.isfinite(x).all() or (x < 0).any():
            raise ValueError("Invalid probability trace")
        torch.testing.assert_close(x.sum(-1), torch.ones(length, dtype=x.dtype), atol=1e-5, rtol=1e-5)
    for key, value in bundle["ordered_provenance"].items():
        if key not in ("unavailable", "provenance_verified") and value is not None and len(value) != n:
            raise ValueError("Provenance/trace length mismatch")
    identities = {(r["architecture"], r["seed"]) for r in bundle["runs"]}
    if identities != {(a, s) for a in ARCHITECTURES for s in bundle["model_seeds"]} or len(bundle["runs"]) != len(identities):
        raise ValueError("Bundle does not cover both architectures and all seeds exactly once")
    rows = {(r["architecture"], r["seed"], r["temporal_method"]): r for r in bundle["experiment_2_rows"]}
    truth = torch.tensor([classes.index(label) for label in bundle["ordered_truth"]])
    for run in bundle["runs"]:
        trace = run["trace"]
        probability(trace["probabilities"], n)
        torch.testing.assert_close(trace["probabilities"], trace["logits"].softmax(-1))
        torch.testing.assert_close(trace["ema_probabilities"], trace["ema_scores"].softmax(-1))
        for method in METHODS:
            selected = trace[method + "_selected"]
            p = trace[{"instantaneous": "probabilities", "ema": "ema_probabilities", "bayes": "bayes_beliefs"}[method]]
            probability(p, n)
            if len(selected) != n or (selected < 0).any() or (selected >= c).any():
                raise ValueError("Invalid emitted-label trace")
            if method != "ema" or bundle["filter_settings"]["ema"]["change_patience"] == 1:
                torch.testing.assert_close(selected, p.argmax(-1))
            matrix = torch.bincount(truth*c+selected, minlength=c*c).reshape(c, c)
            torch.testing.assert_close(matrix, torch.as_tensor(rows[(run["architecture"], run["seed"], method)]["confusion_matrix"]))
        probability(run["structural_probabilities"], len(bundle["structural_truth"]))
        torch.testing.assert_close(run["structural_probabilities"], run["structural_logits"].softmax(-1))
        target = torch.tensor([classes.index(label) for label in bundle["structural_truth"]])
        counts = torch.bincount(target*c+run["structural_probabilities"].argmax(-1), minlength=c*c).reshape(c, c)
        row = next(r for r in bundle["experiment_1_rows"] if r["architecture"] == run["architecture"] and r["seed"] == run["seed"])
        torch.testing.assert_close(counts, torch.as_tensor(row["confusion_matrix"]))
    selections = bundle.get("timeline_selections", [bundle["timeline_selection"]] if bundle["timeline_selection"] else [])
    if "timeline_selections" in bundle and len({x["focus_frame"] for x in selections}) != len(selections):
        raise ValueError("Duplicate sampled transition")
    for timeline in selections:
        ids = bundle["ordered_provenance"]["sequence_ids"]
        chosen = ids[timeline["start"]:timeline["end"]]
        if not all(x == chosen[0] for x in chosen):
            raise ValueError("Timeline crosses episode boundary")
        boundaries = [i for i in range(timeline["start"]+1, timeline["end"])
                      if bundle["ordered_truth"][i] != bundle["ordered_truth"][i-1]]
        if boundaries != timeline["boundaries"]:
            raise ValueError("Timeline does not contain exactly the selected GT boundaries")
    for thumbnails in bundle.get("transition_thumbnails", []):
        timeline = next(t for t in selections if t["sample_id"] == thumbnails["sample_id"])
        indices = thumbnails["row_indices"]
        if len(indices) != len(thumbnails["depth_images"]) or len(indices) != len(thumbnails["labels"]):
            raise ValueError("Thumbnail lengths do not align")
        if any(not timeline["start"] <= i < timeline["end"] for i in indices):
            raise ValueError("Thumbnail outside selected timeline")
        if thumbnails["labels"] != [bundle["ordered_truth"][i] for i in indices]:
            raise ValueError("Thumbnail labels do not match timeline")
        for key, values in thumbnails["provenance"].items():
            expected = bundle["ordered_provenance"][key]
            expected = [expected[i].item() if torch.is_tensor(expected) else expected[i] for i in indices]
            if values != expected:
                raise ValueError("Thumbnail provenance differs from trace")
    return {"trace_lengths": True, "probabilities_normalized": True, "predictions_match_evaluated_confusions": True,
            "shared_confusion_scale": [0., 1.], "num_runs": len(bundle["runs"])}


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle["checks"] = validate_bundle(bundle)
    torch.save(bundle, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    validate_bundle(restored)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return {"bundle": str(path.resolve()), "sha256": digest.hexdigest(), "checks": bundle["checks"]}


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    validate_bundle(bundle)
    return bundle


def plot_bundle(bundle, output):
    """Only consumes local bundle arrays; never loads datasets/checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_bundle(bundle)
    classes, config = bundle["class_ordering"], bundle["plot_config"]
    files, unavailable = [], list(bundle["unavailable"])
    selections_for_count = bundle.get("timeline_selections", [bundle["timeline_selection"]] if bundle["timeline_selection"] else [])
    total_figures = (4*(len(bundle["model_seeds"])+1) + len(bundle["examples"]) + 1
                     + len(selections_for_count)*len(bundle["runs"])*len(METHODS)
                     + len(bundle.get("transition_thumbnails", [])))
    print(f"[paper figures] Rendering {total_figures} figures, each as PNG + PDF, into {output}", flush=True)
    def save(fig, name):
        for suffix in ("png", "pdf"):
            path = figure_path(output, name + "." + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=config["dpi"], bbox_inches="tight")
            files.append(str(path.resolve()))
        plt.close(fig)
        completed = len(files)//2
        if completed == 1 or completed % 25 == 0 or completed == total_figures:
            print(f"[paper figures] {completed}/{total_figures} complete: {name}", flush=True)
    for method in ("structural", *METHODS):
        source = bundle["experiment_1_rows"] if method == "structural" else [r for r in bundle["experiment_2_rows"] if r["temporal_method"] == method]
        for seed in (*bundle["model_seeds"], "aggregate"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for architecture, ax in zip(ARCHITECTURES, axes):
                rows = [r for r in source if r["architecture"] == architecture and (seed == "aggregate" or r["seed"] == seed)]
                counts = np.asarray([r["confusion_matrix"] for r in rows]).sum(0)
                matrix = normalized_confusion(counts)
                im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
                ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
                       xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="Ground truth",
                       title=f"{architecture} / {method} / {seed}")
                ax.tick_params(axis="x", labelrotation=45)
                for y in range(len(classes)):
                    for x in range(len(classes)):
                        value = matrix[y, x]
                        ax.text(x, y, f"{value:.2f}" if np.isfinite(value) else "N/A", ha="center", va="center", fontsize=7,
                                color="white" if value > .5 else "black")
            fig.colorbar(im, ax=axes, label="Fraction within GT class")
            save(fig, f"confusion_{method}_{seed}")
    for index, example in enumerate(bundle["examples"]):
        a = example["arrays"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        axes[0].imshow(a["processed_depth"], cmap="gray")
        y, x = torch.where(a["strong_horizontal_edges"])
        axes[0].scatter(x, y, s=2, c="cyan", label="Strong residual Sobel-y")
        left, right = example["center_columns"]
        axes[0].add_patch(Rectangle((left, example["near_row"]), right-left, a["processed_depth"].shape[0]-example["near_row"],
                                   fill=False, edgecolor="yellow", label="Near/center region"))
        axes[0].legend(fontsize=6)
        axes[0].set_title("Processed depth + extractor cues")
        axes[1].imshow(a["residual"], cmap="coolwarm"); axes[1].set_title("Depth − calibrated reference")
        axes[2].imshow(a["close_mask"].int() + 2*a["far_or_invalid_mask"].int(), vmin=0, vmax=2, cmap="viridis")
        axes[2].set_title("Occupancy: 0 other / 1 close / 2 far-invalid")
        fig.suptitle(f"{example['label']} | row {example['row_index']} | frame {example['provenance'].get('frame_indices', 'unavailable')}")
        save(fig, f"depth_geometry_{index}")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"feature_nn": "#377eb8", "raw_depth_nn": "#e6550d"}
    markers = {"instantaneous": "o", "ema": "s", "bayes": "^"}
    for architecture in ARCHITECTURES:
        for method in METHODS:
            rows = [r for r in bundle["experiment_2_rows"] if r["architecture"] == architecture and r["temporal_method"] == method]
            x = np.array([r[config["delay_metric"]] for r in rows], float)
            y = np.array([r["false_transition_rate"] for r in rows], float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                unavailable.append(f"Delay/FTR scatter unavailable: {architecture}/{method}")
                continue
            ax.scatter(x[mask], y[mask], color=colors[architecture], marker=markers[method], alpha=.4)
            for i in np.flatnonzero(mask):
                ax.annotate(str(rows[i]["seed"]), (x[i], y[i]), fontsize=7)
            ax.errorbar(x[mask].mean(), y[mask].mean(), xerr=x[mask].std(ddof=1) if mask.sum()>1 else 0,
                        yerr=y[mask].std(ddof=1) if mask.sum()>1 else 0, color=colors[architecture],
                        marker=markers[method], capsize=3, label=f"{architecture}/{method}")
    ax.set(xlabel="Mean matched delay (classification frames; segment-bounded v2)", ylabel="False-transition rate")
    ax.legend(fontsize=7); ax.grid(alpha=.2)
    save(fig, "transition_delay_vs_false_transition_rate")
    selections = bundle.get("timeline_selections", [bundle["timeline_selection"]] if bundle["timeline_selection"] else [])
    thumbnail_map = {item["sample_id"]: item for item in bundle.get("transition_thumbnails", [])}
    def show_thumbnail(ax, thumbnails, j, origin):
        ax.imshow(thumbnails["depth_images"][j], cmap="gray")
        frame = thumbnails["provenance"].get("frame_indices", thumbnails["row_indices"])[j]
        ax.set_title(f"t={thumbnails['row_indices'][j]-origin:+d}; frame={frame}\n{thumbnails['labels'][j]}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    for selected in selections:
        thumbnails = thumbnail_map.get(selected.get("sample_id"))
        prefix = f"sample_{selected['sample_id']:03d}_" if "sample_id" in selected else ""
        if thumbnails:
            fig, axes = plt.subplots(1, len(thumbnails["row_indices"]), figsize=(12, 2.2), squeeze=False)
            for j, ax in enumerate(axes[0]):
                show_thumbnail(ax, thumbnails, j, selected["boundaries"][0])
            fig.suptitle(f"Transition sample {selected['sample_id']}; focal row {selected['focus_frame']}")
            save(fig, f"transition_depth_{prefix.rstrip('_')}")
        lo, hi = selected["start"], selected["end"]
        x = np.arange(lo, hi) - selected["boundaries"][0]
        truth = [classes.index(label) for label in bundle["ordered_truth"][lo:hi]]
        for run in bundle["runs"]:
            for method in METHODS:
                trace = run["trace"]
                p = trace[{"instantaneous": "probabilities", "ema": "ema_probabilities", "bayes": "bayes_beliefs"}[method]][lo:hi]
                fig, axes = plt.subplots(3, 1, figsize=(10, 8 if thumbnails else 6), sharex=True)
                if thumbnails:
                    fig.subplots_adjust(bottom=.27, hspace=.25)
                    width = .88 / len(thumbnails["row_indices"])
                    for j in range(len(thumbnails["row_indices"])):
                        ax = fig.add_axes([.08+j*width, .045, width*.92, .13])
                        show_thumbnail(ax, thumbnails, j, selected["boundaries"][0])
                for j, label in enumerate(classes):
                    axes[0].plot(x, trace["probabilities"][lo:hi, j], label=str(label))
                    axes[1].plot(x, p[:, j], label=str(label))
                axes[0].set_ylabel("Instantaneous evidence")
                axes[1].set_ylabel({"instantaneous": "Unfiltered probability", "ema": "EMA probability", "bayes": "Bayes belief"}[method])
                for ax in axes[:2]:
                    ax.set_ylim(0, 1); ax.legend(ncol=len(classes), fontsize=6)
                axes[2].step(x, truth, where="post", color="black", label="Ground truth")
                axes[2].step(x, trace[method+"_selected"][lo:hi], where="post", linestyle="--", label="Emitted terrain / requested skill")
                axes[2].set(yticks=range(len(classes)), yticklabels=classes, xlabel="Classification frames relative to first GT boundary")
                axes[2].legend(fontsize=7)
                for ax in axes:
                    for boundary in selected["boundaries"]:
                        ax.axvline(boundary-selected["boundaries"][0], color="gray", linestyle=":")
                fig.suptitle(f"{prefix}{run['architecture']} seed {run['seed']} / {method}")
                save(fig, f"timeline_{prefix}{run['architecture']}_seed_{run['seed']}_{method}")
    files.extend(str(path.resolve()) for path in sorted(output.glob("figures/results/experiment_*.png")))
    files.extend(str(path.resolve()) for path in sorted(output.glob("figures/results/experiment_*.pdf")))
    write_figure_index(output, files)
    report = {"outputs": files, "unavailable": unavailable, "timeline_selection": bundle["timeline_selection"],
              "browse_index": str((output / "figure_index.html").resolve()),
              "timeline_selections": selections, "sampling_metadata": bundle.get("sampling_metadata"),
              "depth_examples": [{"figure_stem": str(figure_path(Path(), f"depth_geometry_{i}")), "label": e["label"],
                                  "row_index": e["row_index"], "provenance": e["provenance"]} for i, e in enumerate(bundle["examples"])],
              "checks": bundle["checks"] if "checks" in bundle else validate_bundle(bundle)}
    (output / "figure_manifest.json").write_text(json.dumps(report, indent=2, default=str))
    return report
