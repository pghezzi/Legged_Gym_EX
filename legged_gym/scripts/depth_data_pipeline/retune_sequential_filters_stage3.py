#!/usr/bin/env python3
"""
Sequential-only terrain classifier evaluation/search.

Reuses previously trained instantaneous classifiers and evaluates four temporal
approaches:

  Stage 1: fixed-persistence Bayes
  Stage 2: event-conditioned Bayes
  Stage 3: ambiguity-aware event-conditioned Bayes
  Baseline: EMA-logit smoothing + patience gate

Search is staged to keep the expensive temporal recurrence count bounded.

Selection:
  - Each approach is tuned ONLY on ordered validation sequences.
  - The best temporal approach for each classifier is selected by `selection_score`.
  - The global best classifier + temporal approach is also selected by validation
    `selection_score`.
  - The untouched final ordered test set is used only for reporting.

Outputs:
  - per-method detailed JSON containing all search trials
  - root `approach_comparison.csv`
  - root `approach_comparison.json`
  - root `per_classifier_best.json`
  - root `global_best_selection.json`
  - saved runtime temporal-filter artifacts for the best candidate from every family
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    BayesianTerrainFilter,
    estimate_observation_matrix_from_probabilities,
    evaluate_predictions,
    make_persistent_transition_matrix,
    run_filter_sequences,
)

try:
    from legged_gym.utils.depth_terrain_classifier.sequential_terrain_filter_extensions import (
        AmbiguityAwareEventConditionedBayesianTerrainFilter,
        EMALogitPatienceFilter,
        EventConditionedBayesianTerrainFilter,
        run_ambiguity_aware_filter_sequences,
        run_ema_logit_patience_sequences,
        run_event_conditioned_filter_sequences,
    )
except ImportError:
    from sequential_terrain_filter_extensions import (
        AmbiguityAwareEventConditionedBayesianTerrainFilter,
        EMALogitPatienceFilter,
        EventConditionedBayesianTerrainFilter,
        run_ambiguity_aware_filter_sequences,
        run_ema_logit_patience_sequences,
        run_event_conditioned_filter_sequences,
    )

try:
    from .util_func import extract_in_chunks, load_classifier_extractor
except ImportError:
    from util_func import extract_in_chunks, load_classifier_extractor


# =============================================================================
# Constrained staged search
# =============================================================================

# Previous searches repeatedly selected T=0.5 at the lower search boundary.
TEMPERATURES = (0.25, 0.40, 0.50, 0.75, 1.00, 1.50)

# Mostly direct observation evidence:
# O_mix = (1-rho) I + rho O_soft
# likelihood = O_mix @ q = (1-rho)q + rho(O_soft @ q)
OBSERVATION_MIXES = (0.0, 0.25)
OBSERVATION_PSEUDOCOUNT = 0.5

# Stage 1: broad stable-persistence range.
STABLE_STAY_PROBABILITIES = (0.70, 0.85, 0.95, 0.98)

# Stage 2: release temporal inertia on a detected transition event.
SWITCH_STAY_PROBABILITIES = (0.10, 0.30, 0.50, 0.70)
SWITCH_CONFIDENCES = (0.65, 0.80)
CHANGE_PATIENCES = (1, 2)
TOP_K_FIXED = 2

# Stage 3: ambiguous frames hold output, reset switch patience, skip class
# evidence, and flatten internal belief toward the prior.
AMBIGUITY_MARGIN_THRESHOLDS = (0.10, 0.20, 0.30)
AMBIGUITY_FLATTEN_STRENGTHS = (0.10, 0.25, 0.40)
TOP_K_EVENT = 2

# Independent non-Bayesian baseline.
EMA_ALPHAS = (0.40, 0.60, 0.80, 1.00)
EMA_PATIENCES = (1, 2)


# =============================================================================
# Saved model / data helpers
# =============================================================================

class TensorStandardizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.eps = float(eps)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype).clamp_min(self.eps)
        return (x - mean) / std


def _load_standardizer_fallback(run_dir: Path):
    candidates = []
    for pattern in ("*standardizer*.pt", "*standardizer*", "standardizer.pt"):
        candidates.extend(run_dir.glob(pattern))

    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)

        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")

        if hasattr(state, "transform"):
            return state

        if isinstance(state, Mapping):
            mean = state.get("mean", state.get("feature_mean"))
            std = state.get("std", state.get("feature_std"))
            if mean is not None and std is not None:
                return TensorStandardizer(mean, std, state.get("eps", 1e-6))
    return None


def load_saved_model(run_dir: str | Path):
    """
    Support legacy and current utility-loader return formats:
      (classifier, extractor)
      (classifier, extractor, metadata)
      (classifier, extractor, standardizer, metadata)
    """
    run_dir = Path(run_dir)
    loaded = load_classifier_extractor(run_dir)

    if not isinstance(loaded, tuple):
        raise TypeError("load_classifier_extractor() must return a tuple")

    classifier = loaded[0]
    extractor = loaded[1] if len(loaded) > 1 else None
    standardizer = None
    metadata = {}

    if len(loaded) == 3:
        third = loaded[2]
        if isinstance(third, Mapping) and (
            "class" in third or "method" in third or "name" in third
        ):
            metadata = dict(third)
        else:
            standardizer = third

    elif len(loaded) >= 4:
        standardizer = loaded[2]
        if isinstance(loaded[3], Mapping):
            metadata = dict(loaded[3])

    if standardizer is None:
        standardizer = _load_standardizer_fallback(run_dir)

    return classifier, extractor, standardizer, metadata


def sequence_ids_from_data(data: Mapping[str, Any]) -> torch.Tensor:
    if "sequence_ids" in data:
        return torch.as_tensor(data["sequence_ids"]).flatten().cpu()

    if "episode_ids" in data:
        return torch.as_tensor(data["episode_ids"]).flatten().cpu()

    n = len(data["labels"])
    per_eps = int(data["per_eps"])
    if per_eps <= 0 or n % per_eps != 0:
        raise ValueError(
            f"Cannot infer sequence IDs: N={n}, per_eps={per_eps}. "
            "Store sequence_ids/episode_ids explicitly."
        )

    return torch.arange(n // per_eps).repeat_interleave(per_eps)


def _subset_data(data: Mapping[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    idx = torch.where(mask)[0]
    n = len(data["labels"])
    out = {}

    for key, value in data.items():
        if key == "labels":
            out[key] = [value[int(i)] for i in idx]
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
            out[key] = value[idx]
        else:
            out[key] = value
    return out


def split_validation_by_sequence(
    data: Mapping[str, Any],
    calibration_fraction: float = 0.4,
):
    """Fallback only: split complete sequences, never individual frames."""
    ids = sequence_ids_from_data(data)
    unique_ids = torch.unique_consecutive(ids)

    if len(unique_ids) < 2:
        raise ValueError("Need at least two complete sequences for fallback splitting.")

    n_cal = max(1, int(round(len(unique_ids) * calibration_fraction)))
    n_cal = min(n_cal, len(unique_ids) - 1)
    cal_ids = set(unique_ids[:n_cal].tolist())

    cal_mask = torch.tensor([int(v) in cal_ids for v in ids], dtype=torch.bool)
    return _subset_data(data, cal_mask), _subset_data(data, ~cal_mask)


@torch.inference_mode()
def prepare_classifier_inputs(
    classifier,
    extractor,
    standardizer,
    data: Mapping[str, Any],
    *,
    chunk_size: int = 256,
):
    if getattr(classifier, "require_feature", False):
        if extractor is None:
            raise ValueError("Feature classifier requires a saved extractor.")

        features = extract_in_chunks(
            extractor,
            data["depth_images"],
            data["orientation_rpy"],
            data["angular_velocity"],
            chunk_size=chunk_size,
        )

        if standardizer is not None:
            features = standardizer.transform(features)
        return features

    depth = data["depth_images"].float()
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    return depth


@torch.inference_mode()
def decision_scores_in_chunks(
    classifier,
    inputs,
    *,
    chunk_size: int = 4096,
):
    if not torch.is_tensor(inputs):
        return classifier.decision_function(inputs).detach().cpu()

    device = getattr(classifier, "device", torch.device("cpu"))
    out = []

    for start in range(0, inputs.shape[0], chunk_size):
        batch = inputs[start : start + chunk_size].to(device)
        out.append(classifier.decision_function(batch).detach().cpu())

    return torch.cat(out, dim=0)


# =============================================================================
# Metrics and selection objective
# =============================================================================

def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _transition_window_mask(
    truth: Sequence,
    sequence_ids: Sequence,
    radius: int,
) -> torch.Tensor:
    """Mark frames within +/- radius of each true within-sequence transition."""
    n = len(truth)
    ids = list(sequence_ids)
    mask = torch.zeros(n, dtype=torch.bool)

    if radius < 0:
        raise ValueError("transition_window_radius must be >= 0")

    for t in range(1, n):
        if ids[t] == ids[t - 1] and truth[t] != truth[t - 1]:
            lo = max(0, t - radius)
            hi = min(n, t + radius + 1)
            for j in range(lo, hi):
                if ids[j] == ids[t]:
                    mask[j] = True
    return mask


def _mean_true_ambiguity_run_length(mask: torch.Tensor, sequence_ids: Sequence) -> float:
    ids = list(sequence_ids)
    runs = []
    run = 0
    previous_id = object()

    for i, flag in enumerate(mask.tolist()):
        if i == 0 or ids[i] != previous_id:
            if run:
                runs.append(run)
            run = 0

        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0

        previous_id = ids[i]

    if run:
        runs.append(run)

    return float(sum(runs) / len(runs)) if runs else 0.0


def prediction_diagnostics(
    truth: Sequence,
    prediction: Sequence,
    labels: Sequence,
):
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}

    confusion = torch.zeros(len(labels), len(labels), dtype=torch.int64)
    pred_counts = {str(label): 0 for label in labels}

    for y, p in zip(truth, prediction):
        confusion[index[y], index[p]] += 1
        pred_counts[str(p)] += 1

    n = max(len(prediction), 1)
    recall, precision = {}, {}

    for i, label in enumerate(labels):
        tp = float(confusion[i, i])
        fn = float(confusion[i].sum()) - tp
        fp = float(confusion[:, i].sum()) - tp

        recall[str(label)] = tp / max(tp + fn, 1.0)
        precision[str(label)] = tp / max(tp + fp, 1.0)

    return {
        "prediction_counts": pred_counts,
        "prediction_fraction": {k: v / n for k, v in pred_counts.items()},
        "missing_predicted_classes": [k for k, v in pred_counts.items() if v == 0],
        "per_class_recall": recall,
        "per_class_precision": precision,
        "minimum_class_recall": min(recall.values()) if recall else float("nan"),
    }


def sequential_accuracy_metrics(
    truth: Sequence,
    prediction: Sequence,
    sequence_ids: Sequence,
    *,
    transition_window_radius: int,
):
    transition_mask = _transition_window_mask(
        truth, sequence_ids, transition_window_radius
    )
    correct = torch.tensor(
        [y == p for y, p in zip(truth, prediction)], dtype=torch.float32
    )

    if transition_mask.any():
        transition_accuracy = float(correct[transition_mask].mean())
    else:
        transition_accuracy = float("nan")

    steady_mask = ~transition_mask
    if steady_mask.any():
        steady_accuracy = float(correct[steady_mask].mean())
    else:
        steady_accuracy = float("nan")

    return {
        "transition_window_accuracy": transition_accuracy,
        "steady_state_accuracy": steady_accuracy,
        "transition_window_frame_fraction": float(transition_mask.float().mean()),
        "transition_window_frames": int(transition_mask.sum()),
    }


def selection_score(
    *,
    balanced_accuracy: float,
    transition_window_accuracy: float,
    mean_transition_delay: float,
    false_transition_rate: float,
    missing_class_count: int,
    transition_accuracy_weight: float,
    delay_weight: float,
    false_transition_weight: float,
    missing_class_penalty: float,
):
    """
    Deployment-oriented objective.

    First blend overall balanced accuracy with accuracy near true transitions:

      base = (1-w)*balanced_accuracy + w*transition_window_accuracy

    Then penalize transition lag, false state changes, and complete class loss.
    """
    if math.isfinite(transition_window_accuracy):
        base = (
            (1.0 - transition_accuracy_weight) * balanced_accuracy
            + transition_accuracy_weight * transition_window_accuracy
        )
    else:
        base = balanced_accuracy

    score = float(base)

    if math.isfinite(mean_transition_delay):
        score -= delay_weight * mean_transition_delay

    if math.isfinite(false_transition_rate):
        score -= false_transition_weight * false_transition_rate

    score -= missing_class_penalty * int(missing_class_count)
    return float(score)


def evaluate_labels(
    true_labels,
    predicted_labels,
    labels,
    sequence_ids,
    *,
    transition_window_radius,
    transition_accuracy_weight,
    delay_weight,
    false_transition_weight,
    missing_class_penalty,
):
    base = evaluate_predictions(
        true_labels,
        predicted_labels,
        labels,
        sequence_ids=sequence_ids,
    )
    diagnostics = prediction_diagnostics(true_labels, predicted_labels, labels)
    sequential = sequential_accuracy_metrics(
        true_labels,
        predicted_labels,
        sequence_ids,
        transition_window_radius=transition_window_radius,
    )

    score = selection_score(
        balanced_accuracy=float(base.balanced_accuracy),
        transition_window_accuracy=float(sequential["transition_window_accuracy"]),
        mean_transition_delay=float(base.mean_transition_delay),
        false_transition_rate=float(base.false_transition_rate),
        missing_class_count=len(diagnostics["missing_predicted_classes"]),
        transition_accuracy_weight=transition_accuracy_weight,
        delay_weight=delay_weight,
        false_transition_weight=false_transition_weight,
        missing_class_penalty=missing_class_penalty,
    )

    return {
        "selection_score": score,
        **base.as_dict(),
        **sequential,
        **diagnostics,
    }


def probabilities_from_scores(
    scores: torch.Tensor,
    temperature: float,
    probability_floor: float = 1e-8,
):
    probs = F.softmax(scores / float(temperature), dim=1)
    probs = probs.clamp_min(probability_floor)
    return probs / probs.sum(dim=1, keepdim=True)


def mixed_observation_matrix(
    soft_observation: torch.Tensor,
    observation_mix: float,
):
    rho = float(observation_mix)
    eye = torch.eye(soft_observation.shape[0], dtype=soft_observation.dtype)
    return (1.0 - rho) * eye + rho * soft_observation


# =============================================================================
# Candidate evaluators
# =============================================================================

@torch.inference_mode()
def evaluate_fixed_candidate(
    *,
    probs,
    true_labels,
    sequence_ids,
    labels,
    prior,
    transition,
    observation,
    temperature,
    stable_stay_probability,
    observation_mix,
    metric_kwargs,
):
    filt = BayesianTerrainFilter(
        labels=labels,
        prior=prior,
        transition_matrix=transition,
        observation_matrix=observation,
        evidence_power=1.0,
        adaptive_evidence=False,
        min_evidence_power=1.0,
        confidence_gamma=1.0,
        device="cpu",
    )

    predictions, _, _ = run_filter_sequences(
        filt, probs, sequence_ids=sequence_ids
    )

    return {
        "family": "stage1_fixed_bayes",
        "temperature": float(temperature),
        "stable_stay_probability": float(stable_stay_probability),
        "observation_mix": float(observation_mix),
        "transition_matrix": transition.cpu(),
        "observation_matrix": observation.cpu(),
        **evaluate_labels(
            true_labels, predictions, labels, sequence_ids, **metric_kwargs
        ),
    }


@torch.inference_mode()
def evaluate_event_candidate(
    *,
    probs,
    true_labels,
    sequence_ids,
    labels,
    prior,
    stable_transition,
    switch_transition,
    observation,
    temperature,
    stable_stay_probability,
    switch_stay_probability,
    switch_confidence,
    change_patience,
    observation_mix,
    metric_kwargs,
):
    filt = EventConditionedBayesianTerrainFilter(
        labels=labels,
        prior=prior,
        stable_transition_matrix=stable_transition,
        switch_transition_matrix=switch_transition,
        observation_matrix=observation,
        switch_confidence=switch_confidence,
        change_patience=change_patience,
        evidence_power=1.0,
        adaptive_evidence=False,
        min_evidence_power=1.0,
        confidence_gamma=1.0,
        device="cpu",
    )

    predictions, _, _, switch_events = run_event_conditioned_filter_sequences(
        filt, probs, sequence_ids=sequence_ids
    )

    return {
        "family": "stage2_event_bayes",
        "temperature": float(temperature),
        "stable_stay_probability": float(stable_stay_probability),
        "switch_stay_probability": float(switch_stay_probability),
        "switch_confidence": float(switch_confidence),
        "change_patience": int(change_patience),
        "observation_mix": float(observation_mix),
        "stable_transition_matrix": stable_transition.cpu(),
        "switch_transition_matrix": switch_transition.cpu(),
        "observation_matrix": observation.cpu(),
        "switch_transition_fraction": float(switch_events.float().mean()),
        **evaluate_labels(
            true_labels, predictions, labels, sequence_ids, **metric_kwargs
        ),
    }


@torch.inference_mode()
def evaluate_ambiguity_candidate(
    *,
    probs,
    true_labels,
    sequence_ids,
    labels,
    prior,
    stable_transition,
    switch_transition,
    observation,
    seed,
    ambiguity_margin_threshold,
    ambiguity_flatten_strength,
    metric_kwargs,
):
    filt = AmbiguityAwareEventConditionedBayesianTerrainFilter(
        labels=labels,
        prior=prior,
        stable_transition_matrix=stable_transition,
        switch_transition_matrix=switch_transition,
        observation_matrix=observation,
        switch_confidence=seed["switch_confidence"],
        change_patience=seed["change_patience"],
        ambiguity_margin_threshold=ambiguity_margin_threshold,
        ambiguity_flatten_strength=ambiguity_flatten_strength,
        evidence_power=1.0,
        adaptive_evidence=False,
        min_evidence_power=1.0,
        confidence_gamma=1.0,
        device="cpu",
    )

    (
        predictions,
        _,
        _,
        switch_events,
        ambiguous_events,
        margins,
    ) = run_ambiguity_aware_filter_sequences(
        filt, probs, sequence_ids=sequence_ids
    )

    metrics = evaluate_labels(
        true_labels, predictions, labels, sequence_ids, **metric_kwargs
    )

    return {
        "family": "stage3_ambiguity_bayes",
        "temperature": float(seed["temperature"]),
        "stable_stay_probability": float(seed["stable_stay_probability"]),
        "switch_stay_probability": float(seed["switch_stay_probability"]),
        "switch_confidence": float(seed["switch_confidence"]),
        "change_patience": int(seed["change_patience"]),
        "observation_mix": float(seed["observation_mix"]),
        "ambiguity_margin_threshold": float(ambiguity_margin_threshold),
        "ambiguity_flatten_strength": float(ambiguity_flatten_strength),
        "stable_transition_matrix": stable_transition.cpu(),
        "switch_transition_matrix": switch_transition.cpu(),
        "observation_matrix": observation.cpu(),
        "switch_transition_fraction": float(switch_events.float().mean()),
        "ambiguous_frame_fraction": float(ambiguous_events.float().mean()),
        "mean_ambiguity_run_length": _mean_true_ambiguity_run_length(
            ambiguous_events, sequence_ids
        ),
        "mean_top2_margin": float(margins.mean()),
        **metrics,
    }


def search_ema_baseline(
    scores,
    true_labels,
    sequence_ids,
    labels,
    *,
    metric_kwargs,
):
    results = []

    for alpha in EMA_ALPHAS:
        for patience in EMA_PATIENCES:
            filt = EMALogitPatienceFilter(
                labels,
                ema_alpha=alpha,
                change_patience=patience,
                device="cpu",
            )
            predictions = run_ema_logit_patience_sequences(
                filt, scores, sequence_ids=sequence_ids
            )

            results.append(
                {
                    "family": "ema_logit_patience",
                    "ema_alpha": float(alpha),
                    "change_patience": int(patience),
                    **evaluate_labels(
                        true_labels,
                        predictions,
                        labels,
                        sequence_ids,
                        **metric_kwargs,
                    ),
                }
            )

    results.sort(key=lambda r: r["selection_score"], reverse=True)
    return results


# =============================================================================
# Staged validation search
# =============================================================================

@torch.inference_mode()
def search_sequential_parameters(
    classifier,
    calibration_inputs,
    calibration_labels,
    validation_inputs,
    validation_labels,
    validation_sequence_ids,
    *,
    inference_chunk_size,
    top_k_fixed,
    top_k_event,
    metric_kwargs,
):
    labels = list(classifier.class_ids)
    prior = {label: 1.0 / len(labels) for label in labels}

    scores_cal = decision_scores_in_chunks(
        classifier, calibration_inputs, chunk_size=inference_chunk_size
    )
    scores_val = decision_scores_in_chunks(
        classifier, validation_inputs, chunk_size=inference_chunk_size
    )

    probs_cal = {
        t: probabilities_from_scores(scores_cal, t) for t in TEMPERATURES
    }
    probs_val = {
        t: probabilities_from_scores(scores_val, t) for t in TEMPERATURES
    }

    soft_observation = {
        t: estimate_observation_matrix_from_probabilities(
            probs_cal[t],
            calibration_labels,
            labels,
            mode="soft",
            pseudocount=OBSERVATION_PSEUDOCOUNT,
            device="cpu",
        )
        for t in TEMPERATURES
    }

    stable_transitions = {
        stay: make_persistent_transition_matrix(labels, stay, device="cpu")
        for stay in STABLE_STAY_PROBABILITIES
    }

    switch_transitions = {
        stay: make_persistent_transition_matrix(labels, stay, device="cpu")
        for stay in SWITCH_STAY_PROBABILITIES
    }

    # -------------------------------------------------------------------------
    # Stage 1: 6 temperatures x 2 observation mixes x 4 stable stays = 48
    # -------------------------------------------------------------------------
    fixed_results = []

    for temperature in TEMPERATURES:
        probs = probs_val[temperature]

        for obs_mix in OBSERVATION_MIXES:
            observation = mixed_observation_matrix(
                soft_observation[temperature], obs_mix
            )

            for stable_stay in STABLE_STAY_PROBABILITIES:
                fixed_results.append(
                    evaluate_fixed_candidate(
                        probs=probs,
                        true_labels=validation_labels,
                        sequence_ids=validation_sequence_ids,
                        labels=labels,
                        prior=prior,
                        transition=stable_transitions[stable_stay],
                        observation=observation,
                        temperature=temperature,
                        stable_stay_probability=stable_stay,
                        observation_mix=obs_mix,
                        metric_kwargs=metric_kwargs,
                    )
                )

    fixed_results.sort(key=lambda r: r["selection_score"], reverse=True)
    fixed_seeds = fixed_results[: max(1, int(top_k_fixed))]

    # -------------------------------------------------------------------------
    # Stage 2: refine top Stage-1 configurations with event-conditioned T.
    # -------------------------------------------------------------------------
    event_results = []
    seen = set()

    for seed in fixed_seeds:
        temperature = seed["temperature"]
        stable_stay = seed["stable_stay_probability"]
        obs_mix = seed["observation_mix"]

        probs = probs_val[temperature]
        observation = mixed_observation_matrix(
            soft_observation[temperature], obs_mix
        )

        for switch_stay in SWITCH_STAY_PROBABILITIES:
            if switch_stay >= stable_stay:
                continue

            for switch_confidence in SWITCH_CONFIDENCES:
                for patience in CHANGE_PATIENCES:
                    key = (
                        temperature,
                        stable_stay,
                        obs_mix,
                        switch_stay,
                        switch_confidence,
                        patience,
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    event_results.append(
                        evaluate_event_candidate(
                            probs=probs,
                            true_labels=validation_labels,
                            sequence_ids=validation_sequence_ids,
                            labels=labels,
                            prior=prior,
                            stable_transition=stable_transitions[stable_stay],
                            switch_transition=switch_transitions[switch_stay],
                            observation=observation,
                            temperature=temperature,
                            stable_stay_probability=stable_stay,
                            switch_stay_probability=switch_stay,
                            switch_confidence=switch_confidence,
                            change_patience=patience,
                            observation_mix=obs_mix,
                            metric_kwargs=metric_kwargs,
                        )
                    )

    event_results.sort(key=lambda r: r["selection_score"], reverse=True)
    event_seeds = event_results[: max(1, int(top_k_event))]

    # -------------------------------------------------------------------------
    # Stage 3: top event candidates x 3 margins x 3 flatten strengths.
    # -------------------------------------------------------------------------
    ambiguity_results = []
    seen = set()

    for seed in event_seeds:
        temperature = seed["temperature"]
        stable_stay = seed["stable_stay_probability"]
        switch_stay = seed["switch_stay_probability"]
        obs_mix = seed["observation_mix"]

        probs = probs_val[temperature]
        observation = mixed_observation_matrix(
            soft_observation[temperature], obs_mix
        )

        for threshold in AMBIGUITY_MARGIN_THRESHOLDS:
            for flatten_strength in AMBIGUITY_FLATTEN_STRENGTHS:
                key = (
                    temperature,
                    stable_stay,
                    switch_stay,
                    seed["switch_confidence"],
                    seed["change_patience"],
                    obs_mix,
                    threshold,
                    flatten_strength,
                )

                if key in seen:
                    continue
                seen.add(key)

                ambiguity_results.append(
                    evaluate_ambiguity_candidate(
                        probs=probs,
                        true_labels=validation_labels,
                        sequence_ids=validation_sequence_ids,
                        labels=labels,
                        prior=prior,
                        stable_transition=stable_transitions[stable_stay],
                        switch_transition=switch_transitions[switch_stay],
                        observation=observation,
                        seed=seed,
                        ambiguity_margin_threshold=threshold,
                        ambiguity_flatten_strength=flatten_strength,
                        metric_kwargs=metric_kwargs,
                    )
                )

    ambiguity_results.sort(key=lambda r: r["selection_score"], reverse=True)

    # -------------------------------------------------------------------------
    # Independent EMA+patience baseline.
    # -------------------------------------------------------------------------
    ema_results = search_ema_baseline(
        scores_val,
        validation_labels,
        validation_sequence_ids,
        labels,
        metric_kwargs=metric_kwargs,
    )

    return {
        "labels": labels,
        "scores_calibration": scores_cal,
        "scores_validation": scores_val,
        "best_fixed": fixed_results[0],
        "best_event": event_results[0],
        "best_ambiguity": ambiguity_results[0],
        "best_ema": ema_results[0],
        "fixed_results": fixed_results,
        "event_results": event_results,
        "ambiguity_results": ambiguity_results,
        "ema_results": ema_results,
        "search_config": {
            "temperatures": list(TEMPERATURES),
            "observation_mixes": list(OBSERVATION_MIXES),
            "observation_pseudocount": OBSERVATION_PSEUDOCOUNT,
            "stable_stay_probabilities": list(STABLE_STAY_PROBABILITIES),
            "switch_stay_probabilities": list(SWITCH_STAY_PROBABILITIES),
            "switch_confidences": list(SWITCH_CONFIDENCES),
            "change_patiences": list(CHANGE_PATIENCES),
            "ambiguity_margin_thresholds": list(AMBIGUITY_MARGIN_THRESHOLDS),
            "ambiguity_flatten_strengths": list(AMBIGUITY_FLATTEN_STRENGTHS),
            "ema_alphas": list(EMA_ALPHAS),
            "ema_patiences": list(EMA_PATIENCES),
            "top_k_fixed": int(top_k_fixed),
            "top_k_event": int(top_k_event),
            "evidence_power": 1.0,
            "adaptive_evidence": False,
            "metric": dict(metric_kwargs),
        },
    }


# =============================================================================
# Final-test evaluation / artifact construction
# =============================================================================

def _calibrated_observation(
    calibration_scores,
    calibration_labels,
    labels,
    *,
    temperature,
    observation_mix,
):
    cal_probs = probabilities_from_scores(calibration_scores, temperature)

    soft_o = estimate_observation_matrix_from_probabilities(
        cal_probs,
        calibration_labels,
        labels,
        mode="soft",
        pseudocount=OBSERVATION_PSEUDOCOUNT,
        device="cpu",
    )

    return mixed_observation_matrix(soft_o, observation_mix)


@torch.inference_mode()
def evaluate_fixed_on_test(
    best,
    scores,
    true_labels,
    sequence_ids,
    labels,
    calibration_scores,
    calibration_labels,
    metric_kwargs,
):
    probs = probabilities_from_scores(scores, best["temperature"])
    observation = _calibrated_observation(
        calibration_scores,
        calibration_labels,
        labels,
        temperature=best["temperature"],
        observation_mix=best["observation_mix"],
    )
    transition = make_persistent_transition_matrix(
        labels, best["stable_stay_probability"], device="cpu"
    )
    prior = {label: 1.0 / len(labels) for label in labels}

    filt = BayesianTerrainFilter(
        labels=labels,
        prior=prior,
        transition_matrix=transition,
        observation_matrix=observation,
        evidence_power=1.0,
        adaptive_evidence=False,
        min_evidence_power=1.0,
        confidence_gamma=1.0,
        device="cpu",
    )

    predictions, _, _ = run_filter_sequences(
        filt, probs, sequence_ids=sequence_ids
    )

    return filt, evaluate_labels(
        true_labels, predictions, labels, sequence_ids, **metric_kwargs
    )


@torch.inference_mode()
def evaluate_event_on_test(
    best,
    scores,
    true_labels,
    sequence_ids,
    labels,
    calibration_scores,
    calibration_labels,
    metric_kwargs,
):
    probs = probabilities_from_scores(scores, best["temperature"])
    observation = _calibrated_observation(
        calibration_scores,
        calibration_labels,
        labels,
        temperature=best["temperature"],
        observation_mix=best["observation_mix"],
    )
    stable_t = make_persistent_transition_matrix(
        labels, best["stable_stay_probability"], device="cpu"
    )
    switch_t = make_persistent_transition_matrix(
        labels, best["switch_stay_probability"], device="cpu"
    )
    prior = {label: 1.0 / len(labels) for label in labels}

    filt = EventConditionedBayesianTerrainFilter(
        labels=labels,
        prior=prior,
        stable_transition_matrix=stable_t,
        switch_transition_matrix=switch_t,
        observation_matrix=observation,
        switch_confidence=best["switch_confidence"],
        change_patience=best["change_patience"],
        evidence_power=1.0,
        adaptive_evidence=False,
        min_evidence_power=1.0,
        confidence_gamma=1.0,
        device="cpu",
    )

    predictions, _, _, switch_events = run_event_conditioned_filter_sequences(
        filt, probs, sequence_ids=sequence_ids
    )

    metrics = evaluate_labels(
        true_labels, predictions, labels, sequence_ids, **metric_kwargs
    )
    metrics["switch_transition_fraction"] = float(switch_events.float().mean())
    return filt, metrics


@torch.inference_mode()
def evaluate_ambiguity_on_test(
    best,
    scores,
    true_labels,
    sequence_ids,
    labels,
    calibration_scores,
    calibration_labels,
    metric_kwargs,
):
    probs = probabilities_from_scores(scores, best["temperature"])
    observation = _calibrated_observation(
        calibration_scores,
        calibration_labels,
        labels,
        temperature=best["temperature"],
        observation_mix=best["observation_mix"],
    )
    stable_t = make_persistent_transition_matrix(
        labels, best["stable_stay_probability"], device="cpu"
    )
    switch_t = make_persistent_transition_matrix(
        labels, best["switch_stay_probability"], device="cpu"
    )
    prior = {label: 1.0 / len(labels) for label in labels}

    filt = AmbiguityAwareEventConditionedBayesianTerrainFilter(
        labels=labels,
        prior=prior,
        stable_transition_matrix=stable_t,
        switch_transition_matrix=switch_t,
        observation_matrix=observation,
        switch_confidence=best["switch_confidence"],
        change_patience=best["change_patience"],
        ambiguity_margin_threshold=best["ambiguity_margin_threshold"],
        ambiguity_flatten_strength=best["ambiguity_flatten_strength"],
        evidence_power=1.0,
        adaptive_evidence=False,
        min_evidence_power=1.0,
        confidence_gamma=1.0,
        device="cpu",
    )

    (
        predictions,
        _,
        _,
        switch_events,
        ambiguous_events,
        margins,
    ) = run_ambiguity_aware_filter_sequences(
        filt, probs, sequence_ids=sequence_ids
    )

    metrics = evaluate_labels(
        true_labels, predictions, labels, sequence_ids, **metric_kwargs
    )
    metrics.update(
        {
            "switch_transition_fraction": float(switch_events.float().mean()),
            "ambiguous_frame_fraction": float(ambiguous_events.float().mean()),
            "mean_ambiguity_run_length": _mean_true_ambiguity_run_length(
                ambiguous_events, sequence_ids
            ),
            "mean_top2_margin": float(margins.mean()),
        }
    )
    return filt, metrics


def evaluate_ema_on_test(
    best,
    scores,
    true_labels,
    sequence_ids,
    labels,
    metric_kwargs,
):
    filt = EMALogitPatienceFilter(
        labels,
        ema_alpha=best["ema_alpha"],
        change_patience=best["change_patience"],
        device="cpu",
    )

    predictions = run_ema_logit_patience_sequences(
        filt, scores, sequence_ids=sequence_ids
    )

    return filt, evaluate_labels(
        true_labels, predictions, labels, sequence_ids, **metric_kwargs
    )


def instantaneous_from_scores(
    scores,
    true_labels,
    sequence_ids,
    labels,
    metric_kwargs,
):
    indices = scores.argmax(dim=1).tolist()
    predictions = [labels[i] for i in indices]

    return evaluate_labels(
        true_labels, predictions, labels, sequence_ids, **metric_kwargs
    )


# =============================================================================
# Result tables
# =============================================================================

def candidate_config(candidate: Mapping[str, Any]) -> dict:
    keys = (
        "temperature",
        "observation_mix",
        "stable_stay_probability",
        "switch_stay_probability",
        "switch_confidence",
        "change_patience",
        "ambiguity_margin_threshold",
        "ambiguity_flatten_strength",
        "ema_alpha",
    )
    return {k: candidate[k] for k in keys if k in candidate}


def comparison_row(
    *,
    method,
    family,
    validation,
    test,
    config,
    artifact_path,
    selected_per_classifier=False,
    selected_global=False,
):
    return {
        "method": method,
        "family": family,
        "selected_per_classifier": bool(selected_per_classifier),
        "selected_global": bool(selected_global),
        "validation_selection_score": validation.get("selection_score"),
        "validation_accuracy": validation.get("accuracy"),
        "validation_balanced_accuracy": validation.get("balanced_accuracy"),
        "validation_macro_f1": validation.get("macro_f1"),
        "validation_transition_window_accuracy": validation.get(
            "transition_window_accuracy"
        ),
        "validation_steady_state_accuracy": validation.get(
            "steady_state_accuracy"
        ),
        "validation_mean_transition_delay": validation.get(
            "mean_transition_delay"
        ),
        "validation_false_transition_rate": validation.get(
            "false_transition_rate"
        ),
        "validation_minimum_class_recall": validation.get(
            "minimum_class_recall"
        ),
        "validation_missing_classes": json.dumps(
            validation.get("missing_predicted_classes", [])
        ),
        "test_selection_score_report_only": test.get("selection_score"),
        "test_accuracy": test.get("accuracy"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "test_macro_f1": test.get("macro_f1"),
        "test_transition_window_accuracy": test.get(
            "transition_window_accuracy"
        ),
        "test_steady_state_accuracy": test.get("steady_state_accuracy"),
        "test_mean_transition_delay": test.get("mean_transition_delay"),
        "test_false_transition_rate": test.get("false_transition_rate"),
        "test_minimum_class_recall": test.get("minimum_class_recall"),
        "test_missing_classes": json.dumps(
            test.get("missing_predicted_classes", [])
        ),
        "config_json": json.dumps(_jsonable(config), sort_keys=True),
        "artifact_path": str(artifact_path) if artifact_path is not None else "",
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sequential-only search/evaluation for saved terrain classifiers."
        )
    )

    parser.add_argument(
        "--classifier_dirs",
        nargs="+",
        required=True,
        help="Previously trained classifier run directories.",
    )
    parser.add_argument(
        "--bayesian_folder",
        required=True,
        help="Folder containing ordered train.pt/val.pt/test.pt.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--feature_chunk_size", type=int, default=256)
    parser.add_argument("--inference_chunk_size", type=int, default=4096)
    parser.add_argument("--top_k_fixed", type=int, default=TOP_K_FIXED)
    parser.add_argument("--top_k_event", type=int, default=TOP_K_EVENT)

    # Better-informed deployment selection metric.
    parser.add_argument("--transition_window_radius", type=int, default=5)
    parser.add_argument("--transition_accuracy_weight", type=float, default=0.35)
    parser.add_argument("--delay_weight", type=float, default=0.002)
    parser.add_argument("--false_transition_weight", type=float, default=0.05)
    parser.add_argument("--missing_class_penalty", type=float, default=0.05)

    parser.add_argument("--fallback_calibration_fraction", type=float, default=0.4)

    args = parser.parse_args()

    metric_kwargs = {
        "transition_window_radius": args.transition_window_radius,
        "transition_accuracy_weight": args.transition_accuracy_weight,
        "delay_weight": args.delay_weight,
        "false_transition_weight": args.false_transition_weight,
        "missing_class_penalty": args.missing_class_penalty,
    }

    bayes_dir = Path(args.bayesian_folder)
    train_path = bayes_dir / "train.pt"
    val_path = bayes_dir / "val.pt"
    test_path = bayes_dir / "test.pt"

    if not val_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("bayesian_folder must contain val.pt and test.pt")

    val_full = torch.load(val_path, map_location="cpu")
    test_data = torch.load(test_path, map_location="cpu")

    if train_path.is_file():
        calibration_data = torch.load(train_path, map_location="cpu")
        validation_data = val_full
        calibration_source = str(train_path)
    else:
        calibration_data, validation_data = split_validation_by_sequence(
            val_full, args.fallback_calibration_fraction
        )
        calibration_source = (
            f"{val_path} split by complete sequences "
            f"(fraction={args.fallback_calibration_fraction})"
        )

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else bayes_dir / f"sequential_stage3_search_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_classifier_best = []
    detailed_suite = []

    for classifier_dir_string in args.classifier_dirs:
        classifier_dir = Path(classifier_dir_string)
        print(f"\n=== {classifier_dir} ===")

        classifier, extractor, standardizer, metadata = load_saved_model(
            classifier_dir
        )

        cal_inputs = prepare_classifier_inputs(
            classifier,
            extractor,
            standardizer,
            calibration_data,
            chunk_size=args.feature_chunk_size,
        )
        val_inputs = prepare_classifier_inputs(
            classifier,
            extractor,
            standardizer,
            validation_data,
            chunk_size=args.feature_chunk_size,
        )
        test_inputs = prepare_classifier_inputs(
            classifier,
            extractor,
            standardizer,
            test_data,
            chunk_size=args.feature_chunk_size,
        )

        cal_labels = list(calibration_data["labels"])
        val_labels = list(validation_data["labels"])
        test_labels = list(test_data["labels"])

        val_ids = sequence_ids_from_data(validation_data)
        test_ids = sequence_ids_from_data(test_data)

        start = time.perf_counter()

        search = search_sequential_parameters(
            classifier,
            cal_inputs,
            cal_labels,
            val_inputs,
            val_labels,
            val_ids,
            inference_chunk_size=args.inference_chunk_size,
            top_k_fixed=args.top_k_fixed,
            top_k_event=args.top_k_event,
            metric_kwargs=metric_kwargs,
        )

        search_runtime = time.perf_counter() - start
        labels = search["labels"]

        cal_scores = search["scores_calibration"]
        val_scores = search["scores_validation"]
        test_scores = decision_scores_in_chunks(
            classifier,
            test_inputs,
            chunk_size=args.inference_chunk_size,
        )

        instantaneous_validation = instantaneous_from_scores(
            val_scores, val_labels, val_ids, labels, metric_kwargs
        )
        instantaneous_test = instantaneous_from_scores(
            test_scores, test_labels, test_ids, labels, metric_kwargs
        )

        fixed_filter, fixed_test = evaluate_fixed_on_test(
            search["best_fixed"],
            test_scores,
            test_labels,
            test_ids,
            labels,
            cal_scores,
            cal_labels,
            metric_kwargs,
        )

        event_filter, event_test = evaluate_event_on_test(
            search["best_event"],
            test_scores,
            test_labels,
            test_ids,
            labels,
            cal_scores,
            cal_labels,
            metric_kwargs,
        )

        ambiguity_filter, ambiguity_test = evaluate_ambiguity_on_test(
            search["best_ambiguity"],
            test_scores,
            test_labels,
            test_ids,
            labels,
            cal_scores,
            cal_labels,
            metric_kwargs,
        )

        ema_filter, ema_test = evaluate_ema_on_test(
            search["best_ema"],
            test_scores,
            test_labels,
            test_ids,
            labels,
            metric_kwargs,
        )

        method_name = (
            metadata.get("method")
            or metadata.get("name")
            or classifier.__class__.__name__
        )

        safe_name = (
            str(method_name)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        method_dir = output_dir / safe_name
        method_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "stage1_fixed_bayes": method_dir / "best_stage1_fixed_bayes.pt",
            "stage2_event_bayes": method_dir / "best_stage2_event_bayes.pt",
            "stage3_ambiguity_bayes": method_dir / "best_stage3_ambiguity_bayes.pt",
            "ema_logit_patience": method_dir / "best_ema_logit_patience.pt",
        }

        fixed_filter.save(artifacts["stage1_fixed_bayes"])
        event_filter.save(artifacts["stage2_event_bayes"])
        ambiguity_filter.save(artifacts["stage3_ambiguity_bayes"])
        ema_filter.save(artifacts["ema_logit_patience"])

        family_candidates = {
            "stage1_fixed_bayes": search["best_fixed"],
            "stage2_event_bayes": search["best_event"],
            "stage3_ambiguity_bayes": search["best_ambiguity"],
            "ema_logit_patience": search["best_ema"],
        }

        family_tests = {
            "stage1_fixed_bayes": fixed_test,
            "stage2_event_bayes": event_test,
            "stage3_ambiguity_bayes": ambiguity_test,
            "ema_logit_patience": ema_test,
        }

        # Winner chosen strictly from validation metrics.
        selected_family = max(
            family_candidates,
            key=lambda family: family_candidates[family]["selection_score"],
        )
        selected_validation = family_candidates[selected_family]
        selected_test = family_tests[selected_family]
        selected_artifact = artifacts[selected_family]

        selected_copy = method_dir / "selected_best_temporal_filter.pt"
        shutil.copy2(selected_artifact, selected_copy)

        selected_metadata = {
            "method": method_name,
            "classifier_dir": str(classifier_dir.resolve()),
            "selected_family": selected_family,
            "selection_based_on": "ordered validation selection_score",
            "validation_selection_score": selected_validation["selection_score"],
            "selected_config": candidate_config(selected_validation),
            "selected_temporal_filter": str(selected_copy.resolve()),
            "test_metrics_report_only": selected_test,
        }

        with open(method_dir / "selected_best.json", "w") as f:
            json.dump(_jsonable(selected_metadata), f, indent=2)

        result = {
            "method": method_name,
            "classifier_dir": str(classifier_dir.resolve()),
            "calibration_source": calibration_source,
            "search_runtime_sec": search_runtime,
            "search_config": search["search_config"],
            "metric_config": metric_kwargs,
            "instantaneous_validation": instantaneous_validation,
            "instantaneous_test": instantaneous_test,
            "best_stage1_validation": search["best_fixed"],
            "stage1_test": fixed_test,
            "best_stage2_validation": search["best_event"],
            "stage2_test": event_test,
            "best_stage3_validation": search["best_ambiguity"],
            "stage3_test": ambiguity_test,
            "best_ema_validation": search["best_ema"],
            "ema_test": ema_test,
            "selected": selected_metadata,
            "stage1_trials": search["fixed_results"],
            "stage2_trials": search["event_results"],
            "stage3_trials": search["ambiguity_results"],
            "ema_trials": search["ema_results"],
        }

        with open(method_dir / "sequential_search_results.json", "w") as f:
            json.dump(_jsonable(result), f, indent=2)

        per_classifier_best.append(selected_metadata)

        # Instantaneous row is diagnostic only and cannot be selected as the final
        # temporal model requested by this script.
        all_rows.append(
            comparison_row(
                method=method_name,
                family="instantaneous_unfiltered",
                validation=instantaneous_validation,
                test=instantaneous_test,
                config={},
                artifact_path=None,
            )
        )

        for family in family_candidates:
            all_rows.append(
                comparison_row(
                    method=method_name,
                    family=family,
                    validation=family_candidates[family],
                    test=family_tests[family],
                    config=candidate_config(family_candidates[family]),
                    artifact_path=artifacts[family],
                    selected_per_classifier=(family == selected_family),
                )
            )

        detailed_suite.append(
            {
                "method": method_name,
                "classifier_dir": str(classifier_dir.resolve()),
                "selected_family": selected_family,
                "selected_validation_score": selected_validation[
                    "selection_score"
                ],
                "selected_test_balanced_accuracy": selected_test[
                    "balanced_accuracy"
                ],
                "selected_test_transition_window_accuracy": selected_test[
                    "transition_window_accuracy"
                ],
                "search_runtime_sec": search_runtime,
            }
        )

        print(
            json.dumps(
                _jsonable(
                    {
                        "method": method_name,
                        "selected_family": selected_family,
                        "validation_selection_score": selected_validation[
                            "selection_score"
                        ],
                        "selected_config": candidate_config(selected_validation),
                        "test_balanced_accuracy": selected_test[
                            "balanced_accuracy"
                        ],
                        "test_transition_window_accuracy": selected_test[
                            "transition_window_accuracy"
                        ],
                    }
                ),
                indent=2,
            )
        )

        del cal_inputs, val_inputs, test_inputs, test_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Global winner across classifier types + temporal approaches.
    # Still validation-selected; test results are not used for selection.
    # -------------------------------------------------------------------------
    selectable_rows = [
        row for row in all_rows if row["family"] != "instantaneous_unfiltered"
    ]
    global_best_row = max(
        selectable_rows,
        key=lambda row: float(row["validation_selection_score"]),
    )

    global_best_row["selected_global"] = True

    global_best_artifact = Path(global_best_row["artifact_path"])
    global_best_copy = output_dir / "global_best_temporal_filter.pt"
    shutil.copy2(global_best_artifact, global_best_copy)

    global_best = {
        "method": global_best_row["method"],
        "family": global_best_row["family"],
        "selection_based_on": "ordered validation selection_score",
        "validation_selection_score": global_best_row[
            "validation_selection_score"
        ],
        "config_json": global_best_row["config_json"],
        "source_temporal_filter": str(global_best_artifact.resolve()),
        "copied_temporal_filter": str(global_best_copy.resolve()),
        "test_metrics_report_only": {
            key: value
            for key, value in global_best_row.items()
            if key.startswith("test_")
        },
    }

    # Mark matching row in all_rows.
    for row in all_rows:
        if (
            row["method"] == global_best_row["method"]
            and row["family"] == global_best_row["family"]
        ):
            row["selected_global"] = True

    # -------------------------------------------------------------------------
    # Root outputs.
    # -------------------------------------------------------------------------
    with open(output_dir / "approach_comparison.json", "w") as f:
        json.dump(_jsonable(all_rows), f, indent=2)

    with open(output_dir / "per_classifier_best.json", "w") as f:
        json.dump(_jsonable(per_classifier_best), f, indent=2)

    with open(output_dir / "global_best_selection.json", "w") as f:
        json.dump(_jsonable(global_best), f, indent=2)

    with open(output_dir / "suite_summary.json", "w") as f:
        json.dump(_jsonable(detailed_suite), f, indent=2)

    csv_path = output_dir / "approach_comparison.csv"
    fieldnames = list(all_rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(_jsonable(row))

    print("\n=== Global validation-selected winner ===")
    print(json.dumps(_jsonable(global_best), indent=2))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()