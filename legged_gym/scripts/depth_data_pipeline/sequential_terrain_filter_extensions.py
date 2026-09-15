"""
Sequential terrain-filter extensions.

Contains:
  - EventConditionedBayesianTerrainFilter
  - AmbiguityAwareEventConditionedBayesianTerrainFilter
  - EMALogitPatienceFilter
  - sequence runners for the two Bayes extensions

These classes build on the existing BayesianTerrainFilter API.
"""

from __future__ import annotations

from functools import cmp_to_key
from pathlib import Path
import math
from typing import Any, Hashable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    BayesianFilterStep,
    BayesianTerrainFilter,
    estimate_observation_matrix_from_probabilities,
    evaluate_predictions,
    evaluate_transition_accounting,
    make_persistent_transition_matrix,
    run_filter_sequences,
)


def _row_stochastic(
    matrix: torch.Tensor | Sequence,
    size: int,
    device: torch.device,
    eps: float,
    name: str,
) -> torch.Tensor:
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    if x.shape != (size, size):
        raise ValueError(f"{name} must have shape [{size},{size}]")
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    x = x.clamp_min(eps)
    return x / x.sum(dim=1, keepdim=True).clamp_min(eps)


class EventConditionedBayesianTerrainFilter(BayesianTerrainFilter):
    """
    Bayes filter with high persistence during stable operation and a separate,
    lower-persistence transition model during a detected terrain-change event.

    A change event is proposed when the instantaneous classifier's argmax differs
    from the currently emitted filtered state and its maximum probability exceeds
    `switch_confidence`. The same candidate must persist for `change_patience`
    consecutive clear frames before the low-persistence transition matrix is used.

    Patience gates the transition-model release; normal Bayes evidence is still
    processed on clear frames before patience has elapsed.
    """

    def __init__(
        self,
        labels: Sequence[Hashable],
        prior: torch.Tensor | Sequence[float] | Mapping[Hashable, float],
        stable_transition_matrix: torch.Tensor | Sequence,
        switch_transition_matrix: torch.Tensor | Sequence,
        observation_matrix: Optional[torch.Tensor | Sequence] = None,
        *,
        switch_confidence: float = 0.70,
        change_patience: int = 2,
        evidence_power: float = 1.0,
        adaptive_evidence: bool = False,
        min_evidence_power: float = 1.0,
        confidence_gamma: float = 1.0,
        device: str | torch.device = "cpu",
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= switch_confidence <= 1.0:
            raise ValueError("switch_confidence must be in [0,1]")
        if int(change_patience) < 1:
            raise ValueError("change_patience must be >= 1")

        super().__init__(
            labels=labels,
            prior=prior,
            transition_matrix=stable_transition_matrix,
            observation_matrix=observation_matrix,
            evidence_power=evidence_power,
            adaptive_evidence=adaptive_evidence,
            min_evidence_power=min_evidence_power,
            confidence_gamma=confidence_gamma,
            device=device,
            eps=eps,
        )

        self.stable_transition_matrix = self.transition_matrix.clone()
        self.switch_transition_matrix = _row_stochastic(
            switch_transition_matrix,
            self.num_classes,
            self.device,
            self.eps,
            "switch_transition_matrix",
        )
        self.switch_confidence = float(switch_confidence)
        self.change_patience = int(change_patience)

        self.pending_target_index: Optional[int] = None
        self.pending_count = 0
        self.last_used_switch_transition = False
        self.current_output_index = int(self.belief.argmax())

    def _normalize_classifier_probabilities(
        self,
        classifier_probabilities: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        q = torch.as_tensor(
            classifier_probabilities,
            dtype=torch.float32,
            device=self.device,
        ).flatten()
        if q.numel() != self.num_classes:
            raise ValueError(
                f"classifier_probabilities must have {self.num_classes} entries"
            )
        if not torch.isfinite(q).all() or (q < 0).any():
            raise ValueError("classifier_probabilities must be finite and non-negative")
        q = q.clamp_min(self.eps)
        return q / q.sum().clamp_min(self.eps)

    def _select_transition_matrix(self, q: torch.Tensor) -> bool:
        candidate_index = int(q.argmax())
        candidate_confidence = float(q[candidate_index])

        disagreement = (
            candidate_index != self.current_output_index
            and candidate_confidence >= self.switch_confidence
        )

        if disagreement:
            if self.pending_target_index == candidate_index:
                self.pending_count += 1
            else:
                self.pending_target_index = candidate_index
                self.pending_count = 1
        else:
            self.pending_target_index = None
            self.pending_count = 0

        use_switch = disagreement and self.pending_count >= self.change_patience
        self.transition_matrix = (
            self.switch_transition_matrix
            if use_switch
            else self.stable_transition_matrix
        )
        self.last_used_switch_transition = bool(use_switch)
        return bool(use_switch)

    @torch.inference_mode()
    def update(
        self,
        classifier_probabilities: torch.Tensor | Sequence[float],
        *,
        observation_quality: float = 1.0,
    ) -> BayesianFilterStep:
        q = self._normalize_classifier_probabilities(classifier_probabilities)
        previous_output = self.current_output_index

        self._select_transition_matrix(q)
        step = super().update(q, observation_quality=observation_quality)

        new_output = int(step.posterior.argmax())
        self.current_output_index = new_output

        if new_output != previous_output:
            self.pending_target_index = None
            self.pending_count = 0

        # Restore stable T after each step. _select_transition_matrix() will
        # explicitly reactivate switch T on the next qualifying frame.
        self.transition_matrix = self.stable_transition_matrix

        return BayesianFilterStep(
            self.labels[self.current_output_index],
            step.posterior,
            step.predicted_prior,
            step.observation_likelihood,
            step.classifier_probabilities,
            step.confidence,
            step.evidence_power,
        )

    def reset(
        self,
        prior: Optional[
            torch.Tensor | Sequence[float] | Mapping[Hashable, float]
        ] = None,
    ) -> torch.Tensor:
        belief = super().reset(prior)
        self.transition_matrix = self.stable_transition_matrix
        self.pending_target_index = None
        self.pending_count = 0
        self.last_used_switch_transition = False
        self.current_output_index = int(self.belief.argmax())
        return belief

    def _base_state_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "initial_prior": self.initial_prior.cpu(),
            "belief": self.belief.cpu(),
            "stable_transition_matrix": self.stable_transition_matrix.cpu(),
            "switch_transition_matrix": self.switch_transition_matrix.cpu(),
            "observation_matrix": self.observation_matrix.cpu(),
            "switch_confidence": self.switch_confidence,
            "change_patience": self.change_patience,
            "evidence_power": self.evidence_power,
            "adaptive_evidence": self.adaptive_evidence,
            "min_evidence_power": self.min_evidence_power,
            "confidence_gamma": self.confidence_gamma,
            "pending_target_index": self.pending_target_index,
            "pending_count": self.pending_count,
            "current_output_index": self.current_output_index,
            "eps": self.eps,
        }

    def save(self, path: str | Path) -> None:
        state = self._base_state_dict()
        state["filter_class"] = self.__class__.__name__
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "EventConditionedBayesianTerrainFilter":
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)

        filt = cls(
            labels=state["labels"],
            prior=state["initial_prior"],
            stable_transition_matrix=state["stable_transition_matrix"],
            switch_transition_matrix=state["switch_transition_matrix"],
            observation_matrix=state["observation_matrix"],
            switch_confidence=state["switch_confidence"],
            change_patience=state["change_patience"],
            evidence_power=state["evidence_power"],
            adaptive_evidence=state["adaptive_evidence"],
            min_evidence_power=state["min_evidence_power"],
            confidence_gamma=state["confidence_gamma"],
            device=device,
            eps=state["eps"],
        )
        filt.belief = state["belief"].to(filt.device)
        filt.pending_target_index = state.get("pending_target_index")
        filt.pending_count = int(state.get("pending_count", 0))
        filt.current_output_index = int(
            state.get("current_output_index", int(filt.belief.argmax()))
        )
        return filt


class AmbiguityAwareEventConditionedBayesianTerrainFilter(
    EventConditionedBayesianTerrainFilter
):
    """
    Event-conditioned Bayes filter with explicit ambiguous-frame handling.

    A frame is ambiguous when the top-two instantaneous classifier probabilities
    differ by <= `ambiguity_margin_threshold`.

    On an ambiguous frame:
      1) keep the current emitted terrain label;
      2) reset event-switch patience;
      3) do not apply class evidence from the ambiguous observation;
      4) propagate with the stable transition matrix;
      5) flatten the propagated belief toward the initial prior:

           b <- (1-lambda) * predicted + lambda * prior

    This allows uncertainty to reduce accumulated certainty without making
    uncertainty itself evidence for the previous terrain.
    """

    def __init__(
        self,
        labels: Sequence[Hashable],
        prior: torch.Tensor | Sequence[float] | Mapping[Hashable, float],
        stable_transition_matrix: torch.Tensor | Sequence,
        switch_transition_matrix: torch.Tensor | Sequence,
        observation_matrix: Optional[torch.Tensor | Sequence] = None,
        *,
        switch_confidence: float = 0.70,
        change_patience: int = 2,
        ambiguity_margin_threshold: float = 0.20,
        ambiguity_flatten_strength: float = 0.25,
        evidence_power: float = 1.0,
        adaptive_evidence: bool = False,
        min_evidence_power: float = 1.0,
        confidence_gamma: float = 1.0,
        device: str | torch.device = "cpu",
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= ambiguity_margin_threshold <= 1.0:
            raise ValueError("ambiguity_margin_threshold must be in [0,1]")
        if not 0.0 <= ambiguity_flatten_strength <= 1.0:
            raise ValueError("ambiguity_flatten_strength must be in [0,1]")

        super().__init__(
            labels=labels,
            prior=prior,
            stable_transition_matrix=stable_transition_matrix,
            switch_transition_matrix=switch_transition_matrix,
            observation_matrix=observation_matrix,
            switch_confidence=switch_confidence,
            change_patience=change_patience,
            evidence_power=evidence_power,
            adaptive_evidence=adaptive_evidence,
            min_evidence_power=min_evidence_power,
            confidence_gamma=confidence_gamma,
            device=device,
            eps=eps,
        )

        self.ambiguity_margin_threshold = float(ambiguity_margin_threshold)
        self.ambiguity_flatten_strength = float(ambiguity_flatten_strength)
        self.last_was_ambiguous = False
        self.last_top2_margin = 1.0

    def top2_margin(self, q: torch.Tensor) -> float:
        if self.num_classes <= 1:
            return 1.0
        values = torch.topk(q, k=2).values
        return float(values[0] - values[1])

    @torch.inference_mode()
    def update(
        self,
        classifier_probabilities: torch.Tensor | Sequence[float],
        *,
        observation_quality: float = 1.0,
    ) -> BayesianFilterStep:
        q = self._normalize_classifier_probabilities(classifier_probabilities)
        margin = self.top2_margin(q)

        self.last_top2_margin = margin
        self.last_was_ambiguous = margin <= self.ambiguity_margin_threshold

        if not self.last_was_ambiguous:
            return super().update(q, observation_quality=observation_quality)

        # Ambiguous frame: it cannot contribute to a pending state switch.
        self.pending_target_index = None
        self.pending_count = 0
        self.last_used_switch_transition = False
        self.transition_matrix = self.stable_transition_matrix

        predicted = self.belief @ self.stable_transition_matrix
        predicted = predicted / predicted.sum().clamp_min(self.eps)

        lam = self.ambiguity_flatten_strength
        posterior = (1.0 - lam) * predicted + lam * self.initial_prior
        posterior = posterior / posterior.sum().clamp_min(self.eps)
        self.belief = posterior

        # No class-specific evidence was applied. Return an all-ones likelihood
        # and beta=0 for diagnostics while preserving the current output label.
        likelihood = torch.ones(self.num_classes, device=self.device)
        confidence = self.entropy_confidence(q)

        return BayesianFilterStep(
            self.labels[self.current_output_index],
            posterior.clone(),
            predicted.clone(),
            likelihood,
            q.clone(),
            confidence,
            0.0,
        )

    def reset(
        self,
        prior: Optional[
            torch.Tensor | Sequence[float] | Mapping[Hashable, float]
        ] = None,
    ) -> torch.Tensor:
        belief = super().reset(prior)
        self.last_was_ambiguous = False
        self.last_top2_margin = 1.0
        return belief

    def save(self, path: str | Path) -> None:
        state = self._base_state_dict()
        state.update(
            {
                "filter_class": self.__class__.__name__,
                "ambiguity_margin_threshold": self.ambiguity_margin_threshold,
                "ambiguity_flatten_strength": self.ambiguity_flatten_strength,
                "last_was_ambiguous": self.last_was_ambiguous,
                "last_top2_margin": self.last_top2_margin,
            }
        )
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "AmbiguityAwareEventConditionedBayesianTerrainFilter":
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)

        filt = cls(
            labels=state["labels"],
            prior=state["initial_prior"],
            stable_transition_matrix=state["stable_transition_matrix"],
            switch_transition_matrix=state["switch_transition_matrix"],
            observation_matrix=state["observation_matrix"],
            switch_confidence=state["switch_confidence"],
            change_patience=state["change_patience"],
            ambiguity_margin_threshold=state["ambiguity_margin_threshold"],
            ambiguity_flatten_strength=state["ambiguity_flatten_strength"],
            evidence_power=state["evidence_power"],
            adaptive_evidence=state["adaptive_evidence"],
            min_evidence_power=state["min_evidence_power"],
            confidence_gamma=state["confidence_gamma"],
            device=device,
            eps=state["eps"],
        )
        filt.belief = state["belief"].to(filt.device)
        filt.pending_target_index = state.get("pending_target_index")
        filt.pending_count = int(state.get("pending_count", 0))
        filt.current_output_index = int(
            state.get("current_output_index", int(filt.belief.argmax()))
        )
        filt.last_was_ambiguous = bool(state.get("last_was_ambiguous", False))
        filt.last_top2_margin = float(state.get("last_top2_margin", 1.0))
        return filt


class EMALogitPatienceFilter:
    """
    Lightweight non-Bayesian temporal baseline.

    Smooths classifier scores/logits:
        z_bar_t = alpha*z_t + (1-alpha)*z_bar_{t-1}

    and changes the emitted label only after the same new argmax persists for
    `change_patience` consecutive frames. alpha=1 gives a patience-only gate.
    """

    def __init__(
        self,
        labels: Sequence[Hashable],
        *,
        ema_alpha: float = 0.8,
        change_patience: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0,1]")
        if int(change_patience) < 1:
            raise ValueError("change_patience must be >= 1")
        self.labels = list(labels)
        self.ema_alpha = float(ema_alpha)
        self.change_patience = int(change_patience)
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.ema_scores: Optional[torch.Tensor] = None
        self.current_output_index: Optional[int] = None
        self.pending_target_index: Optional[int] = None
        self.pending_count = 0

    @torch.inference_mode()
    def update(self, scores: torch.Tensor | Sequence[float]) -> Hashable:
        z = torch.as_tensor(scores, dtype=torch.float32, device=self.device).flatten()
        if z.numel() != len(self.labels):
            raise ValueError(f"scores must have {len(self.labels)} entries")

        if self.ema_scores is None:
            self.ema_scores = z.clone()
            self.current_output_index = int(self.ema_scores.argmax())
            return self.labels[self.current_output_index]

        a = self.ema_alpha
        self.ema_scores = a * z + (1.0 - a) * self.ema_scores
        candidate = int(self.ema_scores.argmax())

        if candidate == self.current_output_index:
            self.pending_target_index = None
            self.pending_count = 0
        else:
            if self.pending_target_index == candidate:
                self.pending_count += 1
            else:
                self.pending_target_index = candidate
                self.pending_count = 1

            if self.pending_count >= self.change_patience:
                self.current_output_index = candidate
                self.pending_target_index = None
                self.pending_count = 0

        return self.labels[self.current_output_index]

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "labels": list(self.labels),
                "ema_alpha": self.ema_alpha,
                "change_patience": self.change_patience,
                "ema_scores": None if self.ema_scores is None else self.ema_scores.cpu(),
                "current_output_index": self.current_output_index,
                "pending_target_index": self.pending_target_index,
                "pending_count": self.pending_count,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "EMALogitPatienceFilter":
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)
        filt = cls(
            state["labels"],
            ema_alpha=state["ema_alpha"],
            change_patience=state["change_patience"],
            device=device,
        )
        if state.get("ema_scores") is not None:
            filt.ema_scores = state["ema_scores"].to(filt.device)
        filt.current_output_index = state.get("current_output_index")
        filt.pending_target_index = state.get("pending_target_index")
        filt.pending_count = int(state.get("pending_count", 0))
        return filt


@torch.inference_mode()
def run_event_conditioned_filter_sequences(
    terrain_filter: EventConditionedBayesianTerrainFilter,
    classifier_probabilities: torch.Tensor | Sequence,
    *,
    sequence_ids: Optional[Sequence] = None,
    observation_quality: Optional[torch.Tensor | Sequence[float]] = None,
    prior: Optional[
        torch.Tensor | Sequence[float] | Mapping[Hashable, float]
    ] = None,
):
    probabilities = torch.as_tensor(classifier_probabilities, dtype=torch.float32)
    n = probabilities.shape[0]
    ids = list(sequence_ids) if sequence_ids is not None else [0] * n
    qualities = (
        torch.ones(n)
        if observation_quality is None
        else torch.as_tensor(observation_quality, dtype=torch.float32).flatten()
    )

    predictions, posteriors, evidence_powers, switch_events = [], [], [], []
    previous_id = object()

    for i in range(n):
        if i == 0 or ids[i] != previous_id:
            terrain_filter.reset(prior)
        previous_id = ids[i]

        step = terrain_filter.update(
            probabilities[i],
            observation_quality=float(qualities[i]),
        )
        predictions.append(step.label)
        posteriors.append(step.posterior.cpu())
        evidence_powers.append(step.evidence_power)
        switch_events.append(terrain_filter.last_used_switch_transition)

    return (
        predictions,
        torch.stack(posteriors),
        torch.tensor(evidence_powers),
        torch.tensor(switch_events, dtype=torch.bool),
    )


@torch.inference_mode()
def run_ambiguity_aware_filter_sequences(
    terrain_filter: AmbiguityAwareEventConditionedBayesianTerrainFilter,
    classifier_probabilities: torch.Tensor | Sequence,
    *,
    sequence_ids: Optional[Sequence] = None,
    observation_quality: Optional[torch.Tensor | Sequence[float]] = None,
    prior: Optional[
        torch.Tensor | Sequence[float] | Mapping[Hashable, float]
    ] = None,
):
    probabilities = torch.as_tensor(classifier_probabilities, dtype=torch.float32)
    n = probabilities.shape[0]
    ids = list(sequence_ids) if sequence_ids is not None else [0] * n
    qualities = (
        torch.ones(n)
        if observation_quality is None
        else torch.as_tensor(observation_quality, dtype=torch.float32).flatten()
    )

    predictions, posteriors, evidence_powers = [], [], []
    switch_events, ambiguous_events, margins = [], [], []
    previous_id = object()

    for i in range(n):
        if i == 0 or ids[i] != previous_id:
            terrain_filter.reset(prior)
        previous_id = ids[i]

        step = terrain_filter.update(
            probabilities[i],
            observation_quality=float(qualities[i]),
        )
        predictions.append(step.label)
        posteriors.append(step.posterior.cpu())
        evidence_powers.append(step.evidence_power)
        switch_events.append(terrain_filter.last_used_switch_transition)
        ambiguous_events.append(terrain_filter.last_was_ambiguous)
        margins.append(terrain_filter.last_top2_margin)

    return (
        predictions,
        torch.stack(posteriors),
        torch.tensor(evidence_powers),
        torch.tensor(switch_events, dtype=torch.bool),
        torch.tensor(ambiguous_events, dtype=torch.bool),
        torch.tensor(margins, dtype=torch.float32),
    )


@torch.inference_mode()
def run_ema_logit_patience_sequences(
    terrain_filter: EMALogitPatienceFilter,
    classifier_scores: torch.Tensor | Sequence,
    *,
    sequence_ids: Optional[Sequence] = None,
):
    scores = torch.as_tensor(classifier_scores, dtype=torch.float32)
    ids = list(sequence_ids) if sequence_ids is not None else [0] * scores.shape[0]

    predictions = []
    previous_id = object()
    for i in range(scores.shape[0]):
        if i == 0 or ids[i] != previous_id:
            terrain_filter.reset()
        previous_id = ids[i]
        predictions.append(terrain_filter.update(scores[i]))
    return predictions


# Shared staged search used by every terrain-classifier trainer.  Keeping the
# recurrence here also keeps the deployable filter implementations and their
# search semantics in one place.
SEQUENTIAL_METRIC_CONFIG = {
    "transition_window_radius": 5,
    "transition_accuracy_weight": 0.40,
    "delay_weight": 0.003,
    "false_transition_weight": 0.05,
    "missing_class_penalty": 0.05,
}
TEMPERATURES = (0.25, 0.40, 0.50, 0.75, 1.0, 1.5)
OBSERVATION_MIXES = (0.0, 0.25)
STABLE_STAYS = (0.70, 0.85, 0.95, 0.98)
SWITCH_STAYS = (0.10, 0.30, 0.50, 0.70)
SWITCH_CONFIDENCES = (0.65, 0.80)
CHANGE_PATIENCES = (1, 2)
AMBIGUITY_THRESHOLDS = (0.10, 0.20, 0.30)
AMBIGUITY_FLATTEN_STRENGTHS = (0.10, 0.25, 0.40)
EMA_ALPHAS = (0.40, 0.60, 0.80, 1.0)
EMA_PATIENCES = (1, 2)
OBSERVATION_PSEUDOCOUNT = 0.5


def _transition_window_mask(truth: Sequence, sequence_ids: Sequence, radius: int) -> torch.Tensor:
    ids, mask = list(sequence_ids), torch.zeros(len(truth), dtype=torch.bool)
    for t in range(1, len(truth)):
        if ids[t] == ids[t - 1] and truth[t] != truth[t - 1]:
            for j in range(max(0, t - radius), min(len(truth), t + radius + 1)):
                if ids[j] == ids[t]:
                    mask[j] = True
    return mask


def _ambiguity_run_length(mask: torch.Tensor, sequence_ids: Sequence) -> float:
    ids, runs, run = list(sequence_ids), [], 0
    for i, flag in enumerate(mask.tolist()):
        if i and ids[i] != ids[i - 1]:
            if run:
                runs.append(run)
            run = 0
        if flag:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return float(sum(runs) / len(runs)) if runs else 0.0


def evaluate_sequential_predictions(
    truth: Sequence, predictions: Sequence, labels: Sequence, sequence_ids: Sequence,
    metric_config: Mapping[str, float] = SEQUENTIAL_METRIC_CONFIG,
    *, transition_accounting_v2: bool = False,
) -> dict[str, Any]:
    truth, predictions, labels, ids = list(truth), list(predictions), list(labels), list(sequence_ids)
    base = evaluate_predictions(truth, predictions, labels, sequence_ids=ids)
    matrix = torch.zeros(len(labels), len(labels), dtype=torch.int64)
    index = {label: i for i, label in enumerate(labels)}
    for actual, predicted in zip(truth, predictions):
        matrix[index[actual], index[predicted]] += 1
    recalls, precisions = {}, {}
    for i, label in enumerate(labels):
        tp = float(matrix[i, i])
        recalls[str(label)] = tp / max(float(matrix[i].sum()), 1.0)
        precisions[str(label)] = tp / max(float(matrix[:, i].sum()), 1.0)
    missing = [str(label) for i, label in enumerate(labels) if int(matrix[:, i].sum()) == 0]
    window = _transition_window_mask(truth, ids, int(metric_config["transition_window_radius"]))
    correct = torch.tensor([a == b for a, b in zip(truth, predictions)], dtype=torch.float32)
    transition_accuracy = float(correct[window].mean()) if bool(window.any()) else float("nan")
    steady_accuracy = float(correct[~window].mean()) if bool((~window).any()) else float("nan")
    transition_term = transition_accuracy if math.isfinite(transition_accuracy) else float(base.balanced_accuracy)
    score = ((1.0 - float(metric_config["transition_accuracy_weight"])) * float(base.balanced_accuracy)
             + float(metric_config["transition_accuracy_weight"]) * transition_term)
    if math.isfinite(float(base.mean_transition_delay)):
        score -= float(metric_config["delay_weight"]) * float(base.mean_transition_delay)
    if math.isfinite(float(base.false_transition_rate)):
        score -= float(metric_config["false_transition_weight"]) * float(base.false_transition_rate)
    score -= float(metric_config["missing_class_penalty"]) * len(missing)
    legacy_score = 0.65 * float(base.balanced_accuracy) + 0.35 * transition_term
    if math.isfinite(float(base.mean_transition_delay)):
        legacy_score -= 0.002 * float(base.mean_transition_delay)
    if math.isfinite(float(base.false_transition_rate)):
        legacy_score -= 0.05 * float(base.false_transition_rate)
    legacy_score -= 0.05 * len(missing)
    return {
        "selection_score": float(score), "score_v2": float(score),
        "legacy_selection_score": float(legacy_score), **base.as_dict(),
        "transition_window_accuracy": transition_accuracy,
        "steady_state_accuracy": steady_accuracy,
        "transition_window_frame_fraction": float(window.float().mean()),
        "per_class_recall": recalls, "per_class_precision": precisions,
        "minimum_class_recall": min(recalls.values()) if recalls else float("nan"),
        "missing_predicted_classes": missing, "confusion_matrix": matrix,
        **(evaluate_transition_accounting(truth, predictions, ids) if transition_accounting_v2 else {}),
    }


def _probs(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    probabilities = F.softmax(torch.as_tensor(scores).cpu() / float(temperature), dim=1).clamp_min(1e-8)
    return probabilities / probabilities.sum(1, keepdim=True)


def _mixed_observation(soft: torch.Tensor, rho: float) -> torch.Tensor:
    return (1.0 - float(rho)) * torch.eye(soft.shape[0]) + float(rho) * soft


def _params(candidate: Mapping[str, Any]) -> dict[str, Any]:
    metric_keys = {
        "selection_score", "accuracy", "balanced_accuracy", "macro_f1",
        "mean_transition_delay", "false_transition_rate", "transition_window_accuracy",
        "steady_state_accuracy", "transition_window_frame_fraction", "per_class_recall",
        "per_class_precision", "minimum_class_recall", "missing_predicted_classes",
        "confusion_matrix", "switch_transition_fraction", "ambiguous_frame_fraction",
        "mean_ambiguity_run_length",
    }
    return {k: v for k, v in candidate.items() if k not in metric_keys and k != "family"}


def _make_filter(candidate: Mapping[str, Any], labels: Sequence, prior: Mapping, observation: torch.Tensor):
    stable = make_persistent_transition_matrix(labels, candidate["stable_stay"], device="cpu")
    family = candidate["family"]
    if family == "stage1_fixed_bayes":
        return BayesianTerrainFilter(labels, prior, stable, observation, evidence_power=1.0,
                                     adaptive_evidence=False, min_evidence_power=1.0,
                                     confidence_gamma=1.0, device="cpu")
    switch = make_persistent_transition_matrix(labels, candidate["switch_stay"], device="cpu")
    common = dict(labels=labels, prior=prior, stable_transition_matrix=stable,
                  switch_transition_matrix=switch, observation_matrix=observation,
                  switch_confidence=candidate["switch_confidence"],
                  change_patience=candidate["change_patience"], evidence_power=1.0,
                  adaptive_evidence=False, min_evidence_power=1.0,
                  confidence_gamma=1.0, device="cpu")
    if family == "stage2_event_bayes":
        return EventConditionedBayesianTerrainFilter(**common)
    return AmbiguityAwareEventConditionedBayesianTerrainFilter(
        **common, ambiguity_margin_threshold=candidate["ambiguity_margin_threshold"],
        ambiguity_flatten_strength=candidate["ambiguity_flatten_strength"])


@torch.inference_mode()
def _evaluate_candidate(candidate, scores, truth, ids, labels, prior, soft_observations,
                        metric_config, probability_cache=None):
    family = candidate["family"]
    if family == "ema_logit_patience":
        filt = EMALogitPatienceFilter(labels, ema_alpha=candidate["ema_alpha"],
                                      change_patience=candidate["change_patience"])
        predictions = run_ema_logit_patience_sequences(filt, scores, sequence_ids=ids)
        return filt, evaluate_sequential_predictions(truth, predictions, labels, ids, metric_config)
    temperature, rho = candidate["temperature"], candidate["observation_mix"]
    observation = _mixed_observation(soft_observations[temperature], rho)
    filt = _make_filter(candidate, labels, prior, observation)
    probabilities = (_probs(scores, temperature) if probability_cache is None
                     else probability_cache[temperature])
    if family == "stage1_fixed_bayes":
        predictions, _, _ = run_filter_sequences(filt, probabilities, sequence_ids=ids)
        extras = {}
    elif family == "stage2_event_bayes":
        predictions, _, _, switches = run_event_conditioned_filter_sequences(filt, probabilities, sequence_ids=ids)
        extras = {"switch_transition_fraction": float(switches.float().mean())}
    else:
        predictions, _, _, switches, ambiguous, _ = run_ambiguity_aware_filter_sequences(
            filt, probabilities, sequence_ids=ids)
        extras = {"switch_transition_fraction": float(switches.float().mean()),
                  "ambiguous_frame_fraction": float(ambiguous.float().mean()),
                  "mean_ambiguity_run_length": _ambiguity_run_length(ambiguous, ids)}
    return filt, {**evaluate_sequential_predictions(truth, predictions, labels, ids, metric_config), **extras}


@torch.inference_mode()
def staged_sequential_search(
    classifier: Any, calibration_scores: torch.Tensor, calibration_labels: Sequence,
    validation_scores: torch.Tensor, validation_labels: Sequence, validation_sequence_ids: Sequence,
    test_scores: torch.Tensor, test_labels: Sequence, test_sequence_ids: Sequence,
    output_dir: str | Path, metric_config: Optional[Mapping[str, float]] = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Tune all temporal families on ordered validation and evaluate test once selected."""
    output_dir = Path(output_dir)
    config = dict(SEQUENTIAL_METRIC_CONFIG if metric_config is None else metric_config)
    labels = list(classifier.class_ids)
    prior = {label: 1.0 / len(labels) for label in labels}
    validation_labels = classifier._normalize_labels(validation_labels)
    test_labels = classifier._normalize_labels(test_labels)
    calibration_labels = classifier._normalize_labels(calibration_labels)
    val_probs = {t: _probs(validation_scores, t) for t in TEMPERATURES}
    soft = {t: estimate_observation_matrix_from_probabilities(
        _probs(calibration_scores, t), calibration_labels, labels, mode="soft",
        pseudocount=OBSERVATION_PSEUDOCOUNT, device="cpu") for t in TEMPERATURES}

    stage1 = []
    for t in TEMPERATURES:
        for rho in OBSERVATION_MIXES:
            for stay in STABLE_STAYS:
                candidate = {"family": "stage1_fixed_bayes", "temperature": t,
                             "observation_mix": rho, "stable_stay": stay,
                             "evidence_power": 1.0, "adaptive_evidence": False,
                             "observation_pseudocount": OBSERVATION_PSEUDOCOUNT}
                _, metrics = _evaluate_candidate(candidate, validation_scores, validation_labels,
                    validation_sequence_ids, labels, prior, soft, config, val_probs)
                stage1.append({**candidate, **metrics})
    stage1.sort(key=lambda x: x["selection_score"], reverse=True)

    stage2, seen = [], set()
    for seed in stage1[:2]:
        for switch_stay in SWITCH_STAYS:
            if switch_stay >= seed["stable_stay"]:
                continue
            for confidence in SWITCH_CONFIDENCES:
                for patience in CHANGE_PATIENCES:
                    key = (seed["temperature"], seed["observation_mix"], seed["stable_stay"],
                           switch_stay, confidence, patience)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = {**_params(seed), "family": "stage2_event_bayes",
                                 "switch_stay": switch_stay, "switch_confidence": confidence,
                                 "change_patience": patience}
                    _, metrics = _evaluate_candidate(candidate, validation_scores, validation_labels,
                        validation_sequence_ids, labels, prior, soft, config, val_probs)
                    stage2.append({**candidate, **metrics})
    stage2.sort(key=lambda x: x["selection_score"], reverse=True)

    stage3, seen = [], set()
    for seed in stage2[:2]:
        for threshold in AMBIGUITY_THRESHOLDS:
            for strength in AMBIGUITY_FLATTEN_STRENGTHS:
                key = (seed["temperature"], seed["observation_mix"], seed["stable_stay"],
                       seed["switch_stay"], seed["switch_confidence"],
                       seed["change_patience"], threshold, strength)
                if key in seen:
                    continue
                seen.add(key)
                candidate = {**_params(seed), "family": "stage3_ambiguity_bayes",
                             "ambiguity_margin_threshold": threshold,
                             "ambiguity_flatten_strength": strength}
                _, metrics = _evaluate_candidate(candidate, validation_scores, validation_labels,
                    validation_sequence_ids, labels, prior, soft, config, val_probs)
                stage3.append({**candidate, **metrics})
    stage3.sort(key=lambda x: x["selection_score"], reverse=True)

    ema = []
    for alpha in EMA_ALPHAS:
        for patience in EMA_PATIENCES:
            candidate = {"family": "ema_logit_patience", "ema_alpha": alpha,
                         "change_patience": patience}
            _, metrics = _evaluate_candidate(candidate, validation_scores, validation_labels,
                validation_sequence_ids, labels, prior, soft, config)
            ema.append({**candidate, **metrics})
    ema.sort(key=lambda x: x["selection_score"], reverse=True)

    bests = [stage1[0], stage2[0], stage3[0], ema[0]]
    test_soft = soft  # observation calibration remains structural calibration only
    test_probs = {t: _probs(test_scores, t) for t in TEMPERATURES}
    stage_records, filters = {}, {}
    names = ("stage1_fixed_bayes", "stage2_event_bayes", "stage3_ambiguity_bayes", "ema_logit_patience")
    for name, best in zip(names, bests):
        filt, test_metrics = _evaluate_candidate(best, test_scores, test_labels, test_sequence_ids,
                                                  labels, prior, test_soft, config, test_probs)
        filters[name] = filt
        stage_records[name] = {"best_validation": best, "test_metrics": test_metrics,
                               "parameters": _params(best)}
        filt.reset()
        filt.save(output_dir / f"{name}_filter.pt")
    best_bayes = max(bests[:3], key=lambda x: x["selection_score"])
    best_temporal = max(bests, key=lambda x: x["selection_score"])
    best_bayes_name = best_bayes["family"]
    filters[best_bayes_name].save(output_dir / "bayes_filter.pt")
    filters["ema_logit_patience"].save(output_dir / "ema_filter.pt")
    filters[best_temporal["family"]].save(output_dir / "best_temporal_filter.pt")
    schema = {"metric_config": config, "stages": stage_records,
              "best_bayes_stage": best_bayes_name, "best_bayes": best_bayes,
              "best_ema": ema[0], "best_temporal_overall": best_temporal}
    return schema, {"stage1_fixed_bayes": stage1, "stage2_event_bayes": stage2,
                    "stage3_ambiguity_bayes": stage3, "ema_logit_patience": ema}


# =============================================================================
# Uncertainty-aware candidate-directed filter/search
# =============================================================================

FILTER_TEMPERATURES = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
FILTER_STAYS = (0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997)
RELEASE_STRENGTHS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.70)
RELEASE_MARGINS = (0.05, 0.10, 0.20, 0.30, 0.40)
RELEASE_PATIENCES = (1, 2)
EPISTEMIC_PERCENTILES = (75, 90, 95, 100)
AMBIGUITY_MARGINS = (0.05, 0.10, 0.15, 0.20, 0.30)
AMBIGUITY_FLATTEN = (0.0, 0.05, 0.10, 0.25, 0.40)
BETA_MIN_VALUES = (0.25, 0.40, 0.50, 0.65, 0.75, 0.90, 1.0)
MI_SCALE_PERCENTILES = (75, 90, 95)
EVIDENCE_DECAYS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90)
EVIDENCE_THRESHOLDS = (0.02, 0.05, 0.075, 0.10, 0.20, 0.35, 0.50)

STAGE_SEARCH_BOUNDARIES = {
    "stage1_fixed_bayes": {"T_filter": FILTER_TEMPERATURES, "stable_stay": FILTER_STAYS},
    "stage2_candidate_release": {"release_strength": RELEASE_STRENGTHS,
                                 "switch_margin": RELEASE_MARGINS,
                                 "change_patience": RELEASE_PATIENCES},
    "stage3_mi_release": {"epistemic_percentile": EPISTEMIC_PERCENTILES},
    "stage4_ambiguity": {"ambiguity_margin": AMBIGUITY_MARGINS,
                         "flatten_strength": AMBIGUITY_FLATTEN},
    "u1_adaptive_beta": {"beta_min": BETA_MIN_VALUES,
                         "mi_scale_percentile": MI_SCALE_PERCENTILES},
    "u2_accumulated_evidence": {"evidence_decay": EVIDENCE_DECAYS,
                                "evidence_threshold": EVIDENCE_THRESHOLDS},
    "controlled_C1": {"epistemic_percentile": EPISTEMIC_PERCENTILES},
    "controlled_C2": {"beta_min": BETA_MIN_VALUES,
                      "mi_scale_percentile": MI_SCALE_PERCENTILES},
    "controlled_C3": {"evidence_decay": EVIDENCE_DECAYS,
                      "evidence_threshold": EVIDENCE_THRESHOLDS},
    "ema": {"ema_alpha": EMA_ALPHAS, "change_patience": EMA_PATIENCES},
}
for _stage_name, _stage_space in STAGE_SEARCH_BOUNDARIES.items():
    if _stage_name not in {"stage1_fixed_bayes", "ema"}:
        _stage_space.update(T_filter=FILTER_TEMPERATURES, stable_stay=FILTER_STAYS)


def _search_boundary_metadata(stage, parameters):
    space = STAGE_SEARCH_BOUNDARIES.get(stage, {})
    lower = [name for name, values in space.items()
             if name in parameters and parameters[name] == min(values)]
    upper = [name for name, values in space.items()
             if name in parameters and parameters[name] == max(values)]
    return {"selected_at_lower_boundary": bool(lower),
            "selected_at_upper_boundary": bool(upper),
            "lower_boundary_parameters": lower,
            "upper_boundary_parameters": upper}


def mc_candidate_agreement(mc_probabilities):
    probabilities = torch.as_tensor(mc_probabilities).float()
    if probabilities.ndim != 3:
        raise ValueError("mc_probabilities must have shape [K,B,C]")
    candidate = probabilities.mean(0).argmax(-1)
    return (probabilities.argmax(-1) == candidate.unsqueeze(0)).float().mean(0)


def validation_mi_threshold(mutual_information, percentile):
    values = torch.as_tensor(mutual_information).float()
    return float(torch.quantile(values, float(percentile) / 100.0))


def uncertainty_adaptive_beta(mutual_information, mi_scale, beta_min):
    values = torch.as_tensor(mutual_information).float()
    scale = max(float(mi_scale), torch.finfo(values.dtype).eps)
    uncertainty = (values / scale).clamp(0.0, 1.0)
    return float(beta_min) + (1.0 - uncertainty) * (1.0 - float(beta_min))


def accumulate_transition_evidence(previous_evidence, previous_candidate, candidate,
                                   instant_evidence, evidence_decay, valid=True):
    if not valid:
        return 0.0, None
    value = float(instant_evidence)
    if previous_candidate == candidate:
        value += float(evidence_decay) * float(previous_evidence)
    return value, candidate


class CandidateReleaseBayesianTerrainFilter(BayesianTerrainFilter):
    """Persistent Bayes filter with candidate-directed release and uncertainty handling."""

    def __init__(
        self, labels, prior, stable_transition_matrix, *, release_strength=0.0,
        switch_margin=0.0, change_patience=1, epistemic_threshold=None,
        ambiguity_margin=None, flatten_strength=0.0,
        beta_min=1.0, mi_scale=None, use_accumulated_evidence=False,
        evidence_decay=0.0, evidence_threshold=0.0, device="cpu", eps=1e-8,
    ):
        super().__init__(labels, prior, stable_transition_matrix, torch.eye(len(labels)),
                         evidence_power=1.0, adaptive_evidence=False,
                         min_evidence_power=1.0, confidence_gamma=1.0,
                         device=device, eps=eps)
        self.stable_transition_matrix = self.transition_matrix.clone()
        self.release_strength = float(release_strength)
        self.switch_margin = float(switch_margin)
        self.change_patience = int(change_patience)
        self.epistemic_threshold = None if epistemic_threshold is None else float(epistemic_threshold)
        self.ambiguity_margin = None if ambiguity_margin is None else float(ambiguity_margin)
        self.flatten_strength = float(flatten_strength)
        self.beta_min = float(beta_min)
        self.mi_scale = None if mi_scale is None else float(mi_scale)
        self.use_accumulated_evidence = bool(use_accumulated_evidence)
        self.evidence_decay = float(evidence_decay)
        self.evidence_threshold = float(evidence_threshold)
        if not 0.0 < self.beta_min <= 1.0:
            raise ValueError("beta_min must be in (0,1]")
        self.current_output_index = int(self.belief.argmax())
        self.pending_target_index = None
        self.pending_count = 0
        self.last_event = False
        self.last_ambiguous = False
        self.last_high_epistemic = False
        self.accumulated_evidence = 0.0
        self.evidence_candidate_index = None
        self.last_beta = 1.0
        self.last_accumulated_evidence = 0.0
        self.last_candidate_index = self.current_output_index
        self.last_candidate_margin = 0.0

    def reset(self, prior=None):
        belief = super().reset(prior)
        self.current_output_index = int(self.belief.argmax())
        self.pending_target_index = None
        self.pending_count = 0
        self.last_event = self.last_ambiguous = self.last_high_epistemic = False
        self.accumulated_evidence = 0.0
        self.evidence_candidate_index = None
        self.last_beta = 1.0
        self.last_accumulated_evidence = 0.0
        self.last_candidate_index = self.current_output_index
        self.last_candidate_margin = 0.0
        return belief

    def _q(self, value):
        q = torch.as_tensor(value, dtype=torch.float32, device=self.device).flatten().clamp_min(self.eps)
        if q.numel() != self.num_classes:
            raise ValueError("probability vector has wrong class count")
        return q / q.sum().clamp_min(self.eps)

    @torch.inference_mode()
    def update(self, classifier_probabilities, *, event_probabilities=None,
               mutual_information=0.0, observation_quality=1.0):
        del observation_quality
        q = self._q(classifier_probabilities)
        q_event = q if event_probabilities is None else self._q(event_probabilities)
        current = self.current_output_index
        event_candidate = int(q_event.argmax())
        event_margin = float(q_event[event_candidate] - q_event[current])
        self.last_candidate_index = event_candidate
        self.last_candidate_margin = event_margin
        predicted = self.belief @ self.stable_transition_matrix
        predicted = predicted / predicted.sum().clamp_min(self.eps)
        values, indices = torch.topk(q, min(2, self.num_classes))
        ambiguity = self.num_classes > 1 and float(values[0] - values[1]) <= (
            -1.0 if self.ambiguity_margin is None else self.ambiguity_margin)
        high_mi = self.epistemic_threshold is not None and float(mutual_information) > self.epistemic_threshold
        self.last_event = False
        self.last_ambiguous = bool(ambiguity)
        self.last_high_epistemic = bool(self.ambiguity_margin is not None and high_mi)
        scale = self.mi_scale if self.mi_scale is not None else float("inf")
        uncertainty = min(max(float(mutual_information) / max(scale, self.eps), 0.0), 1.0)
        self.last_beta = float(uncertainty_adaptive_beta(
            float(mutual_information), scale, self.beta_min))
        self.last_accumulated_evidence = self.accumulated_evidence

        if ambiguity:
            candidate = int(indices[0])
            if candidate == current and self.num_classes > 1:
                candidate = int(indices[1])
            plausible = set(indices.tolist())
            if (self.pending_target_index is not None
                    and self.pending_target_index not in plausible):
                self.pending_target_index = None
                self.pending_count = 0
            if self.use_accumulated_evidence:
                if (self.evidence_candidate_index is not None
                        and self.evidence_candidate_index in plausible
                        and self.evidence_candidate_index != current):
                    self.accumulated_evidence *= self.evidence_decay
                else:
                    self.evidence_candidate_index = None
                    self.accumulated_evidence = 0.0
                self.last_accumulated_evidence = self.accumulated_evidence
            u = torch.zeros_like(q)
            u[current], u[candidate] = q[current], q[candidate]
            u = u / u.sum().clamp_min(self.eps)
            posterior = (1.0 - self.flatten_strength) * predicted + self.flatten_strength * u
            posterior = posterior / posterior.sum().clamp_min(self.eps)
            self.belief = posterior
            return BayesianFilterStep(self.labels[current], posterior.clone(), predicted.clone(),
                                      torch.ones_like(q), q.clone(), self.entropy_confidence(q), 0.0)

        candidate = event_candidate
        margin = event_margin
        if self.use_accumulated_evidence:
            plausible_candidate = candidate != current and margin > 0.0
            valid = plausible_candidate and not high_mi
            if valid:
                uncertainty_weight = max(1.0 - uncertainty, 0.0)
                instant = max(margin, 0.0) * uncertainty_weight
                self.accumulated_evidence, self.evidence_candidate_index = (
                    accumulate_transition_evidence(
                        self.accumulated_evidence, self.evidence_candidate_index,
                        candidate, instant, self.evidence_decay))
            elif (plausible_candidate
                  and self.evidence_candidate_index == candidate):
                # An epistemically uncertain frame pauses release; it does not
                # erase already accumulated candidate-specific support.
                self.accumulated_evidence *= self.evidence_decay
            else:
                self.accumulated_evidence, self.evidence_candidate_index = (
                    accumulate_transition_evidence(
                        self.accumulated_evidence, self.evidence_candidate_index,
                        candidate, 0.0, self.evidence_decay, valid=False))
            self.pending_target_index, self.pending_count = None, 0
            accepted = valid and self.accumulated_evidence >= self.evidence_threshold
        else:
            eligible = (candidate != current and margin >= self.switch_margin
                        and not high_mi)
            if eligible:
                if self.pending_target_index == candidate:
                    self.pending_count += 1
                else:
                    self.pending_target_index, self.pending_count = candidate, 1
            else:
                self.pending_target_index, self.pending_count = None, 0
            accepted = eligible and self.pending_count >= self.change_patience
        self.last_accumulated_evidence = self.accumulated_evidence
        if accepted and self.release_strength > 0:
            directed = torch.zeros_like(predicted)
            directed[candidate] = 1.0
            predicted = (1.0 - self.release_strength) * predicted + self.release_strength * directed
            predicted = predicted / predicted.sum().clamp_min(self.eps)
            self.last_event = True
        likelihood = q.pow(self.last_beta)
        posterior = predicted * likelihood
        posterior = posterior / posterior.sum().clamp_min(self.eps)
        self.belief = posterior
        new_output = int(posterior.argmax())
        self.current_output_index = new_output
        if new_output != current:
            self.pending_target_index, self.pending_count = None, 0
            self.evidence_candidate_index = None
            self.accumulated_evidence = 0.0
        return BayesianFilterStep(self.labels[new_output], posterior.clone(), predicted.clone(),
                                  likelihood.clone(), q.clone(), self.entropy_confidence(q),
                                  self.last_beta)

    def save(self, path):
        torch.save({"filter_class": self.__class__.__name__, "labels": self.labels,
                    "initial_prior": self.initial_prior.cpu(),
                    "stable_transition_matrix": self.stable_transition_matrix.cpu(),
                    "release_strength": self.release_strength, "switch_margin": self.switch_margin,
                    "change_patience": self.change_patience,
                    "epistemic_threshold": self.epistemic_threshold,
                    "ambiguity_margin": self.ambiguity_margin,
                    "flatten_strength": self.flatten_strength,
                    "beta_min": self.beta_min, "mi_scale": self.mi_scale,
                    "use_accumulated_evidence": self.use_accumulated_evidence,
                    "evidence_decay": self.evidence_decay,
                    "evidence_threshold": self.evidence_threshold, "eps": self.eps}, path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=True)
        return cls(state["labels"], state["initial_prior"], state["stable_transition_matrix"],
                   release_strength=state["release_strength"], switch_margin=state["switch_margin"],
                   change_patience=state["change_patience"],
                   epistemic_threshold=state.get("epistemic_threshold"),
                   ambiguity_margin=state.get("ambiguity_margin"),
                   flatten_strength=state.get("flatten_strength", 0.0),
                   beta_min=state.get("beta_min", 1.0), mi_scale=state.get("mi_scale"),
                   use_accumulated_evidence=state.get("use_accumulated_evidence", False),
                   evidence_decay=state.get("evidence_decay", 0.0),
                   evidence_threshold=state.get("evidence_threshold", 0.0),
                   device=device, eps=state.get("eps", 1e-8))


@torch.inference_mode()
def run_candidate_release_sequences(terrain_filter, filter_probabilities, event_probabilities,
                                    sequence_ids, mutual_information=None,
                                    diagnostic_agreement=None, return_diagnostics=False):
    qf, qe, ids = torch.as_tensor(filter_probabilities), torch.as_tensor(event_probabilities), list(sequence_ids)
    mi = torch.zeros(qf.shape[0]) if mutual_information is None else torch.as_tensor(mutual_information)
    agreement = (torch.ones(qf.shape[0]) if diagnostic_agreement is None
                 else torch.as_tensor(diagnostic_agreement))
    predictions, events, ambiguities, high_mi = [], [], [], []
    betas, evidence, posteriors, candidates, margins, emitted = [], [], [], [], [], []
    for i in range(qf.shape[0]):
        if i == 0 or ids[i] != ids[i - 1]:
            terrain_filter.reset()
        step = terrain_filter.update(
            qf[i], event_probabilities=qe[i], mutual_information=float(mi[i]))
        predictions.append(step.label)
        events.append(terrain_filter.last_event)
        ambiguities.append(terrain_filter.last_ambiguous)
        high_mi.append(terrain_filter.last_high_epistemic)
        betas.append(terrain_filter.last_beta)
        evidence.append(terrain_filter.last_accumulated_evidence)
        posteriors.append(step.posterior.detach().cpu())
        candidates.append(terrain_filter.last_candidate_index)
        margins.append(terrain_filter.last_candidate_margin)
        emitted.append(terrain_filter.current_output_index)
    base = (predictions, torch.tensor(events), torch.tensor(ambiguities), torch.tensor(high_mi))
    if not return_diagnostics:
        return base
    return (*base, {"agreement": agreement.float().cpu(), "beta": torch.tensor(betas),
                    "accumulated_evidence": torch.tensor(evidence),
                    "posterior": torch.stack(posteriors),
                    "candidate_index": torch.tensor(candidates, dtype=torch.long),
                    "candidate_margin": torch.tensor(margins, dtype=torch.float32),
                    "emitted_index": torch.tensor(emitted, dtype=torch.long)})


def _probabilities_from_cached_logits(logits, temperature):
    values = torch.as_tensor(logits).float()
    if values.ndim == 2:
        values = values.unsqueeze(0)
    return F.softmax(values / float(temperature), dim=-1).mean(0)


def _candidate_metrics(truth, predictions, labels, ids, events=None, ambiguities=None,
                       high_epistemic=None, diagnostics=None,
                       instantaneous_correct=None, evidence_threshold=None):
    metrics = evaluate_sequential_predictions(truth, predictions, labels, ids)
    window = _transition_window_mask(truth, ids, 5)
    if events is not None:
        events = torch.as_tensor(events).bool()
        metrics["event_fraction"] = float(events.float().mean())
        metrics["event_precision"] = float(window[events].float().mean()) if events.any() else 0.0
        transition_indices = [i for i in range(1, len(truth)) if ids[i] == ids[i-1] and truth[i] != truth[i-1]]
        event_indices = events.nonzero(as_tuple=False).flatten().tolist()
        metrics["event_recall"] = (float(sum(any(ids[e] == ids[t] and abs(e - t) <= 5
            for e in event_indices) for t in transition_indices) / len(transition_indices))
            if transition_indices else float("nan"))
        metrics["switch_event_precision"] = metrics["event_precision"]
        metrics["switch_event_recall"] = metrics["event_recall"]
        metrics["true_transition_detection_recall"] = metrics["event_recall"]
        offsets = []
        for i in event_indices:
            same = [t for t in transition_indices if ids[t] == ids[i]]
            if same:
                offsets.append(min(same, key=lambda t: abs(t-i)) - i)
        metrics["mean_event_offset"] = float(sum(offsets) / len(offsets)) if offsets else float("nan")
        metrics["mean_switch_event_offset"] = metrics["mean_event_offset"]
    if ambiguities is not None:
        ambiguities = torch.as_tensor(ambiguities).bool()
        metrics["ambiguous_frame_fraction"] = float(ambiguities.float().mean())
        metrics["mean_ambiguity_run_length"] = _ambiguity_run_length(ambiguities, ids)
        metrics["ambiguity_inside_transition_fraction"] = float(ambiguities[window].float().mean()) if window.any() else float("nan")
        metrics["ambiguity_outside_transition_fraction"] = float(ambiguities[~window].float().mean()) if (~window).any() else float("nan")
    if high_epistemic is not None:
        metrics["high_epistemic_frame_fraction"] = float(torch.as_tensor(high_epistemic).float().mean())
    if diagnostics is not None:
        agreement = torch.as_tensor(diagnostics["agreement"]).float()
        beta = torch.as_tensor(diagnostics["beta"]).float()
        evidence = torch.as_tensor(diagnostics["accumulated_evidence"]).float()
        correct = (torch.ones_like(agreement, dtype=torch.bool) if instantaneous_correct is None
                   else torch.as_tensor(instantaneous_correct).bool())
        events_mask = (torch.zeros_like(correct) if events is None
                       else torch.as_tensor(events).bool())

        def mean_where(values, mask):
            return float(values[mask].mean()) if mask.any() else float("nan")

        metrics.update(
            mean_mc_agreement=float(agreement.mean()),
            agreement_correct_frames=mean_where(agreement, correct),
            agreement_incorrect_frames=mean_where(agreement, ~correct),
            agreement_true_switch_events=mean_where(agreement, events_mask & window),
            agreement_false_switch_events=mean_where(agreement, events_mask & ~window),
            mean_beta=float(beta.mean()), beta_std=float(beta.std(unbiased=False)),
            beta_correct_frames=mean_where(beta, correct),
            beta_incorrect_frames=mean_where(beta, ~correct),
            beta_inside_transition_window=mean_where(beta, window),
            beta_outside_transition_window=mean_where(beta, ~window),
            mean_accumulated_evidence=float(evidence.mean()),
            evidence_true_switch_events=mean_where(evidence, events_mask & window),
            evidence_false_switch_events=mean_where(evidence, events_mask & ~window),
        )
        if evidence_threshold is not None:
            above = evidence >= float(evidence_threshold)
            metrics["evidence_frames_above_threshold"] = int(above.sum())
            metrics["evidence_fraction_above_threshold"] = float(above.float().mean())
    return metrics


def _evaluate_release_config(config, logits, truth, ids, labels, mi_event=None,
                             diagnostic_agreement=None, return_outputs=False):
    q_filter = _probabilities_from_cached_logits(logits, config["T_filter"])
    q_event = _probabilities_from_cached_logits(logits, 1.0)
    prior = {label: 1.0 / len(labels) for label in labels}
    filt = CandidateReleaseBayesianTerrainFilter(
        labels, prior, make_persistent_transition_matrix(labels, config["stable_stay"]),
        release_strength=config.get("release_strength", 0.0),
        switch_margin=config.get("switch_margin", 0.0),
        change_patience=config.get("change_patience", 1),
        epistemic_threshold=config.get("epistemic_threshold"),
        ambiguity_margin=config.get("ambiguity_margin"),
        flatten_strength=config.get("flatten_strength", 0.0),
        beta_min=config.get("beta_min", 1.0), mi_scale=config.get("mi_scale"),
        use_accumulated_evidence=config.get("use_accumulated_evidence", False),
        evidence_decay=config.get("evidence_decay", 0.0),
        evidence_threshold=config.get("evidence_threshold", 0.0), device="cpu")
    predictions, events, ambiguous, high_epistemic, diagnostics = run_candidate_release_sequences(
        filt, q_filter, q_event, ids, mutual_information=mi_event,
        diagnostic_agreement=diagnostic_agreement, return_diagnostics=True)
    label_index = {label: i for i, label in enumerate(labels)}
    truth_indices = torch.tensor([label_index[value] for value in truth])
    instantaneous_correct = q_event.argmax(1).cpu() == truth_indices
    metrics = _candidate_metrics(
        truth, predictions, labels, ids, events, ambiguous, high_epistemic,
        diagnostics, instantaneous_correct, config.get("evidence_threshold")
        if config.get("use_accumulated_evidence") else None)
    if not return_outputs:
        return filt, metrics
    return filt, metrics, {
        "q_filter": q_filter.cpu(), "q_event": q_event.cpu(),
        "predictions": predictions, "events": events.cpu(),
        "ambiguity": ambiguous.cpu(), "high_epistemic": high_epistemic.cpu(),
        **diagnostics,
    }


def _ema_scores_from_cached_logits(logits):
    """Use logits for deterministic inference and MC-mean scores for MC inference."""
    values = torch.as_tensor(logits).float()
    if values.ndim == 2:
        return values
    return F.softmax(values, dim=-1).mean(0)


def _evaluate_ema_config(config, logits, truth, ids, labels):
    filt = EMALogitPatienceFilter(
        labels, ema_alpha=config["ema_alpha"],
        change_patience=config["change_patience"], device="cpu")
    predictions = run_ema_logit_patience_sequences(
        filt, _ema_scores_from_cached_logits(logits), sequence_ids=ids)
    return filt, _candidate_metrics(truth, predictions, labels, ids)


def _trace_frame_fields(truth, ids, labels):
    ids, truth = list(ids), list(truth)
    label_index = {label: index for index, label in enumerate(labels)}
    frame_in_sequence, transition_mask = [], []
    position = 0
    for index in range(len(truth)):
        if index == 0 or ids[index] != ids[index - 1]:
            position = 0
        frame_in_sequence.append(position)
        transition_mask.append(
            index > 0 and ids[index] == ids[index - 1] and truth[index] != truth[index - 1])
        position += 1
    transition_mask = torch.tensor(transition_mask, dtype=torch.bool)
    return {
        "sequence_id": ids,
        "frame_index": torch.arange(len(truth), dtype=torch.long),
        "sequence_frame_index": torch.tensor(frame_in_sequence, dtype=torch.long),
        "ground_truth_class": truth,
        "ground_truth_index": torch.tensor([label_index[value] for value in truth], dtype=torch.long),
        "true_transition_mask": transition_mask,
        "true_transition_index": torch.where(
            transition_mask, torch.arange(len(truth), dtype=torch.long),
            torch.full((len(truth),), -1, dtype=torch.long)),
    }


def _cached_uncertainty_for_trace(logits, inference_mode):
    values = torch.as_tensor(logits).float()
    if values.ndim == 2:
        values = values.unsqueeze(0)
    samples = F.softmax(values, dim=-1)
    probabilities = samples.mean(0)
    predictive_entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
    expected_entropy = -(samples * samples.clamp_min(1e-8).log()).sum(-1).mean(0)
    if inference_mode != "mc":
        predictive_entropy = torch.full_like(predictive_entropy, float("nan"))
        mutual_information = torch.full_like(predictive_entropy, float("nan"))
    else:
        mutual_information = predictive_entropy - expected_entropy
    return probabilities.cpu(), predictive_entropy.cpu(), mutual_information.cpu()


def _validate_temporal_trace(trace):
    n = len(trace["ground_truth_class"])
    length_fields = (
        "sequence_id", "frame_index", "sequence_frame_index", "ground_truth_index",
        "true_transition_mask", "true_transition_index", "instantaneous_class_probabilities",
        "instantaneous_predicted_index", "filtered_posterior", "emitted_filtered_index",
        "mutual_information", "predictive_entropy", "beta_t", "candidate_index",
        "candidate_vs_current_margin", "ambiguity_flag", "release_event_flag",
        "accumulated_transition_evidence",
    )
    for name in length_fields:
        if len(trace[name]) != n:
            raise ValueError(f"trace field {name!r} has length {len(trace[name])}, expected {n}")
    posterior = torch.as_tensor(trace["filtered_posterior"]).float()
    if not torch.allclose(posterior.sum(1), torch.ones(n), atol=1e-5, rtol=1e-5):
        raise ValueError("trace posterior rows do not sum to one")
    expected = _trace_frame_fields(
        trace["ground_truth_class"], trace["sequence_id"], trace["metadata"]["class_ordering"])
    if not torch.equal(torch.as_tensor(trace["true_transition_mask"]), expected["true_transition_mask"]):
        raise ValueError("trace transition indices do not match within-sequence GT changes")
    if not torch.equal(torch.as_tensor(trace["true_transition_index"]), expected["true_transition_index"]):
        raise ValueError("trace transition-index values do not match the transition mask")
    if trace["metadata"]["inference_mode"] != "mc":
        if not torch.isnan(torch.as_tensor(trace["mutual_information"]).float()).all():
            raise ValueError("deterministic trace mutual information must be NaN")
        if not torch.isnan(torch.as_tensor(trace["predictive_entropy"]).float()).all():
            raise ValueError("deterministic trace predictive entropy must be NaN")
    return trace


def save_release_temporal_trace(path, trial, logits, truth, ids, labels, *,
                                inference_mode, mc_samples, mutual_information=None,
                                diagnostic_agreement=None):
    """Save compact cached-score/filter diagnostics without depth-image tensors."""
    _, _, outputs = _evaluate_release_config(
        trial, logits, truth, ids, labels, mutual_information,
        diagnostic_agreement, return_outputs=True)
    probabilities, entropy, cached_mi = _cached_uncertainty_for_trace(logits, inference_mode)
    if inference_mode == "mc" and mutual_information is not None:
        cached_mi = torch.as_tensor(mutual_information).float().cpu()
    candidate_indices = torch.as_tensor(outputs["candidate_index"]).long()
    emitted_indices = torch.as_tensor(outputs["emitted_index"]).long()
    trace = {
        **_trace_frame_fields(truth, ids, labels),
        "instantaneous_class_probabilities": probabilities,
        "instantaneous_predicted_index": probabilities.argmax(1),
        "instantaneous_predicted_class": [labels[index] for index in probabilities.argmax(1).tolist()],
        "filtered_posterior": torch.as_tensor(outputs["posterior"]).float(),
        "emitted_filtered_index": emitted_indices,
        "emitted_filtered_class": [labels[index] for index in emitted_indices.tolist()],
        "mutual_information": cached_mi,
        "predictive_entropy": entropy,
        "beta_t": torch.as_tensor(outputs["beta"]).float(),
        "candidate_index": candidate_indices,
        "candidate_class": [labels[index] for index in candidate_indices.tolist()],
        "candidate_vs_current_margin": torch.as_tensor(outputs["candidate_margin"]).float(),
        "ambiguity_flag": torch.as_tensor(outputs["ambiguity"]).bool(),
        "release_event_flag": torch.as_tensor(outputs["events"]).bool(),
        "accumulated_transition_evidence": torch.as_tensor(
            outputs["accumulated_evidence"]).float(),
        "metadata": {
            "class_ordering": list(labels), "trial_id": trial["trial_id"],
            "parent_trial_id": trial.get("parent_trial_id"), "stage": trial.get("stage"),
            "family": trial.get("family"), "filter_parameters": _config_only(trial),
            "inference_mode": inference_mode, "mc_samples": int(mc_samples),
        },
    }
    _validate_temporal_trace(trace)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    _validate_temporal_trace(restored)
    if (not torch.equal(restored["emitted_filtered_index"], trace["emitted_filtered_index"])
            or not torch.allclose(restored["filtered_posterior"], trace["filtered_posterior"])):
        raise AssertionError("trace serialization changed filter outputs")
    return str(path)


def save_ema_temporal_trace(path, trial, logits, truth, ids, labels, *,
                            inference_mode, mc_samples):
    scores = _ema_scores_from_cached_logits(logits)
    instantaneous, entropy, mi = _cached_uncertainty_for_trace(logits, inference_mode)
    filt = EMALogitPatienceFilter(
        labels, ema_alpha=trial["ema_alpha"],
        change_patience=trial["change_patience"], device="cpu")
    emitted, posterior, candidates, margins, events = [], [], [], [], []
    previous_id, previous_output = object(), None
    for index, score in enumerate(scores):
        if index == 0 or ids[index] != previous_id:
            filt.reset()
            previous_output = None
        previous_id = ids[index]
        label = filt.update(score)
        probability = F.softmax(filt.ema_scores, dim=0).cpu()
        candidate = int(probability.argmax())
        output = int(filt.current_output_index)
        emitted.append(output)
        posterior.append(probability)
        candidates.append(candidate)
        margins.append(float(probability[candidate] - probability[output]))
        events.append(previous_output is not None and output != previous_output)
        previous_output = output
    emitted = torch.tensor(emitted, dtype=torch.long)
    candidates = torch.tensor(candidates, dtype=torch.long)
    n = len(truth)
    trace = {
        **_trace_frame_fields(truth, ids, labels),
        "instantaneous_class_probabilities": instantaneous,
        "instantaneous_predicted_index": instantaneous.argmax(1),
        "instantaneous_predicted_class": [labels[i] for i in instantaneous.argmax(1).tolist()],
        "filtered_posterior": torch.stack(posterior),
        "emitted_filtered_index": emitted,
        "emitted_filtered_class": [labels[i] for i in emitted.tolist()],
        "mutual_information": mi, "predictive_entropy": entropy,
        "beta_t": torch.ones(n), "candidate_index": candidates,
        "candidate_class": [labels[i] for i in candidates.tolist()],
        "candidate_vs_current_margin": torch.tensor(margins),
        "ambiguity_flag": torch.zeros(n, dtype=torch.bool),
        "release_event_flag": torch.tensor(events, dtype=torch.bool),
        "accumulated_transition_evidence": torch.zeros(n),
        "metadata": {"class_ordering": list(labels), "trial_id": trial["trial_id"],
                     "parent_trial_id": trial.get("parent_trial_id"),
                     "stage": trial.get("stage"), "family": trial.get("family"),
                     "filter_parameters": _config_only(trial),
                     "inference_mode": inference_mode, "mc_samples": int(mc_samples)},
    }
    _validate_temporal_trace(trace)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    _validate_temporal_trace(restored)
    if (not torch.equal(restored["emitted_filtered_index"], trace["emitted_filtered_index"])
            or not torch.allclose(restored["filtered_posterior"], trace["filtered_posterior"])):
        raise AssertionError("trace serialization changed EMA outputs")
    return str(path)


def load_temporal_trace(path, *, validate=True):
    """Load a saved trace for offline metrics, error analysis, or plotting."""
    trace = torch.load(path, map_location="cpu", weights_only=False)
    return _validate_temporal_trace(trace) if validate else trace


def recompute_temporal_trace_metrics(trace_or_path, *, transition_window_radius=5):
    """Recompute temporal/confusion/uncertainty metrics without NN inference."""
    trace = (load_temporal_trace(trace_or_path) if isinstance(trace_or_path, (str, Path))
             else _validate_temporal_trace(trace_or_path))
    config = dict(SEQUENTIAL_METRIC_CONFIG)
    config["transition_window_radius"] = int(transition_window_radius)
    metrics = evaluate_sequential_predictions(
        trace["ground_truth_class"], trace["emitted_filtered_class"],
        trace["metadata"]["class_ordering"], trace["sequence_id"], config)
    incorrect = (torch.as_tensor(trace["instantaneous_predicted_index"])
                 != torch.as_tensor(trace["ground_truth_index"]))
    mi = torch.as_tensor(trace["mutual_information"]).float()
    metrics["instantaneous_accuracy"] = float((~incorrect).float().mean())
    instantaneous_labels = [trace["metadata"]["class_ordering"][index]
                            for index in trace["instantaneous_predicted_index"].tolist()]
    instantaneous_metrics = evaluate_sequential_predictions(
        trace["ground_truth_class"], instantaneous_labels,
        trace["metadata"]["class_ordering"], trace["sequence_id"], config)
    metrics["instantaneous_confusion_matrix"] = instantaneous_metrics["confusion_matrix"]
    metrics["instantaneous_per_class_precision"] = instantaneous_metrics["per_class_precision"]
    metrics["instantaneous_per_class_recall"] = instantaneous_metrics["per_class_recall"]
    metrics.update(_probability_losses(
        torch.as_tensor(trace["instantaneous_class_probabilities"]),
        trace["ground_truth_class"], trace["metadata"]["class_ordering"]))
    metrics["uncertainty_error_auroc"] = (
        uncertainty_error_auroc(mi, incorrect) if torch.isfinite(mi).all() else float("nan"))
    return metrics


def _sequence_cv(config, logits, truth, ids, labels, mi_event=None,
                 diagnostic_agreement=None):
    unique = list(dict.fromkeys(list(ids)))
    folds = [unique[i::3] for i in range(3)]
    scores, fold_metrics = [], []
    for fold in folds:
        if not fold:
            continue
        mask = torch.tensor([value in set(fold) for value in ids])
        values = torch.as_tensor(logits)
        fold_logits = values[mask] if values.ndim == 2 else values[:, mask]
        _, metrics = _evaluate_release_config(config, fold_logits,
            [truth[i] for i, keep in enumerate(mask.tolist()) if keep],
            [ids[i] for i, keep in enumerate(mask.tolist()) if keep], labels,
            None if mi_event is None else torch.as_tensor(mi_event)[mask],
            None if diagnostic_agreement is None
            else torch.as_tensor(diagnostic_agreement)[mask])
        scores.append(metrics["selection_score"])
        fold_metrics.append(metrics)
    return {"fold_selection_scores": scores,
            "fold_metrics": fold_metrics,
            "mean_selection_score": float(sum(scores) / len(scores)) if scores else float("nan")}


def _sequence_cv_ema(config, logits, truth, ids, labels):
    unique = list(dict.fromkeys(list(ids)))
    folds = [unique[i::3] for i in range(3)]
    scores, fold_metrics = [], []
    values = torch.as_tensor(logits)
    for fold in folds:
        if not fold:
            continue
        selected_ids = set(fold)
        mask = torch.tensor([value in selected_ids for value in ids])
        fold_logits = values[mask] if values.ndim == 2 else values[:, mask]
        _, metrics = _evaluate_ema_config(
            config, fold_logits,
            [truth[i] for i, keep in enumerate(mask.tolist()) if keep],
            [ids[i] for i, keep in enumerate(mask.tolist()) if keep], labels)
        scores.append(metrics["selection_score"])
        fold_metrics.append(metrics)
    return {"fold_selection_scores": scores, "fold_metrics": fold_metrics,
            "mean_selection_score": float(sum(scores) / len(scores)) if scores else float("nan")}


def _finite(value, fallback):
    value = float(value)
    return fallback if not math.isfinite(value) else value


def _compare_score_v2(left, right):
    delta = _finite(left.get("score_v2", left.get("selection_score")), -float("inf")) - _finite(
        right.get("score_v2", right.get("selection_score")), -float("inf"))
    if abs(delta) >= 0.003:
        return -1 if delta > 0 else 1
    for key, higher in (("mean_transition_delay", False),
                        ("transition_window_accuracy", True),
                        ("false_transition_rate", False),
                        ("balanced_accuracy", True)):
        lv = _finite(left.get(key, float("nan")), -float("inf") if higher else float("inf"))
        rv = _finite(right.get(key, float("nan")), -float("inf") if higher else float("inf"))
        if lv != rv:
            return -1 if (lv > rv if higher else lv < rv) else 1
    return 0


def rank_score_v2(trials):
    return sorted(trials, key=cmp_to_key(_compare_score_v2))


def _parameter_signature(value):
    ignored = set(_METRIC_NAMES) | {
        "trial_id", "parent_trial_id", "parent_stage", "lineage", "on_stage_frontier",
        "on_pareto_frontier", "stage_best", "selected", "cv", "ordered_test_metrics",
        "filter_path", "trace_paths", "stage", "family", "is_noop_baseline",
        "noop_expected_exact", "noop_verified", "noop_max_posterior_abs_error",
        "metric_deltas_vs_parent", "selected_at_lower_boundary",
        "selected_at_upper_boundary", "lower_boundary_parameters",
        "upper_boundary_parameters",
    }
    return tuple(sorted((key, repr(item)) for key, item in value.items() if key not in ignored))


def _deduplicate_trials(trials):
    unique = {}
    for trial in rank_score_v2(trials):
        unique.setdefault(_parameter_signature(trial), trial)
    return list(unique.values())


def select_stage_frontier(trials):
    """Select score, responsive, transition-accuracy, and low-false-event representatives."""
    trials = _deduplicate_trials(trials)
    if not trials:
        return []
    ranked = rank_score_v2(trials)
    best_score = ranked[0]["score_v2"]
    near = [trial for trial in ranked if best_score - trial["score_v2"] <= 0.01]
    choices = [ranked[0]]
    choices.append(min(near, key=lambda x: _finite(x["mean_transition_delay"], float("inf"))))
    choices.append(max(near, key=lambda x: _finite(x["transition_window_accuracy"], -float("inf"))))
    choices.append(min(near, key=lambda x: _finite(x["false_transition_rate"], float("inf"))))
    frontier, seen = [], set()
    for trial in choices:
        signature = _parameter_signature(trial)
        if signature not in seen:
            seen.add(signature)
            frontier.append(trial)
    # Fill tied/collapsed objective representatives from validation rank. This
    # prevents a single fixed-Bayes winner from becoming the sole Stage-2
    # ancestor while keeping the frontier bounded and parameter-unique.
    for trial in ranked:
        if len(frontier) >= 4:
            break
        signature = _parameter_signature(trial)
        if signature not in seen:
            seen.add(signature)
            frontier.append(trial)
    selected_ids = {trial["trial_id"] for trial in frontier}
    for trial in trials:
        trial["on_stage_frontier"] = trial["trial_id"] in selected_ids
    return frontier


def pareto_frontier(trials):
    trials = _deduplicate_trials(trials)
    frontier = []
    for trial in trials:
        dominated = False
        for other in trials:
            if other is trial:
                continue
            no_worse = (other["balanced_accuracy"] >= trial["balanced_accuracy"]
                        and other["transition_window_accuracy"] >= trial["transition_window_accuracy"]
                        and _finite(other["mean_transition_delay"], float("inf"))
                        <= _finite(trial["mean_transition_delay"], float("inf"))
                        and _finite(other["false_transition_rate"], float("inf"))
                        <= _finite(trial["false_transition_rate"], float("inf")))
            strictly = (other["balanced_accuracy"] > trial["balanced_accuracy"]
                        or other["transition_window_accuracy"] > trial["transition_window_accuracy"]
                        or _finite(other["mean_transition_delay"], float("inf"))
                        < _finite(trial["mean_transition_delay"], float("inf"))
                        or _finite(other["false_transition_rate"], float("inf"))
                        < _finite(trial["false_transition_rate"], float("inf")))
            if no_worse and strictly:
                dominated = True
                break
        trial["on_pareto_frontier"] = not dominated
        if not dominated:
            frontier.append(trial)
    return rank_score_v2(frontier)


def uncertainty_error_auroc(mutual_information, incorrect):
    scores = torch.as_tensor(mutual_information).float().flatten()
    target = torch.as_tensor(incorrect).bool().flatten()
    positives, negatives = int(target.sum()), int((~target).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort()
    sorted_scores = scores[order]
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    start = 0
    while start < scores.numel():
        stop = start + 1
        while stop < scores.numel() and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = ranks[start:stop].mean()
        start = stop
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    positive_rank_sum = float(ranks[inverse][target].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def search_uncertainty_aware_temporal(candidates, validation_labels, validation_ids,
                                      test_labels, test_ids, labels, output_dir):
    """Structured validation-only temporal search with frozen paper A-C lineages."""
    output_dir = Path(output_dir)
    truth_val, truth_test = list(validation_labels), list(test_labels)
    ids_val = torch.as_tensor(validation_ids).flatten().cpu().tolist() if torch.is_tensor(validation_ids) else list(validation_ids)
    ids_test = torch.as_tensor(test_ids).flatten().cpu().tolist() if torch.is_tensor(test_ids) else list(test_ids)

    for candidate in candidates:
        q, pe, ee, mi = _uncertainty(candidate["validation_logits"], 1.0)
        predicted_indices = q.argmax(1)
        predicted = [labels[index] for index in predicted_indices.tolist()]
        incorrect = torch.tensor([prediction != actual for prediction, actual in zip(predicted, truth_val)])
        runtime = float(candidate["inference_runtime_seconds"]["ordered_validation"])
        latency = runtime / max(len(truth_val), 1)
        candidate["validation_mi"] = mi
        candidate["stage0_trial_id"] = f"{candidate['id']}:stage0"
        candidate["stage0_validation"] = {
            **_candidate_metrics(truth_val, predicted, labels, ids_val),
            **_probability_losses(q, truth_val, labels),
            "predictive_entropy_mean": float(pe.mean()), "expected_entropy_mean": float(ee.mean()),
            "mutual_information_mean": float(mi.mean()),
            "uncertainty_error_auroc": (uncertainty_error_auroc(mi, incorrect)
                                        if candidate["inference_mode"] == "mc" else float("nan")),
            "inference_latency_seconds": latency,
            "effective_hz": 1.0 / latency if latency > 0 else float("inf"),
        }
    deterministic = next(candidate for candidate in candidates if candidate["inference_mode"] == "deterministic")
    mc_candidates = [candidate for candidate in candidates if candidate["inference_mode"] == "mc"]
    mc_selected = sorted(mc_candidates, key=lambda c: (
        -c["stage0_validation"]["balanced_accuracy"], c["stage0_validation"]["nll"],
        c["stage0_validation"]["inference_latency_seconds"]))[0]
    selected = [deterministic, mc_selected]

    # Test is touched only after the Stage-0 K selection is frozen.
    for candidate in candidates:
        q, pe, ee, mi = _uncertainty(candidate["test_logits"], 1.0)
        predicted = [labels[index] for index in q.argmax(1).tolist()]
        incorrect = torch.tensor([prediction != actual for prediction, actual in zip(predicted, truth_test)])
        runtime = float(candidate["inference_runtime_seconds"]["ordered_test"])
        latency = runtime / max(len(truth_test), 1)
        candidate["stage0_ordered_test"] = {
            **_candidate_metrics(truth_test, predicted, labels, ids_test),
            **_probability_losses(q, truth_test, labels),
            "predictive_entropy_mean": float(pe.mean()), "expected_entropy_mean": float(ee.mean()),
            "mutual_information_mean": float(mi.mean()),
            "uncertainty_error_auroc": (uncertainty_error_auroc(mi, incorrect)
                                        if candidate["inference_mode"] == "mc" else float("nan")),
            "inference_latency_seconds": latency,
            "effective_hz": 1.0 / latency if latency > 0 else float("inf"),
        }

    all_trials, per_config = {}, {}
    for candidate in selected:
        cid, logits, test_logits = candidate["id"], candidate["validation_logits"], candidate["test_logits"]
        is_mc = candidate["inference_mode"] == "mc"
        mi = candidate["validation_mi"] if is_mc else None
        test_mi = _uncertainty(test_logits, 1.0)[3] if is_mc else None
        agreement = mc_candidate_agreement(F.softmax(torch.as_tensor(logits).float(), dim=-1)) if is_mc else None
        test_agreement = mc_candidate_agreement(F.softmax(torch.as_tensor(test_logits).float(), dim=-1)) if is_mc else None
        counter = 0

        def make_trial(stage, family, config, parent=None, *, is_noop=False,
                       noop_expected_exact=False):
            nonlocal counter
            counter += 1
            if noop_expected_exact:
                _, metrics, outputs = _evaluate_release_config(
                    config, logits, truth_val, ids_val, labels, mi, agreement,
                    return_outputs=True)
            else:
                _, metrics = _evaluate_release_config(
                    config, logits, truth_val, ids_val, labels, mi, agreement)
                outputs = None
            record = {**config, **metrics, "stage": stage, "family": family,
                      "trial_id": f"{cid}:{stage}:{counter:05d}",
                      "parent_trial_id": None if parent is None else parent["trial_id"],
                      "parent_stage": None if parent is None else parent["stage"],
                      "lineage": family if parent is None else f"{parent['lineage']}>{family}",
                      "on_stage_frontier": False, "on_pareto_frontier": False,
                      "is_noop_baseline": bool(is_noop),
                      "noop_expected_exact": bool(noop_expected_exact),
                      **_search_boundary_metadata(stage, config)}
            if is_noop:
                if parent is None:
                    raise ValueError("a no-op/reference trial requires a parent")
                delta_names = ("score_v2", "balanced_accuracy", "transition_window_accuracy",
                               "mean_transition_delay", "false_transition_rate")
                record["metric_deltas_vs_parent"] = {
                    name: float(record[name]) - float(parent[name])
                    if math.isfinite(float(record[name])) and math.isfinite(float(parent[name]))
                    else float("nan") for name in delta_names}
                record["noop_verified"] = None
                if noop_expected_exact:
                    _, parent_metrics, parent_outputs = _evaluate_release_config(
                        parent, logits, truth_val, ids_val, labels, mi, agreement,
                        return_outputs=True)
                    posterior_error = float((outputs["posterior"]
                                             - parent_outputs["posterior"]).abs().max())
                    predictions_match = outputs["predictions"] == parent_outputs["predictions"]
                    metrics_match = all(
                        (not math.isfinite(float(metrics[name]))
                         and not math.isfinite(float(parent_metrics[name])))
                        or math.isclose(float(metrics[name]), float(parent_metrics[name]),
                                        rel_tol=1e-6, abs_tol=1e-7)
                        for name in delta_names)
                    record["noop_max_posterior_abs_error"] = posterior_error
                    record["noop_verified"] = bool(
                        predictions_match and metrics_match and posterior_error <= 1e-6)
                    if not record["noop_verified"]:
                        raise AssertionError(
                            f"{record['trial_id']} failed exact no-op verification")
            return record

        def inherited(parent):
            return _config_only(parent)

        def unique_parents(values):
            return _deduplicate_trials(values)

        def best_by_cv(values):
            ranked = rank_score_v2([
                {**trial, "score_v2": trial["cv"]["mean_selection_score"]}
                for trial in values])
            trial_id = ranked[0]["trial_id"]
            return next(trial for trial in values if trial["trial_id"] == trial_id)

        trace_cache = {}

        def assert_trace_metrics(path, expected):
            recomputed = recompute_temporal_trace_metrics(path)
            for name in ("score_v2", "balanced_accuracy", "transition_window_accuracy",
                         "mean_transition_delay", "false_transition_rate"):
                left, right = float(recomputed[name]), float(expected[name])
                if math.isfinite(left) or math.isfinite(right):
                    if not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7):
                        raise AssertionError(
                            f"trace metric {name} changed for {path}: {left} != {right}")

        def ensure_release_traces(trial):
            if trial["trial_id"] in trace_cache:
                return trace_cache[trial["trial_id"]]
            safe_id = trial["trial_id"].replace(":", "_")
            trace_root = output_dir / "temporal_traces" / cid / safe_id
            paths = {
                "validation": save_release_temporal_trace(
                    trace_root / "ordered_validation.pt", trial, logits, truth_val,
                    ids_val, labels, inference_mode=candidate["inference_mode"],
                    mc_samples=candidate["mc_samples"], mutual_information=mi,
                    diagnostic_agreement=agreement),
                "ordered_test": save_release_temporal_trace(
                    trace_root / "ordered_test.pt", trial, test_logits, truth_test,
                    ids_test, labels, inference_mode=candidate["inference_mode"],
                    mc_samples=candidate["mc_samples"], mutual_information=test_mi,
                    diagnostic_agreement=test_agreement),
            }
            assert_trace_metrics(paths["validation"], trial)
            trace_cache[trial["trial_id"]] = paths
            return paths

        def ensure_ema_traces(trial):
            key = f"ema:{trial['trial_id']}"
            if key in trace_cache:
                return trace_cache[key]
            safe_id = trial["trial_id"].replace(":", "_")
            trace_root = output_dir / "temporal_traces" / cid / safe_id
            paths = {
                "validation": save_ema_temporal_trace(
                    trace_root / "ordered_validation.pt", trial, logits, truth_val,
                    ids_val, labels, inference_mode=candidate["inference_mode"],
                    mc_samples=candidate["mc_samples"]),
                "ordered_test": save_ema_temporal_trace(
                    trace_root / "ordered_test.pt", trial, test_logits, truth_test,
                    ids_test, labels, inference_mode=candidate["inference_mode"],
                    mc_samples=candidate["mc_samples"]),
            }
            assert_trace_metrics(paths["validation"], trial)
            trace_cache[key] = paths
            return paths

        def trial_record_metadata(trial):
            keys = ("parent_trial_id", "parent_stage", "stage", "family", "lineage",
                    "is_noop_baseline", "noop_expected_exact", "noop_verified",
                    "noop_max_posterior_abs_error", "metric_deltas_vs_parent",
                    "selected_at_lower_boundary", "selected_at_upper_boundary",
                    "lower_boundary_parameters", "upper_boundary_parameters")
            return {key: trial.get(key) for key in keys if key in trial}

        def stage_payload(stage, trials, frontier, parent_baselines=None):
            best = rank_score_v2(frontier)[0]
            records = []
            for index, trial in enumerate(frontier):
                filt, test_metrics = _evaluate_release_config(
                    trial, test_logits, truth_test, ids_test, labels, test_mi, test_agreement)
                filt.reset()
                path = output_dir / f"{cid}_{stage}_frontier_{index}.pt"
                filt.save(path)
                trace_paths = ensure_release_traces(trial)
                assert_trace_metrics(trace_paths["ordered_test"], test_metrics)
                records.append({"trial_id": trial["trial_id"], "parameters": _config_only(trial),
                                "validation_metrics": _metrics_only(trial),
                                "ordered_test_metrics": test_metrics, "filter_path": str(path),
                                "trace_paths": trace_paths,
                                **trial_record_metadata(trial)})
            best_record = next(record for record in records if record["trial_id"] == best["trial_id"])
            baseline_records = []
            for index, trial in enumerate(parent_baselines or []):
                filt, test_metrics = _evaluate_release_config(
                    trial, test_logits, truth_test, ids_test, labels, test_mi, test_agreement)
                filt.reset()
                path = output_dir / f"{cid}_{stage}_parent_baseline_{index}.pt"
                filt.save(path)
                trace_paths = ensure_release_traces(trial)
                assert_trace_metrics(trace_paths["ordered_test"], test_metrics)
                baseline_records.append({
                    "trial_id": trial["trial_id"], "parameters": _config_only(trial),
                    "validation_metrics": _metrics_only(trial),
                    "ordered_test_metrics": test_metrics, "filter_path": str(path),
                    "trace_paths": trace_paths,
                    **trial_record_metadata(trial)})
            return {"best": best_record, "frontier": records,
                    "parent_baselines": baseline_records,
                    "pareto_frontier_trial_ids": [trial["trial_id"] for trial in pareto_frontier(trials)]}

        instantaneous_trial = {
            "trial_id": candidate["stage0_trial_id"], "parent_trial_id": None,
            "parent_stage": None, "stage": "stage0_instantaneous",
            "family": "instantaneous", "lineage": "instantaneous",
            "ema_alpha": 1.0, "change_patience": 1,
            **candidate["stage0_validation"],
        }
        candidate["stage0_trace_paths"] = ensure_ema_traces(instantaneous_trial)

        stage1 = [make_trial("stage1_fixed_bayes", "fixed_bayes", {
            "T_filter": temperature, "T_event": 1.0, "stable_stay": stay,
            "observation_mix": 0.0, "evidence_power": 1.0, "adaptive_evidence": False,
        }) for temperature in FILTER_TEMPERATURES for stay in FILTER_STAYS]
        frontier1 = select_stage_frontier(stage1)

        ema = []
        for alpha in EMA_ALPHAS:
            for patience in EMA_PATIENCES:
                config = {"family": "ema_logit_patience", "ema_alpha": alpha,
                          "change_patience": patience}
                _, metrics = _evaluate_ema_config(config, logits, truth_val, ids_val, labels)
                counter += 1
                ema.append({**config, **metrics, "stage": "ema", "family": "ema_logit_patience",
                            "trial_id": f"{cid}:ema:{counter:05d}", "parent_trial_id": None,
                            "parent_stage": None, "lineage": "ema_logit_patience",
                            "on_stage_frontier": False, "on_pareto_frontier": False,
                            "is_noop_baseline": False,
                            **_search_boundary_metadata("ema", config)})
        ema_finalists = select_stage_frontier(ema)
        for trial in ema_finalists:
            trial["cv"] = _sequence_cv_ema(trial, logits, truth_val, ids_val, labels)
        best_ema = best_by_cv(ema_finalists)

        stage2 = []
        stage2_parent_baselines = [
            make_trial("stage2_candidate_release", "fixed_bayes_parent", inherited(parent),
                       parent, is_noop=True, noop_expected_exact=True)
            for parent in frontier1]
        for parent in frontier1:
            for release in RELEASE_STRENGTHS:
                for margin in RELEASE_MARGINS:
                    for patience in RELEASE_PATIENCES:
                        config = {**inherited(parent), "T_event": 1.0,
                                  "release_strength": release, "switch_margin": margin,
                                  "change_patience": patience}
                        stage2.append(make_trial("stage2_candidate_release", "candidate_release", config, parent))
        frontier2 = select_stage_frontier(stage2)
        responsive2 = min(
            [trial for trial in frontier2 if rank_score_v2(frontier2)[0]["score_v2"] - trial["score_v2"] <= 0.01],
            key=lambda trial: _finite(trial["mean_transition_delay"], float("inf")))

        stage3 = []
        if is_mc:
            for parent in frontier2:
                for percentile in EPISTEMIC_PERCENTILES:
                    config = {**inherited(parent), "epistemic_percentile": percentile,
                              "epistemic_threshold": validation_mi_threshold(mi, percentile)}
                    stage3.append(make_trial(
                        "stage3_mi_release", "mi_gated_release", config, parent,
                        is_noop=percentile == 100, noop_expected_exact=percentile == 100))
        frontier3 = select_stage_frontier(stage3) if stage3 else []

        stage4_parents = unique_parents(frontier2 + frontier3)
        stage4 = []
        for parent in stage4_parents:
            for ambiguity_margin in AMBIGUITY_MARGINS:
                for flatten_strength in AMBIGUITY_FLATTEN:
                    config = {**inherited(parent), "ambiguity_margin": ambiguity_margin,
                              "flatten_strength": flatten_strength}
                    stage4.append(make_trial(
                        "stage4_ambiguity", "ambiguity_aware_release", config, parent,
                        is_noop=flatten_strength == 0.0, noop_expected_exact=False))
        frontier4 = select_stage_frontier(stage4)

        uncertainty = None
        controlled = None
        unrestricted_trials = []
        if is_mc:
            u0_frontier = unique_parents(frontier2 + frontier3 + frontier4)
            adaptive = []
            for parent in u0_frontier:
                for beta_min in BETA_MIN_VALUES:
                    for scale_percentile in MI_SCALE_PERCENTILES:
                        config = {**inherited(parent), "beta_min": beta_min, "beta_max": 1.0,
                                  "mi_scale_percentile": scale_percentile,
                                  "mi_scale": validation_mi_threshold(mi, scale_percentile)}
                        adaptive.append(make_trial(
                            "u1_adaptive_beta", "adaptive_beta", config, parent,
                            is_noop=beta_min == 1.0, noop_expected_exact=beta_min == 1.0))
            adaptive_frontier = select_stage_frontier(adaptive)
            accumulated_parents = unique_parents(
                [responsive2] + ([rank_score_v2(frontier3)[0]] if frontier3 else [])
                + [rank_score_v2(frontier4)[0]] + adaptive_frontier)
            accumulated = []
            accumulated_parent_baselines = [
                make_trial("u2_accumulated_evidence", "patience_parent_baseline",
                           inherited(parent), parent, is_noop=True,
                           noop_expected_exact=True)
                for parent in accumulated_parents]
            for parent in accumulated_parents:
                for decay in EVIDENCE_DECAYS:
                    for threshold in EVIDENCE_THRESHOLDS:
                        config = {**inherited(parent), "use_accumulated_evidence": True,
                                  "discrete_patience_enabled": False,
                                  "evidence_decay": decay, "evidence_threshold": threshold}
                        accumulated.append(make_trial(
                            "u2_accumulated_evidence", "accumulated_evidence", config, parent))
            accumulated_frontier = select_stage_frontier(accumulated)
            unrestricted_trials = u0_frontier + adaptive_frontier + accumulated_frontier
            unrestricted_best = rank_score_v2(unrestricted_trials)[0]
            unrestricted_near = [trial for trial in unrestricted_trials
                                 if unrestricted_best["score_v2"] - trial["score_v2"] <= 0.01]
            unrestricted_low_delay = min(
                unrestricted_near, key=lambda trial: _finite(trial["mean_transition_delay"], float("inf")))

            def controlled_record(name, trial, parent, searched):
                filt, test_metrics = _evaluate_release_config(
                    trial, test_logits, truth_test, ids_test, labels, test_mi, test_agreement)
                filt.reset()
                path = output_dir / f"{cid}_{name}_filter.pt"
                filt.save(path)
                trace_paths = ensure_release_traces(trial)
                assert_trace_metrics(trace_paths["ordered_test"], test_metrics)
                return {"trial_id": trial["trial_id"],
                        "parent_id": None if parent is None else parent["trial_id"],
                        "inherited_frozen_parameters": {} if parent is None else _config_only(parent),
                        "newly_searched_parameters": searched,
                        "parameters": _config_only(trial), "validation_metrics": _metrics_only(trial),
                        "ordered_test_metrics": test_metrics,
                        "delta_score_v2_vs_parent": None if parent is None else trial["score_v2"] - parent["score_v2"],
                        "filter_path": str(path), "trace_paths": trace_paths,
                        **trial_record_metadata(trial)}

            c0_pool = frontier2 + [trial for trial in stage4 if "epistemic_threshold" not in trial]
            c0 = rank_score_v2(c0_pool)[0]
            c1_trials = [make_trial("controlled_C1", "controlled_mi", {
                **inherited(c0), "epistemic_percentile": percentile,
                "epistemic_threshold": validation_mi_threshold(mi, percentile)}, c0,
                is_noop=percentile == 100, noop_expected_exact=percentile == 100)
                for percentile in EPISTEMIC_PERCENTILES]
            c1 = rank_score_v2(c1_trials)[0]
            c2_parent = c1 if c1["score_v2"] > c0["score_v2"] else c0
            c2_trials = [make_trial("controlled_C2", "controlled_adaptive_beta", {
                **inherited(c2_parent), "beta_min": beta_min, "beta_max": 1.0,
                "mi_scale_percentile": percentile,
                "mi_scale": validation_mi_threshold(mi, percentile)}, c2_parent,
                is_noop=beta_min == 1.0, noop_expected_exact=beta_min == 1.0)
                for beta_min in BETA_MIN_VALUES for percentile in MI_SCALE_PERCENTILES]
            c2 = rank_score_v2(c2_trials)[0]
            c3_parent = c2 if c2["score_v2"] > c2_parent["score_v2"] else c2_parent
            c3_trials = [make_trial("controlled_C3", "controlled_accumulated_evidence", {
                **inherited(c3_parent), "use_accumulated_evidence": True,
                "discrete_patience_enabled": False, "evidence_decay": decay,
                "evidence_threshold": threshold}, c3_parent)
                for decay in EVIDENCE_DECAYS for threshold in EVIDENCE_THRESHOLDS]
            c3 = rank_score_v2(c3_trials)[0]
            final_controlled = c3 if c3["score_v2"] > c3_parent["score_v2"] else c3_parent
            controlled = {
                "C0": controlled_record("C0", c0, None, []),
                "C1_MI": controlled_record("C1", c1, c0, ["epistemic_percentile"]),
                "C2_adaptive_beta": controlled_record(
                    "C2", c2, c2_parent, ["beta_min", "mi_scale_percentile"]),
                "C3_accumulated_evidence": controlled_record(
                    "C3", c3, c3_parent, ["evidence_decay", "evidence_threshold"]),
                "final_beneficial_stage": final_controlled["stage"],
                "final_beneficial_trial_id": final_controlled["trial_id"],
            }
            accumulated_improved = (
                rank_score_v2(accumulated_frontier)[0]["score_v2"]
                > rank_score_v2(accumulated_parents)[0]["score_v2"])
            accumulated_payload = stage_payload(
                "accumulated_evidence", accumulated, accumulated_frontier,
                accumulated_parent_baselines)
            if accumulated_improved:
                accumulated_payload["selected"] = accumulated_payload["best"]
            else:
                best_parent_id = rank_score_v2(accumulated_parents)[0]["trial_id"]
                accumulated_payload["selected"] = next(
                    record for record in accumulated_payload["parent_baselines"]
                    if record["parent_trial_id"] == best_parent_id)
            uncertainty = {
                "u0_frontier_trial_ids": [trial["trial_id"] for trial in u0_frontier],
                "adaptive_beta": stage_payload("adaptive_beta", adaptive, adaptive_frontier),
                "accumulated_evidence": accumulated_payload,
                "best_unrestricted_trial_id": unrestricted_best["trial_id"],
                "best_low_delay_trial_id": unrestricted_low_delay["trial_id"],
                "accumulated_evidence_improved": accumulated_improved,
            }
            extra_trials = {"u1_adaptive_beta": adaptive,
                            "u2_accumulated_evidence": accumulated,
                            "u2_parent_baselines": accumulated_parent_baselines,
                            "controlled_C1": c1_trials, "controlled_C2": c2_trials,
                            "controlled_C3": c3_trials}
        else:
            # Fixed Bayes remains eligible for deployment, but it never prunes
            # the protected candidate-release ancestry used by Stage 4.
            deterministic_finalists = unique_parents(frontier1 + frontier2 + frontier4)
            unrestricted_trials = deterministic_finalists
            unrestricted_best = rank_score_v2(deterministic_finalists)[0]
            unrestricted_low_delay = min(
                [trial for trial in deterministic_finalists
                 if unrestricted_best["score_v2"] - trial["score_v2"] <= 0.01],
                key=lambda trial: _finite(trial["mean_transition_delay"], float("inf")))
            extra_trials = {}

        stage_records = {
            "stage1": stage_payload("fixed_bayes", stage1, frontier1),
            "stage2": stage_payload(
                "candidate_release", stage2, frontier2, stage2_parent_baselines),
            "stage3": stage_payload("mi_gated_release", stage3, frontier3) if stage3 else None,
            "stage4": stage_payload("ambiguity_aware", stage4, frontier4),
        }
        if stage_records["stage3"] is not None:
            stage_records["stage3"]["preserved_no_mi_parent_trial_ids"] = [
                trial["trial_id"] for trial in frontier2]
        ema_frontier_records = []
        for index, trial in enumerate(ema_finalists):
            frontier_filter, frontier_test = _evaluate_ema_config(
                trial, test_logits, truth_test, ids_test, labels)
            frontier_filter.reset()
            frontier_path = output_dir / f"{cid}_ema_frontier_{index}.pt"
            frontier_filter.save(frontier_path)
            trace_paths = ensure_ema_traces(trial)
            assert_trace_metrics(trace_paths["ordered_test"], frontier_test)
            ema_frontier_records.append({
                "trial_id": trial["trial_id"], "parameters": _config_only(trial),
                "validation_metrics": _metrics_only(trial),
                "ordered_test_metrics": frontier_test, "filter_path": str(frontier_path),
                "trace_paths": trace_paths,
                **trial_record_metadata(trial)})
        ema_filter, ema_test = _evaluate_ema_config(best_ema, test_logits, truth_test, ids_test, labels)
        ema_filter.reset()
        ema_path = output_dir / f"{cid}_ema_filter.pt"
        ema_filter.save(ema_path)
        ema_record = {"parameters": _config_only(best_ema),
                      "validation_metrics": _metrics_only(best_ema),
                      "ordered_test_metrics": ema_test, "filter_path": str(ema_path),
                      "trace_paths": ensure_ema_traces(best_ema),
                      "cv": best_ema["cv"], "frontier": ema_frontier_records,
                      "best": {"trial_id": best_ema["trial_id"]},
                      **trial_record_metadata(best_ema)}

        # Rerank only the compact finalist set by whole-sequence CV. Ordered
        # test remains untouched until this selection is frozen.
        cv_finalists = select_stage_frontier(unrestricted_trials)
        for trial in cv_finalists:
            trial["cv"] = _sequence_cv(
                trial, logits, truth_val, ids_val, labels, mi, agreement)
        selected_trial = best_by_cv(cv_finalists)
        selected_filter, selected_test = _evaluate_release_config(
            selected_trial, test_logits, truth_test, ids_test, labels, test_mi, test_agreement)
        selected_filter.reset()
        selected_path = output_dir / f"{cid}_temporal_filter.pt"
        selected_filter.save(selected_path)
        low_filter, low_test = _evaluate_release_config(
            unrestricted_low_delay, test_logits, truth_test, ids_test, labels, test_mi, test_agreement)
        low_filter.reset()
        low_path = output_dir / f"{cid}_low_delay_filter.pt"
        low_filter.save(low_path)

        def winner_record(trial, test_metrics, path):
            trace_paths = ensure_release_traces(trial)
            assert_trace_metrics(trace_paths["ordered_test"], test_metrics)
            return {"trial_id": trial["trial_id"], "parameters": _config_only(trial),
                    "validation_metrics": _metrics_only(trial),
                    "cv": trial.get("cv", {"mean_selection_score": trial["score_v2"]}),
                    "ordered_test_metrics": test_metrics, "filter_path": str(path),
                    "trace_paths": trace_paths,
                    **trial_record_metadata(trial)}

        per_config[cid] = {
            "candidate": _candidate_metadata(candidate), **stage_records,
            "winner": winner_record(selected_trial, selected_test, selected_path),
            "best_bayes": winner_record(selected_trial, selected_test, selected_path),
            "best_low_delay": winner_record(unrestricted_low_delay, low_test, low_path),
            "ema": ema_record, "uncertainty_search": uncertainty,
            "controlled_C_chain": controlled,
            "experiment_B": {"B0_instantaneous": {
                                 "parameters": {"family": "instantaneous"},
                                 "validation_metrics": candidate["stage0_validation"],
                                 "ordered_test_metrics": candidate["stage0_ordered_test"],
                                 "trace_paths": candidate["stage0_trace_paths"]},
                             "B1_ema": ema_record, "B2_fixed_bayes": stage_records["stage1"]["best"],
                             "B3_candidate_release": stage_records["stage2"]["best"],
                             "supplementary_ambiguity": stage_records["stage4"]["best"]},
        }
        all_trials[cid] = {"stage1": stage1, "stage2": stage2,
                           "stage2_parent_baselines": stage2_parent_baselines,
                           "stage3": stage3,
                           "stage4": stage4, **extra_trials, "ema": ema}

    det_result = per_config[deterministic["id"]]
    mc_result = per_config[mc_selected["id"]]
    return {
            "stage0": {"deterministic_selected": deterministic["id"],
                        "mc_selected": mc_selected["id"],
                        "candidates": {c["id"]: {
                            **_candidate_metadata(c), 
                            "validation_metrics": c["stage0_validation"],
                            # Stage-0 test metrics are calculated above after the
                            # validation-only candidate selection is frozen.  Keep
                            # them in the result schema for Experiment-A reporting.
                            "ordered_test_metrics": c["stage0_ordered_test"],
                        } for c in candidates}},
            "configs": per_config, 
            "best_deterministic": det_result,
            "best_mc": mc_result
        }, all_trials


def _uncertainty(logits, temperature):
    values = torch.as_tensor(logits).float()
    if values.ndim == 2:
        values = values.unsqueeze(0)
    qs = F.softmax(values / temperature, dim=-1)
    mean = qs.mean(0)
    pe = -(mean * mean.clamp_min(1e-8).log()).sum(-1)
    ee = -(qs * qs.clamp_min(1e-8).log()).sum(-1).mean(0)
    return mean, pe, ee, pe - ee


def _probability_losses(q, truth, labels):
    index = {label: i for i, label in enumerate(labels)}
    y = torch.tensor([index[v] for v in truth])
    targets = F.one_hot(y, len(labels)).float()
    return {"nll": float(F.nll_loss(q.clamp_min(1e-8).log(), y)),
            "brier": float((q - targets).square().sum(1).mean())}


_METRIC_NAMES = {"selection_score", "score_v2", "legacy_selection_score",
    "accuracy", "balanced_accuracy", "macro_f1",
    "mean_transition_delay", "false_transition_rate", "transition_window_accuracy",
    "steady_state_accuracy", "per_class_recall", "per_class_precision",
    "minimum_class_recall", "missing_predicted_classes", "confusion_matrix",
    "event_fraction", "event_precision", "event_recall", "mean_event_offset",
    "mean_switch_event_offset",
    "switch_event_precision", "switch_event_recall", "true_transition_detection_recall",
    "ambiguous_frame_fraction", "mean_ambiguity_run_length",
    "ambiguity_inside_transition_fraction", "ambiguity_outside_transition_fraction",
    "high_epistemic_frame_fraction", "mean_mc_agreement",
    "agreement_correct_frames", "agreement_incorrect_frames",
    "agreement_true_switch_events", "agreement_false_switch_events",
    "mean_beta", "beta_std", "beta_correct_frames", "beta_incorrect_frames",
    "beta_inside_transition_window", "beta_outside_transition_window",
    "mean_accumulated_evidence", "evidence_true_switch_events",
    "evidence_false_switch_events", "evidence_frames_above_threshold",
    "evidence_fraction_above_threshold", "nll", "brier",
    "predictive_entropy_mean", "expected_entropy_mean", "mutual_information_mean",
    "uncertainty_error_auroc", "inference_latency_seconds", "effective_hz"}


def _config_only(value):
    return {k: v for k, v in value.items()
            if k not in _METRIC_NAMES and k not in {
                "cv", "selected", "stage_best", "trial_id", "parent_trial_id",
                "parent_stage", "lineage", "on_stage_frontier", "on_pareto_frontier",
                "stage", "ordered_test_metrics", "filter_path", "trace_paths",
                "is_noop_baseline", "noop_expected_exact", "noop_verified",
                "noop_max_posterior_abs_error", "metric_deltas_vs_parent",
                "selected_at_lower_boundary", "selected_at_upper_boundary",
                "lower_boundary_parameters", "upper_boundary_parameters"}}


def _metrics_only(value):
    return {k: v for k, v in value.items() if k in _METRIC_NAMES}


def _candidate_metadata(candidate):
    return {k: v for k, v in candidate.items() if not k.endswith("_logits") and k != "validation_mi"}


def _stage_summary(results, candidate, test_logits, truth, ids, labels, test_mi):
    best = results[0]
    _, metrics = _evaluate_release_config(best, test_logits, truth, ids, labels, test_mi)
    return {"parameters": _config_only(best), "validation_metrics": _metrics_only(best),
            "ordered_test_metrics": metrics}
