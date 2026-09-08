"""
terrain_classifier_bayes.py

Classifier-agnostic terrain classification and discrete Bayesian filtering.

Included components
-------------------
1. ProbabilisticClassifier
   Abstract API for classifiers that expose ordered class scores/distributions.
2. RBFSVM
   PyTorch one-vs-rest Gaussian RBF SVM for standardized features.
3. PCAWhitenedRBFPrototypeClassifier
   Incrementally extensible multi-prototype Gaussian RBF classifier with
   Euclidean or diagonal-Mahalanobis local metrics and full-data or streaming
   DataLoader-based mini-batch k-means fitting.
4. NeuralClassifierAdapter
   Adapter that lets a PyTorch neural classifier use the same inference/filter API,
   optionally with a completely different input preprocessing function.
5. BayesianTerrainFilter
   Persistent, confusion-aware discrete Bayes filter with adaptive evidence.
6. Search/calibration utilities
   - SVM-specific grid search.
   - Temperature calibration.
   - Observation/transition matrix estimation.
   - Joint Bayes-filter + prediction-temperature search on ordered sequences.

The module intentionally separates the instantaneous classifier from the temporal
filter. The classifier maps one observation to a distribution over classes; the
Bayes filter recursively combines that distribution with temporal context.
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Result containers
# =============================================================================


@dataclass
class ClassifierPrediction:
    """Instantaneous classifier output for a batch."""

    labels: List[Hashable]
    predicted_labels: List[Hashable]
    probabilities: torch.Tensor
    scores: torch.Tensor


@dataclass
class SVMSearchResult:
    """One SVM hyperparameter-search result."""

    params: Dict[str, Any]
    validation_accuracy: float
    validation_nll: float
    validation_brier: float


@dataclass
class PrototypeRBFSearchResult:
    """One prototype-RBF hyperparameter-search result."""

    params: Dict[str, Any]
    validation_accuracy: float
    validation_nll: float
    validation_brier: float
    num_prototypes: int


@dataclass
class HybridTransitionCache:
    """Classifier-independent empirical/persistent transition candidates."""

    labels: List[Hashable]
    empirical_matrix: torch.Tensor
    matrices_by_parameters: Dict[Tuple[float, float], torch.Tensor]
    unique_candidates: List[Dict[str, Any]]
    transition_pseudocount: float


@dataclass
class BayesianFilterStep:
    """Diagnostic output for one Bayes-filter update."""

    label: Hashable
    posterior: torch.Tensor
    predicted_prior: torch.Tensor
    observation_likelihood: torch.Tensor
    classifier_probabilities: torch.Tensor
    confidence: float
    evidence_power: float


@dataclass
class FilterEvaluation:
    """Metrics for an ordered sequence evaluation."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: torch.Tensor
    labels: List[Hashable]
    mean_transition_delay: float = float("nan")
    false_transition_rate: float = float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "mean_transition_delay": self.mean_transition_delay,
            "false_transition_rate": self.false_transition_rate,
        }


# =============================================================================
# Classifier-independent API
# =============================================================================


class ProbabilisticClassifier(ABC):
    """Abstract classifier API consumed by the Bayesian filter utilities.

    A subclass may use engineered SVM features, images, temporal tensors, or any
    other input representation. The only strict interface requirement is that
    ``decision_function(inputs)`` returns a tensor ``[B, C]`` whose columns follow
    ``class_ids``. The base class then supplies temperature-scaled class
    distributions, predictions, calibration, and common metrics.

    Neural classifiers can therefore use the same raw inputs as the SVM or a
    different input representation entirely. The Bayes filter never inspects the
    classifier input; it only receives the ordered probability vector.
    """

    def __init__(
        self,
        class_ids: Optional[Sequence[Hashable]] = None,
        *,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.class_ids: List[Hashable] = list(class_ids or [])
        self.class_to_index: Dict[Hashable, int] = {
            label: i for i, label in enumerate(self.class_ids)
        }
        self.temperature = float(temperature)
        self.temperature_fitted = False
        self.temperature_calibration: Dict[str, Any] = {}
        self.device = torch.device(device)
        self.dtype = dtype
        self.eps = float(eps)
        self.require_feature = False

    @abstractmethod
    def fit(self, inputs: Any, labels: Sequence[Hashable] | torch.Tensor, **kwargs: Any) -> "ProbabilisticClassifier":
        """Fit the classifier."""

    @abstractmethod
    def decision_function(self, inputs: Any) -> torch.Tensor:
        """Return unnormalized multiclass scores with shape [B, C]."""

    def set_class_ids(self, labels: Sequence[Hashable]) -> None:
        labels = list(labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("class labels must be non-empty and unique")
        self.class_ids = labels
        self.class_to_index = {label: i for i, label in enumerate(labels)}

    def predict_class_distribution(
        self,
        inputs: Any,
        *,
        temperature: Optional[float] = None,
        probability_floor: float = 0.0,
    ) -> Tuple[torch.Tensor, List[Hashable]]:
        """Return an ordered distribution over known classes.

        This is the canonical classifier-to-filter interface. ``temperature`` may
        be supplied without changing the stored calibration value, which is useful
        during Bayes-filter hyperparameter search.
        """
        if probability_floor < 0:
            raise ValueError("probability_floor must be non-negative")
        scores = self.decision_function(inputs)
        temp = self.temperature if temperature is None else float(temperature)
        if temp <= 0:
            raise ValueError("temperature must be positive")
        probabilities = F.softmax(scores / temp, dim=1)
        if probability_floor > 0:
            probabilities = probabilities.clamp_min(probability_floor)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return probabilities, list(self.class_ids)

    @torch.inference_mode()
    def predict(self, inputs: Any, *, temperature: Optional[float] = None) -> List[Hashable]:
        probabilities, _ = self.predict_class_distribution(inputs, temperature=temperature)
        return [self.class_ids[i] for i in probabilities.argmax(dim=1).tolist()]

    @torch.inference_mode()
    def predict_proba(
        self, inputs: Any, *, temperature: Optional[float] = None
    ) -> Tuple[List[Hashable], torch.Tensor]:
        probabilities, _ = self.predict_class_distribution(inputs, temperature=temperature)
        labels = [self.class_ids[i] for i in probabilities.argmax(dim=1).tolist()]
        return labels, probabilities

    @torch.inference_mode()
    def predict_details(self, inputs: Any, *, temperature: Optional[float] = None) -> ClassifierPrediction:
        scores = self.decision_function(inputs)
        temp = self.temperature if temperature is None else float(temperature)
        probabilities = F.softmax(scores / temp, dim=1)
        predicted = [self.class_ids[i] for i in probabilities.argmax(dim=1).tolist()]
        return ClassifierPrediction(list(self.class_ids), predicted, probabilities, scores)

    @torch.inference_mode()
    def score(
        self,
        inputs: Any,
        labels: Sequence[Hashable] | torch.Tensor,
        *,
        temperature: Optional[float] = None,
    ) -> float:
        """Return closed-set classification accuracy."""
        truth = self._normalize_labels(labels)
        predicted = self.predict(inputs, temperature=temperature)
        if len(truth) != len(predicted):
            raise ValueError("input and label counts differ")
        return sum(a == b for a, b in zip(truth, predicted)) / max(len(truth), 1)

    @torch.inference_mode()
    def evaluate(
        self,
        inputs: Any,
        labels: Sequence[Hashable] | torch.Tensor,
        *,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return accuracy, macro metrics, confusion matrix, and predictions."""
        truth = self._normalize_labels(labels)
        predicted = self.predict(inputs, temperature=temperature)
        metrics = evaluate_predictions(truth, predicted, self.class_ids)
        result = metrics.as_dict()
        result["predictions"] = predicted
        return result

    def fit_temperature(
        self,
        calibration_inputs: Any,
        calibration_labels: Sequence[Hashable] | torch.Tensor,
        *,
        objective: str = "nll",
        max_iter: int = 200,
        learning_rate: float = 0.05,
        min_temperature: float = 0.05,
        max_temperature: float = 20.0,
    ) -> float:
        """Fit one global post-hoc temperature on held-out data.

        This changes only the class distribution, not the score ordering or the
        instantaneous argmax prediction. ``objective`` may be ``'nll'`` or
        ``'brier'``.
        """
        if objective not in {"nll", "brier"}:
            raise ValueError("objective must be 'nll' or 'brier'")
        y = self._encode_labels(calibration_labels)
        with torch.no_grad():
            scores = self.decision_function(calibration_inputs).detach()
        log_t = torch.tensor(
            math.log(min(max(self.temperature, min_temperature), max_temperature)),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([log_t], lr=learning_rate)
        best_loss, best_t = math.inf, self.temperature
        history: List[float] = []
        for _ in range(max_iter):
            t = log_t.exp().clamp(min_temperature, max_temperature)
            probs = F.softmax(scores / t, dim=1)
            if objective == "nll":
                loss = F.nll_loss(probs.clamp_min(self.eps).log(), y)
            else:
                targets = F.one_hot(y, num_classes=len(self.class_ids)).to(self.dtype)
                loss = (probs - targets).square().sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                log_t.clamp_(math.log(min_temperature), math.log(max_temperature))
            value = float(loss.detach())
            history.append(value)
            if value < best_loss:
                best_loss = value
                best_t = float(log_t.detach().exp())
        self.temperature = best_t
        self.temperature_fitted = True
        self.temperature_calibration = {
            "objective": objective,
            "loss": best_loss,
            "iterations": max_iter,
            "history": history,
        }
        return best_t

    def _normalize_labels(self, labels: Sequence[Hashable] | torch.Tensor) -> List[Hashable]:
        if torch.is_tensor(labels):
            if labels.ndim != 1:
                raise ValueError("labels must be one-dimensional")
            return labels.detach().cpu().tolist()
        return list(labels)

    def _encode_labels(self, labels: Sequence[Hashable] | torch.Tensor) -> torch.Tensor:
        values = self._normalize_labels(labels)
        unknown = [v for v in values if v not in self.class_to_index]
        if unknown:
            raise ValueError(f"Unknown labels: {sorted(set(unknown), key=str)}")
        return torch.tensor(
            [self.class_to_index[v] for v in values],
            dtype=torch.long,
            device=self.device,
        )


# =============================================================================
# Feature standardization
# =============================================================================


class FeatureStandardizer:
    """Per-feature standardization fitted with streaming, batchwise Welford stats.

    The loader must be re-iterable and yield a feature tensor ``[B,D]`` directly,
    as the first tuple item, or under a ``features`` mapping key. Accumulation is
    performed in float64 and only the final mean/std are stored.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.count = 0

    @staticmethod
    def _features(batch: Any) -> torch.Tensor:
        if isinstance(batch, Mapping):
            if "features" not in batch:
                raise ValueError("mapping batches require a 'features' entry")
            batch = batch["features"]
        elif isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError("empty DataLoader batch")
            batch = batch[0]
        value = torch.as_tensor(batch)
        if value.ndim != 2:
            raise ValueError(f"features must have shape [B,D], got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError("features contain non-finite values")
        return value

    @torch.no_grad()
    def fit(self, dataloader: Iterable) -> "FeatureStandardizer":
        count = 0
        mean: Optional[torch.Tensor] = None
        m2: Optional[torch.Tensor] = None
        for batch in dataloader:
            x = self._features(batch).detach().to(device="cpu", dtype=torch.float64)
            if x.shape[0] == 0:
                continue
            batch_count = x.shape[0]
            batch_mean = x.mean(0)
            batch_m2 = (x - batch_mean).square().sum(0)
            if mean is None:
                count, mean, m2 = batch_count, batch_mean, batch_m2
                continue
            delta = batch_mean - mean
            total = count + batch_count
            m2 = m2 + batch_m2 + delta.square() * count * batch_count / total
            mean = mean + delta * batch_count / total
            count = total
        if count == 0 or mean is None or m2 is None:
            raise ValueError("dataloader yielded no feature rows")
        variance = m2 / max(count, 1)
        self.mean = mean.to(torch.float32)
        self.std = variance.sqrt().clamp_min(self.eps).to(torch.float32)
        self.count = int(count)
        return self

    def transform(self, features: torch.Tensor | Sequence) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("standardizer is not fitted")
        x = torch.as_tensor(features)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.mean.numel():
            raise ValueError(f"expected [B,{self.mean.numel()}], got {tuple(x.shape)}")
        if not x.is_floating_point():
            x = x.float()
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def state_dict(self) -> Dict[str, Any]:
        if self.mean is None or self.std is None:
            raise ValueError("standardizer is not fitted")
        return {"eps": self.eps, "count": self.count, "mean": self.mean, "std": self.std}

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "FeatureStandardizer":
        state = torch.load(Path(path), map_location=map_location, weights_only=False)
        result = cls(state["eps"])
        result.count = int(state.get("count", 0))
        result.mean = state["mean"].detach().cpu()
        result.std = state["std"].detach().cpu().clamp_min(result.eps)
        return result


# =============================================================================
# Gaussian RBF SVM
# =============================================================================


class RBFSVM(ProbabilisticClassifier):
    """Multiclass Gaussian RBF SVM for externally standardized features.

    For feature vectors ``x`` and kernel basis vectors ``x_i``,

        K(x, x_i) = exp(-gamma ||x-x_i||^2).

    Each class has a one-vs-rest score

        f_c(z) = sum_i alpha[i,c] K(z,z_i) + b_c.

    The coefficient matrix is trained with hinge loss and RKHS regularization.
    ``max_kernel_samples`` bounds the basis size and therefore bounds prediction
    cost, which is useful for fixed-rate deployment.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        gamma: float | str = "scale",
        max_kernel_samples: Optional[int] = 512,
        learning_rate: float = 1e-2,
        epochs: int = 300,
        batch_size: int = 256,
        weight_decay: float = 1e-3,
        squared_hinge: bool = True,
        class_balance: bool = True,
        temperature: float = 1.0,
        early_stopping_patience: int = 30,
        random_seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(None, temperature=temperature, device=device, dtype=dtype, eps=eps)
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
            raise ValueError("gamma must be positive, 'scale', or 'auto'")
        if not isinstance(gamma, str) and float(gamma) <= 0:
            raise ValueError("numeric gamma must be positive")
        if max_kernel_samples is not None and max_kernel_samples <= 0:
            raise ValueError("max_kernel_samples must be positive or None")
        self.feature_dim = int(feature_dim)
        self.gamma = gamma
        self.resolved_gamma: Optional[float] = None
        self.max_kernel_samples = max_kernel_samples
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.squared_hinge = bool(squared_hinge)
        self.class_balance = bool(class_balance)
        self.early_stopping_patience = int(early_stopping_patience)
        self.random_seed = int(random_seed)

        self.model_fitted = False
        self.kernel_basis: Optional[torch.Tensor] = None
        self.dual_coefficients: Optional[torch.Tensor] = None
        self.bias_vector: Optional[torch.Tensor] = None
        self.class_counts: Optional[torch.Tensor] = None
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [], "validation_loss": [], "validation_accuracy": []
        }
        self.require_feature = True

    def fit(
        self,
        inputs: torch.Tensor | Sequence,
        labels: Sequence[Hashable] | torch.Tensor,
        *,
        validation_inputs: Optional[torch.Tensor | Sequence] = None,
        validation_labels: Optional[Sequence[Hashable] | torch.Tensor] = None,
        verbose: bool = False,
        _kernel_cache: Optional[Mapping[str, torch.Tensor]] = None,
        **_: Any,
    ) -> "RBFSVM":
        X = self._validate_input(inputs)
        y_labels = self._normalize_labels(labels)
        if len(y_labels) != X.shape[0]:
            raise ValueError("input and label counts differ")
        self.set_class_ids(list(dict.fromkeys(y_labels)))
        if len(self.class_ids) < 2:
            raise ValueError("At least two classes are required")
        y = self._encode_labels(y_labels)
        self._resolve_gamma(X)

        if validation_inputs is not None or validation_labels is not None:
            if validation_inputs is None or validation_labels is None:
                raise ValueError("validation_inputs and validation_labels must be supplied together")
            Xv = self._validate_input(validation_inputs)
            yv = self._encode_labels(validation_labels)
        else:
            Xv = yv = None
        self._fit_kernel_model(X, y, Xv, yv, verbose, _kernel_cache)
        self.class_counts = torch.bincount(y, minlength=len(self.class_ids))
        self.model_fitted = True
        self.temperature_fitted = False
        return self

    def _fit_kernel_model(
        self,
        Z: torch.Tensor,
        y: torch.Tensor,
        Zv: Optional[torch.Tensor],
        yv: Optional[torch.Tensor],
        verbose: bool,
        kernel_cache: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> None:
        torch.manual_seed(self.random_seed)
        if kernel_cache is not None:
            basis = kernel_cache["basis"].to(self.device)
        elif self.max_kernel_samples is not None and Z.shape[0] > self.max_kernel_samples:
            basis_indices = self._stratified_basis_indices(y, self.max_kernel_samples)
            basis = Z[basis_indices]
        else:
            basis = Z
        if kernel_cache is None:
            K_train = self._rbf_kernel(Z, basis)
            K_basis = self._rbf_kernel(basis, basis)
            K_val = self._rbf_kernel(Zv, basis) if Zv is not None else None
        else:
            gamma = float(self.resolved_gamma)
            K_train = torch.exp(-gamma * kernel_cache["train_distances"].to(self.device))
            K_basis = torch.exp(-gamma * kernel_cache["basis_distances"].to(self.device))
            K_val = None if Zv is None else torch.exp(-gamma * kernel_cache["validation_distances"].to(self.device))
        n, m, c = Z.shape[0], basis.shape[0], len(self.class_ids)
        alpha = torch.zeros(m, c, device=self.device, dtype=self.dtype, requires_grad=True)
        bias = torch.zeros(c, device=self.device, dtype=self.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([alpha, bias], lr=self.learning_rate)
        targets = -torch.ones(n, c, device=self.device, dtype=self.dtype)
        targets[torch.arange(n, device=self.device), y] = 1.0
        weights = self._binary_term_weights(y, c)
        best, best_loss, patience = None, math.inf, 0
        self.training_history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            loss_sum = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                scores = K_train[idx] @ alpha + bias
                hinge = (1.0 - targets[idx] * scores).clamp_min(0.0)
                if self.squared_hinge:
                    hinge = hinge.square()
                data_loss = (hinge * weights[idx]).mean()
                rkhs = torch.sum(alpha * (K_basis @ alpha))
                loss = data_loss + 0.5 * self.weight_decay * rkhs / max(m, 1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * idx.numel()
            train_loss = loss_sum / n
            self.training_history["train_loss"].append(train_loss)
            if K_val is not None and yv is not None:
                with torch.no_grad():
                    val_scores = K_val @ alpha + bias
                    val_loss = float(self._hinge_loss(val_scores, yv))
                    val_acc = float((val_scores.argmax(1) == yv).float().mean())
                self.training_history["validation_loss"].append(val_loss)
                self.training_history["validation_accuracy"].append(val_acc)
                monitored = val_loss
            else:
                monitored = train_loss
            if monitored < best_loss - 1e-8:
                best_loss = monitored
                best = (alpha.detach().clone(), bias.detach().clone())
                patience = 0
            else:
                patience += 1
            if verbose and (epoch == 0 or (epoch + 1) % 25 == 0):
                print(f"epoch={epoch+1} train_loss={train_loss:.6f}")
            if self.early_stopping_patience > 0 and patience >= self.early_stopping_patience:
                break
        if best is None:
            best = (alpha.detach(), bias.detach())
        self.kernel_basis = basis.detach().clone()
        self.dual_coefficients, self.bias_vector = best

    def decision_function(self, inputs: torch.Tensor | Sequence) -> torch.Tensor:
        if not self.model_fitted:
            raise ValueError("SVM is not fitted")
        Z = self._validate_input(inputs)
        return self._rbf_kernel(Z, self.kernel_basis) @ self.dual_coefficients + self.bias_vector

    def to(self, device: str | torch.device) -> "RBFSVM":
        """Move fitted SVM state without changing its public interface."""
        self.device = torch.device(device)
        for name in ("kernel_basis", "dual_coefficients", "bias_vector", "class_counts"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(self.device))
        return self

    def _resolve_gamma(self, Z: torch.Tensor) -> None:
        if self.gamma == "auto":
            self.resolved_gamma = 1.0 / Z.shape[1]
        elif self.gamma == "scale":
            variance = float(Z.var(unbiased=False).clamp_min(self.eps))
            self.resolved_gamma = 1.0 / max(Z.shape[1] * variance, self.eps)
        else:
            self.resolved_gamma = float(self.gamma)

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        squared = (
            X.square().sum(1, keepdim=True)
            + Y.square().sum(1).unsqueeze(0)
            - 2.0 * X @ Y.T
        ).clamp_min(0.0)
        gamma = self.resolved_gamma if self.resolved_gamma is not None else 1.0 / X.shape[1]
        return torch.exp(-gamma * squared)

    def _stratified_basis_indices(self, y: torch.Tensor, maximum: int) -> torch.Tensor:
        per_class = max(maximum // len(self.class_ids), 1)
        pieces = []
        for i in range(len(self.class_ids)):
            idx = torch.where(y == i)[0]
            if idx.numel() > per_class:
                idx = idx[torch.randperm(idx.numel(), device=self.device)[:per_class]]
            pieces.append(idx)
        result = torch.cat(pieces)
        return result[:maximum]

    def _binary_term_weights(self, y: torch.Tensor, c: int) -> torch.Tensor:
        weights = torch.ones(y.shape[0], c, device=self.device, dtype=self.dtype)
        if not self.class_balance:
            return weights
        for i in range(c):
            pos = y == i
            neg = ~pos
            if pos.any() and neg.any():
                weights[pos, i] = y.numel() / (2.0 * float(pos.sum()))
                weights[neg, i] = y.numel() / (2.0 * float(neg.sum()))
        return weights

    def _hinge_loss(self, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        targets = -torch.ones_like(scores)
        targets[torch.arange(y.shape[0], device=self.device), y] = 1.0
        hinge = (1.0 - targets * scores).clamp_min(0.0)
        if self.squared_hinge:
            hinge = hinge.square()
        return (hinge * self._binary_term_weights(y, scores.shape[1])).mean()

    def _validate_input(self, X: torch.Tensor | Sequence) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError(f"Expected [N,{self.feature_dim}], got {tuple(X.shape)}")
        if not torch.isfinite(X).all():
            raise ValueError("input contains non-finite values")
        return X

    def get_config(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "gamma": self.gamma,
            "max_kernel_samples": self.max_kernel_samples,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "squared_hinge": self.squared_hinge,
            "class_balance": self.class_balance,
            "temperature": self.temperature,
            "early_stopping_patience": self.early_stopping_patience,
            "random_seed": self.random_seed,
            "device": str(self.device),
            "dtype": self.dtype,
            "eps": self.eps,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.get_config(),
            "class_ids": self.class_ids,
            "resolved_gamma": self.resolved_gamma,
            "kernel_basis": self.kernel_basis,
            "dual_coefficients": self.dual_coefficients,
            "bias_vector": self.bias_vector,
            "class_counts": self.class_counts,
            "model_fitted": self.model_fitted,
            "temperature_fitted": self.temperature_fitted,
            "temperature_calibration": self.temperature_calibration,
            "training_history": self.training_history,
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "RBFSVM":
        state = torch.load(Path(path), map_location=map_location, weights_only=False)
        config = dict(state["config"])
        config["device"] = str(map_location)
        model = cls(**config)
        model.set_class_ids(state["class_ids"])
        for key in [
            "kernel_basis", "dual_coefficients", "bias_vector", "class_counts"
        ]:
            value = state[key]
            setattr(model, key, None if value is None else value.to(model.device))
        model.resolved_gamma = state["resolved_gamma"]
        model.model_fitted = state["model_fitted"]
        model.temperature_fitted = state.get("temperature_fitted", False)
        model.temperature_calibration = state.get("temperature_calibration", {})
        model.training_history = state.get("training_history", {})
        return model


# Source compatibility for callers importing the historical name. New checkpoints
# contain no PCA state and should be loaded through either name.
PCAWhitenedRBFSVM = RBFSVM



# =============================================================================
# PCA-whitened RBF prototype classifier
# =============================================================================


class PCAWhitenedRBFPrototypeClassifier(ProbabilisticClassifier):
    """Multi-prototype Gaussian RBF classifier with PCA whitening.

    Each class is represented by several local prototypes rather than one global
    mean. For transformed input ``z`` and prototype ``mu[c,k]``, the local response
    is

        r[c,k](z) = exp(-gamma * d[c,k](z)),

    where ``d`` is squared Euclidean distance or a diagonal Mahalanobis distance.
    Per-prototype responses are combined into one class score using weighted
    log-sum-exp, maximum response, or mean response. The resulting score tensor is
    consumed by the common ``ProbabilisticClassifier`` temperature and Bayes-filter
    APIs.

    The PCA transform is normally frozen after initial fitting. A new class can
    then be added by fitting only its local prototypes; existing class prototypes
    are untouched. This makes class addition substantially cheaper than updating a
    one-vs-rest RBF SVM.

    ``PCAWhitenedPrototypeRBFNetwork`` is provided as an alias. The word "network"
    is reasonable because the prototypes behave as an RBF hidden layer, although
    "classifier" is more precise because this implementation uses fixed analytic
    aggregation rather than a separately trained output layer.
    """

    VALID_METRICS = {"euclidean", "diag_mahalanobis"}
    VALID_AGGREGATIONS = {"logsumexp", "max", "mean"}
    VALID_INIT = {"kmeans++", "farthest", "random"}
    VALID_KMEANS_FIT_MODES = {"full", "mini_batch"}

    def __init__(
        self,
        feature_dim: int,
        pca_dim: Optional[int] = None,
        *,
        prototypes_per_class: int = 8,
        gamma: float | str = "scale",
        metric_type: str = "diag_mahalanobis",
        aggregation: str = "logsumexp",
        kmeans_iterations: int = 40,
        prototype_init: str = "kmeans++",
        kmeans_fit_mode: str = "full",
        prototype_batch_size: int = 256,
        prototype_epochs: int = 10,
        initialization_sample_size: int = 2048,
        reset_counts_each_epoch: bool = True,
        min_variance: float = 1e-3,
        variance_shrinkage: float = 0.10,
        min_cluster_samples: int = 2,
        store_training_data: bool = True,
        temperature: float = 1.0,
        random_seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(None, temperature=temperature, device=device, dtype=dtype, eps=eps)
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if pca_dim is not None and pca_dim <= 0:
            raise ValueError("pca_dim must be positive or None")
        if prototypes_per_class <= 0:
            raise ValueError("prototypes_per_class must be positive")
        if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
            raise ValueError("gamma must be positive, 'scale', or 'auto'")
        if not isinstance(gamma, str) and float(gamma) <= 0:
            raise ValueError("numeric gamma must be positive")
        if metric_type not in self.VALID_METRICS:
            raise ValueError(f"metric_type must be one of {sorted(self.VALID_METRICS)}")
        if aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {sorted(self.VALID_AGGREGATIONS)}")
        if prototype_init not in self.VALID_INIT:
            raise ValueError(f"prototype_init must be one of {sorted(self.VALID_INIT)}")
        if kmeans_fit_mode not in self.VALID_KMEANS_FIT_MODES:
            raise ValueError(
                f"kmeans_fit_mode must be one of {sorted(self.VALID_KMEANS_FIT_MODES)}"
            )
        if prototype_batch_size <= 0:
            raise ValueError("prototype_batch_size must be positive")
        if prototype_epochs <= 0:
            raise ValueError("prototype_epochs must be positive")
        if initialization_sample_size <= 0:
            raise ValueError("initialization_sample_size must be positive")
        if kmeans_iterations <= 0:
            raise ValueError("kmeans_iterations must be positive")
        if min_variance <= 0:
            raise ValueError("min_variance must be positive")
        if not 0.0 <= variance_shrinkage <= 1.0:
            raise ValueError("variance_shrinkage must be in [0,1]")

        self.feature_dim = int(feature_dim)
        self.pca_dim = int(pca_dim) if pca_dim is not None else int(feature_dim)
        self.prototypes_per_class = int(prototypes_per_class)
        self.gamma = gamma
        self.resolved_gamma: Optional[float] = None
        self.metric_type = metric_type
        self.aggregation = aggregation
        self.kmeans_iterations = int(kmeans_iterations)
        self.prototype_init = prototype_init
        self.kmeans_fit_mode = kmeans_fit_mode
        self.prototype_batch_size = int(prototype_batch_size)
        self.prototype_epochs = int(prototype_epochs)
        self.initialization_sample_size = int(initialization_sample_size)
        self.reset_counts_each_epoch = bool(reset_counts_each_epoch)
        self.min_variance = float(min_variance)
        self.variance_shrinkage = float(variance_shrinkage)
        self.min_cluster_samples = int(min_cluster_samples)
        self.store_training_data = bool(store_training_data)
        self.random_seed = int(random_seed)

        self.pca_fitted = False
        self.model_fitted = False
        self.global_mean: Optional[torch.Tensor] = None
        self.pca_components: Optional[torch.Tensor] = None
        self.pca_eigenvalues: Optional[torch.Tensor] = None

        self.prototypes: Dict[Hashable, torch.Tensor] = {}
        self.prototype_variances: Dict[Hashable, torch.Tensor] = {}
        self.prototype_log_weights: Dict[Hashable, torch.Tensor] = {}
        self.class_counts: Dict[Hashable, int] = {}
        self.raw_class_data: Dict[Hashable, torch.Tensor] = {}
        self.require_feature = True

    @torch.no_grad()
    def fit_initial_pca(
        self,
        X: torch.Tensor | Sequence,
        pca_dim: Optional[int] = None,
    ) -> "PCAWhitenedRBFPrototypeClassifier":
        X = self._validate_input(X)
        if X.shape[0] < 2:
            raise ValueError("At least two samples are required to fit PCA")
        if pca_dim is not None:
            self.pca_dim = int(pca_dim)
        mean = X.mean(dim=0)
        centered = X - mean
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        k = min(self.pca_dim, X.shape[0] - 1, X.shape[1])
        if k <= 0:
            raise ValueError("PCA retained zero components")
        self.global_mean = mean
        self.pca_components = vh[:k].contiguous()
        self.pca_eigenvalues = (
            singular_values[:k].square() / max(X.shape[0] - 1, 1)
        ).clamp_min(self.eps)
        self.pca_fitted = True
        self.model_fitted = False
        return self

    def transform(self, X: torch.Tensor | Sequence) -> torch.Tensor:
        if not self.pca_fitted:
            raise ValueError("PCA is not fitted")
        X = self._validate_input(X)
        projected = (X - self.global_mean) @ self.pca_components.T
        return projected / torch.sqrt(self.pca_eigenvalues.unsqueeze(0) + self.eps)

    def fit(
        self,
        inputs: torch.Tensor | Sequence,
        labels: Sequence[Hashable] | torch.Tensor,
        *,
        fit_pca: bool = True,
        validation_inputs: Optional[torch.Tensor | Sequence] = None,
        validation_labels: Optional[Sequence[Hashable] | torch.Tensor] = None,
        **_: Any,
    ) -> "PCAWhitenedRBFPrototypeClassifier":
        del validation_inputs, validation_labels
        X = self._validate_input(inputs)
        y_labels = self._normalize_labels(labels)
        if len(y_labels) != X.shape[0]:
            raise ValueError("input and label counts differ")
        if fit_pca:
            self.fit_initial_pca(X)
        elif not self.pca_fitted:
            raise ValueError("fit_pca=False requires a fitted PCA transform")

        self._fit_from_transformed(X, y_labels, self.transform(X))
        return self

    @torch.no_grad()
    def _fit_from_transformed(
        self, X: torch.Tensor, y_labels: Sequence[Hashable], Z_all: torch.Tensor
    ) -> None:
        """Fit prototypes from rows transformed by this model's frozen PCA."""
        self.set_class_ids(list(dict.fromkeys(y_labels)))
        if len(self.class_ids) < 2:
            raise ValueError("At least two classes are required")

        self.prototypes.clear()
        self.prototype_variances.clear()
        self.prototype_log_weights.clear()
        self.class_counts.clear()
        self.raw_class_data.clear()

        self._resolve_gamma(Z_all)
        for class_id in self.class_ids:
            mask = torch.tensor(
                [label == class_id for label in y_labels],
                dtype=torch.bool,
                device=self.device,
            )
            self._fit_class_from_transformed(class_id, Z_all[mask])
            self.class_counts[class_id] = int(mask.sum())
            if self.store_training_data:
                self.raw_class_data[class_id] = X[mask].detach().clone()

        self.model_fitted = True
        self.temperature_fitted = False


    @torch.no_grad()
    def fit_dataloader(
        self,
        dataloader: Iterable,
        *,
        fit_pca: bool = True,
        class_ids: Optional[Sequence[Hashable]] = None,
    ) -> "PCAWhitenedRBFPrototypeClassifier":
        """Fit from a re-iterable data loader without materializing all samples.

        Each loader batch must be ``(features, labels)`` or a mapping containing
        ``features`` and ``labels``. Features must have shape ``[B, feature_dim]``.
        The loader is traversed multiple times: once for streaming PCA (when
        requested), once for class discovery/initialization, ``prototype_epochs``
        times for mini-batch center updates, and once for final variance/weight
        statistics. A standard ``torch.utils.data.DataLoader`` is re-iterable.

        This path intentionally does not populate ``raw_class_data`` because doing
        so would defeat its bounded-RAM purpose. Consequently, exact global PCA
        refitting through ``update_class(..., refit_pca=True)`` is unavailable unless
        training data are managed externally.
        """
        if self.kmeans_fit_mode != "mini_batch":
            raise ValueError(
                "fit_dataloader requires kmeans_fit_mode='mini_batch'; use fit() "
                "for full-dataset Lloyd k-means"
            )
        if fit_pca:
            self._fit_pca_from_dataloader(dataloader)
        elif not self.pca_fitted:
            raise ValueError("fit_pca=False requires a fitted PCA transform")

        discovered, reservoirs, class_counts = self._collect_stream_metadata(
            dataloader, class_ids=class_ids
        )
        self.set_class_ids(discovered)
        if len(self.class_ids) < 2:
            raise ValueError("At least two classes are required")

        self.prototypes.clear()
        self.prototype_variances.clear()
        self.prototype_log_weights.clear()
        self.class_counts = dict(class_counts)
        self.raw_class_data.clear()

        centers: Dict[Hashable, torch.Tensor] = {}
        for class_id in self.class_ids:
            sample = reservoirs[class_id]
            k = min(self.prototypes_per_class, class_counts[class_id])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.random_seed + len(centers))
            centers[class_id] = self._initialize_centers(sample, k, generator)

        # Mini-batch k-means updates. Counts may reset each epoch to keep later
        # assignments influential, or persist for strict online running means.
        persistent_counts = {
            c: torch.zeros(centers[c].shape[0], dtype=torch.float64, device=self.device)
            for c in self.class_ids
        }
        for _epoch in range(self.prototype_epochs):
            epoch_counts = (
                {
                    c: torch.zeros_like(persistent_counts[c])
                    for c in self.class_ids
                }
                if self.reset_counts_each_epoch
                else persistent_counts
            )
            for X_batch, y_batch in self._iterate_loader(dataloader):
                Z_batch = self.transform(X_batch)
                for class_id in self.class_ids:
                    mask = self._label_mask(y_batch, class_id)
                    if not bool(mask.any()):
                        continue
                    Zc = Z_batch[mask]
                    C = centers[class_id]
                    assignment = torch.cdist(Zc, C).square().argmin(dim=1)
                    for k in range(C.shape[0]):
                        members = Zc[assignment == k]
                        if members.shape[0] == 0:
                            continue
                        m = float(members.shape[0])
                        batch_mean = members.mean(dim=0)
                        n = float(epoch_counts[class_id][k])
                        C[k] = (n * C[k] + m * batch_mean) / max(n + m, 1.0)
                        epoch_counts[class_id][k] += m
            if self.reset_counts_each_epoch:
                persistent_counts = epoch_counts

        self.prototypes = {c: v.contiguous() for c, v in centers.items()}
        self._finalize_stream_statistics(dataloader)
        self._resolve_gamma_from_dataloader(dataloader)
        self.model_fitted = True
        self.temperature_fitted = False

    @torch.no_grad()
    def add_class_dataloader(
        self,
        class_id: Hashable,
        dataloader: Iterable,
    ) -> "PCAWhitenedRBFPrototypeClassifier":
        """Add one class from a loader while keeping PCA and old classes fixed.

        The loader may contain only the new class or mixed labels; only rows whose
        label equals ``class_id`` are used. Existing prototypes are untouched.
        """
        if class_id in self.class_ids:
            raise ValueError(f"Class {class_id!r} already exists")
        if not self.pca_fitted:
            raise ValueError("Fit PCA before adding a class")
        if self.kmeans_fit_mode != "mini_batch":
            raise ValueError("add_class_dataloader requires kmeans_fit_mode='mini_batch'")

        _, reservoirs, counts = self._collect_stream_metadata(
            dataloader, class_ids=[class_id], strict_requested_classes=True
        )
        count = counts[class_id]
        k = min(self.prototypes_per_class, count)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + len(self.class_ids))
        center = self._initialize_centers(reservoirs[class_id], k, generator)
        running = torch.zeros(k, dtype=torch.float64, device=self.device)
        for _epoch in range(self.prototype_epochs):
            if self.reset_counts_each_epoch:
                running.zero_()
            for X_batch, y_batch in self._iterate_loader(dataloader):
                mask = self._label_mask(y_batch, class_id)
                if not bool(mask.any()):
                    continue
                Z = self.transform(X_batch)[mask]
                assignment = torch.cdist(Z, center).square().argmin(dim=1)
                for j in range(k):
                    members = Z[assignment == j]
                    if members.shape[0] == 0:
                        continue
                    m = float(members.shape[0])
                    n = float(running[j])
                    center[j] = (n * center[j] + m * members.mean(0)) / max(n + m, 1.0)
                    running[j] += m

        self.class_ids.append(class_id)
        self.class_to_index = {v: i for i, v in enumerate(self.class_ids)}
        self.prototypes[class_id] = center.contiguous()
        self.class_counts[class_id] = count
        self._finalize_one_class_stream_statistics(class_id, dataloader)
        self.model_fitted = True
        self.temperature_fitted = False
        return self

    @torch.no_grad()
    def _fit_pca_from_dataloader(self, dataloader: Iterable) -> None:
        count = 0
        sum_x = torch.zeros(self.feature_dim, dtype=torch.float64, device=self.device)
        sum_xx = torch.zeros(
            self.feature_dim, self.feature_dim, dtype=torch.float64, device=self.device
        )
        for X, _ in self._iterate_loader(dataloader):
            X64 = X.to(torch.float64)
            count += X64.shape[0]
            sum_x += X64.sum(dim=0)
            sum_xx += X64.T @ X64
        if count < 2:
            raise ValueError("At least two samples are required to fit PCA")
        mean = sum_x / count
        covariance = (sum_xx - count * torch.outer(mean, mean)) / max(count - 1, 1)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        k = min(self.pca_dim, self.feature_dim, count - 1)
        selected = order[:k]
        self.global_mean = mean.to(self.dtype)
        self.pca_components = eigenvectors[:, selected].T.to(self.dtype).contiguous()
        self.pca_eigenvalues = eigenvalues[selected].clamp_min(self.eps).to(self.dtype)
        self.pca_fitted = True
        self.model_fitted = False

    @torch.no_grad()
    def _collect_stream_metadata(
        self,
        dataloader: Iterable,
        *,
        class_ids: Optional[Sequence[Hashable]],
        strict_requested_classes: bool = False,
    ) -> Tuple[List[Hashable], Dict[Hashable, torch.Tensor], Dict[Hashable, int]]:
        requested = list(class_ids) if class_ids is not None else None
        discovered: List[Hashable] = list(requested or [])
        counts: Dict[Hashable, int] = {c: 0 for c in discovered}
        reservoirs: Dict[Hashable, List[torch.Tensor]] = {c: [] for c in discovered}
        seen_for_reservoir: Dict[Hashable, int] = {c: 0 for c in discovered}
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed)

        for X, labels in self._iterate_loader(dataloader):
            Z = self.transform(X)
            for i, label in enumerate(labels):
                if requested is not None and label not in requested:
                    if strict_requested_classes:
                        continue
                    continue
                if label not in counts:
                    discovered.append(label)
                    counts[label] = 0
                    reservoirs[label] = []
                    seen_for_reservoir[label] = 0
                counts[label] += 1
                seen_for_reservoir[label] += 1
                reservoir = reservoirs[label]
                item = Z[i].detach().clone()
                if len(reservoir) < self.initialization_sample_size:
                    reservoir.append(item)
                else:
                    j = int(torch.randint(seen_for_reservoir[label], (1,), generator=generator))
                    if j < self.initialization_sample_size:
                        reservoir[j] = item

        if requested is not None:
            missing = [c for c in requested if counts.get(c, 0) == 0]
            if missing:
                raise ValueError(f"No samples found for requested classes: {missing}")
        if not discovered:
            raise ValueError("No classes found in dataloader")
        tensors = {c: torch.stack(reservoirs[c]) for c in discovered}
        return discovered, tensors, counts

    @torch.no_grad()
    def _finalize_stream_statistics(self, dataloader: Iterable) -> None:
        stats = self._empty_cluster_stats(self.class_ids)
        for X, labels in self._iterate_loader(dataloader):
            Z = self.transform(X)
            self._accumulate_cluster_stats(stats, Z, labels, self.class_ids)
        self._install_cluster_stats(stats, self.class_ids)

    @torch.no_grad()
    def _finalize_one_class_stream_statistics(
        self, class_id: Hashable, dataloader: Iterable
    ) -> None:
        stats = self._empty_cluster_stats([class_id])
        for X, labels in self._iterate_loader(dataloader):
            Z = self.transform(X)
            self._accumulate_cluster_stats(stats, Z, labels, [class_id])
        self._install_cluster_stats(stats, [class_id])

    def _empty_cluster_stats(self, class_ids: Sequence[Hashable]) -> Dict[Hashable, Dict[str, torch.Tensor]]:
        result = {}
        d = self.pca_components.shape[0]
        for c in class_ids:
            k = self.prototypes[c].shape[0]
            result[c] = {
                "count": torch.zeros(k, dtype=torch.float64, device=self.device),
                "sum": torch.zeros(k, d, dtype=torch.float64, device=self.device),
                "sumsq": torch.zeros(k, d, dtype=torch.float64, device=self.device),
                "class_count": torch.zeros((), dtype=torch.float64, device=self.device),
                "class_sum": torch.zeros(d, dtype=torch.float64, device=self.device),
                "class_sumsq": torch.zeros(d, dtype=torch.float64, device=self.device),
            }
        return result

    @torch.no_grad()
    def _accumulate_cluster_stats(
        self,
        stats: Dict[Hashable, Dict[str, torch.Tensor]],
        Z: torch.Tensor,
        labels: List[Hashable],
        class_ids: Sequence[Hashable],
    ) -> None:
        for c in class_ids:
            mask = self._label_mask(labels, c)
            if not bool(mask.any()):
                continue
            Zc = Z[mask].to(torch.float64)
            C = self.prototypes[c]
            assignment = torch.cdist(Zc.to(self.dtype), C).square().argmin(dim=1)
            st = stats[c]
            st["class_count"] += Zc.shape[0]
            st["class_sum"] += Zc.sum(0)
            st["class_sumsq"] += Zc.square().sum(0)
            for k in range(C.shape[0]):
                members = Zc[assignment == k]
                if members.shape[0] == 0:
                    continue
                st["count"][k] += members.shape[0]
                st["sum"][k] += members.sum(0)
                st["sumsq"][k] += members.square().sum(0)

    @torch.no_grad()
    def _install_cluster_stats(
        self,
        stats: Dict[Hashable, Dict[str, torch.Tensor]],
        class_ids: Sequence[Hashable],
    ) -> None:
        for c in class_ids:
            st = stats[c]
            class_n = st["class_count"].clamp_min(1.0)
            class_mean = st["class_sum"] / class_n
            class_var = (
                st["class_sumsq"] / class_n - class_mean.square()
            ).clamp_min(self.min_variance)
            variances = []
            safe_counts = st["count"].clamp_min(1.0)
            for k in range(self.prototypes[c].shape[0]):
                n = float(st["count"][k])
                if n >= self.min_cluster_samples:
                    mean = st["sum"][k] / safe_counts[k]
                    local_var = st["sumsq"][k] / safe_counts[k] - mean.square()
                else:
                    local_var = class_var
                local_var = (
                    (1.0 - self.variance_shrinkage) * local_var
                    + self.variance_shrinkage * class_var
                ).clamp_min(self.min_variance)
                variances.append(local_var.to(self.dtype))
            weights = safe_counts / safe_counts.sum().clamp_min(self.eps)
            self.prototype_variances[c] = torch.stack(variances).contiguous()
            self.prototype_log_weights[c] = weights.to(self.dtype).clamp_min(self.eps).log()

    @torch.no_grad()
    def _resolve_gamma_from_dataloader(self, dataloader: Iterable) -> None:
        if self.gamma != "scale":
            if self.gamma == "auto":
                self.resolved_gamma = 1.0 / max(self.pca_components.shape[0], 1)
            else:
                self.resolved_gamma = float(self.gamma)
            return
        count = 0
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        total_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        dimensions = self.pca_components.shape[0]
        for X, _ in self._iterate_loader(dataloader):
            Z = self.transform(X).to(torch.float64)
            count += Z.numel()
            total += Z.sum()
            total_sq += Z.square().sum()
        mean = total / max(count, 1)
        variance = (total_sq / max(count, 1) - mean.square()).clamp_min(self.eps)
        self.resolved_gamma = 1.0 / max(dimensions * float(variance), self.eps)

    def _iterate_loader(self, dataloader: Iterable) -> Iterator[Tuple[torch.Tensor, List[Hashable]]]:
        yielded = False
        for batch in dataloader:
            yielded = True
            if isinstance(batch, Mapping):
                if "features" not in batch or "labels" not in batch:
                    raise ValueError("mapping batches require 'features' and 'labels'")
                X, labels = batch["features"], batch["labels"]
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                X, labels = batch[0], batch[1]
            else:
                raise ValueError("Each dataloader batch must provide features and labels")
            X_t = self._validate_input(X)
            labels_list = self._normalize_labels(labels)
            if len(labels_list) != X_t.shape[0]:
                raise ValueError("batch feature and label counts differ")
            yield X_t, labels_list
        if not yielded:
            raise ValueError("dataloader yielded no batches")

    def _label_mask(self, labels: Sequence[Hashable], class_id: Hashable) -> torch.Tensor:
        return torch.tensor(
            [label == class_id for label in labels], dtype=torch.bool, device=self.device
        )

    @torch.no_grad()
    def add_class(
        self,
        class_id: Hashable,
        inputs: torch.Tensor | Sequence,
        *,
        refit_pca: bool = False,
    ) -> "PCAWhitenedRBFPrototypeClassifier":
        """Add one class without changing old prototypes when PCA remains frozen.

        If ``refit_pca=True``, exact refitting requires ``store_training_data=True``;
        all stored classes are then re-fit under the new PCA basis.
        """
        if class_id in self.class_ids:
            raise ValueError(f"Class {class_id!r} already exists; use update_class")
        X_new = self._validate_input(inputs)
        if X_new.shape[0] == 0:
            raise ValueError("new class requires samples")
        if not self.pca_fitted:
            raise ValueError("Fit PCA before adding a class")

        if refit_pca:
            if not self.store_training_data:
                raise ValueError("refit_pca=True requires store_training_data=True")
            combined = [v for v in self.raw_class_data.values()] + [X_new]
            self.fit_initial_pca(torch.cat(combined, dim=0))
            self.raw_class_data[class_id] = X_new.detach().clone()
            self.class_ids.append(class_id)
            self.class_to_index = {v: i for i, v in enumerate(self.class_ids)}
            self._refit_all_classes_from_memory()
        else:
            Z_new = self.transform(X_new)
            self.class_ids.append(class_id)
            self.class_to_index = {v: i for i, v in enumerate(self.class_ids)}
            self._fit_class_from_transformed(class_id, Z_new)
            self.class_counts[class_id] = X_new.shape[0]
            if self.store_training_data:
                self.raw_class_data[class_id] = X_new.detach().clone()

        self.model_fitted = True
        self.temperature_fitted = False
        return self

    @torch.no_grad()
    def update_class(
        self,
        class_id: Hashable,
        inputs: torch.Tensor | Sequence,
        *,
        refit_pca: bool = False,
    ) -> "PCAWhitenedRBFPrototypeClassifier":
        """Refit one class using its stored samples plus a new batch."""
        if class_id not in self.class_ids:
            return self.add_class(class_id, inputs, refit_pca=refit_pca)
        if not self.store_training_data:
            raise ValueError("update_class requires store_training_data=True")
        X_new = self._validate_input(inputs)
        X_class = torch.cat([self.raw_class_data[class_id], X_new], dim=0)
        self.raw_class_data[class_id] = X_class
        self.class_counts[class_id] = X_class.shape[0]
        if refit_pca:
            self.fit_initial_pca(torch.cat(list(self.raw_class_data.values()), dim=0))
            self._refit_all_classes_from_memory()
        else:
            self._fit_class_from_transformed(class_id, self.transform(X_class))
        self.model_fitted = True
        self.temperature_fitted = False
        return self

    def decision_function(self, inputs: torch.Tensor | Sequence) -> torch.Tensor:
        if not self.model_fitted:
            raise ValueError("Prototype RBF classifier is not fitted")
        Z = self.transform(inputs)
        return self._decision_function_transformed(Z)

    def to(self, device: str | torch.device) -> "PCAWhitenedRBFPrototypeClassifier":
        """Move PCA/prototype state, primarily for low-memory search staging."""
        self.device = torch.device(device)
        for name in ("global_mean", "pca_components", "pca_eigenvalues"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(self.device))
        for name in ("prototypes", "prototype_variances", "prototype_log_weights", "raw_class_data"):
            setattr(self, name, {key: value.to(self.device) for key, value in getattr(self, name).items()})
        return self

    def _decision_function_transformed(self, Z: torch.Tensor) -> torch.Tensor:
        """Score already PCA-transformed rows (used by cached searches)."""
        scores = [self._class_score(Z, class_id) for class_id in self.class_ids]
        return torch.stack(scores, dim=1)

    def _class_score(self, Z: torch.Tensor, class_id: Hashable) -> torch.Tensor:
        centers = self.prototypes[class_id]
        diff = Z[:, None, :] - centers[None, :, :]
        if self.metric_type == "diag_mahalanobis":
            variances = self.prototype_variances[class_id]
            distances = (diff.square() / variances[None, :, :]).sum(dim=-1)
        else:
            distances = diff.square().sum(dim=-1)
        local_log_scores = -float(self.resolved_gamma) * distances
        local_log_scores = local_log_scores + self.prototype_log_weights[class_id][None, :]
        if self.aggregation == "logsumexp":
            return torch.logsumexp(local_log_scores, dim=1)
        if self.aggregation == "max":
            return local_log_scores.max(dim=1).values
        return torch.exp(local_log_scores).mean(dim=1).clamp_min(self.eps).log()

    @torch.no_grad()
    def _fit_class_from_transformed(self, class_id: Hashable, Z: torch.Tensor) -> None:
        if Z.ndim != 2 or Z.shape[0] == 0:
            raise ValueError("class has no transformed samples")
        k = min(self.prototypes_per_class, Z.shape[0])
        if self.kmeans_fit_mode == "full":
            centers, assignment = self._kmeans(Z, k)
        else:
            centers, assignment = self._mini_batch_kmeans_tensor(Z, k)
        global_var = Z.var(dim=0, unbiased=False).clamp_min(self.min_variance)
        variances, counts = [], []
        for cluster in range(k):
            members = Z[assignment == cluster]
            count = members.shape[0]
            counts.append(max(count, 1))
            if count >= self.min_cluster_samples:
                local_var = members.var(dim=0, unbiased=False)
            else:
                local_var = global_var
            local_var = (
                (1.0 - self.variance_shrinkage) * local_var
                + self.variance_shrinkage * global_var
            ).clamp_min(self.min_variance)
            variances.append(local_var)
        weights = torch.tensor(counts, dtype=self.dtype, device=self.device)
        weights = weights / weights.sum().clamp_min(self.eps)
        self.prototypes[class_id] = centers.contiguous()
        self.prototype_variances[class_id] = torch.stack(variances).contiguous()
        self.prototype_log_weights[class_id] = weights.clamp_min(self.eps).log()


    @torch.no_grad()
    def _mini_batch_kmeans_tensor(
        self, Z: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mini-batch k-means for tensor inputs while preserving the fit() API."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + len(self.prototypes))
        init_n = min(Z.shape[0], self.initialization_sample_size)
        init_idx = torch.randperm(Z.shape[0], generator=generator)[:init_n].to(self.device)
        centers = self._initialize_centers(Z[init_idx], k, generator)
        persistent = torch.zeros(k, dtype=torch.float64, device=self.device)
        for _ in range(self.prototype_epochs):
            if self.reset_counts_each_epoch:
                persistent.zero_()
            order = torch.randperm(Z.shape[0], generator=generator).to(self.device)
            for start in range(0, Z.shape[0], self.prototype_batch_size):
                batch = Z[order[start : start + self.prototype_batch_size]]
                assignment = torch.cdist(batch, centers).square().argmin(dim=1)
                for cluster in range(k):
                    members = batch[assignment == cluster]
                    if members.shape[0] == 0:
                        continue
                    m = float(members.shape[0])
                    n = float(persistent[cluster])
                    centers[cluster] = (
                        n * centers[cluster] + m * members.mean(dim=0)
                    ) / max(n + m, 1.0)
                    persistent[cluster] += m
        assignment = torch.cdist(Z, centers).square().argmin(dim=1)
        return centers, assignment

    @torch.no_grad()
    def _kmeans(self, Z: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + len(self.prototypes))
        centers = self._initialize_centers(Z, k, generator)
        assignment = torch.zeros(Z.shape[0], dtype=torch.long, device=self.device)
        for _ in range(self.kmeans_iterations):
            distances = torch.cdist(Z, centers).square()
            new_assignment = distances.argmin(dim=1)
            if torch.equal(new_assignment, assignment):
                assignment = new_assignment
                break
            assignment = new_assignment
            new_centers = []
            for cluster in range(k):
                members = Z[assignment == cluster]
                if members.shape[0] == 0:
                    farthest = distances.min(dim=1).values.argmax()
                    new_centers.append(Z[farthest])
                else:
                    new_centers.append(members.mean(dim=0))
            centers = torch.stack(new_centers)
        return centers, assignment

    @torch.no_grad()
    def _initialize_centers(
        self, Z: torch.Tensor, k: int, generator: torch.Generator
    ) -> torch.Tensor:
        if k == Z.shape[0]:
            return Z.clone()
        if self.prototype_init == "random":
            idx = torch.randperm(Z.shape[0], generator=generator)[:k].to(self.device)
            return Z[idx].clone()
        first = int(torch.randint(Z.shape[0], (1,), generator=generator))
        indices = [first]
        min_dist = (Z - Z[first]).square().sum(dim=1)
        for _ in range(1, k):
            if self.prototype_init == "farthest":
                next_idx = int(min_dist.argmax())
            else:
                probs = min_dist.clamp_min(self.eps)
                probs = (probs / probs.sum()).cpu()
                next_idx = int(torch.multinomial(probs, 1, generator=generator))
            indices.append(next_idx)
            new_dist = (Z - Z[next_idx]).square().sum(dim=1)
            min_dist = torch.minimum(min_dist, new_dist)
        return Z[torch.tensor(indices, device=self.device)].clone()

    def _resolve_gamma(self, Z: torch.Tensor) -> None:
        if self.gamma == "auto":
            self.resolved_gamma = 1.0 / max(Z.shape[1], 1)
        elif self.gamma == "scale":
            variance = float(Z.var(unbiased=False).clamp_min(self.eps))
            self.resolved_gamma = 1.0 / max(Z.shape[1] * variance, self.eps)
        else:
            self.resolved_gamma = float(self.gamma)

    @torch.no_grad()
    def _refit_all_classes_from_memory(self) -> None:
        if not self.raw_class_data:
            raise ValueError("No stored class data")
        all_z = self.transform(torch.cat(list(self.raw_class_data.values()), dim=0))
        self._resolve_gamma(all_z)
        self.prototypes.clear()
        self.prototype_variances.clear()
        self.prototype_log_weights.clear()
        for class_id in self.class_ids:
            X = self.raw_class_data[class_id]
            self._fit_class_from_transformed(class_id, self.transform(X))
            self.class_counts[class_id] = X.shape[0]

    def _validate_input(self, X: torch.Tensor | Sequence) -> torch.Tensor:
        tensor = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.feature_dim:
            raise ValueError(f"Expected [N,{self.feature_dim}] inputs")
        if not torch.isfinite(tensor).all():
            raise ValueError("inputs contain non-finite values")
        return tensor

    def get_config(self) -> Dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "pca_dim": self.pca_dim,
            "prototypes_per_class": self.prototypes_per_class,
            "gamma": self.gamma,
            "metric_type": self.metric_type,
            "aggregation": self.aggregation,
            "kmeans_iterations": self.kmeans_iterations,
            "prototype_init": self.prototype_init,
            "kmeans_fit_mode": self.kmeans_fit_mode,
            "prototype_batch_size": self.prototype_batch_size,
            "prototype_epochs": self.prototype_epochs,
            "initialization_sample_size": self.initialization_sample_size,
            "reset_counts_each_epoch": self.reset_counts_each_epoch,
            "min_variance": self.min_variance,
            "variance_shrinkage": self.variance_shrinkage,
            "min_cluster_samples": self.min_cluster_samples,
            "store_training_data": self.store_training_data,
            "temperature": self.temperature,
            "random_seed": self.random_seed,
            "device": str(self.device),
            "dtype": self.dtype,
            "eps": self.eps,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.get_config(),
            "class_ids": list(self.class_ids),
            "temperature_fitted": self.temperature_fitted,
            "temperature_calibration": copy.deepcopy(self.temperature_calibration),
            "pca_fitted": self.pca_fitted,
            "model_fitted": self.model_fitted,
            "global_mean": self.global_mean,
            "pca_components": self.pca_components,
            "pca_eigenvalues": self.pca_eigenvalues,
            "resolved_gamma": self.resolved_gamma,
            "prototypes": self.prototypes,
            "prototype_variances": self.prototype_variances,
            "prototype_log_weights": self.prototype_log_weights,
            "class_counts": self.class_counts,
            "raw_class_data": self.raw_class_data,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.set_class_ids(state["class_ids"])
        self.temperature_fitted = bool(state.get("temperature_fitted", False))
        self.temperature_calibration = copy.deepcopy(state.get("temperature_calibration", {}))
        self.pca_fitted = bool(state["pca_fitted"])
        self.model_fitted = bool(state["model_fitted"])
        self.global_mean = self._move(state["global_mean"])
        self.pca_components = self._move(state["pca_components"])
        self.pca_eigenvalues = self._move(state["pca_eigenvalues"])
        self.resolved_gamma = state["resolved_gamma"]
        self.prototypes = {k: self._move(v) for k, v in state["prototypes"].items()}
        self.prototype_variances = {
            k: self._move(v) for k, v in state["prototype_variances"].items()
        }
        self.prototype_log_weights = {
            k: self._move(v) for k, v in state["prototype_log_weights"].items()
        }
        self.class_counts = dict(state["class_counts"])
        self.raw_class_data = {k: self._move(v) for k, v in state.get("raw_class_data", {}).items()}

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "PCAWhitenedRBFPrototypeClassifier":
        state = torch.load(Path(path), map_location=map_location, weights_only=False)
        config = dict(state["config"])
        config["device"] = str(map_location)
        model = cls(**config)
        model.load_state_dict(state)
        return model

    def _move(self, value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if value is None else value.to(self.device)


# Naming alias: valid terminology, while the primary name emphasizes that this
# implementation is a classifier rather than a trainable multilayer network.
PCAWhitenedPrototypeRBFNetwork = PCAWhitenedRBFPrototypeClassifier


# =============================================================================
# Neural classifier adapter
# =============================================================================


class NeuralClassifierAdapter(ProbabilisticClassifier):
    """Expose a PyTorch neural model through the common classifier API.

    Parameters
    ----------
    model:
        A module returning logits ``[B,C]``.
    class_ids:
        Output-column labels.
    input_transform:
        Optional callable that converts arbitrary deployment input into the neural
        model's required tensor. This is how a neural classifier may use images or
        histories while the SVM uses engineered feature vectors.
    fit_callback:
        Optional external training routine. It is called as
        ``fit_callback(adapter, inputs, labels, **kwargs)``. This avoids imposing a
        single training objective or dataloader design on future neural models.
    """

    def __init__(
        self,
        model: nn.Module,
        class_ids: Sequence[Hashable],
        *,
        input_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        fit_callback: Optional[Callable[..., Any]] = None,
        temperature: float = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(class_ids, temperature=temperature, device=device, dtype=dtype, eps=eps)
        self.model = model.to(self.device)
        self.input_transform = input_transform
        self.fit_callback = fit_callback
        self.training_history: Dict[str, List[float]] = {}
        self.require_feature = False

    def _prepare(self, inputs: Any) -> torch.Tensor:
        value = self.input_transform(inputs) if self.input_transform is not None else inputs
        tensor = torch.as_tensor(value, device=self.device)
        if tensor.is_floating_point():
            tensor = tensor.to(self.dtype)
        return tensor

    def fit(self, inputs: Any, labels: Sequence[Hashable] | torch.Tensor, **kwargs: Any) -> "NeuralClassifierAdapter":
        if self.fit_callback is None:
            raise NotImplementedError("Supply fit_callback or train the neural model externally")
        history = self.fit_callback(self, inputs, labels, **kwargs)
        if history is not None:
            self.training_history = history
        return self

    def decision_function(self, inputs: Any) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(self._prepare(inputs))
        if logits.ndim != 2 or logits.shape[1] != len(self.class_ids):
            raise RuntimeError(f"Neural model must return [B,{len(self.class_ids)}] logits")
        return logits

    def to(self, device: str | torch.device) -> "NeuralClassifierAdapter":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def save(self, path: str | Path) -> None: 
        """Save the adapter's model weights and configuration.""" 
        torch.save( { 
            "model_state_dict": self.model.state_dict(), 
            "class_ids": list(self.class_ids), 
            "temperature": self.temperature, 
            "eps": self.eps, 
            "require_feature": self.require_feature,
            "training_history": self.training_history,
        }, path, )
    
    @classmethod
    def load(
        cls,
        path: str | Path,
        model: nn.Module,
        *,
        input_transform: Optional[Callable[[Any], torch.Tensor]] = None,
        fit_callback: Optional[Callable[..., Any]] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "NeuralClassifierAdapter":
        """Load an adapter from a saved checkpoint."""
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        adapter = cls(
            model=model,
            class_ids=checkpoint["class_ids"],
            input_transform=input_transform,
            fit_callback=fit_callback,
            temperature=checkpoint["temperature"],
            device=device,
            dtype=dtype,
            eps=checkpoint["eps"],
        )

        adapter.model.load_state_dict(checkpoint["model_state_dict"])
        adapter.require_feature = bool(checkpoint.get("require_feature", False))
        adapter.training_history = checkpoint.get("training_history", {})
        return adapter


def fit_nn(
    adapter: NeuralClassifierAdapter,
    inputs: Any,
    labels: Sequence[Hashable] | torch.Tensor,
    *,
    val: Optional[Tuple[Any, Sequence[Hashable] | torch.Tensor]] = None,
    epochs: int = 20,
    batch_size: int = 64,
    validation_batch_size: int = 256,
    optimizer: str | type[torch.optim.Optimizer] | Callable[..., torch.optim.Optimizer] = "adam",
    lr: float = 1e-3,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    shuffle: bool = True,
    verbose: bool = True,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    restore_best_weights: bool = False,
) -> Dict[str, Any]:
    """Train a :class:`NeuralClassifierAdapter` with cross-entropy.

    Inputs remain on CPU in the DataLoaders and are prepared by the adapter for
    each device-safe batch. The same callback works for ``[N,D]`` engineered
    features and ``[N,C,H,W]`` raw-depth tensors.
    """
    from torch.utils.data import DataLoader, TensorDataset

    if epochs <= 0 or batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive or None")
    label_values = adapter._normalize_labels(labels)
    if not adapter.class_ids:
        adapter.set_class_ids(list(dict.fromkeys(label_values)))
    encoded = adapter._encode_labels(label_values).cpu()
    train_x = torch.as_tensor(inputs).cpu()
    if train_x.shape[0] != encoded.numel():
        raise ValueError("input and label counts differ")
    train_loader = DataLoader(
        TensorDataset(train_x, encoded), batch_size=batch_size, shuffle=shuffle
    )

    validation_loader = None
    if val is not None:
        val_x, val_labels = val
        val_encoded = adapter._encode_labels(val_labels).cpu()
        val_x = torch.as_tensor(val_x).cpu()
        if val_x.shape[0] != val_encoded.numel():
            raise ValueError("validation input and label counts differ")
        validation_loader = DataLoader(
            TensorDataset(val_x, val_encoded), batch_size=validation_batch_size
        )

    kwargs = dict(optimizer_kwargs or {})
    if isinstance(optimizer, str):
        optimizer_types = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        try:
            optimizer_type = optimizer_types[optimizer.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported optimizer {optimizer!r}") from exc
        opt = optimizer_type(adapter.model.parameters(), lr=lr, **kwargs)
    else:
        opt = optimizer(adapter.model.parameters(), lr=lr, **kwargs)
    criterion = nn.CrossEntropyLoss()
    history: Dict[str, Any] = {
        "train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": [],
        "optimizer_updates": 0,
    }
    best_state, best_loss, best_epoch, stale_epochs = None, float("inf"), None, 0
    stopped_early = False

    for epoch in range(epochs):
        adapter.model.train()
        loss_sum = correct = total = 0
        for x_batch, y_batch in train_loader:
            x_batch = adapter._prepare(x_batch)
            y_batch = y_batch.to(adapter.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = adapter.model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            opt.step()
            history["optimizer_updates"] += 1
            total += y_batch.numel()
            loss_sum += float(loss.detach()) * y_batch.numel()
            correct += int((logits.detach().argmax(1) == y_batch).sum())
        history["train_loss"].append(loss_sum / max(total, 1))
        history["train_accuracy"].append(correct / max(total, 1))

        if validation_loader is not None:
            adapter.model.eval()
            val_loss = val_correct = val_total = 0
            with torch.inference_mode():
                for x_batch, y_batch in validation_loader:
                    x_batch = adapter._prepare(x_batch)
                    y_batch = y_batch.to(adapter.device, non_blocking=True)
                    logits = adapter.model(x_batch)
                    loss = criterion(logits, y_batch)
                    val_total += y_batch.numel()
                    val_loss += float(loss) * y_batch.numel()
                    val_correct += int((logits.argmax(1) == y_batch).sum())
            history["validation_loss"].append(val_loss / max(val_total, 1))
            history["validation_accuracy"].append(val_correct / max(val_total, 1))
            current_loss = history["validation_loss"][-1]
            if ((early_stopping_patience is not None or restore_best_weights)
                    and current_loss < best_loss - float(early_stopping_min_delta)):
                best_loss, best_epoch, stale_epochs = current_loss, epoch, 0
                best_state = {key: value.detach().clone()
                              for key, value in adapter.model.state_dict().items()}
            elif early_stopping_patience is not None or restore_best_weights:
                stale_epochs += 1
        if verbose:
            message = (
                f"Epoch {epoch + 1:3d} | train loss {history['train_loss'][-1]:.4f} "
                f"| train acc {history['train_accuracy'][-1]:.4f}"
            )
            if history["validation_loss"]:
                message += (
                    f" | val loss {history['validation_loss'][-1]:.4f}"
                    f" | val acc {history['validation_accuracy'][-1]:.4f}"
                )
            print(message)
        if (validation_loader is not None and early_stopping_patience is not None
                and stale_epochs >= early_stopping_patience):
            stopped_early = True
            break
    if restore_best_weights and best_state is not None:
        adapter.model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = None if best_epoch is None else best_loss
    history["stopped_early"] = stopped_early
    history["best_weights_restored"] = bool(restore_best_weights and best_state is not None)
    history["epochs_completed"] = len(history["train_loss"])
    return history

# =============================================================================
# Discrete Bayesian filter
# =============================================================================


class BayesianTerrainFilter:
    """Discrete Bayes filter over an ordered terrain class set.

    Conventions
    -----------
    ``transition[i,j] = P(x_t=j | x_{t-1}=i)``.
    ``observation[i,j] = P(classifier observation=j | true state=i)``.

    Given classifier distribution q, the confusion-aware compatibility is

        L(i) = sum_j observation[i,j] q(j).

    The likelihood is tempered by an evidence exponent before correction.
    """

    def __init__(
        self,
        labels: Sequence[Hashable],
        prior: torch.Tensor | Sequence[float] | Mapping[Hashable, float],
        transition_matrix: torch.Tensor | Sequence,
        observation_matrix: Optional[torch.Tensor | Sequence] = None,
        *,
        evidence_power: float = 0.75,
        adaptive_evidence: bool = True,
        min_evidence_power: float = 0.20,
        confidence_gamma: float = 1.0,
        transition_alpha: Optional[float] = None,
        stay_probability: Optional[float] = None,
        transition_source: Optional[str] = None,
        device: str | torch.device = "cpu",
        eps: float = 1e-8,
    ) -> None:
        self.labels = list(labels)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        self.device = torch.device(device)
        self.eps = float(eps)
        self.num_classes = len(self.labels)
        self.evidence_power = float(evidence_power)
        self.adaptive_evidence = bool(adaptive_evidence)
        self.min_evidence_power = float(min_evidence_power)
        self.confidence_gamma = float(confidence_gamma)
        self.transition_alpha = None if transition_alpha is None else float(transition_alpha)
        self.stay_probability = None if stay_probability is None else float(stay_probability)
        self.transition_source = transition_source
        if self.min_evidence_power > self.evidence_power:
            raise ValueError("min_evidence_power cannot exceed evidence_power")
        self.initial_prior = make_manual_prior(self.labels, prior, device=self.device, eps=self.eps)
        self.transition_matrix = _validate_row_stochastic_matrix(
            transition_matrix, self.num_classes, "transition_matrix", self.device, self.eps
        )
        if observation_matrix is None:
            observation_matrix = torch.eye(self.num_classes)
        self.observation_matrix = _validate_row_stochastic_matrix(
            observation_matrix, self.num_classes, "observation_matrix", self.device, self.eps
        )
        self.belief = self.initial_prior.clone()

    @torch.inference_mode()
    def update(
        self,
        classifier_probabilities: torch.Tensor | Sequence[float],
        *,
        observation_quality: float = 1.0,
    ) -> BayesianFilterStep:
        q = _normalize_vector(
            classifier_probabilities, self.num_classes, self.device, self.eps, "classifier_probabilities"
        )
        quality = float(observation_quality)
        if not 0 <= quality <= 1:
            raise ValueError("observation_quality must be in [0,1]")
        predicted = self.belief @ self.transition_matrix
        predicted = predicted / predicted.sum().clamp_min(self.eps)
        likelihood = (self.observation_matrix @ q).clamp_min(self.eps)
        confidence = self.entropy_confidence(q)
        beta = self._effective_evidence_power(confidence, quality)
        posterior = predicted * likelihood.pow(beta)
        posterior = posterior / posterior.sum().clamp_min(self.eps)
        self.belief = posterior
        index = int(posterior.argmax())
        return BayesianFilterStep(
            self.labels[index], posterior.clone(), predicted.clone(), likelihood.clone(),
            q.clone(), confidence, beta
        )

    def reset(self, prior: Optional[torch.Tensor | Sequence[float] | Mapping[Hashable, float]] = None) -> torch.Tensor:
        self.belief = self.initial_prior.clone() if prior is None else make_manual_prior(
            self.labels, prior, device=self.device, eps=self.eps
        )
        return self.belief.clone()

    def predict_label(
        self,
        *,
        min_posterior: float = 0.0,
        min_margin: float = 0.0,
        fallback_label: Optional[Hashable] = None,
    ) -> Hashable:
        values, indices = torch.topk(self.belief, min(2, self.num_classes))
        label = self.labels[int(indices[0])]
        margin = float(values[0] - values[1]) if self.num_classes > 1 else 1.0
        if float(values[0]) < min_posterior or margin < min_margin:
            return label if fallback_label is None else fallback_label
        return label

    def entropy_confidence(self, probabilities: torch.Tensor | Sequence[float]) -> float:
        q = _normalize_vector(probabilities, self.num_classes, self.device, self.eps, "probabilities")
        if self.num_classes == 1:
            return 1.0
        entropy = -(q * q.clamp_min(self.eps).log()).sum()
        return float((1.0 - entropy / math.log(self.num_classes)).clamp(0.0, 1.0))

    def _effective_evidence_power(self, confidence: float, quality: float) -> float:
        if self.adaptive_evidence:
            beta = self.min_evidence_power + confidence ** self.confidence_gamma * (
                self.evidence_power - self.min_evidence_power
            )
        else:
            beta = self.evidence_power
        return float(beta * quality)

    def save(self, path: str | Path) -> None:
        """Save the filter's configuration and current belief state."""
        torch.save(
            {
                "labels": list(self.labels),
                "initial_prior": self.initial_prior.cpu(),
                "belief": self.belief.cpu(),
                "transition_matrix": self.transition_matrix.cpu(),
                "observation_matrix": self.observation_matrix.cpu(),
                "evidence_power": self.evidence_power,
                "adaptive_evidence": self.adaptive_evidence,
                "min_evidence_power": self.min_evidence_power,
                "confidence_gamma": self.confidence_gamma,
                "transition_alpha": self.transition_alpha,
                "stay_probability": self.stay_probability,
                "transition_source": self.transition_source,
                "eps": self.eps,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "BayesianTerrainFilter":
        """Load a filter from a saved checkpoint, restoring its belief state."""
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        filt = cls(
            labels=checkpoint["labels"],
            prior=checkpoint["initial_prior"],
            transition_matrix=checkpoint["transition_matrix"],
            observation_matrix=checkpoint["observation_matrix"],
            evidence_power=checkpoint["evidence_power"],
            adaptive_evidence=checkpoint["adaptive_evidence"],
            min_evidence_power=checkpoint["min_evidence_power"],
            confidence_gamma=checkpoint["confidence_gamma"],
            transition_alpha=checkpoint.get("transition_alpha"),
            stay_probability=checkpoint.get("stay_probability"),
            transition_source=checkpoint.get("transition_source"),
            device=device,
            eps=checkpoint["eps"],
        )

        filt.belief = checkpoint["belief"].to(filt.device)
        return filt

class BayesianFilteredClassifier:
    """Runtime composition of any ProbabilisticClassifier and a Bayes filter."""

    def __init__(self, classifier: ProbabilisticClassifier, bayes_filter: BayesianTerrainFilter):
        if list(classifier.class_ids) != list(bayes_filter.labels):
            raise ValueError("classifier and filter label order differ")
        self.classifier = classifier
        self.filter = bayes_filter

    @torch.inference_mode()
    def predict(
        self,
        inputs: Any,
        *,
        observation_quality: float = 1.0,
        temperature: Optional[float] = None,
        probability_floor: float = 1e-8,
    ) -> BayesianFilterStep:
        probabilities, _ = self.classifier.predict_class_distribution(
            inputs, temperature=temperature, probability_floor=probability_floor
        )
        if probabilities.shape[0] != 1:
            raise ValueError("Runtime predict expects one observation; use run_filter_sequences for batches")
        return self.filter.update(probabilities[0], observation_quality=observation_quality)

    def reset(self, prior: Optional[torch.Tensor | Sequence[float] | Mapping[Hashable, float]] = None) -> torch.Tensor:
        return self.filter.reset(prior)


# =============================================================================
# Construction, calibration, and evaluation utilities
# =============================================================================


def make_manual_prior(
    labels: Sequence[Hashable],
    prior: torch.Tensor | Sequence[float] | Mapping[Hashable, float],
    *,
    device: str | torch.device = "cpu",
    eps: float = 1e-8,
) -> torch.Tensor:
    labels = list(labels)
    if isinstance(prior, Mapping):
        unknown = set(prior) - set(labels)
        if unknown:
            raise ValueError(f"prior contains unknown labels: {unknown}")
        values = [float(prior.get(label, eps)) for label in labels]
    else:
        values = prior
    return _normalize_vector(values, len(labels), torch.device(device), eps, "prior")


def make_persistent_transition_matrix(
    labels: Sequence[Hashable],
    stay_probability: float = 0.95,
    *,
    transition_weights: Optional[Mapping[Hashable, Mapping[Hashable, float]]] = None,
    device: str | torch.device = "cpu",
    eps: float = 1e-8,
) -> torch.Tensor:
    labels = list(labels)
    c = len(labels)
    if c == 1:
        return torch.ones(1, 1, device=device)
    if not 0 <= stay_probability < 1:
        raise ValueError("stay_probability must be in [0,1)")
    matrix = torch.zeros(c, c, device=device)
    index = {v: i for i, v in enumerate(labels)}
    for source in labels:
        i = index[source]
        matrix[i, i] = stay_probability
        remaining = 1.0 - stay_probability
        if transition_weights is None or source not in transition_weights:
            for j in range(c):
                if j != i:
                    matrix[i, j] = remaining / (c - 1)
        else:
            row = torch.tensor(
                [0.0 if d == source else float(transition_weights[source].get(d, 0.0)) for d in labels],
                device=device,
            )
            if float(row.sum()) <= 0:
                raise ValueError(f"no positive outgoing weights for {source}")
            matrix[i] += remaining * row / row.sum()
            matrix[i, i] = stay_probability
    return _validate_row_stochastic_matrix(matrix, c, "transition_matrix", torch.device(device), eps)


def estimate_transition_matrix_from_sequences(
    true_labels: Sequence[Hashable],
    labels: Sequence[Hashable],
    *,
    sequence_ids: Optional[Sequence[Any]] = None,
    pseudocount: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    labels = list(labels)
    true_labels = (
        true_labels.detach().cpu().tolist() if torch.is_tensor(true_labels) else list(true_labels)
    )
    sequence_ids = (
        sequence_ids.detach().cpu().tolist()
        if torch.is_tensor(sequence_ids)
        else list(sequence_ids) if sequence_ids is not None else [0] * len(true_labels)
    )
    if len(sequence_ids) != len(true_labels):
        raise ValueError("transition labels and sequence_ids must have equal length")
    index = {v: i for i, v in enumerate(labels)}
    counts = torch.full((len(labels), len(labels)), float(pseudocount), device=device)
    for t in range(1, len(true_labels)):
        if sequence_ids[t] == sequence_ids[t - 1]:
            counts[index[true_labels[t - 1]], index[true_labels[t]]] += 1
    return counts / counts.sum(1, keepdim=True)


def build_hybrid_transition_cache(
    labels: Sequence[Hashable],
    transition_training_labels: Sequence[Hashable] | torch.Tensor,
    *,
    transition_training_sequence_ids: Optional[Sequence[Any]] = None,
    stay_probabilities: Sequence[float] = (0.90, 0.94, 0.97),
    transition_alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    transition_pseudocount: float = 1.0,
    device: str | torch.device = "cpu",
) -> HybridTransitionCache:
    """Build reusable, deduplicated hybrid transition candidates.

    This cache depends only on label ordering and ordered ground-truth sequences,
    so one instance may be passed to searches for several classifiers. Candidate
    matrices are ``alpha*T_empirical + (1-alpha)*T_persistent``. Numerically
    identical matrices share storage and are evaluated once; in particular, the
    pure empirical endpoint is independent of ``stay_probability``.
    """
    labels = list(labels)
    training_labels = (
        transition_training_labels.detach().cpu().tolist()
        if torch.is_tensor(transition_training_labels)
        else list(transition_training_labels)
    )
    if transition_pseudocount < 0:
        raise ValueError("transition_pseudocount must be non-negative")
    unknown = set(training_labels) - set(labels)
    if unknown:
        raise ValueError(f"transition training labels contain unknown classes: {unknown}")
    for alpha in transition_alphas:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("transition_alphas must be in [0,1]")

    target_device = torch.device(device)
    empirical = estimate_transition_matrix_from_sequences(
        training_labels, labels, sequence_ids=transition_training_sequence_ids,
        pseudocount=transition_pseudocount, device=target_device,
    )
    persistent_cache = {
        float(stay): make_persistent_transition_matrix(labels, float(stay), device=target_device)
        for stay in dict.fromkeys(stay_probabilities)
    }
    matrices_by_parameters: Dict[Tuple[float, float], torch.Tensor] = {}
    unique_candidates: List[Dict[str, Any]] = []
    for stay in dict.fromkeys(float(value) for value in stay_probabilities):
        persistent = persistent_cache[stay]
        for alpha in dict.fromkeys(float(value) for value in transition_alphas):
            combined = alpha * empirical + (1.0 - alpha) * persistent
            combined = combined / combined.sum(1, keepdim=True)
            canonical = next(
                (candidate for candidate in unique_candidates
                 if torch.allclose(candidate["transition_matrix"], combined, rtol=1e-6, atol=1e-7)),
                None,
            )
            if canonical is None:
                canonical = {
                    "stay_probability": stay,
                    "transition_alpha": alpha,
                    "transition_matrix": combined,
                }
                unique_candidates.append(canonical)
            matrices_by_parameters[(stay, alpha)] = canonical["transition_matrix"]
    if not unique_candidates:
        raise ValueError("transition search produced no candidates")
    return HybridTransitionCache(
        labels, empirical, matrices_by_parameters, unique_candidates,
        float(transition_pseudocount),
    )


def estimate_observation_matrix_from_probabilities(
    probabilities: torch.Tensor | Sequence,
    true_labels: Sequence[Hashable],
    labels: Sequence[Hashable],
    *,
    mode: str = "soft",
    pseudocount: float = 0.5,
    device: str | torch.device = "cpu",
    eps: float = 1e-8,
    encoded_true_labels: Optional[torch.Tensor | Sequence[int]] = None,
) -> torch.Tensor:
    labels = list(labels)
    probs = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
    probs = probs.clamp_min(0)
    probs = probs / probs.sum(1, keepdim=True).clamp_min(eps)
    if probs.shape != (len(true_labels), len(labels)):
        raise ValueError("probability matrix shape does not match labels")
    index = {v: i for i, v in enumerate(labels)}
    counts = torch.full((len(labels), len(labels)), float(pseudocount), device=device)
    encoded = (
        torch.as_tensor(encoded_true_labels, dtype=torch.long, device=device)
        if encoded_true_labels is not None
        else torch.tensor([index[y] for y in true_labels], dtype=torch.long, device=device)
    )
    if encoded.shape != (len(true_labels),):
        raise ValueError("encoded_true_labels must have one index per label")
    if mode == "soft":
        counts.index_add_(0, encoded, probs)
    elif mode == "hard":
        observed = probs.argmax(1)
        flat = encoded * len(labels) + observed
        counts += torch.bincount(flat, minlength=len(labels) ** 2).reshape_as(counts)
    else:
        raise ValueError("mode must be 'soft' or 'hard'")
    return counts / counts.sum(1, keepdim=True)


@torch.inference_mode()
def collect_classifier_probabilities(
    classifier: ProbabilisticClassifier,
    inputs: Any,
    *,
    temperature: Optional[float] = None,
    probability_floor: float = 0.0,
) -> Tuple[torch.Tensor, List[Hashable]]:
    return classifier.predict_class_distribution(
        inputs, temperature=temperature, probability_floor=probability_floor
    )


@torch.inference_mode()
def collect_classifier_scores_batched(
    classifier: ProbabilisticClassifier,
    inputs: Any,
    *,
    batch_size: int = 512,
    cache_device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Run classifier inference in bounded batches and retain only ``[N,C]`` scores.

    ``inputs`` may be an indexable tensor/array or a DataLoader-style iterable.
    Iterable batches may provide inputs directly, as their first tuple item, or
    under ``inputs``/``features``. Scores are transferred to ``cache_device`` after
    every batch so a GPU classifier never retains the full dataset in VRAM.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def batch_input(batch: Any) -> Any:
        if isinstance(batch, Mapping):
            for key in ("inputs", "features"):
                if key in batch:
                    return batch[key]
            raise ValueError("mapping batches require an 'inputs' or 'features' entry")
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            return batch[0]
        return batch

    if torch.is_tensor(inputs) or hasattr(inputs, "shape"):
        total = int(inputs.shape[0])
        batches = (inputs[start:start + batch_size] for start in range(0, total, batch_size))
    else:
        batches = (batch_input(batch) for batch in inputs)

    pieces: List[torch.Tensor] = []
    for batch in batches:
        scores = classifier.decision_function(batch)
        if scores.ndim != 2 or scores.shape[1] != len(classifier.class_ids):
            raise RuntimeError("classifier scores must have shape [B,C]")
        pieces.append(scores.detach().to(cache_device))
        del scores
    if not pieces:
        raise ValueError("classifier score input yielded no samples")
    return torch.cat(pieces, dim=0)


@torch.inference_mode()
def run_filter_sequences(
    terrain_filter: BayesianTerrainFilter,
    classifier_probabilities: torch.Tensor | Sequence,
    *,
    sequence_ids: Optional[Sequence[Any]] = None,
    observation_quality: Optional[torch.Tensor | Sequence[float]] = None,
    prior: Optional[torch.Tensor | Sequence[float] | Mapping[Hashable, float]] = None,
    return_traces: bool = True,
) -> Tuple[List[Hashable], torch.Tensor, torch.Tensor]:
    probabilities = torch.as_tensor(classifier_probabilities, dtype=torch.float32)
    n = probabilities.shape[0]
    sequence_ids = list(sequence_ids) if sequence_ids is not None else [0] * n
    qualities = torch.ones(n) if observation_quality is None else torch.as_tensor(observation_quality).flatten()
    predicted, posteriors, powers = [], [], []
    previous = object()
    for i in range(n):
        if i == 0 or sequence_ids[i] != previous:
            terrain_filter.reset(prior)
        previous = sequence_ids[i]
        step = terrain_filter.update(probabilities[i], observation_quality=float(qualities[i]))
        predicted.append(step.label)
        if return_traces:
            posteriors.append(step.posterior.cpu())
            powers.append(step.evidence_power)
    if return_traces:
        return predicted, torch.stack(posteriors), torch.tensor(powers)
    return predicted, torch.empty(0), torch.empty(0)


def evaluate_predictions(
    true_labels: Sequence[Hashable],
    predicted_labels: Sequence[Hashable],
    labels: Optional[Sequence[Hashable]] = None,
    *,
    sequence_ids: Optional[Sequence[Any]] = None,
) -> FilterEvaluation:
    truth, prediction = list(true_labels), list(predicted_labels)
    labels = list(labels or dict.fromkeys(truth + prediction))
    index = {v: i for i, v in enumerate(labels)}
    confusion = torch.zeros(len(labels), len(labels), dtype=torch.int64)
    for y, yh in zip(truth, prediction):
        confusion[index[y], index[yh]] += 1
    accuracy = float(confusion.diag().sum()) / max(int(confusion.sum()), 1)
    recalls, f1s = [], []
    for i in range(len(labels)):
        tp = float(confusion[i, i])
        fn = float(confusion[i].sum()) - tp
        fp = float(confusion[:, i].sum()) - tp
        recall = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    delay = _mean_transition_delay(truth, prediction, sequence_ids)
    false_rate = _false_transition_rate(truth, prediction, sequence_ids)
    return FilterEvaluation(
        accuracy,
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s),
        confusion,
        labels,
        delay,
        false_rate,
    )


# =============================================================================
# Hyperparameter search
# =============================================================================


def search_rbf_svm_hyperparameters(
    train_features: torch.Tensor | Sequence,
    train_labels: Sequence[Hashable] | torch.Tensor,
    validation_features: torch.Tensor | Sequence,
    validation_labels: Sequence[Hashable] | torch.Tensor,
    *,
    base_config: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    scoring: str = "validation_accuracy",
    fit_best: bool = True,
    verbose: bool = False,
) -> Tuple[RBFSVM, List[SVMSearchResult]]:
    """Grid-search direct-feature RBF-SVM structural parameters.

    Typical searchable parameters are ``gamma``, ``max_kernel_samples``,
    ``weight_decay``, ``learning_rate``,
    ``squared_hinge``, and ``class_balance``. Prediction temperature is omitted
    deliberately; calibrate it afterward or tune it jointly with the Bayes filter.
    """
    if scoring not in {"validation_accuracy", "validation_nll", "validation_brier"}:
        raise ValueError("unsupported scoring")
    keys = list(search_space)
    if "pca_dim" in base_config or "pca_dim" in search_space:
        raise ValueError("RBFSVM has no pca_dim; standardize features before search")
    results: List[SVMSearchResult] = []
    best_model, best_key = None, None
    train = torch.as_tensor(train_features)
    validation = torch.as_tensor(validation_features)
    labels_list = list(train_labels.detach().cpu().tolist() if torch.is_tensor(train_labels) else train_labels)
    class_ids = list(dict.fromkeys(labels_list))
    class_index = {label: i for i, label in enumerate(class_ids)}
    encoded = torch.tensor([class_index[label] for label in labels_list], dtype=torch.long)
    distance_cache: Dict[Tuple[Any, ...], Dict[str, torch.Tensor]] = {}

    def cached_distances(config: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        maximum = config.get("max_kernel_samples", 512)
        seed = int(config.get("random_seed", 0))
        cache_key = (
            maximum, seed, str(config.get("device", "cpu")), config.get("dtype", torch.float32)
        )
        if cache_key in distance_cache:
            return distance_cache[cache_key]
        helper = RBFSVM(**config)
        helper.set_class_ids(class_ids)
        x = train.to(helper.device, helper.dtype)
        y = encoded.to(helper.device)
        torch.manual_seed(seed)
        if maximum is not None and x.shape[0] > maximum:
            basis = x[helper._stratified_basis_indices(y, maximum)]
        else:
            basis = x
        val = validation.to(helper.device, helper.dtype)
        squared = lambda a, b: (
            a.square().sum(1, keepdim=True) + b.square().sum(1).unsqueeze(0) - 2 * a @ b.T
        ).clamp_min(0).cpu()
        cached = {
            "basis": basis.detach().cpu(),
            "train_distances": squared(x, basis),
            "basis_distances": squared(basis, basis),
            "validation_distances": squared(val, basis),
        }
        distance_cache[cache_key] = cached
        return cached

    for values in itertools.product(*(search_space[k] for k in keys)):
        params = dict(zip(keys, values))
        config = dict(base_config)
        config.update(params)
        model = RBFSVM(**config)
        model.fit(
            train_features,
            train_labels,
            validation_inputs=validation_features,
            validation_labels=validation_labels,
            _kernel_cache=cached_distances(config),
        )
        probabilities, _ = model.predict_class_distribution(validation_features, temperature=1.0)
        y = model._encode_labels(validation_labels)
        acc = float((probabilities.argmax(1) == y).float().mean())
        nll = float(F.nll_loss(probabilities.clamp_min(model.eps).log(), y))
        targets = F.one_hot(y, len(model.class_ids)).to(model.dtype)
        brier = float((probabilities - targets).square().sum(1).mean())
        result = SVMSearchResult(params, acc, nll, brier)
        results.append(result)
        key = (-acc, nll, brier) if scoring == "validation_accuracy" else (nll,) if scoring == "validation_nll" else (brier,)
        if best_key is None or key < best_key:
            best_key, best_model = key, model
        if verbose:
            print(params, acc, nll, brier)
    if scoring == "validation_accuracy":
        results.sort(key=lambda r: (-r.validation_accuracy, r.validation_nll, r.validation_brier))
    elif scoring == "validation_nll":
        results.sort(key=lambda r: r.validation_nll)
    else:
        results.sort(key=lambda r: r.validation_brier)
    if best_model is None:
        raise RuntimeError("no SVM search trials")
    # Trials are released as the loop advances; only the selected fitted model is
    # retained. ``fit_best`` remains accepted for API compatibility.
    return best_model, results



def search_prototype_rbf_hyperparameters(
    train_features: torch.Tensor | Sequence,
    train_labels: Sequence[Hashable] | torch.Tensor,
    validation_features: torch.Tensor | Sequence,
    validation_labels: Sequence[Hashable] | torch.Tensor,
    *,
    base_config: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    scoring: str = "validation_accuracy",
    fit_best: bool = True,
    calibrate_temperature: bool = False,
    calibration_objective: str = "nll",
    verbose: bool = False,
) -> Tuple[PCAWhitenedRBFPrototypeClassifier, List[PrototypeRBFSearchResult]]:
    """Grid-search PCA and prototype-RBF structural hyperparameters.

    Useful parameters include ``pca_dim``, ``prototypes_per_class``, ``gamma``,
    ``metric_type``, ``aggregation``, ``prototype_init``, ``kmeans_fit_mode``,
    ``prototype_batch_size``, ``prototype_epochs``, ``min_variance``, and
    ``variance_shrinkage``. Temperature is normally calibrated after structural
    selection or tuned jointly with the downstream Bayes filter.
    """
    if scoring not in {"validation_accuracy", "validation_nll", "validation_brier"}:
        raise ValueError("unsupported scoring")
    keys = list(search_space)
    results: List[PrototypeRBFSearchResult] = []
    best_model: Optional[PCAWhitenedRBFPrototypeClassifier] = None
    best_key: Optional[Tuple[float, ...]] = None
    pca_cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for values in itertools.product(*(search_space[k] for k in keys)):
        params = dict(zip(keys, values))
        config = dict(base_config)
        config.update(params)
        model = PCAWhitenedRBFPrototypeClassifier(**config)
        X = model._validate_input(train_features)
        pca_key = (model.pca_dim, model.feature_dim, str(model.device), model.dtype, model.eps)
        if pca_key not in pca_cache:
            model.fit_initial_pca(X)
            pca_cache[pca_key] = (
                model.global_mean.detach().clone(), model.pca_components.detach().clone(),
                model.pca_eigenvalues.detach().clone(), model.transform(X).detach(),
                model.transform(validation_features).detach(),
            )
        mean, components, eigenvalues, Z_train, Z_validation = pca_cache[pca_key]
        model.global_mean = mean.detach().clone()
        model.pca_components = components.detach().clone()
        model.pca_eigenvalues = eigenvalues.detach().clone()
        model.pca_fitted = True
        y_labels = model._normalize_labels(train_labels)
        model._fit_from_transformed(X, y_labels, Z_train)
        if calibrate_temperature:
            # Calibration still uses the public API; structural searches normally
            # leave this disabled and tune temperature with the Bayes filter.
            model.fit_temperature(validation_features, validation_labels, objective=calibration_objective)
        scores = model._decision_function_transformed(Z_validation)
        probabilities = F.softmax(scores / model.temperature, dim=1)
        y = model._encode_labels(validation_labels)
        acc = float((probabilities.argmax(1) == y).float().mean())
        nll = float(F.nll_loss(probabilities.clamp_min(model.eps).log(), y))
        targets = F.one_hot(y, len(model.class_ids)).to(model.dtype)
        brier = float((probabilities - targets).square().sum(1).mean())
        num_prototypes = sum(v.shape[0] for v in model.prototypes.values())
        result = PrototypeRBFSearchResult(params, acc, nll, brier, num_prototypes)
        results.append(result)
        key = (
            (-acc, nll, brier, num_prototypes)
            if scoring == "validation_accuracy"
            else (nll, num_prototypes)
            if scoring == "validation_nll"
            else (brier, num_prototypes)
        )
        if best_key is None or key < best_key:
            best_key, best_model = key, model
        if verbose:
            print(params, acc, nll, brier, num_prototypes)

    if scoring == "validation_accuracy":
        results.sort(key=lambda r: (-r.validation_accuracy, r.validation_nll, r.validation_brier, r.num_prototypes))
    elif scoring == "validation_nll":
        results.sort(key=lambda r: (r.validation_nll, r.num_prototypes))
    else:
        results.sort(key=lambda r: (r.validation_brier, r.num_prototypes))
    if best_model is None:
        raise RuntimeError("no prototype-RBF search trials")
    return best_model, results



def search_prototype_rbf_hyperparameters_dataloader(
    train_loader_factory: Callable[[], Iterable],
    validation_loader_factory: Callable[[], Iterable],
    *,
    base_config: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    scoring: str = "validation_accuracy",
    verbose: bool = False,
) -> Tuple[PCAWhitenedRBFPrototypeClassifier, List[PrototypeRBFSearchResult]]:
    """Grid-search prototype-RBF models without materializing train/validation data.

    Loader factories are used rather than loader instances so every trial receives
    a fresh, re-iterable loader. Each training loader must yield ``(features,
    labels)`` batches. The utility forces ``kmeans_fit_mode='mini_batch'`` unless a
    trial explicitly supplies another value; full mode is incompatible with the
    streaming fitting entry point.

    Temperature is intentionally not calibrated inside this structural search.
    Calibrate the selected model afterward on a separate held-out calibration set.
    """
    if scoring not in {"validation_accuracy", "validation_nll", "validation_brier"}:
        raise ValueError("unsupported scoring")
    keys = list(search_space)
    results: List[PrototypeRBFSearchResult] = []
    best_model: Optional[PCAWhitenedRBFPrototypeClassifier] = None
    best_key: Optional[Tuple[float, ...]] = None
    pca_cache: Dict[Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    validation_cache: Dict[Tuple[Any, ...], List[Tuple[torch.Tensor, List[Hashable]]]] = {}

    for values in itertools.product(*(search_space[k] for k in keys)):
        params = dict(zip(keys, values))
        config = dict(base_config)
        config.update(params)
        config.setdefault("kmeans_fit_mode", "mini_batch")
        if config["kmeans_fit_mode"] != "mini_batch":
            raise ValueError(
                "DataLoader search requires kmeans_fit_mode='mini_batch'"
            )
        model = PCAWhitenedRBFPrototypeClassifier(**config)
        pca_key = (
            model.pca_dim, model.feature_dim, str(model.device), model.dtype, model.eps
        )
        if pca_key not in pca_cache:
            model._fit_pca_from_dataloader(train_loader_factory())
            pca_cache[pca_key] = (
                model.global_mean.detach().clone(),
                model.pca_components.detach().clone(),
                model.pca_eigenvalues.detach().clone(),
            )
        else:
            model.global_mean, model.pca_components, model.pca_eigenvalues = (
                value.detach().clone().to(model.device) for value in pca_cache[pca_key]
            )
            model.pca_fitted = True
        model.fit_dataloader(train_loader_factory(), fit_pca=False)

        if pca_key not in validation_cache:
            validation_cache[pca_key] = [
                (model.transform(x).detach(), list(labels))
                for x, labels in model._iterate_loader(validation_loader_factory())
            ]

        correct = 0
        total = 0
        nll_sum = 0.0
        brier_sum = 0.0
        for Z_batch, labels in validation_cache[pca_key]:
            probabilities = F.softmax(model._decision_function_transformed(Z_batch), dim=1)
            y = model._encode_labels(labels)
            correct += int((probabilities.argmax(1) == y).sum())
            total += y.numel()
            nll_sum += float(
                F.nll_loss(
                    probabilities.clamp_min(model.eps).log(), y, reduction="sum"
                )
            )
            targets = F.one_hot(y, len(model.class_ids)).to(model.dtype)
            brier_sum += float((probabilities - targets).square().sum(1).sum())
        if total == 0:
            raise ValueError("validation loader yielded no samples")
        acc = correct / total
        nll = nll_sum / total
        brier = brier_sum / total
        num_prototypes = sum(v.shape[0] for v in model.prototypes.values())
        result = PrototypeRBFSearchResult(params, acc, nll, brier, num_prototypes)
        results.append(result)
        key = (
            (-acc, nll, brier, num_prototypes)
            if scoring == "validation_accuracy"
            else (nll, num_prototypes)
            if scoring == "validation_nll"
            else (brier, num_prototypes)
        )
        if best_key is None or key < best_key:
            best_key, best_model = key, model
        if verbose:
            print(params, acc, nll, brier, num_prototypes)

    if scoring == "validation_accuracy":
        results.sort(key=lambda r: (-r.validation_accuracy, r.validation_nll, r.validation_brier, r.num_prototypes))
    elif scoring == "validation_nll":
        results.sort(key=lambda r: (r.validation_nll, r.num_prototypes))
    else:
        results.sort(key=lambda r: (r.validation_brier, r.num_prototypes))
    if best_model is None:
        raise RuntimeError("no prototype-RBF DataLoader search trials")
    return best_model, results


@torch.inference_mode()
def search_bayes_filter_hyperparameters(
    classifier: ProbabilisticClassifier,
    filter_inputs: Any,
    true_labels: Sequence[Hashable],
    manual_prior: torch.Tensor | Sequence[float] | Mapping[Hashable, float],
    *,
    sequence_ids: Optional[Sequence[Any]] = None,
    observation_calibration_inputs: Optional[Any] = None,
    observation_calibration_labels: Optional[Sequence[Hashable]] = None,
    filter_scores: Optional[torch.Tensor | Sequence] = None,
    observation_calibration_scores: Optional[torch.Tensor | Sequence] = None,
    score_batch_size: int = 512,
    score_cache_device: str | torch.device = "cpu",
    temperatures: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    stay_probabilities: Sequence[float] = (0.90, 0.94, 0.97),
    transition_alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    transition_pseudocount: float = 1.0,
    transition_training_labels: Optional[Sequence[Hashable] | torch.Tensor] = None,
    transition_training_sequence_ids: Optional[Sequence[Any]] = None,
    transition_cache: Optional[HybridTransitionCache] = None,
    evidence_powers: Sequence[float] = (0.50, 0.75, 1.0),
    min_evidence_powers: Sequence[float] = (0.10, 0.25),
    confidence_gammas: Sequence[float] = (1.0, 2.0),
    observation_modes: Sequence[str] = ("soft",),
    observation_pseudocounts: Sequence[float] = (0.5,),
    transition_matrix: Optional[torch.Tensor | Sequence] = None,
    scoring: str = "balanced_accuracy",
    transition_delay_weight: float = 0.0,
    false_transition_weight: float = 0.0,
    probability_floor: float = 1e-8,
    device: str | torch.device = "cpu",
) -> List[Dict[str, Any]]:
    """Tune a Bayes filter around an already-trained classifier.

    The trained classifier weights remain fixed. Classifier inference is performed
    once in bounded batches and only compact ``[N,C]`` scores are cached (on CPU by
    default). Callers that need to release large source tensors before the search
    can precompute them with :func:`collect_classifier_scores_batched` and supply
    ``filter_scores``/``observation_calibration_scores``. Only one temperature's
    ``[N,C]`` probability matrices exist at a time.

    If separate observation-calibration data are supplied, an observation matrix
    is estimated independently for every candidate temperature. Otherwise the
    filter-validation data are reused, which can yield optimistic scores.

    The empirical transition matrix is estimated once from
    ``transition_training_labels`` and combined with each persistent matrix using
    ``transition_alpha``. If separate transition data are omitted, the ordered
    filter-validation truth is reused and a warning is emitted because transition
    selection can then be optimistic. A :class:`HybridTransitionCache` can be
    shared by searches for multiple classifiers with the same class order.

    ``transition_matrix`` is retained only for old callers. Supplying it bypasses
    hybrid generation and emits a deprecation warning.

    ``scoring`` selects accuracy, balanced_accuracy, or macro_f1. Optional penalties
    can favor lower transition delay and fewer false state changes:

        objective = score - delay_weight*delay - false_weight*false_rate.
    """
    if scoring not in {"accuracy", "balanced_accuracy", "macro_f1"}:
        raise ValueError("unsupported scoring")
    labels = list(classifier.class_ids)
    true_labels = classifier._normalize_labels(true_labels)
    if filter_scores is None and filter_inputs is None:
        raise ValueError("provide filter_inputs or precomputed filter_scores")
    scores_filter = (
        collect_classifier_scores_batched(
            classifier, filter_inputs, batch_size=score_batch_size,
            cache_device=score_cache_device,
        )
        if filter_scores is None
        else torch.as_tensor(filter_scores).detach().to(score_cache_device)
    )
    if scores_filter.shape[0] != len(true_labels):
        raise ValueError("filter input and label counts differ")
    if observation_calibration_inputs is not None or observation_calibration_scores is not None:
        if observation_calibration_labels is None:
            raise ValueError("observation_calibration_labels are required")
        observation_calibration_labels = classifier._normalize_labels(observation_calibration_labels)
        if observation_calibration_scores is None and observation_calibration_inputs is None:
            raise ValueError("provide observation calibration inputs or scores")
        scores_cal = (
            collect_classifier_scores_batched(
                classifier, observation_calibration_inputs, batch_size=score_batch_size,
                cache_device=score_cache_device,
            )
            if observation_calibration_scores is None
            else torch.as_tensor(observation_calibration_scores).detach().to(score_cache_device)
        )
    else:
        scores_cal = scores_filter
        observation_calibration_labels = true_labels
    if scores_cal.shape[0] != len(observation_calibration_labels):
        raise ValueError("observation-calibration score and label counts differ")

    if transition_matrix is not None:
        warnings.warn(
            "transition_matrix is a legacy fixed-matrix path; use transition training "
            "labels and transition_alphas for hybrid search",
            DeprecationWarning,
            stacklevel=2,
        )
        fixed_matrix = _validate_row_stochastic_matrix(
            transition_matrix, len(labels), "transition_matrix", torch.device(device), 1e-8
        )
        transition_candidates = [{
            "stay_probability": None,
            "transition_alpha": None,
            "transition_matrix": fixed_matrix,
            "transition_matrix_cpu": fixed_matrix.cpu(),
            "transition_source": "fixed",
        }]
    else:
        if transition_cache is None:
            if transition_training_labels is None:
                warnings.warn(
                    "Estimating empirical transitions from filter-validation labels; "
                    "transition selection may be optimistic. Supply separate "
                    "transition_training_labels/sequence_ids when available.",
                    UserWarning,
                    stacklevel=2,
                )
                transition_training_labels = true_labels
                transition_training_sequence_ids = sequence_ids
            transition_cache = build_hybrid_transition_cache(
                labels, transition_training_labels,
                transition_training_sequence_ids=transition_training_sequence_ids,
                stay_probabilities=stay_probabilities,
                transition_alphas=transition_alphas,
                transition_pseudocount=transition_pseudocount,
                device=device,
            )
        elif list(transition_cache.labels) != labels:
            raise ValueError("transition_cache label ordering differs from classifier.class_ids")
        transition_candidates = [
            {
                **candidate,
                "transition_matrix": candidate["transition_matrix"].to(device),
                "transition_matrix_cpu": candidate["transition_matrix"].cpu(),
                "transition_source": "hybrid",
            }
            for candidate in transition_cache.unique_candidates
        ]

    results: List[Dict[str, Any]] = []
    observation_cache: Dict[Tuple[float, str, float], torch.Tensor] = {}
    calibration_index = {label: i for i, label in enumerate(labels)}
    encoded_calibration_labels = torch.tensor(
        [calibration_index[label] for label in observation_calibration_labels],
        dtype=torch.long,
        device=device,
    )
    evidence_candidates = [
        (max_power, min_power, gamma)
        for max_power, min_power, gamma in itertools.product(
            evidence_powers, min_evidence_powers, confidence_gammas
        )
        if min_power <= max_power
    ]
    for temperature in temperatures:
        if temperature <= 0:
            continue
        temperature = float(temperature)
        # Keep only the current temperature's probabilities. This preserves reuse
        # across its trials without retaining one [N,C] tensor per temperature.
        probs_filter = F.softmax(scores_filter / temperature, dim=1)
        probs_filter = probs_filter.clamp_min(probability_floor)
        probs_filter = probs_filter / probs_filter.sum(1, keepdim=True)
        probs_cal = (
            probs_filter
            if scores_cal is scores_filter
            else F.softmax(scores_cal / temperature, dim=1)
        )

        for mode, pseudocount in itertools.product(observation_modes, observation_pseudocounts):
            observation_key = (temperature, mode, float(pseudocount))
            if observation_key not in observation_cache:
                observation_cache[observation_key] = estimate_observation_matrix_from_probabilities(
                    probs_cal, observation_calibration_labels, labels, mode=mode,
                    pseudocount=pseudocount, device=device,
                    encoded_true_labels=encoded_calibration_labels,
                )
            observation = observation_cache[observation_key]
            for transition_candidate in transition_candidates:
                transition = transition_candidate["transition_matrix"]
                for max_power, min_power, gamma in evidence_candidates:
                    filt = BayesianTerrainFilter(
                        labels,
                        manual_prior,
                        transition,
                        observation,
                        evidence_power=max_power,
                        adaptive_evidence=True,
                        min_evidence_power=min_power,
                        confidence_gamma=gamma,
                        transition_alpha=transition_candidate["transition_alpha"],
                        stay_probability=transition_candidate["stay_probability"],
                        transition_source=transition_candidate["transition_source"],
                        device=device,
                    )
                    predictions, _, _ = run_filter_sequences(
                        filt, probs_filter, sequence_ids=sequence_ids,
                        return_traces=False,
                    )
                    metrics = evaluate_predictions(
                        true_labels, predictions, labels, sequence_ids=sequence_ids
                    )
                    base_score = getattr(metrics, scoring)
                    delay = 0.0 if math.isnan(metrics.mean_transition_delay) else metrics.mean_transition_delay
                    false_rate = 0.0 if math.isnan(metrics.false_transition_rate) else metrics.false_transition_rate
                    objective = base_score - transition_delay_weight * delay - false_transition_weight * false_rate
                    results.append({
                        "temperature": float(temperature),
                        "stay_probability": transition_candidate["stay_probability"],
                        "transition_alpha": transition_candidate["transition_alpha"],
                        "transition_source": transition_candidate["transition_source"],
                        "transition_matrix": transition_candidate["transition_matrix_cpu"],
                        "evidence_power": float(max_power),
                        "min_evidence_power": float(min_power),
                        "confidence_gamma": float(gamma),
                        "observation_mode": mode,
                        "observation_pseudocount": float(pseudocount),
                        "objective": float(objective),
                        "observation_matrix": observation.cpu(),
                        **metrics.as_dict(),
                    })
        del probs_filter, probs_cal
    return sorted(results, key=lambda r: r["objective"], reverse=True)


def build_filter_from_search_result(
    classifier: ProbabilisticClassifier,
    result: Mapping[str, Any],
    manual_prior: torch.Tensor | Sequence[float] | Mapping[Hashable, float],
    *,
    device: str | torch.device = "cpu",
) -> Tuple[BayesianTerrainFilter, float]:
    """Construct the selected filter using its stored transition matrix directly."""
    labels = list(classifier.class_ids)
    if "transition_matrix" in result:
        transition = result["transition_matrix"]
    else:
        # Compatibility with search results created before hybrid transitions.
        transition = make_persistent_transition_matrix(
            labels, result["stay_probability"], device=device
        )
    filt = BayesianTerrainFilter(
        labels,
        manual_prior,
        transition,
        result["observation_matrix"],
        evidence_power=result["evidence_power"],
        min_evidence_power=result["min_evidence_power"],
        confidence_gamma=result["confidence_gamma"],
        adaptive_evidence=True,
        transition_alpha=result.get("transition_alpha", 0.0),
        stay_probability=result.get("stay_probability"),
        transition_source=result.get("transition_source", "persistent_legacy"),
        device=device,
    )
    return filt, float(result["temperature"])


# =============================================================================
# Internal helpers
# =============================================================================


def _normalize_vector(
    values: torch.Tensor | Sequence[float],
    expected_size: int,
    device: torch.device,
    eps: float,
    name: str,
) -> torch.Tensor:
    vector = torch.as_tensor(values, dtype=torch.float32, device=device).flatten()
    if vector.numel() != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    if not torch.isfinite(vector).all() or (vector < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    vector = vector.clamp_min(eps)
    return vector / vector.sum().clamp_min(eps)


def _validate_row_stochastic_matrix(
    matrix: torch.Tensor | Sequence,
    expected_size: int,
    name: str,
    device: torch.device,
    eps: float,
) -> torch.Tensor:
    tensor = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    if tensor.shape != (expected_size, expected_size):
        raise ValueError(f"{name} must have shape [{expected_size},{expected_size}]")
    if not torch.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    tensor = tensor.clamp_min(eps)
    return tensor / tensor.sum(1, keepdim=True).clamp_min(eps)


def _mean_transition_delay(
    truth: Sequence[Hashable],
    prediction: Sequence[Hashable],
    sequence_ids: Optional[Sequence[Any]],
) -> float:
    if len(truth) < 2:
        return float("nan")
    ids = list(sequence_ids) if sequence_ids is not None else [0] * len(truth)
    delays = []
    for t in range(1, len(truth)):
        if ids[t] != ids[t - 1] or truth[t] == truth[t - 1]:
            continue
        target = truth[t]
        end = t
        while end < len(truth) and ids[end] == ids[t] and prediction[end] != target:
            end += 1
        if end < len(truth) and ids[end] == ids[t]:
            delays.append(end - t)
    return sum(delays) / len(delays) if delays else float("nan")


def _false_transition_rate(
    truth: Sequence[Hashable],
    prediction: Sequence[Hashable],
    sequence_ids: Optional[Sequence[Any]],
) -> float:
    if len(truth) < 2:
        return float("nan")
    ids = list(sequence_ids) if sequence_ids is not None else [0] * len(truth)
    false_changes, opportunities = 0, 0
    for t in range(1, len(truth)):
        if ids[t] != ids[t - 1]:
            continue
        opportunities += 1
        if prediction[t] != prediction[t - 1] and truth[t] == truth[t - 1]:
            false_changes += 1
    return false_changes / max(opportunities, 1)


__all__ = [
    "ProbabilisticClassifier",
    "FeatureStandardizer",
    "RBFSVM",
    "PCAWhitenedRBFSVM",
    "PCAWhitenedRBFPrototypeClassifier",
    "PCAWhitenedPrototypeRBFNetwork",
    "NeuralClassifierAdapter",
    "fit_nn",
    "ClassifierPrediction",
    "SVMSearchResult",
    "PrototypeRBFSearchResult",
    "HybridTransitionCache",
    "BayesianTerrainFilter",
    "BayesianFilteredClassifier",
    "BayesianFilterStep",
    "FilterEvaluation",
    "make_manual_prior",
    "make_persistent_transition_matrix",
    "estimate_transition_matrix_from_sequences",
    "build_hybrid_transition_cache",
    "estimate_observation_matrix_from_probabilities",
    "collect_classifier_probabilities",
    "collect_classifier_scores_batched",
    "run_filter_sequences",
    "evaluate_predictions",
    "search_rbf_svm_hyperparameters",
    "search_prototype_rbf_hyperparameters",
    "search_prototype_rbf_hyperparameters_dataloader",
    "search_bayes_filter_hyperparameters",
    "build_filter_from_search_result",
]
