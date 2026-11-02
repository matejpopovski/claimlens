"""Calibration measures and temperature scaling, implemented on plain arrays."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = softmax(logits)
    picked = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[labels]
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
    tolerance: float = 1e-5,
) -> float:
    """Minimise NLL over T with a golden-section search on log T.

    One parameter fitted on held-out data cannot change the predicted class, so
    calibration cannot quietly alter any accuracy or F1 reported downstream.
    """
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    low, high = np.log(bounds[0]), np.log(bounds[1])

    def objective(log_t: float) -> float:
        return negative_log_likelihood(logits / np.exp(log_t), labels)

    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    value_left, value_right = objective(left), objective(right)
    while high - low > tolerance:
        if value_left < value_right:
            high, right, value_right = right, left, value_left
            left = high - ratio * (high - low)
            value_left = objective(left)
        else:
            low, left, value_left = left, right, value_right
            right = low + ratio * (high - low)
            value_right = objective(right)
    return float(np.exp((low + high) / 2.0))


def fit_logistic(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float = 1.0,
    iterations: int = 50,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """L2-regularised logistic regression by Newton steps; the intercept is not penalised.

    Returns weights with the intercept last. Written here because the service otherwise
    has no use for scikit-learn, and with four parameters the exact solve is short.
    """
    design = np.column_stack([features, np.ones(len(features))])
    penalty = np.eye(design.shape[1]) * l2
    penalty[-1, -1] = 0.0
    weights = np.zeros(design.shape[1])
    for _ in range(iterations):
        predicted = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (predicted - targets) + penalty @ weights
        curvature = predicted * (1.0 - predicted)
        hessian = (design * curvature[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.abs(step).max() < tolerance:
            break
    return weights


def reliability_bins(
    confidences: np.ndarray, correct: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-width confidence bins over (0, 1], empty bins omitted."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lower, upper in pairwise(edges):
        inside = (confidences > lower) & (confidences <= upper)
        if inside.any():
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(inside.sum()),
                    "confidence": float(confidences[inside].mean()),
                    "accuracy": float(correct[inside].mean()),
                }
            )
    return bins


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, *, n_bins: int = 10
) -> float:
    """Top-label ECE: the count-weighted |accuracy - confidence| over the bins."""
    total = len(confidences)
    return sum(
        b["count"] / total * abs(b["accuracy"] - b["confidence"])
        for b in reliability_bins(confidences, correct, n_bins=n_bins)
    )


def precision_coverage_curve(
    confidences: np.ndarray, correct: np.ndarray
) -> list[dict[str, float]]:
    """Answer the most confident fraction of claims and measure accuracy on those.

    Claims with equal confidence enter together, because no threshold can separate
    them. Each point also records the threshold that produces it.
    """
    order = np.argsort(-confidences, kind="stable")
    sorted_confidence = confidences[order]
    hits = np.cumsum(correct[order].astype(np.float64))
    last_of_tie = np.flatnonzero(np.r_[np.diff(sorted_confidence) != 0, True])
    return [
        {
            "threshold": float(sorted_confidence[i]),
            "coverage": float((i + 1) / len(order)),
            "precision": float(hits[i] / (i + 1)),
            "answered": int(i + 1),
        }
        for i in last_of_tie
    ]


def area_under_precision_coverage(curve: Sequence[dict[str, float]]) -> float:
    """Trapezoidal area over coverage, starting from the first answerable point."""
    coverage = np.array([p["coverage"] for p in curve])
    precision = np.array([p["precision"] for p in curve])
    return float(np.sum((coverage[1:] - coverage[:-1]) * (precision[1:] + precision[:-1]) / 2))


def bootstrap_interval(
    statistic,
    n_items: int,
    *,
    n_resamples: int = 2000,
    seed: int = 13,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval, resampling claims with replacement."""
    rng = np.random.default_rng(seed)
    values = np.array([statistic(rng.integers(0, n_items, n_items)) for _ in range(n_resamples)])
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)
