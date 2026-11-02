"""The decision to answer or decline, kept apart from the model that produced the scores.

NOT_ENOUGH_INFO and abstention are different outcomes. The first is a finding: related
abstracts were read and they do not bear on the claim. The second is a refusal: the
system cannot vouch for any answer, and it says which check failed.

The confidence compared with the threshold is an estimate of P(this verdict is correct),
not the top class probability. On held-out splits the top probability alone ranked
answers poorly, largely because REFUTES predictions are right much less often than
SUPPORTS at the same probability. A small logistic model over the probability and the
predicted class fixes that, and it cannot change which label is predicted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

from claimlens.data import LABELS, NEI, REFUTES, SUPPORTS


class Reason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    WEAK_RETRIEVAL = "weak_retrieval"


def confidence_features(probabilities: np.ndarray) -> np.ndarray:
    """Log-odds of the top probability, plus indicators for the predicted class."""
    top = np.clip(probabilities.max(axis=1), 1e-6, 1.0 - 1e-6)
    predicted = probabilities.argmax(axis=1)
    return np.column_stack(
        [
            np.log(top / (1.0 - top)),
            predicted == LABELS.index(REFUTES),
            predicted == LABELS.index(NEI),
        ]
    ).astype(np.float64)


@dataclass(frozen=True)
class AbstentionPolicy:
    """Thresholds, all chosen on the calibration split. None disables a check."""

    min_confidence: float
    min_retrieval_score: float | None = None
    conflict_probability: float | None = None
    # Weights over confidence_features, intercept last. None uses the top probability.
    confidence_weights: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """P(correct) for each row of calibrated class probabilities."""
        if self.confidence_weights is None:
            return probabilities.max(axis=-1)
        features = confidence_features(np.atleast_2d(probabilities))
        design = np.column_stack([features, np.ones(len(features))])
        estimate = 1.0 / (1.0 + np.exp(-design @ np.array(self.confidence_weights)))
        return estimate.reshape(probabilities.shape[:-1])


@dataclass(frozen=True)
class Abstention:
    reason: Reason
    message: str


@dataclass(frozen=True)
class Decision:
    best_guess: str
    confidence: float
    probabilities: dict[str, float]
    abstentions: tuple[Abstention, ...]

    @property
    def abstained(self) -> bool:
        return bool(self.abstentions)

    @property
    def verdict(self) -> str | None:
        return None if self.abstained else self.best_guess


def has_conflict(abstract_probabilities: np.ndarray, threshold: float) -> bool:
    """True when one abstract confidently supports and another confidently refutes."""
    if abstract_probabilities.size == 0:
        return False
    support = LABELS.index(SUPPORTS)
    refute = LABELS.index(REFUTES)
    return bool(
        abstract_probabilities[:, support].max() >= threshold
        and abstract_probabilities[:, refute].max() >= threshold
    )


def decide(
    probabilities: Sequence[float],
    *,
    top_retrieval_score: float | None,
    abstract_probabilities: np.ndarray,
    policy: AbstentionPolicy,
) -> Decision:
    """Apply every check and report all that fail, not only the first."""
    calibrated = np.asarray(probabilities, dtype=np.float64)
    label = LABELS[int(calibrated.argmax())]
    confidence = float(policy.confidence(calibrated))

    reasons: list[Abstention] = []
    if policy.min_retrieval_score is not None and (
        top_retrieval_score is None or top_retrieval_score < policy.min_retrieval_score
    ):
        found = "no abstract" if top_retrieval_score is None else f"{top_retrieval_score:.2f}"
        reasons.append(
            Abstention(
                Reason.WEAK_RETRIEVAL,
                f"The best abstract scored {found} with the reranker, below the required "
                f"{policy.min_retrieval_score:.2f}, so the corpus may simply not cover this claim.",
            )
        )
    if policy.conflict_probability is not None and has_conflict(
        abstract_probabilities, policy.conflict_probability
    ):
        reasons.append(
            Abstention(
                Reason.CONFLICTING_EVIDENCE,
                f"Separate abstracts support and refute the claim, each with probability at "
                f"least {policy.conflict_probability:.2f}.",
            )
        )
    if confidence < policy.min_confidence:
        reasons.append(
            Abstention(
                Reason.LOW_CONFIDENCE,
                f"The leading verdict has an estimated {confidence:.2f} chance of being "
                f"correct, below the required {policy.min_confidence:.2f}.",
            )
        )
    return Decision(
        best_guess=label,
        confidence=confidence,
        probabilities={name: float(p) for name, p in zip(LABELS, calibrated, strict=True)},
        abstentions=tuple(reasons),
    )


def threshold_for_precision(
    confidences: np.ndarray,
    correct: np.ndarray,
    *,
    target: float,
    min_answered: int = 1,
) -> float | None:
    """Threshold giving the largest coverage whose selective accuracy meets target.

    ``min_answered`` keeps a lucky handful of top claims from counting as evidence that
    the target is reachable. None means no threshold qualifies.
    """
    order = np.argsort(-confidences, kind="stable")
    ranked = confidences[order]
    hits = np.cumsum(correct[order].astype(np.float64))
    precision = hits / np.arange(1, len(order) + 1)
    for i in range(len(order) - 1, min_answered - 2, -1):
        tied_with_next = i + 1 < len(order) and ranked[i + 1] == ranked[i]
        if not tied_with_next and precision[i] >= target:
            return float(ranked[i])
    return None
