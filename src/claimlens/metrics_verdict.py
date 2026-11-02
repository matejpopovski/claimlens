"""Verdict metrics: claim-level macro-F1 and the official SciFact scores.

The abstract-level and sentence-level definitions follow the SciFact evaluator
(allenai/scifact-evaluator) and doc/evaluation.md in allenai/scifact, including its
quirk that abstract-level scoring keeps only the first three predicted sentences
unless the shortest gold rationale is longer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from claimlens.data import LABELS, REFUTES, SUPPORTS, Claim


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def classification_report(
    gold: Sequence[str], predicted: Sequence[str], *, labels: Sequence[str] = LABELS
) -> dict[str, object]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted differ in length")
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "n": tp + fn}
    correct = sum(g == p for g, p in zip(gold, predicted, strict=True))
    return {
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(labels),
        "accuracy": correct / len(gold) if gold else 0.0,
        "n": len(gold),
        "per_class": per_class,
    }


@dataclass(frozen=True)
class AbstractPrediction:
    """A decisive prediction for one abstract, with sentences most confident first."""

    label: str
    sentences: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.label not in (SUPPORTS, REFUTES):
            raise ValueError(f"only decisive labels are scored, got {self.label}")


def _contains_evidence(gold_sets: Iterable[frozenset[int]], predicted: set[int]) -> bool:
    return any(gold <= predicted for gold in gold_sets)


def scifact_scores(
    claims: Sequence[Claim],
    predictions: Mapping[int, Mapping[int, AbstractPrediction]],
    *,
    max_abstract_sentences: int = 3,
) -> dict[str, dict[str, float]]:
    """Official SciFact P/R/F1. ``predictions`` maps claim id to {doc_id: prediction}."""
    counts: Counter[str] = Counter()
    for claim in claims:
        gold_evidence = claim.evidence
        counts["abstract_relevant"] += len(gold_evidence)
        counts["sentence_relevant"] += sum(
            len(r.sentences) for rationales in gold_evidence.values() for r in rationales
        )
        for doc_id, prediction in predictions.get(claim.id, {}).items():
            counts["abstract_retrieved"] += 1
            counts["sentence_retrieved"] += len(prediction.sentences)
            rationales = gold_evidence.get(doc_id)
            if rationales is None:
                continue
            gold_sets = [frozenset(r.sentences) for r in rationales]
            label_ok = prediction.label == rationales[0].label

            if label_ok:
                counts["abstract_label_only"] += 1
                keep = max(max_abstract_sentences, min(len(g) for g in gold_sets))
                if _contains_evidence(gold_sets, set(prediction.sentences[:keep])):
                    counts["abstract_rationalized"] += 1

            predicted = set(prediction.sentences)
            n_correct = sum(
                1 for index in predicted for g in gold_sets if index in g and g <= predicted
            )
            counts["sentence_selection"] += n_correct
            if label_ok:
                counts["sentence_label"] += n_correct

    def score(correct: str, level: str) -> dict[str, float]:
        precision, recall, f1 = f1_from_counts(
            counts[correct],
            counts[f"{level}_retrieved"] - counts[correct],
            counts[f"{level}_relevant"] - counts[correct],
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    return {
        "abstract_label_only": score("abstract_label_only", "abstract"),
        "abstract_label_rationale": score("abstract_rationalized", "abstract"),
        "sentence_selection": score("sentence_selection", "sentence"),
        "sentence_selection_label": score("sentence_label", "sentence"),
    }
