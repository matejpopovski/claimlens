"""Scoring helpers shared by the tuning, calibration and final evaluation scripts.

Scoring sentences once and reusing the result keeps a threshold sweep cheap, so that
step is separated from the choices made on top of it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from claimlens.abstain import AbstentionPolicy, Decision, decide
from claimlens.calibration import softmax
from claimlens.data import LABELS, NEI, Abstract, Claim
from claimlens.encoders import CrossEncoder
from claimlens.metrics_verdict import AbstractPrediction, classification_report
from claimlens.retrieval import Hit, best_rerank_score
from claimlens.text import tokenize
from claimlens.verdict import (
    AbstractJudgement,
    NLIModel,
    judge_abstracts,
    pool_claim_logits,
    score_sentences,
)

JudgementMap = dict[int, list[AbstractJudgement]]


def save_hits(path: Path, hits: Mapping[int, Sequence[Hit]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): [asdict(h) for h in v] for k, v in hits.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_hits(path: Path) -> dict[int, list[Hit]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): [Hit(**h) for h in v] for k, v in payload.items()}


@dataclass(frozen=True)
class SentencePass:
    """Selector output for a set of (claim, abstract) requests."""

    claim_ids: tuple[int, ...]
    requests: tuple[tuple[str, Abstract], ...]
    scores: tuple[np.ndarray, ...]

    def judge(
        self, nli: NLIModel, *, threshold: float, max_sentences: int, nei_weight: float = 1.0
    ) -> JudgementMap:
        judged = judge_abstracts(
            nli,
            self.requests,
            self.scores,
            threshold=threshold,
            max_sentences=max_sentences,
            nei_weight=nei_weight,
        )
        grouped: JudgementMap = {}
        for claim_id, judgement in zip(self.claim_ids, judged, strict=True):
            grouped.setdefault(claim_id, []).append(judgement)
        return grouped


def run_sentence_pass(
    selector: CrossEncoder,
    claims: Sequence[Claim],
    documents: Mapping[int, Sequence[int]],
    corpus: Mapping[int, Abstract],
) -> SentencePass:
    claim_ids, requests = [], []
    for claim in claims:
        for doc_id in documents.get(claim.id, ()):
            claim_ids.append(claim.id)
            requests.append((claim.text, corpus[doc_id]))
    return SentencePass(
        tuple(claim_ids), tuple(requests), tuple(score_sentences(selector, requests))
    )


def claim_logits(
    claims: Sequence[Claim],
    judgements: JudgementMap,
    *,
    top_k: int,
    nei_bias: float,
) -> np.ndarray:
    rows = []
    for claim in claims:
        selected = judgements.get(claim.id, [])[:top_k]
        probabilities = np.array([j.probabilities for j in selected]).reshape(-1, len(LABELS))
        rows.append(pool_claim_logits(probabilities, nei_bias=nei_bias))
    return np.array(rows)


def gold_indices(claims: Sequence[Claim]) -> np.ndarray:
    return np.array([LABELS.index(c.verdict) for c in claims])


def claim_report(claims: Sequence[Claim], logits: np.ndarray) -> dict[str, object]:
    predicted = [LABELS[i] for i in logits.argmax(axis=1)]
    return classification_report([c.verdict for c in claims], predicted)


def abstract_predictions(
    claims: Sequence[Claim],
    judgements: JudgementMap,
    logits: np.ndarray,
    *,
    top_k: int,
) -> dict[int, dict[int, AbstractPrediction]]:
    """What the system asserts per abstract, for the official SciFact scores.

    An abstract counts only when the claim-level verdict is decisive and agrees with
    it, so these scores describe the answer a user would actually receive.
    """
    predictions: dict[int, dict[int, AbstractPrediction]] = {}
    for claim, row in zip(claims, logits, strict=True):
        verdict = LABELS[int(row.argmax())]
        if verdict == NEI:
            continue
        kept = {
            j.doc_id: AbstractPrediction(verdict, j.rationale)
            for j in judgements.get(claim.id, [])[:top_k]
            if j.label == verdict
        }
        if kept:
            predictions[claim.id] = kept
    return predictions


def confidence_and_correct(
    claims: Sequence[Claim], logits: np.ndarray, *, temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = softmax(logits / temperature)
    return probabilities.max(axis=1), probabilities.argmax(axis=1) == gold_indices(claims)


def apply_policy(
    claims: Sequence[Claim],
    judgements: JudgementMap,
    hits: Mapping[int, Sequence[Hit]],
    logits: np.ndarray,
    *,
    top_k: int,
    temperature: float,
    policy: AbstentionPolicy,
) -> list[Decision]:
    probabilities = softmax(logits / temperature)
    decisions = []
    for claim, row in zip(claims, probabilities, strict=True):
        selected = judgements.get(claim.id, [])[:top_k]
        decisions.append(
            decide(
                row,
                top_retrieval_score=best_rerank_score(hits.get(claim.id, ())[:top_k]),
                abstract_probabilities=np.array([j.probabilities for j in selected]).reshape(
                    -1, len(LABELS)
                ),
                policy=policy,
            )
        )
    return decisions


def selective_summary(claims: Sequence[Claim], decisions: Sequence[Decision]) -> dict[str, float]:
    """Coverage and accuracy of the claims actually answered."""
    answered = [
        (c.verdict, d.best_guess) for c, d in zip(claims, decisions, strict=True) if not d.abstained
    ]
    correct = sum(gold == guess for gold, guess in answered)
    decisive = [(gold, guess) for gold, guess in answered if guess != NEI]
    decisive_correct = sum(gold == guess for gold, guess in decisive)
    return {
        "n_claims": len(claims),
        "answered": len(answered),
        "coverage": len(answered) / len(claims) if claims else 0.0,
        "selective_accuracy": correct / len(answered) if answered else 0.0,
        "decisive_answers": len(decisive),
        "decisive_precision": decisive_correct / len(decisive) if decisive else 0.0,
    }


def tfidf_top_sentences(
    claim: str, abstract: Abstract, idf: Mapping[str, float], n: int
) -> list[int]:
    """Indices of the n sentences most similar to the claim, by TF-IDF cosine."""

    def vector(text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        return {t: c * idf.get(t, 0.0) for t, c in counts.items()}

    def cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        dot = sum(v * b.get(t, 0.0) for t, v in a.items())
        norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
        return dot / norm if norm else 0.0

    query = vector(claim)
    scored = [(-cosine(query, vector(s)), i) for i, s in enumerate(abstract.sentences)]
    return [i for _, i in sorted(scored)[:n]]


def sentence_idf(corpus: Mapping[int, Abstract]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    total = 0
    for abstract in corpus.values():
        for sentence in abstract.sentences:
            document_frequency.update(set(tokenize(sentence)))
            total += 1
    return {t: math.log((1 + total) / (1 + df)) + 1.0 for t, df in document_frequency.items()}
