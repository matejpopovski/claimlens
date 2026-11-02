"""Turn a claim and an abstract into a verdict and the sentences behind it.

Two models answer two different questions. A cross-encoder selector asks whether a
sentence is evidence for the claim at all. An NLI model then reads the selected sentences
together and asks which way they point. They combine as

    P(label) = P(evidence) * P(label | rationale) + P(no evidence) * [label is NEI]

An earlier design used one NLI model for both jobs, and given the gold rationale it still
answered NOT_ENOUGH_INFO on about 40% of validation abstracts, because that class had to
absorb every near-miss sentence in the corpus (results/one_model_design). Splitting the
questions lets each model train on the examples that fit it.

The NLI model keeps a NOT_ENOUGH_INFO output, but the selector already answers that
question, so ``nli_nei_weight`` scales how much of that output counts. At zero the NLI
model only chooses a direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimlens.calibration import softmax
from claimlens.data import LABELS, NEI, REFUTES, SUPPORTS, Abstract
from claimlens.devices import pick_device
from claimlens.encoders import CrossEncoder, length_sorted_batches

_NLI_NAMES = {SUPPORTS: "entailment", REFUTES: "contradiction", NEI: "neutral"}


def label_order(id2label: dict[int, str]) -> list[int]:
    """Model output indices, listed in the project's LABELS order."""
    by_name = {name.lower(): index for index, name in id2label.items()}
    return [by_name[_NLI_NAMES[label]] for label in LABELS]


class NLIModel:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        batch_size: int = 32,
        max_length: int = 256,
        device: torch.device,
    ) -> None:
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.order = label_order(model.config.id2label)
        self.batch_size = batch_size
        self.max_length = max_length

    @classmethod
    def load(
        cls, path: Path | str, *, device: str | torch.device | None = None, **options
    ) -> NLIModel:
        resolved = torch.device(device) if device else pick_device()
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(model, AutoTokenizer.from_pretrained(path), device=resolved, **options)

    @torch.inference_mode()
    def logits(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Logits in LABELS order for (premise, hypothesis) pairs."""
        output = np.zeros((len(pairs), len(LABELS)), dtype=np.float64)
        lengths = [len(p) + len(h) for p, h in pairs]
        for batch in length_sorted_batches(lengths, self.batch_size):
            tokens = self.tokenizer(
                [pairs[i][0] for i in batch],
                [pairs[i][1] for i in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**tokens).logits[:, self.order]
            output[batch] = logits.float().cpu().numpy()
        return output


@dataclass(frozen=True)
class VerdictConfig:
    rationale_threshold: float = 0.5
    max_rationale_sentences: int = 3
    top_k: int = 3
    nei_bias: float = 0.0
    nli_nei_weight: float = 1.0
    temperature: float = 1.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AbstractJudgement:
    doc_id: int
    probabilities: tuple[float, float, float]
    rationale: tuple[int, ...]
    sentence_evidence: tuple[float, ...]
    nli_probabilities: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def label(self) -> str:
        return LABELS[int(np.argmax(self.probabilities))]

    @property
    def is_decisive(self) -> bool:
        return self.label != NEI


def format_premise(title: str, sentences: Sequence[str]) -> str:
    """Sentences alone often say "these mice"; the title says which mice."""
    return f"{title} {' '.join(sentences)}".strip()


def select_rationale(
    evidence_scores: np.ndarray, *, threshold: float, max_sentences: int
) -> tuple[int, ...]:
    """Indices above threshold, strongest first, or the best sentence if none qualify.

    The fallback means the NLI model always has something to read, and a low evidence
    score still pulls the verdict to NOT_ENOUGH_INFO. Order matters: the official
    abstract-level score reads only the first three.
    """
    order = np.lexsort((np.arange(len(evidence_scores)), -evidence_scores))
    chosen = [int(i) for i in order if evidence_scores[i] >= threshold][:max_sentences]
    return tuple(chosen) if chosen else (int(order[0]),)


def score_sentences(
    selector: CrossEncoder, requests: Sequence[tuple[str, Abstract]]
) -> list[np.ndarray]:
    """P(evidence) for every sentence of every requested abstract.

    The title stays out of this question: it matches the claim for every sentence of
    a relevant abstract, so it would tell sentences apart by nothing.
    """
    pairs, owners = [], []
    for index, (claim, abstract) in enumerate(requests):
        for sentence in abstract.sentences:
            pairs.append((claim, sentence))
            owners.append(index)
    scores = 1.0 / (1.0 + np.exp(-selector.score_pairs(pairs))) if pairs else np.zeros(0)
    grouped: list[list[float]] = [[] for _ in requests]
    for owner, score in zip(owners, scores, strict=True):
        grouped[owner].append(float(score))
    return [np.array(values) for values in grouped]


def marginalise_evidence(
    nli_probabilities: np.ndarray, evidence: float, *, nei_weight: float = 1.0
) -> np.ndarray:
    nei = LABELS.index(NEI)
    adjusted = np.array(nli_probabilities, dtype=np.float64)
    adjusted[nei] *= nei_weight
    combined = evidence * adjusted / adjusted.sum()
    combined[nei] += 1.0 - evidence
    return combined


def with_nei_weight(judgement: AbstractJudgement, nei_weight: float) -> AbstractJudgement:
    """Recompute the combined probabilities, which needs no further model calls."""
    combined = marginalise_evidence(
        np.array(judgement.nli_probabilities),
        judgement.sentence_evidence[judgement.rationale[0]],
        nei_weight=nei_weight,
    )
    return replace(judgement, probabilities=tuple(float(p) for p in combined))


def judge_abstracts(
    nli: NLIModel,
    requests: Sequence[tuple[str, Abstract]],
    sentence_evidence: Sequence[np.ndarray],
    *,
    threshold: float,
    max_sentences: int,
    nei_weight: float = 1.0,
) -> list[AbstractJudgement]:
    """Read each abstract's selected sentences together and combine with P(evidence)."""
    rationales, pairs = [], []
    for (claim, abstract), evidence in zip(requests, sentence_evidence, strict=True):
        rationale = select_rationale(evidence, threshold=threshold, max_sentences=max_sentences)
        rationales.append(rationale)
        chosen = [abstract.sentences[i] for i in sorted(rationale)]
        pairs.append((format_premise(abstract.title, chosen), claim))
    probabilities = softmax(nli.logits(pairs)) if pairs else np.zeros((0, len(LABELS)))
    judgements = []
    for (_, abstract), evidence, rationale, row in zip(
        requests, sentence_evidence, rationales, probabilities, strict=True
    ):
        strongest = float(evidence[rationale[0]])
        combined = marginalise_evidence(row, strongest, nei_weight=nei_weight)
        judgements.append(
            AbstractJudgement(
                doc_id=abstract.doc_id,
                probabilities=tuple(float(p) for p in combined),
                rationale=rationale,
                sentence_evidence=tuple(float(e) for e in evidence),
                nli_probabilities=tuple(float(p) for p in row),
            )
        )
    return judgements


def pool_claim_logits(abstract_probabilities: np.ndarray, *, nei_bias: float = 0.0) -> np.ndarray:
    """Combine abstract probabilities into claim-level log-scores.

    One abstract with real evidence settles a claim however many others are silent,
    so the decisive classes take the maximum over abstracts and NOT_ENOUGH_INFO takes
    the minimum. That pooling favours decisive answers as abstracts are added, and
    ``nei_bias`` is the single correction for it, tuned on the validation split.
    """
    if abstract_probabilities.size == 0:
        scores = np.array([0.0, 0.0, 1.0])
        return np.log(np.clip(scores, 1e-12, 1.0))
    log_probabilities = np.log(np.clip(abstract_probabilities, 1e-12, 1.0))
    nei = LABELS.index(NEI)
    pooled = log_probabilities.max(axis=0)
    pooled[nei] = log_probabilities[:, nei].min() + nei_bias
    return pooled
