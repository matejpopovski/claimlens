"""The full path from a claim's text to a decision, as the service runs it."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from claimlens.abstain import AbstentionPolicy, Decision, decide
from claimlens.calibration import softmax
from claimlens.data import Abstract
from claimlens.encoders import CrossEncoder, TextEncoder
from claimlens.models import MODELS, local_path
from claimlens.retrieval import Hit, HybridRetriever, RetrievalConfig, best_rerank_score
from claimlens.verdict import (
    AbstractJudgement,
    NLIModel,
    VerdictConfig,
    judge_abstracts,
    pool_claim_logits,
    score_sentences,
)

FINETUNED = "finetuned"


def reranker_path(name: str, runs: Path = Path("runs")) -> Path:
    return runs / "reranker" / "best" if name == FINETUNED else local_path(name)


def load_retrieval_config(configs: Path = Path("configs")) -> RetrievalConfig:
    return RetrievalConfig(**json.loads((configs / "retrieval.json").read_text()))


def load_verdict_config(configs: Path = Path("configs")) -> VerdictConfig:
    return VerdictConfig(**json.loads((configs / "verdict.json").read_text()))


def load_policy(configs: Path = Path("configs")) -> AbstentionPolicy:
    values = json.loads((configs / "abstention.json").read_text())
    if values.get("confidence_weights") is not None:
        values["confidence_weights"] = tuple(values["confidence_weights"])
    return AbstentionPolicy(**values)


def build_retriever(
    corpus: Mapping[int, Abstract],
    config: RetrievalConfig,
    *,
    index_dir: Path = Path("runs/index"),
    runs: Path = Path("runs"),
    device: str | None = None,
    reranker: Path | None = None,
) -> HybridRetriever:
    spec = MODELS[config.encoder]
    encoder = TextEncoder(
        local_path(config.encoder),
        pooling=spec.pooling,
        query_prefix=spec.query_prefix,
        max_length=384,
        device=device,
    )
    return HybridRetriever(
        corpus,
        config,
        encoder=encoder,
        embeddings=np.load(index_dir / f"{config.encoder}.npy"),
        reranker=CrossEncoder.load(reranker or reranker_path(config.reranker, runs), device=device),
    )


@dataclass(frozen=True)
class EvidenceItem:
    hit: Hit
    abstract: Abstract
    judgement: AbstractJudgement


@dataclass(frozen=True)
class Verification:
    claim: str
    decision: Decision
    evidence: tuple[EvidenceItem, ...]
    seconds: float


class ClaimVerifier:
    def __init__(
        self,
        retriever: HybridRetriever,
        selector: CrossEncoder,
        nli: NLIModel,
        *,
        verdict_config: VerdictConfig,
        policy: AbstentionPolicy,
    ) -> None:
        self.retriever = retriever
        self.selector = selector
        self.nli = nli
        self.verdict_config = verdict_config
        self.policy = policy

    def verify(self, claim: str) -> Verification:
        started = time.perf_counter()
        config = self.verdict_config
        hits = self.retriever.search(claim, config.top_k)
        requests = [(claim, self.retriever.corpus[hit.doc_id]) for hit in hits]
        judgements = judge_abstracts(
            self.nli,
            requests,
            score_sentences(self.selector, requests),
            threshold=config.rationale_threshold,
            max_sentences=config.max_rationale_sentences,
            nei_weight=config.nli_nei_weight,
        )
        abstract_probabilities = np.array([j.probabilities for j in judgements]).reshape(-1, 3)
        pooled = pool_claim_logits(abstract_probabilities, nei_bias=config.nei_bias)
        decision = decide(
            softmax(pooled / config.temperature),
            top_retrieval_score=best_rerank_score(hits),
            abstract_probabilities=abstract_probabilities,
            policy=self.policy,
        )
        evidence = tuple(
            EvidenceItem(hit, abstract, judgement)
            for hit, (_, abstract), judgement in zip(hits, requests, judgements, strict=True)
        )
        return Verification(claim, decision, evidence, time.perf_counter() - started)
