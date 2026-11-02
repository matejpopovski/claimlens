"""HTTP layer. Everything here is a translation between JSON and the pipeline."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from importlib import resources
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from claimlens.abstain import AbstentionPolicy
from claimlens.data import LABELS
from claimlens.pipeline import ClaimVerifier, EvidenceItem, Verification

ABSTAINED = "ABSTAINED"


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=3, max_length=500)


class AbstentionOut(BaseModel):
    reason: str
    message: str


class SentenceOut(BaseModel):
    index: int
    text: str
    evidence_score: float
    rationale_rank: int | None


class RetrievalTrail(BaseModel):
    rerank_score: float
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None


class EvidenceOut(BaseModel):
    rank: int
    doc_id: int
    title: str
    verdict: str
    probabilities: dict[str, float]
    retrieval: RetrievalTrail
    sentences: list[SentenceOut]


class VerifyResponse(BaseModel):
    claim: str
    outcome: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "ABSTAINED"]
    best_guess: str
    confidence: float
    probabilities: dict[str, float]
    abstentions: list[AbstentionOut]
    evidence: list[EvidenceOut]
    seconds: float


class HealthResponse(BaseModel):
    status: str
    corpus_size: int
    weights: str
    weights_note: str | None
    temperature: float
    top_k: int
    policy: dict[str, object]


class Example(BaseModel):
    claim: str
    source: str


def _evidence(rank: int, item: EvidenceItem) -> EvidenceOut:
    judgement = item.judgement
    rationale_rank = {index: r for r, index in enumerate(judgement.rationale, start=1)}
    sentences = [
        SentenceOut(
            index=index,
            text=text,
            evidence_score=judgement.sentence_evidence[index],
            rationale_rank=rationale_rank.get(index),
        )
        for index, text in enumerate(item.abstract.sentences)
    ]
    return EvidenceOut(
        rank=rank,
        doc_id=item.abstract.doc_id,
        title=item.abstract.title,
        verdict=judgement.label,
        probabilities=dict(zip(LABELS, judgement.probabilities, strict=True)),
        retrieval=RetrievalTrail(
            rerank_score=item.hit.rerank_score,
            rrf_score=item.hit.rrf_score,
            bm25_rank=item.hit.bm25_rank,
            dense_rank=item.hit.dense_rank,
        ),
        sentences=sentences,
    )


def to_response(result: Verification) -> VerifyResponse:
    decision = result.decision
    return VerifyResponse(
        claim=result.claim,
        outcome=decision.verdict if decision.verdict is not None else ABSTAINED,
        best_guess=decision.best_guess,
        confidence=decision.confidence,
        probabilities=decision.probabilities,
        abstentions=[
            AbstentionOut(reason=a.reason.value, message=a.message) for a in decision.abstentions
        ],
        evidence=[_evidence(rank, item) for rank, item in enumerate(result.evidence, start=1)],
        seconds=result.seconds,
    )


def create_app(
    verifier: ClaimVerifier,
    *,
    weights: str,
    weights_note: str | None = None,
    examples: Sequence[Example] = (),
) -> FastAPI:
    app = FastAPI(title="claimlens", version="0.1.0")
    static = resources.files("claimlens.service") / "static"
    # PyTorch modules are not safe to share across concurrent forward passes on one
    # device, and FastAPI runs sync handlers in a thread pool.
    inference_lock = threading.Lock()

    @app.post("/api/verify")
    def verify(request: VerifyRequest) -> VerifyResponse:
        with inference_lock:
            return to_response(verifier.verify(request.claim.strip()))

    @app.get("/api/health")
    def health() -> HealthResponse:
        policy: AbstentionPolicy = verifier.policy
        return HealthResponse(
            status="ok",
            corpus_size=len(verifier.retriever.corpus),
            weights=weights,
            weights_note=weights_note,
            temperature=verifier.verdict_config.temperature,
            top_k=verifier.verdict_config.top_k,
            policy=policy.to_dict(),
        )

    @app.get("/api/examples")
    def list_examples() -> list[Example]:
        return list(examples)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(static / "index.html"))

    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    return app
