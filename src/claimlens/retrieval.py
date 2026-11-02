"""Hybrid retrieval: BM25 and dense lists, fused by rank, reranked by a cross-encoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from claimlens.bm25 import BM25Index
from claimlens.data import Abstract
from claimlens.encoders import CrossEncoder, DenseIndex, TextEncoder
from claimlens.fusion import reciprocal_rank_fusion
from claimlens.text import tokenize


@dataclass(frozen=True)
class RetrievalConfig:
    k1: float = 1.2
    b: float = 0.75
    stem: bool = True
    remove_stopwords: bool = True
    encoder: str = "bge-small"
    rrf_k: int = 60
    dense_weight: float = 1.0
    fusion_depth: int = 100
    rerank_depth: int = 50
    reranker: str = "ms-marco-minilm"
    # None keeps the reranker's order. A value fuses that order with the first-stage
    # order, which guards against a reranker that discards a good first stage.
    final_fusion_k: int | None = None
    final_fusion_weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Hit:
    """One retrieved abstract, carrying the trail of how it got there."""

    doc_id: int
    rerank_score: float
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None


def best_rerank_score(hits: Sequence[Hit]) -> float | None:
    """The strongest reranker score among hits.

    Final fusion means the first hit need not hold the highest score.
    """
    return max((h.rerank_score for h in hits), default=None)


def build_bm25(abstracts: Sequence[Abstract], config: RetrievalConfig) -> BM25Index:
    tokenized = [
        tokenize(a.full_text, stem=config.stem, remove_stopwords=config.remove_stopwords)
        for a in abstracts
    ]
    return BM25Index(tokenized, k1=config.k1, b=config.b)


class HybridRetriever:
    def __init__(
        self,
        corpus: Mapping[int, Abstract],
        config: RetrievalConfig,
        *,
        encoder: TextEncoder,
        embeddings: np.ndarray,
        reranker: CrossEncoder,
    ) -> None:
        self.corpus = corpus
        self.config = config
        self.doc_ids = list(corpus)
        abstracts = [corpus[d] for d in self.doc_ids]
        self.bm25 = build_bm25(abstracts, config)
        self.encoder = encoder
        self.dense = DenseIndex(embeddings)
        self.reranker = reranker

    def bm25_ranking(self, claim: str) -> list[int]:
        tokens = tokenize(
            claim, stem=self.config.stem, remove_stopwords=self.config.remove_stopwords
        )
        hits = self.bm25.top_k(tokens, self.config.fusion_depth)
        return [self.doc_ids[i] for i, _ in hits]

    def dense_ranking(self, claim: str) -> list[int]:
        query = self.encoder.encode([claim], is_query=True)[0]
        hits = self.dense.top_k(query, self.config.fusion_depth)
        return [self.doc_ids[i] for i, _ in hits]

    def search(self, claim: str, top_k: int) -> list[Hit]:
        lexical = self.bm25_ranking(claim)
        semantic = self.dense_ranking(claim)
        fused = reciprocal_rank_fusion(
            [lexical, semantic],
            k=self.config.rrf_k,
            weights=[1.0, self.config.dense_weight],
        )
        pool = fused[: self.config.rerank_depth]
        if not pool:
            return []
        passages = [self.corpus[doc_id].full_text for doc_id, _ in pool]
        scores = self.reranker.score(claim, passages)
        lexical_rank = {d: r for r, d in enumerate(lexical, start=1)}
        semantic_rank = {d: r for r, d in enumerate(semantic, start=1)}
        hits = [
            Hit(doc_id, float(score), rrf, lexical_rank.get(doc_id), semantic_rank.get(doc_id))
            for (doc_id, rrf), score in zip(pool, scores, strict=True)
        ]
        hits.sort(key=lambda h: (-h.rerank_score, -h.rrf_score, h.doc_id))
        if self.config.final_fusion_k is not None:
            order = reciprocal_rank_fusion(
                [[h.doc_id for h in hits], [doc_id for doc_id, _ in pool]],
                k=self.config.final_fusion_k,
                weights=[1.0, self.config.final_fusion_weight],
            )
            by_id = {h.doc_id: h for h in hits}
            hits = [by_id[doc_id] for doc_id, _ in order]
        return hits[:top_k]
