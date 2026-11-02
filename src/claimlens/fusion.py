"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion[T: Hashable](
    rankings: Sequence[Sequence[T]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[T, float]]:
    """Fuse ranked lists using only ranks: score(d) = sum_i w_i / (k + rank_i(d)).

    Ranks are 1-based. BM25 scores and cosine similarities live on incomparable
    scales, which is why the fusion ignores scores altogether. Ties break on first
    appearance across the input lists, which keeps the result deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("need one weight per ranking")

    fused: dict[T, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if len(set(ranking)) != len(ranking):
            raise ValueError("a ranking contains a duplicate document")
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])
