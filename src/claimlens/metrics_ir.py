"""Ranking metrics, with the definitions pytrec_eval uses for BEIR.

Gain is linear in the relevance grade and the discount is log2(rank + 1). SciFact
qrels are binary, so the gain choice does not change any number reported here.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Sequence[Hashable], qrels: Mapping[Hashable, int], k: int) -> float:
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    if not ideal:
        return 0.0
    return dcg([qrels.get(d, 0) for d in ranking[:k]]) / dcg(ideal)


def recall_at_k(ranking: Sequence[Hashable], qrels: Mapping[Hashable, int], k: int) -> float:
    relevant = {d for d, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: Sequence[Hashable], qrels: Mapping[Hashable, int], k: int) -> float:
    for rank, doc in enumerate(ranking[:k], start=1):
        if qrels.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


def evaluate_run(
    run: Mapping[Hashable, Sequence[Hashable]],
    qrels: Mapping[Hashable, Mapping[Hashable, int]],
    *,
    ks: Sequence[int] = (3, 5, 10, 100),
) -> dict[str, float]:
    """Macro-average over the queries in qrels; a query missing from run scores zero."""
    if not qrels:
        raise ValueError("empty qrels")
    totals: dict[str, float] = {}
    for query_id, relevant in qrels.items():
        ranking = run.get(query_id, ())
        for k in ks:
            for name, metric in (("ndcg", ndcg_at_k), ("recall", recall_at_k)):
                key = f"{name}@{k}"
                totals[key] = totals.get(key, 0.0) + metric(ranking, relevant, k)
        totals["mrr@10"] = totals.get("mrr@10", 0.0) + mrr_at_k(ranking, relevant, 10)
    return {name: value / len(qrels) for name, value in sorted(totals.items())}


def micro_recall_at_k(
    run: Mapping[Hashable, Sequence[Hashable]],
    relevant: Mapping[Hashable, Sequence[Hashable]],
    k: int,
) -> float:
    """Pooled recall over all relevant documents, as VerT5erini's Table 6 reports it."""
    hit = total = 0
    for query_id, docs in relevant.items():
        top = set(run.get(query_id, ())[:k])
        hit += sum(d in top for d in docs)
        total += len(docs)
    return hit / total if total else 0.0
