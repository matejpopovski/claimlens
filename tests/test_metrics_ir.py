import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from claimlens.metrics_ir import (
    evaluate_run,
    micro_recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_ndcg_hand_computed():
    qrels = {"a": 1, "b": 1}
    ranking = ["x", "a", "b"]
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranking, qrels, 10) == pytest.approx(expected)


def test_ndcg_is_one_for_a_perfect_ranking_and_zero_for_a_miss():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, 10) == 1.0
    assert ndcg_at_k(["x"], {"a": 1}, 10) == 0.0


def test_ndcg_truncates_the_ideal_list_at_k():
    assert ndcg_at_k(["a"], {"a": 1, "b": 1, "c": 1}, 1) == 1.0


def test_ndcg_agrees_with_sklearn_on_random_rankings():
    rng = np.random.default_rng(0)
    for _ in range(20):
        relevance = (rng.random(30) < 0.2).astype(int)
        if not relevance.any():
            continue
        scores = rng.random(30)
        ranking = list(np.argsort(-scores))
        qrels = {i: int(r) for i, r in enumerate(relevance) if r}
        reference = ndcg_score([relevance], [scores], k=10, ignore_ties=True)
        assert ndcg_at_k(ranking, qrels, 10) == pytest.approx(reference)


def test_recall_and_mrr():
    qrels = {"a": 1, "b": 1}
    assert recall_at_k(["a", "x"], qrels, 2) == 0.5
    assert recall_at_k(["x", "a"], qrels, 1) == 0.0
    assert mrr_at_k(["x", "y", "b"], qrels, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x"], qrels, 10) == 0.0


def test_evaluate_run_macro_averages_and_counts_missing_queries_as_zero():
    result = evaluate_run({1: ["a"]}, {1: {"a": 1}, 2: {"b": 1}}, ks=(10,))
    assert result["recall@10"] == 0.5
    assert result["ndcg@10"] == 0.5
    assert result["mrr@10"] == 0.5


def test_micro_recall_pools_documents_across_queries():
    run = {1: ["a", "b"], 2: ["z"]}
    relevant = {1: ["a", "b", "c"], 2: ["d"]}
    assert micro_recall_at_k(run, relevant, 2) == pytest.approx(2 / 4)
