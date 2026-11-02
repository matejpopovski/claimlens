"""BM25 is checked three independent ways: by hand, by brute force, and by rank_bm25."""

import math
import random

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from claimlens.bm25 import BM25Index

DOCS = [
    ["aspirin", "reduces", "fever", "in", "adults"],
    ["aspirin", "aspirin", "thins", "blood"],
    ["vitamin", "c", "cures", "scurvy", "in", "sailors", "at", "sea", "on", "long", "voyages"],
    ["fever", "and", "chills", "are", "symptoms"],
    ["blood", "pressure", "in", "adults"],
]


def test_hand_computed_score():
    index = BM25Index(DOCS, k1=1.5, b=0.75)
    n, df = 5, 2  # "aspirin" appears in documents 0 and 1.
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    average_length = (5 + 4 + 11 + 5 + 4) / 5
    tf, length = 2, 4
    expected = idf * tf * (1.5 + 1) / (tf + 1.5 * (1 - 0.75 + 0.75 * length / average_length))
    assert index.scores(["aspirin"])[1] == pytest.approx(expected)


def brute_force_lucene(docs, query, k1, b):
    """A deliberately naive reference that shares no code with the index."""
    n = len(docs)
    average_length = sum(len(d) for d in docs) / n
    scores = []
    for doc in docs:
        total = 0.0
        for term in query:
            df = sum(term in other for other in docs)
            if df == 0:
                continue
            tf = doc.count(term)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / average_length))
        scores.append(total)
    return scores


def random_corpus(seed, n_docs=60, vocabulary=40):
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocabulary)]
    docs = [rng.choices(words, k=rng.randint(1, 40)) for _ in range(n_docs)]
    query = rng.choices([*words, "unseen"], k=6)
    return docs, query


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_reference(seed):
    docs, query = random_corpus(seed)
    index = BM25Index(docs, k1=0.9, b=0.4)
    assert index.scores(query) == pytest.approx(brute_force_lucene(docs, query, 0.9, 0.4))


@pytest.mark.parametrize("seed", range(5))
def test_matches_rank_bm25_including_negative_idf_floor(seed):
    docs, query = random_corpus(seed)
    reference = BM25Okapi(docs, k1=1.5, b=0.75, epsilon=0.25)
    index = BM25Index(docs, k1=1.5, b=0.75, idf="robertson", epsilon=0.25)
    assert index.scores(query) == pytest.approx(reference.get_scores(query))


def test_rank_bm25_fixture_exercises_the_negative_idf_branch():
    # A term in most documents has negative Robertson IDF, so without this the
    # comparison above could pass while the floor was wrong.
    docs, _ = random_corpus(0)
    reference = BM25Okapi(docs, k1=1.5, b=0.75, epsilon=0.25)
    assert any(
        math.log(len(docs) - sum(w in d for d in docs) + 0.5)
        - math.log(sum(w in d for d in docs) + 0.5)
        < 0
        for w in reference.idf
    )


def test_top_k_orders_by_score_and_breaks_ties_by_index():
    index = BM25Index([["a"], ["b"], ["a"], ["a", "a", "b"]], k1=1.2, b=0.0)
    ranked = [doc for doc, _ in index.top_k(["a"], 10)]
    assert ranked == [3, 0, 2]


def test_top_k_excludes_documents_with_no_matching_term():
    index = BM25Index(DOCS)
    assert index.top_k(["nonexistent"], 5) == []


def test_repeated_query_terms_count_twice():
    index = BM25Index(DOCS)
    once = index.scores(["fever"])
    twice = index.scores(["fever", "fever"])
    np.testing.assert_allclose(twice, 2 * once)


def test_rejects_empty_corpus():
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Index([])
