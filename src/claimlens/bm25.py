"""BM25 over an in-memory inverted index, written out rather than imported.

Two IDF variants are provided. ``lucene`` is the default and matches what
Elasticsearch computes. ``robertson`` replicates the negative-IDF floor used by
the ``rank_bm25`` package, which exists so that an independent implementation can
check this one term for term (see tests/test_bm25.py).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np

IdfVariant = Literal["lucene", "robertson"]


class BM25Index:
    def __init__(
        self,
        tokenized_docs: Sequence[Sequence[str]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        idf: IdfVariant = "lucene",
        epsilon: float = 0.25,
    ) -> None:
        if not tokenized_docs:
            raise ValueError("cannot index an empty corpus")
        if idf not in ("lucene", "robertson"):
            raise ValueError(f"unknown idf variant: {idf}")
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_docs)
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float64)
        self.avg_doc_len = float(self.doc_len.mean())

        doc_ids: dict[str, list[int]] = {}
        term_freqs: dict[str, list[int]] = {}
        for doc_index, tokens in enumerate(tokenized_docs):
            for term, tf in Counter(tokens).items():
                doc_ids.setdefault(term, []).append(doc_index)
                term_freqs.setdefault(term, []).append(tf)
        self._postings = {
            term: (np.array(ids, dtype=np.int32), np.array(term_freqs[term], dtype=np.float64))
            for term, ids in doc_ids.items()
        }
        self.idf = self._compute_idf(idf, epsilon)
        # The length normaliser is per-document and independent of the query, so
        # computing it once turns scoring into a few vector operations per term.
        self._length_norm = k1 * (1.0 - b + b * self.doc_len / self.avg_doc_len)

    def _compute_idf(self, variant: IdfVariant, epsilon: float) -> dict[str, float]:
        n = self.n_docs
        doc_freqs = {term: len(ids) for term, (ids, _) in self._postings.items()}
        if variant == "lucene":
            return {t: math.log(1.0 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freqs.items()}
        idf = {t: math.log(n - df + 0.5) - math.log(df + 0.5) for t, df in doc_freqs.items()}
        floor = epsilon * (sum(idf.values()) / len(idf))
        return {t: (floor if value < 0 else value) for t, value in idf.items()}

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        """Score every document. A repeated query term contributes once per occurrence."""
        total = np.zeros(self.n_docs, dtype=np.float64)
        for term in query_tokens:
            posting = self._postings.get(term)
            if posting is None:
                continue
            ids, tf = posting
            total[ids] += self.idf[term] * tf * (self.k1 + 1.0) / (tf + self._length_norm[ids])
        return total

    def top_k(self, query_tokens: Sequence[str], k: int) -> list[tuple[int, float]]:
        """Return up to k (doc_index, score) pairs with a positive score, best first.

        Ties break on the lower document index so a run is reproducible.
        """
        scores = self.scores(query_tokens)
        matched = np.flatnonzero(scores > 0)
        order = matched[np.lexsort((matched, -scores[matched]))][:k]
        return [(int(i), float(scores[i])) for i in order]
