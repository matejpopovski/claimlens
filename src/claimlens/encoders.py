"""Bi-encoder embeddings and the cross-encoder reranker, on plain transformers.

Using transformers directly keeps tokenisation, truncation and pooling in view,
and saves the sentence-transformers dependency tree in a service that needs neither
its training API nor its model zoo.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from claimlens.devices import pick_device


def length_sorted_batches(lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Group similar lengths so padding, and therefore compute, stays small."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


class TextEncoder:
    def __init__(
        self,
        path: Path | str,
        *,
        pooling: str = "mean",
        query_prefix: str = "",
        max_length: int = 256,
        batch_size: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        if pooling not in ("mean", "cls"):
            raise ValueError(f"unknown pooling: {pooling}")
        self.device = torch.device(device) if device else pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.max_length = max_length
        self.batch_size = batch_size

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        """Return L2-normalised float32 embeddings in input order."""
        if is_query:
            texts = [self.query_prefix + text for text in texts]
        output: np.ndarray | None = None
        for batch in length_sorted_batches([len(t) for t in texts], self.batch_size):
            tokens = self.tokenizer(
                [texts[i] for i in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state
            pooled = self._pool(hidden, tokens["attention_mask"])
            vectors = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
            if output is None:
                output = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
            output[batch] = vectors
        if output is None:
            raise ValueError("no texts to encode")
        return output


class DenseIndex:
    """Exact cosine search. 5,183 abstracts do not justify an approximate index."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings

    def top_k(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        similarities = self.embeddings @ query
        order = np.lexsort((np.arange(len(similarities)), -similarities))[:k]
        return [(int(i), float(similarities[i])) for i in order]


class CrossEncoder:
    """Scores (query, passage) pairs with a single-logit sequence classifier."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_length: int = 384,
        batch_size: int = 32,
        device: torch.device,
    ) -> None:
        if model.config.num_labels != 1:
            raise ValueError("a reranker must have exactly one output logit")
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    @classmethod
    def load(
        cls, path: Path | str, *, device: str | torch.device | None = None, **options
    ) -> CrossEncoder:
        resolved = torch.device(device) if device else pick_device()
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return cls(model, AutoTokenizer.from_pretrained(path), device=resolved, **options)

    @torch.inference_mode()
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        scores = np.zeros(len(pairs), dtype=np.float64)
        lengths = [len(passage) for _, passage in pairs]
        for batch in length_sorted_batches(lengths, self.batch_size):
            tokens = self.tokenizer(
                [pairs[i][0] for i in batch],
                [pairs[i][1] for i in batch],
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            scores[batch] = self.model(**tokens).logits.squeeze(-1).float().cpu().numpy()
        return scores

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        return self.score_pairs([(query, passage) for passage in passages])
