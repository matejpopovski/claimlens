"""Every pretrained model the project can use, pinned to a commit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    licence: str
    role: str
    pooling: str = "mean"
    query_prefix: str = ""


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "minilm",
            "sentence-transformers/all-MiniLM-L6-v2",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "apache-2.0",
            "dense retrieval candidate",
        ),
        ModelSpec(
            "bge-base",
            "BAAI/bge-base-en-v1.5",
            "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
            "mit",
            "dense retrieval candidate",
            pooling="cls",
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        ModelSpec(
            "bge-small",
            "BAAI/bge-small-en-v1.5",
            "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            "mit",
            "dense retrieval candidate",
            pooling="cls",
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        ModelSpec(
            "ms-marco-minilm",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "233902d25c440f23af6f7d6e94d2946bac0bee0a",
            "apache-2.0",
            "cross-encoder reranker, starting point for fine-tuning",
        ),
        ModelSpec(
            "deberta-nli",
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            "6f5cf0a2b59cabb106aca4c287eed12e357e90eb",
            "mit",
            "NLI model, starting point for fine-tuning",
        ),
    )
}


def models_dir() -> Path:
    return Path(os.environ.get("CLAIMLENS_MODELS", "models"))


def local_path(key: str) -> Path:
    return models_dir() / key
