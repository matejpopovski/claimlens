"""Embed every abstract with the chosen bi-encoders and cache the matrices in runs/index."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from claimlens.data import DataPaths, load_corpus
from claimlens.encoders import TextEncoder
from claimlens.models import MODELS, local_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=["minilm", "bge-small", "bge-base"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/index"))
    args = parser.parse_args()

    corpus = load_corpus(DataPaths(args.data).corpus)
    texts = [a.full_text for a in corpus.values()]
    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.encoders:
        spec = MODELS[key]
        started = time.monotonic()
        encoder = TextEncoder(
            local_path(key), pooling=spec.pooling, query_prefix=spec.query_prefix, max_length=384
        )
        embeddings = encoder.encode(texts)
        temporary = args.out / f"{key}.tmp.npy"
        np.save(temporary, embeddings)
        os.replace(temporary, args.out / f"{key}.npy")
        print(f"{key}: {embeddings.shape} on {encoder.device} in {time.monotonic() - started:.0f}s")
        del encoder


if __name__ == "__main__":
    main()
