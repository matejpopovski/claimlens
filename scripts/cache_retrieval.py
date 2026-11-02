"""Run the frozen retriever over the tuning splits and cache its ranked lists."""

from __future__ import annotations

import argparse
from pathlib import Path

from claimlens.data import DataPaths, load_corpus, load_tuning_splits
from claimlens.evaluation import save_hits
from claimlens.pipeline import build_retriever, load_retrieval_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/cache"))
    parser.add_argument("--depth", type=int, default=20)
    args = parser.parse_args()

    corpus = load_corpus(DataPaths(args.data).corpus)
    retriever = build_retriever(corpus, load_retrieval_config())
    for name, claims in load_tuning_splits(args.data).items():
        hits = {claim.id: retriever.search(claim.text, args.depth) for claim in claims}
        save_hits(args.out / f"{name}_hits.json", hits)
        print(f"{name}: {len(hits)} claims")


if __name__ == "__main__":
    main()
