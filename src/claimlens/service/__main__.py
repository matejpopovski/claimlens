"""Run the service: python -m claimlens.service [--weights finetuned|stock]."""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path

import uvicorn

from claimlens.data import DataPaths, load_corpus
from claimlens.encoders import CrossEncoder
from claimlens.models import local_path
from claimlens.pipeline import (
    ClaimVerifier,
    build_retriever,
    load_policy,
    load_retrieval_config,
    load_verdict_config,
    reranker_path,
)
from claimlens.service.app import Example, create_app
from claimlens.verdict import NLIModel

STOCK_NOTE = (
    "Running on stock pretrained weights, without fine-tuning. The abstention thresholds "
    "were calibrated for the fine-tuned model, so the probabilities shown here are not "
    "calibrated and no reported metric applies to this configuration."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", choices=("finetuned", "stock"), default="finetuned")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=45411)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--nli", type=Path, default=Path("runs/nli/best"))
    parser.add_argument("--selector", type=Path, default=Path("runs/selector/best"))
    args = parser.parse_args()

    finetuned = args.weights == "finetuned"
    retrieval_config = load_retrieval_config()
    stock_cross_encoder = local_path("ms-marco-minilm")
    if finetuned:
        nli_path, selector_path = args.nli, args.selector
        reranker = reranker_path(retrieval_config.reranker)
    else:
        nli_path, selector_path = local_path("deberta-nli"), stock_cross_encoder
        reranker = stock_cross_encoder
    for path in (nli_path, selector_path, reranker):
        if not path.exists():
            raise SystemExit(
                f"{path} is missing. Train it (see the README), or start with --weights stock."
            )

    corpus = load_corpus(DataPaths(args.data).corpus)
    verifier = ClaimVerifier(
        build_retriever(corpus, retrieval_config, reranker=reranker),
        CrossEncoder.load(selector_path),
        NLIModel.load(nli_path),
        verdict_config=load_verdict_config(),
        policy=load_policy(),
    )
    examples_file = resources.files("claimlens.service") / "examples.json"
    examples = [Example(**row) for row in json.loads(examples_file.read_text())]
    app = create_app(
        verifier,
        weights=args.weights,
        weights_note=None if finetuned else STOCK_NOTE,
        examples=examples,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
