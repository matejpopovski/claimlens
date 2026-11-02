"""Fine-tune the cross-encoder on the fit split, with hard negatives from fusion.

Negatives are abstracts that hybrid retrieval ranks highly for a claim without the
claim citing them. Those are the only confusions a reranker ever has to resolve.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimlens.data import DataPaths, load_corpus, load_qrels, load_tuning_splits
from claimlens.devices import pick_device
from claimlens.encoders import CrossEncoder, DenseIndex, TextEncoder
from claimlens.fusion import reciprocal_rank_fusion
from claimlens.metrics_ir import evaluate_run
from claimlens.models import MODELS, local_path
from claimlens.retrieval import RetrievalConfig, build_bm25
from claimlens.text import tokenize
from claimlens.training import PairExample, TrainConfig, fit


def candidate_lists(claims, doc_ids, abstracts, config, index_dir, depth):
    spec = MODELS[config.encoder]
    bm25 = build_bm25(abstracts, config)
    encoder = TextEncoder(
        local_path(config.encoder),
        pooling=spec.pooling,
        query_prefix=spec.query_prefix,
        max_length=384,
    )
    dense = DenseIndex(np.load(index_dir / f"{config.encoder}.npy"))
    vectors = encoder.encode([c.text for c in claims], is_query=True)
    candidates = {}
    for claim, vector in zip(claims, vectors, strict=True):
        tokens = tokenize(claim.text, stem=config.stem, remove_stopwords=config.remove_stopwords)
        lexical = [doc_ids[i] for i, _ in bm25.top_k(tokens, depth)]
        semantic = [doc_ids[i] for i, _ in dense.top_k(vector, depth)]
        fused = reciprocal_rank_fusion(
            [lexical, semantic], k=config.rrf_k, weights=[1.0, config.dense_weight]
        )
        candidates[claim.id] = [d for d, _ in fused[:depth]]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--index", type=Path, default=Path("runs/index"))
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/reranker"))
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--hard-pool", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/reranker_training.json"))
    args = parser.parse_args()

    config = RetrievalConfig(**json.loads(args.config.read_text()))
    paths = DataPaths(args.data)
    corpus = load_corpus(paths.corpus)
    doc_ids = list(corpus)
    abstracts = [corpus[d] for d in doc_ids]
    splits = load_tuning_splits(args.data)
    qrels = load_qrels(paths.qrels_train)

    claims = splits["fit"] + splits["val"]
    candidates = candidate_lists(claims, doc_ids, abstracts, config, args.index, 100)

    rng = random.Random(13)
    examples = []
    for claim in splits["fit"]:
        relevant = qrels[claim.id]
        hard = [d for d in candidates[claim.id][: args.hard_pool] if d not in relevant]
        for doc_id in relevant:
            examples.append(PairExample(claim.text, corpus[doc_id].full_text, 1.0))
        for doc_id in rng.sample(hard, min(args.negatives, len(hard))):
            examples.append(PairExample(claim.text, corpus[doc_id].full_text, 0.0))
    positives = sum(e.label == 1.0 for e in examples)
    print(f"{len(examples)} pairs, {positives} positive")

    device = pick_device()
    base = local_path("ms-marco-minilm")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base)
    pos_weight = torch.tensor((len(examples) - positives) / positives, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return criterion(logits.squeeze(-1), targets)

    val_qrels = {c.id: qrels[c.id] for c in splits["val"]}

    def evaluate(current) -> float:
        scorer = CrossEncoder(current, tokenizer, device=device)
        run = {}
        for claim in splits["val"]:
            pool = candidates[claim.id][:50]
            scores = scorer.score(claim.text, [corpus[d].full_text for d in pool])
            run[claim.id] = [pool[i] for i in np.lexsort((np.arange(len(pool)), -scores))]
        return evaluate_run(run, val_qrels)["ndcg@10"]

    result = fit(
        model,
        tokenizer,
        examples,
        config=TrainConfig(
            epochs=args.epochs,
            batch_size=16,
            learning_rate=args.learning_rate,
            max_length=384,
            checkpoint_every=0,
        ),
        loss_fn=loss_fn,
        run_dir=args.run_dir,
        device=device,
        evaluate=evaluate,
        resume=args.resume,
    )
    report = {
        "base_model": MODELS["ms-marco-minilm"].repo,
        "selection_split": "val",
        "selection_metric": "ndcg@10 reranking the top 50 fused abstracts",
        "pairs": len(examples),
        "positives": positives,
        **result,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
