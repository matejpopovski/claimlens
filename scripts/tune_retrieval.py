"""Choose every retrieval setting on the validation split and freeze the result.

The stages run in order and each fixes its winner before the next: BM25 parameters,
dense encoder, fusion, then reranking. Relevance for these claims comes from the BEIR
train qrels, which list the same claims as SciFact's train split.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from claimlens.data import DataPaths, load_corpus, load_qrels, load_tuning_splits
from claimlens.encoders import CrossEncoder, DenseIndex, TextEncoder
from claimlens.fusion import reciprocal_rank_fusion
from claimlens.metrics_ir import evaluate_run
from claimlens.models import MODELS, local_path
from claimlens.pipeline import FINETUNED
from claimlens.retrieval import RetrievalConfig, build_bm25
from claimlens.text import tokenize

TARGET = "ndcg@10"


def bm25_run(abstracts, doc_ids, claims, config, depth):
    index = build_bm25(abstracts, config)
    run = {}
    for claim in claims:
        tokens = tokenize(claim.text, stem=config.stem, remove_stopwords=config.remove_stopwords)
        run[claim.id] = [doc_ids[i] for i, _ in index.top_k(tokens, depth)]
    return run


def dense_run(key, doc_ids, claims, depth, index_dir):
    spec = MODELS[key]
    encoder = TextEncoder(
        local_path(key), pooling=spec.pooling, query_prefix=spec.query_prefix, max_length=384
    )
    index = DenseIndex(np.load(index_dir / f"{key}.npy"))
    vectors = encoder.encode([c.text for c in claims], is_query=True)
    return {
        claim.id: [doc_ids[i] for i, _ in index.top_k(vector, depth)]
        for claim, vector in zip(claims, vectors, strict=True)
    }


def fuse(lexical, dense, claims, rrf_k, depth, dense_weight):
    fused = {}
    for claim in claims:
        lists = [lexical[claim.id][:depth], dense[claim.id][:depth]]
        ranked = reciprocal_rank_fusion(lists, k=rrf_k, weights=[1.0, dense_weight])
        fused[claim.id] = [doc_id for doc_id, _ in ranked]
    return fused


def rerank_run(model_path, corpus, claims, candidates, depth):
    reranker = CrossEncoder.load(model_path)
    run = {}
    for claim in claims:
        pool = candidates[claim.id][:depth]
        scores = reranker.score(claim.text, [corpus[d].full_text for d in pool])
        order = np.lexsort((np.arange(len(pool)), -scores))
        run[claim.id] = [pool[i] for i in order]
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--index", type=Path, default=Path("runs/index"))
    parser.add_argument("--encoders", nargs="+", default=["minilm", "bge-small", "bge-base"])
    parser.add_argument("--finetuned-reranker", type=Path, default=Path("runs/reranker/best"))
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_tuning_val.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.json"))
    args = parser.parse_args()

    paths = DataPaths(args.data)
    corpus = load_corpus(paths.corpus)
    doc_ids = list(corpus)
    abstracts = [corpus[d] for d in doc_ids]
    val = load_tuning_splits(args.data)["val"]
    train_qrels = load_qrels(paths.qrels_train)
    qrels = {c.id: train_qrels[c.id] for c in val}
    report: dict[str, object] = {"split": "val", "n_claims": len(val), "target": TARGET}

    bm25_rows = []
    for k1, b, stem, stop in itertools.product(
        (0.3, 0.45, 0.6, 0.9, 1.2, 1.5), (0.25, 0.4, 0.6, 0.75), (True, False), (True, False)
    ):
        config = RetrievalConfig(k1=k1, b=b, stem=stem, remove_stopwords=stop)
        scores = evaluate_run(bm25_run(abstracts, doc_ids, val, config, 100), qrels)
        bm25_rows.append({"k1": k1, "b": b, "stem": stem, "remove_stopwords": stop, **scores})
    best = max(bm25_rows, key=lambda r: r[TARGET])
    config = RetrievalConfig(
        k1=best["k1"], b=best["b"], stem=best["stem"], remove_stopwords=best["remove_stopwords"]
    )
    report["bm25_grid"] = bm25_rows
    report["bm25_best"] = best
    print(f"bm25 best {best[TARGET]:.4f} with k1={config.k1} b={config.b}")

    lexical = bm25_run(abstracts, doc_ids, val, config, 200)
    dense_runs = {k: dense_run(k, doc_ids, val, 200, args.index) for k in args.encoders}
    report["dense"] = {k: evaluate_run(run, qrels) for k, run in dense_runs.items()}
    for key, scores in report["dense"].items():
        print(f"dense {key}: {scores[TARGET]:.4f}")

    fusion_rows = []
    for key, rrf_k, depth, weight in itertools.product(
        args.encoders, (3, 10, 30, 60, 100), (50, 100, 200), (1.0, 2.0, 3.0, 5.0)
    ):
        run = fuse(lexical, dense_runs[key], val, rrf_k, depth, weight)
        scores = evaluate_run(run, qrels)
        fusion_rows.append(
            {"encoder": key, "rrf_k": rrf_k, "depth": depth, "dense_weight": weight, **scores}
        )
    best_fusion = max(fusion_rows, key=lambda r: r[TARGET])
    report["fusion_grid"] = fusion_rows
    report["fusion_best"] = best_fusion
    config = replace(
        config,
        encoder=best_fusion["encoder"],
        rrf_k=best_fusion["rrf_k"],
        dense_weight=best_fusion["dense_weight"],
        fusion_depth=best_fusion["depth"],
    )
    print(
        f"fusion best {best_fusion[TARGET]:.4f}: {config.encoder} k={config.rrf_k} "
        f"weight={config.dense_weight} depth={config.fusion_depth}"
    )

    fused = fuse(
        lexical,
        dense_runs[config.encoder],
        val,
        config.rrf_k,
        config.fusion_depth,
        config.dense_weight,
    )
    models = {"ms-marco-minilm": local_path("ms-marco-minilm")}
    if args.finetuned_reranker.exists():
        models[FINETUNED] = args.finetuned_reranker
    rerank_rows = []
    for (name, path), depth in itertools.product(models.items(), (20, 50, 100)):
        scores = evaluate_run(rerank_run(path, corpus, val, fused, depth), qrels)
        rerank_rows.append({"reranker": name, "depth": depth, **scores})
        print(f"rerank {name} depth {depth}: {scores[TARGET]:.4f}")
    report["rerank_grid"] = rerank_rows
    best_rerank = max(rerank_rows, key=lambda r: r[TARGET])
    config = replace(config, reranker=best_rerank["reranker"], rerank_depth=best_rerank["depth"])

    late_rows = [{"final_fusion_k": None, "final_fusion_weight": None, **best_rerank}]
    reranked = rerank_run(models[config.reranker], corpus, val, fused, config.rerank_depth)
    for final_k, weight in itertools.product((3, 10, 30, 60), (0.5, 1.0, 2.0, 3.0, 5.0)):
        late = {}
        for claim in val:
            lists = [reranked[claim.id], fused[claim.id][: config.rerank_depth]]
            ranked = reciprocal_rank_fusion(lists, k=final_k, weights=[1.0, weight])
            late[claim.id] = [doc_id for doc_id, _ in ranked]
        scores = evaluate_run(late, qrels)
        late_rows.append({"final_fusion_k": final_k, "final_fusion_weight": weight, **scores})
    report["final_fusion_grid"] = late_rows
    report["first_stage_only"] = evaluate_run(fused, qrels)
    best_late = max(late_rows, key=lambda r: r[TARGET])
    report["final_best"] = best_late
    print(
        f"late fusion best {best_late[TARGET]:.4f}: k={best_late['final_fusion_k']} "
        f"weight={best_late['final_fusion_weight']}"
    )
    if best_late["final_fusion_k"] is not None:
        config = replace(
            config,
            final_fusion_k=best_late["final_fusion_k"],
            final_fusion_weight=best_late["final_fusion_weight"],
        )

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    args.config.parent.mkdir(exist_ok=True)
    args.config.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    print(json.dumps(config.to_dict()))


if __name__ == "__main__":
    main()
