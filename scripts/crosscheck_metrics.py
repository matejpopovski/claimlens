"""Compare this repo's evaluators with the reference implementations on tuning data.

Two references are used, neither of them a dependency of the project:

* --official: a local copy of evaluator/eval.py from allenai/scifact-evaluator, scored on
  randomised predictions built from the fit split so that every branch gets exercised.
* --trec-run-output: writes a BM25 run and qrels for the validation split as JSON, which
  a separate environment with pytrec_eval scores for comparison with --trec-scores.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

from claimlens.data import (
    REFUTES,
    SUPPORTS,
    DataPaths,
    load_corpus,
    load_qrels,
    load_tuning_splits,
)
from claimlens.metrics_ir import evaluate_run
from claimlens.metrics_verdict import AbstractPrediction, scifact_scores
from claimlens.retrieval import RetrievalConfig, build_bm25
from claimlens.text import tokenize

TO_RAW = {SUPPORTS: "SUPPORT", REFUTES: "CONTRADICT"}


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("scifact_official_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_predictions(claims, rng):
    predictions = {}
    for claim in claims:
        documents = [*claim.evidence, 10_000_000 + rng.randrange(3)]
        kept = {}
        for doc_id in documents:
            if rng.random() < 0.3:
                continue
            gold = [i for r in claim.evidence.get(doc_id, ()) for i in r.sentences]
            noise = [rng.randrange(25) for _ in range(rng.randrange(4))]
            chosen = rng.sample(gold, k=rng.randint(0, len(gold)))
            sentences = list(dict.fromkeys([*chosen, *noise]))
            rng.shuffle(sentences)
            kept[doc_id] = AbstractPrediction(rng.choice((SUPPORTS, REFUTES)), tuple(sentences))
        predictions[claim.id] = kept
    return predictions


def official_gold(claim):
    return {
        "id": claim.id,
        "evidence": {
            str(doc_id): [
                {"sentences": list(r.sentences), "label": TO_RAW[r.label]} for r in rationales
            ]
            for doc_id, rationales in claim.evidence.items()
        },
    }


def official_prediction(claim_id, predicted):
    return {
        "id": claim_id,
        "evidence": {
            str(doc_id): {"sentences": list(p.sentences), "label": TO_RAW[p.label]}
            for doc_id, p in predicted.items()
        },
    }


def compare_with_official(path: Path, claims, trials: int) -> dict[str, float]:
    module = load_official(path)
    evaluator = module.Evaluator(verbose=True)
    golds = [module.unify_label(official_gold(c)) for c in claims]
    worst = 0.0
    names = {
        "abstract_label_only": "abstract_label_only",
        "abstract_rationalized": "abstract_label_rationale",
        "sentence_selection": "sentence_selection",
        "sentence_label": "sentence_selection_label",
    }
    for trial in range(trials):
        rng = random.Random(trial)
        predictions = random_predictions(claims, rng)
        official = evaluator.evaluate(
            golds, [official_prediction(c.id, predictions[c.id]) for c in claims]
        )
        ours = scifact_scores(claims, predictions)
        for theirs, mine in names.items():
            for metric in ("precision", "recall", "f1"):
                worst = max(worst, abs(official[f"{theirs}_{metric}"] - ours[mine][metric]))
    return {"trials": trials, "n_claims": len(claims), "largest_absolute_difference": worst}


def bm25_run(args, claims, corpus):
    config = RetrievalConfig(**json.loads(args.config.read_text()))
    doc_ids = list(corpus)
    index = build_bm25([corpus[d] for d in doc_ids], config)
    run = {}
    for claim in claims:
        tokens = tokenize(claim.text, stem=config.stem, remove_stopwords=config.remove_stopwords)
        run[claim.id] = index.top_k(tokens, 100)
    return {claim_id: [(doc_ids[i], score) for i, score in hits] for claim_id, hits in run.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval.json"))
    parser.add_argument("--official", type=Path)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--trec-run-output", type=Path)
    parser.add_argument("--trec-scores", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/metric_crosscheck.json"))
    args = parser.parse_args()

    splits = load_tuning_splits(args.data)
    paths = DataPaths(args.data)
    report = json.loads(args.out.read_text()) if args.out.exists() else {}

    if args.official:
        report["official_scifact_evaluator"] = compare_with_official(
            args.official, splits["fit"], args.trials
        )

    if args.trec_run_output or args.trec_scores:
        corpus = load_corpus(paths.corpus)
        qrels_all = load_qrels(paths.qrels_train)
        qrels = {c.id: qrels_all[c.id] for c in splits["val"]}
        run = bm25_run(args, splits["val"], corpus)
        if args.trec_run_output:
            payload = {
                "qrels": {str(q): {str(d): g for d, g in v.items()} for q, v in qrels.items()},
                "run": {str(q): {str(d): s for d, s in v} for q, v in run.items()},
            }
            args.trec_run_output.write_text(json.dumps(payload))
        if args.trec_scores:
            reference = json.loads(args.trec_scores.read_text())
            ours = evaluate_run({q: [d for d, _ in v] for q, v in run.items()}, qrels)
            report["pytrec_eval"] = {
                "n_queries": len(qrels),
                "ours": {k: ours[k] for k in reference},
                "pytrec_eval": reference,
                "largest_absolute_difference": max(abs(ours[k] - v) for k, v in reference.items()),
            }

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
