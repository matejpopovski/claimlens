"""Test, on tuning data only, one explanation for low official abstract-level recall.

scripts/final_eval.py asserts an abstract only when its own label equals the claim
verdict, while nei_bias is applied after pooling, so an abstract behind a decisive verdict
can still read as NOT_ENOUGH_INFO on its own. This script asks whether applying the same
bias to abstract labels would raise the official scores, and counts what happens to gold
abstracts, to separate abstracts judged wrongly from abstracts left unjudged. The held-out
set is not read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from claimlens.data import LABELS, NEI, DataPaths, load_corpus, load_tuning_splits
from claimlens.encoders import CrossEncoder
from claimlens.evaluation import (
    abstract_predictions,
    claim_logits,
    load_hits,
    run_sentence_pass,
)
from claimlens.metrics_verdict import AbstractPrediction, scifact_scores
from claimlens.pipeline import load_verdict_config
from claimlens.verdict import NLIModel


def biased_predictions(claims, judged, logits, *, top_k: int, nei_bias: float):
    """Assert abstracts by their label after the same NEI bias the claim verdict uses."""
    nei = LABELS.index(NEI)
    predictions = {}
    for claim, row in zip(claims, logits, strict=True):
        verdict = LABELS[int(row.argmax())]
        if verdict == NEI:
            continue
        kept = {}
        for judgement in judged.get(claim.id, [])[:top_k]:
            scores = np.log(np.clip(judgement.probabilities, 1e-12, 1.0))
            scores[nei] += nei_bias
            if LABELS[int(scores.argmax())] == verdict:
                kept[judgement.doc_id] = AbstractPrediction(verdict, judgement.rationale)
        if kept:
            predictions[claim.id] = kept
    return predictions


def gold_abstract_outcomes(claims, corpus, selector, nli, config) -> dict[str, int]:
    subset = [c for c in claims if c.evidence]
    documents = {c.id: list(c.evidence) for c in subset}
    judged = run_sentence_pass(selector, subset, documents, corpus).judge(
        nli,
        threshold=config.rationale_threshold,
        max_sentences=config.max_rationale_sentences,
        nei_weight=config.nli_nei_weight,
    )
    outcomes = {"abstracts": 0, "correct": 0, "wrong_direction": 0, "not_enough_info": 0}
    for claim in subset:
        for judgement in judged[claim.id]:
            gold = claim.evidence[judgement.doc_id][0].label
            outcomes["abstracts"] += 1
            if judgement.label == NEI:
                outcomes["not_enough_info"] += 1
            elif judgement.label == gold:
                outcomes["correct"] += 1
            else:
                outcomes["wrong_direction"] += 1
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/abstract_label_check.json"))
    args = parser.parse_args()

    corpus = load_corpus(DataPaths(args.data).corpus)
    config = load_verdict_config()
    nli = NLIModel.load("runs/nli/best")
    selector = CrossEncoder.load("runs/selector/best", max_length=128)
    report = {
        "note": "Tuning splits only. as_evaluated matches the behaviour behind results/final."
    }
    for name in ("val", "calib"):
        claims = load_tuning_splits(args.data)[name]
        hits = load_hits(Path(f"runs/cache/{name}_hits.json"))
        documents = {c.id: [h.doc_id for h in hits[c.id][: config.top_k]] for c in claims}
        judged = run_sentence_pass(selector, claims, documents, corpus).judge(
            nli,
            threshold=config.rationale_threshold,
            max_sentences=config.max_rationale_sentences,
            nei_weight=config.nli_nei_weight,
        )
        logits = claim_logits(claims, judged, top_k=config.top_k, nei_bias=config.nei_bias)
        as_evaluated = scifact_scores(
            claims, abstract_predictions(claims, judged, logits, top_k=config.top_k)
        )
        with_bias = scifact_scores(
            claims,
            biased_predictions(
                claims, judged, logits, top_k=config.top_k, nei_bias=config.nei_bias
            ),
        )
        report[name] = {
            "as_evaluated": as_evaluated,
            "with_bias_on_abstract_labels": with_bias,
            "gold_abstract_outcomes": gold_abstract_outcomes(claims, corpus, selector, nli, config),
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({n: report[n]["gold_abstract_outcomes"] for n in ("val", "calib")}))


if __name__ == "__main__":
    main()
