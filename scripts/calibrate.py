"""Fit the temperature, confidence model and thresholds on the calibration split.

The split played no part in training, epoch selection or verdict tuning, so these few
parameters are fitted on data the models have not seen.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from claimlens.abstain import AbstentionPolicy, confidence_features, threshold_for_precision
from claimlens.calibration import (
    brier_score,
    expected_calibration_error,
    fit_logistic,
    fit_temperature,
    negative_log_likelihood,
    reliability_bins,
    softmax,
)
from claimlens.data import DataPaths, load_corpus, load_tuning_splits
from claimlens.encoders import CrossEncoder
from claimlens.evaluation import (
    apply_policy,
    claim_logits,
    claim_report,
    confidence_and_correct,
    gold_indices,
    load_hits,
    run_sentence_pass,
    selective_summary,
)
from claimlens.pipeline import load_verdict_config
from claimlens.retrieval import best_rerank_score
from claimlens.verdict import NLIModel

CONFLICT_PROBABILITY = 0.5
RETRIEVAL_QUANTILE = 0.05
MIN_ANSWERED = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--hits", type=Path, default=Path("runs/cache/calib_hits.json"))
    parser.add_argument("--model", type=Path, default=Path("runs/nli/best"))
    parser.add_argument("--selector", type=Path, default=Path("runs/selector/best"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--target-precision", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=Path("results/calibration_calib.json"))
    args = parser.parse_args()

    corpus = load_corpus(DataPaths(args.data).corpus)
    claims = load_tuning_splits(args.data)["calib"]
    hits = load_hits(args.hits)
    config = load_verdict_config(args.configs)

    nli = NLIModel.load(args.model)
    documents = {c.id: [h.doc_id for h in hits[c.id][: config.top_k]] for c in claims}
    selector = CrossEncoder.load(args.selector, max_length=128)
    judged = run_sentence_pass(selector, claims, documents, corpus).judge(
        nli,
        threshold=config.rationale_threshold,
        max_sentences=config.max_rationale_sentences,
        nei_weight=config.nli_nei_weight,
    )
    logits = claim_logits(claims, judged, top_k=config.top_k, nei_bias=config.nei_bias)
    gold = gold_indices(claims)

    temperature = fit_temperature(logits, gold)
    before, correct = confidence_and_correct(claims, logits)
    scaled, _ = confidence_and_correct(claims, logits, temperature=temperature)

    probabilities = softmax(logits / temperature)
    weights = fit_logistic(confidence_features(probabilities), correct.astype(np.float64))
    estimator = AbstentionPolicy(0.0, confidence_weights=tuple(float(w) for w in weights))
    estimated = estimator.confidence(probabilities)

    threshold = threshold_for_precision(
        estimated, correct, target=args.target_precision, min_answered=MIN_ANSWERED
    )
    if threshold is None:
        raise SystemExit(f"selective accuracy never reaches {args.target_precision} on calib")

    # A claim whose cited abstract was retrieved shows what a healthy top score looks
    # like. Requiring that level treats low scores as "the corpus may not cover this",
    # with no reference to whether the verdict happened to be right.
    answerable = [
        best_rerank_score(hits[c.id][: config.top_k])
        for c in claims
        if set(c.cited_doc_ids) & set(documents[c.id])
    ]
    retrieval_floor = float(np.quantile(answerable, RETRIEVAL_QUANTILE))

    policy = AbstentionPolicy(
        min_confidence=threshold,
        min_retrieval_score=retrieval_floor,
        conflict_probability=CONFLICT_PROBABILITY,
        confidence_weights=estimator.confidence_weights,
    )
    decisions = apply_policy(
        claims,
        judged,
        hits,
        logits,
        top_k=config.top_k,
        temperature=temperature,
        policy=policy,
    )
    reasons: dict[str, int] = {}
    for decision in decisions:
        for abstention in decision.abstentions:
            reasons[abstention.reason.value] = reasons.get(abstention.reason.value, 0) + 1

    report = {
        "split": "calib",
        "n_claims": len(claims),
        "note": "Measured on the data the temperature was fitted to; test figures are in "
        "results/final.",
        "temperature": temperature,
        "target_precision": args.target_precision,
        "policy": policy.to_dict(),
        "report": claim_report(claims, logits),
        "uncalibrated": {
            "ece": expected_calibration_error(before, correct),
            "nll": negative_log_likelihood(logits, gold),
            "brier": brier_score(softmax(logits), gold),
            "bins": reliability_bins(before, correct),
        },
        "temperature_scaled": {
            "ece": expected_calibration_error(scaled, correct),
            "nll": negative_log_likelihood(logits / temperature, gold),
            "brier": brier_score(softmax(logits / temperature), gold),
            "bins": reliability_bins(scaled, correct),
        },
        "confidence_model": {
            "ece": expected_calibration_error(estimated, correct),
            "bins": reliability_bins(estimated, correct),
        },
        "operating_point": selective_summary(claims, decisions),
        "abstention_reasons": reasons,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    (args.configs / "verdict.json").write_text(
        json.dumps(replace(config, temperature=temperature).to_dict(), indent=2) + "\n"
    )
    (args.configs / "abstention.json").write_text(json.dumps(policy.to_dict(), indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("temperature", "policy", "operating_point")}
            | {
                "ece_before": report["uncalibrated"]["ece"],
                "ece_after": report["confidence_model"]["ece"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
