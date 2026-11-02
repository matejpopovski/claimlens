"""Choose the rationale threshold, abstract count and NEI bias on the validation split."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from claimlens.data import DataPaths, load_corpus, load_tuning_splits
from claimlens.encoders import CrossEncoder
from claimlens.evaluation import (
    abstract_predictions,
    claim_logits,
    claim_report,
    load_hits,
    run_sentence_pass,
)
from claimlens.metrics_verdict import scifact_scores
from claimlens.verdict import NLIModel, VerdictConfig, with_nei_weight

THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)
TOP_KS = (1, 2, 3, 5)
BIASES = (-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
NEI_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)

# On about 120 claims, macro-F1 differences this small are noise. Settings inside the
# margin are separated by the official abstract-level score, so the choice cannot favour
# a setting that only looks good on one metric, such as reading a single abstract.
MACRO_F1_MARGIN = 0.02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--hits", type=Path, default=Path("runs/cache/val_hits.json"))
    parser.add_argument("--model", type=Path, default=Path("runs/nli/best"))
    parser.add_argument("--selector", type=Path, default=Path("runs/selector/best"))
    parser.add_argument("--out", type=Path, default=Path("results/verdict_tuning_val.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/verdict.json"))
    parser.add_argument("--max-sentences", type=int, default=3)
    args = parser.parse_args()

    corpus = load_corpus(DataPaths(args.data).corpus)
    claims = load_tuning_splits(args.data)["val"]
    hits = load_hits(args.hits)
    documents = {c.id: [h.doc_id for h in hits[c.id][: max(TOP_KS)]] for c in claims}
    nli = NLIModel.load(args.model)
    selector = CrossEncoder.load(args.selector, max_length=128)
    sentence_pass = run_sentence_pass(selector, claims, documents, corpus)

    rows = []
    for threshold in THRESHOLDS:
        judged = sentence_pass.judge(nli, threshold=threshold, max_sentences=args.max_sentences)
        for nei_weight, top_k, bias in itertools.product(NEI_WEIGHTS, TOP_KS, BIASES):
            weighted = {
                claim_id: [with_nei_weight(j, nei_weight) for j in items]
                for claim_id, items in judged.items()
            }
            logits = claim_logits(claims, weighted, top_k=top_k, nei_bias=bias)
            report = claim_report(claims, logits)
            official = scifact_scores(
                claims, abstract_predictions(claims, weighted, logits, top_k=top_k)
            )
            rows.append(
                {
                    "rationale_threshold": threshold,
                    "nli_nei_weight": nei_weight,
                    "top_k": top_k,
                    "nei_bias": bias,
                    "claim_macro_f1": report["macro_f1"],
                    "claim_accuracy": report["accuracy"],
                    "abstract_label_only_f1": official["abstract_label_only"]["f1"],
                    "sentence_selection_label_f1": official["sentence_selection_label"]["f1"],
                }
            )
    top = max(r["claim_macro_f1"] for r in rows)
    contenders = [r for r in rows if r["claim_macro_f1"] >= top - MACRO_F1_MARGIN]
    best = max(contenders, key=lambda r: (r["abstract_label_only_f1"], r["claim_macro_f1"]))
    print(json.dumps(best, indent=2))

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "split": "val",
                "n_claims": len(claims),
                "selection_rule": f"highest abstract-level F1 within {MACRO_F1_MARGIN} of the "
                "best claim macro-F1",
                "best_macro_f1_in_grid": top,
                "best": best,
                "grid": rows,
            },
            indent=2,
        )
        + "\n"
    )
    config = VerdictConfig(
        rationale_threshold=best["rationale_threshold"],
        max_rationale_sentences=args.max_sentences,
        top_k=best["top_k"],
        nei_bias=best["nei_bias"],
        nli_nei_weight=best["nli_nei_weight"],
    )
    args.config.write_text(json.dumps(config.to_dict(), indent=2) + "\n")


if __name__ == "__main__":
    main()
