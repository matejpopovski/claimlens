"""Fine-tune the rationale selector: a cross-encoder scoring (claim, sentence) pairs.

The loss is unweighted so the output stays a usable probability of evidence, which the
verdict formula relies on. Sentence-level positives are rare, but only the ordering and a
threshold tuned on validation decide what gets selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimlens.data import DataPaths, load_corpus, load_tuning_splits
from claimlens.devices import pick_device
from claimlens.encoders import CrossEncoder
from claimlens.evaluation import load_hits
from claimlens.models import MODELS, local_path
from claimlens.training import PairExample, TrainConfig, fit
from claimlens.verdict_data import build_selector_examples


def rationale_recall(scorer: CrossEncoder, claims, corpus) -> dict[str, float]:
    """How often a whole gold rationale sits inside the top 1 and top 3 sentences."""
    total = top1 = top3 = 0
    for claim in claims:
        for doc_id, rationales in claim.evidence.items():
            sentences = corpus[doc_id].sentences
            order = np.argsort(-scorer.score(claim.text, sentences), kind="stable")
            total += 1
            top1 += any(set(r.sentences) <= set(order[:1].tolist()) for r in rationales)
            top3 += any(set(r.sentences) <= set(order[:3].tolist()) for r in rationales)
    return {"rationale_in_top1": top1 / total, "rationale_in_top3": top3 / total, "n": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--hits", type=Path, default=Path("runs/cache"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/selector"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/selector_training.json"))
    args = parser.parse_args()

    corpus = load_corpus(DataPaths(args.data).corpus)
    splits = load_tuning_splits(args.data)
    hits = load_hits(args.hits / "fit_hits.json")
    retrieved = {k: [h.doc_id for h in v] for k, v in hits.items()}
    triples = build_selector_examples(splits["fit"], corpus, retrieved)
    examples = [PairExample(claim, sentence, label) for claim, sentence, label in triples]
    positives = sum(e.label == 1.0 for e in examples)
    print(f"{len(examples)} sentences, {positives} rationale")

    device = pick_device()
    base = local_path("ms-marco-minilm")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base)
    criterion = torch.nn.BCEWithLogitsLoss()

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return criterion(logits.squeeze(-1), targets)

    def evaluate(current) -> float:
        scorer = CrossEncoder(current, tokenizer, device=device, max_length=128)
        scores = rationale_recall(scorer, splits["val"], corpus)
        return (scores["rationale_in_top1"] + scores["rationale_in_top3"]) / 2

    result = fit(
        model,
        tokenizer,
        examples,
        config=TrainConfig(epochs=args.epochs, batch_size=32, max_length=128, checkpoint_every=0),
        loss_fn=loss_fn,
        run_dir=args.run_dir,
        device=device,
        evaluate=evaluate,
        resume=args.resume,
    )
    best = CrossEncoder.load(args.run_dir / "best", device=device, max_length=128)
    report = {
        "base_model": MODELS["ms-marco-minilm"].repo,
        "selection_split": "val",
        "selection_metric": "mean of rationale_in_top1 and rationale_in_top3 on gold abstracts",
        "sentences": len(examples),
        "positives": positives,
        "val_gold_abstracts": rationale_recall(best, splits["val"], corpus),
        "val_gold_abstracts_stock": rationale_recall(
            CrossEncoder.load(base, device=device, max_length=128), splits["val"], corpus
        ),
        **result,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k.startswith("val_")}))


if __name__ == "__main__":
    main()
