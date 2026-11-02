"""Fine-tune the pretrained NLI model on SciFact rationales. Resumable with --resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimlens.data import LABELS, DataPaths, load_corpus, load_tuning_splits
from claimlens.devices import pick_device
from claimlens.evaluation import load_hits
from claimlens.metrics_verdict import classification_report
from claimlens.models import MODELS, local_path
from claimlens.training import PairExample, TrainConfig, fit
from claimlens.verdict import NLIModel, label_order
from claimlens.verdict_data import balanced_class_weights, build_nli_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--hits", type=Path, default=Path("runs/cache"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/nli"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--nei-per-decisive", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/nli_training.json"))
    args = parser.parse_args()

    corpus = load_corpus(DataPaths(args.data).corpus)
    splits = load_tuning_splits(args.data)

    def examples_for(name: str):
        hits = load_hits(args.hits / f"{name}_hits.json")
        retrieved = {k: [h.doc_id for h in v] for k, v in hits.items()}
        return build_nli_examples(
            splits[name], corpus, retrieved, nei_per_decisive=args.nei_per_decisive
        )

    train_examples = examples_for("fit")
    val_examples = examples_for("val")
    counts = {label: sum(e.label == label for e in train_examples) for label in LABELS}
    print(f"{len(train_examples)} training examples: {counts}")

    device = pick_device()
    base = local_path("deberta-nli")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base)
    order = label_order(model.config.id2label)
    index_of = dict(zip(LABELS, order, strict=True))
    pairs = [PairExample(e.premise, e.claim, index_of[e.label]) for e in train_examples]

    # Weights follow the model's own output order, not the project's LABELS order.
    weights = torch.zeros(len(LABELS), dtype=torch.float32)
    for label_index, weight in zip(order, balanced_class_weights(train_examples), strict=True):
        weights[label_index] = weight
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    val_pairs = [(e.premise, e.claim) for e in val_examples]
    val_gold = [e.label for e in val_examples]

    def evaluate(current) -> float:
        scorer = NLIModel(current, tokenizer, device=device, max_length=args.max_length)
        predicted = np.argmax(scorer.logits(val_pairs), axis=1)
        report = classification_report(val_gold, [LABELS[i] for i in predicted])
        if device.type == "mps":
            torch.mps.empty_cache()
        return float(report["macro_f1"])

    result = fit(
        model,
        tokenizer,
        pairs,
        config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            checkpoint_every=args.checkpoint_every,
        ),
        loss_fn=criterion,
        run_dir=args.run_dir,
        device=device,
        evaluate=evaluate,
        resume=args.resume,
    )
    report = {
        "base_model": MODELS["deberta-nli"].repo,
        "device": str(device),
        "selection_split": "val",
        "selection_metric": "macro-F1 over validation premise/claim examples",
        "training_examples": counts,
        "validation_examples": len(val_examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "nei_per_decisive": args.nei_per_decisive,
        **result,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
