#!/usr/bin/env bash
# Rebuild every model and result from public sources. Fine-tuned weights are too large
# for the repo, so this is how they are obtained. Needs a GPU or Apple MPS to finish in
# reasonable time, and uses about 3 GB of disk.
#
# Results go to runs/repro so the committed results/final stays untouched. Fine-tuning
# is seeded, but GPU kernels are not bit-exact across machines, so expect close numbers
# and not identical ones.
set -euo pipefail

uv sync --frozen
mkdir -p runs/repro
run="uv run --frozen python"

$run scripts/download_data.py
$run scripts/fetch_models.py --provenance runs/repro/provenance.json
$run scripts/encode_corpus.py

$run scripts/tune_retrieval.py --out runs/repro/retrieval_tuning_stock.json
$run scripts/train_reranker.py --out runs/repro/reranker_training.json
$run scripts/tune_retrieval.py --out runs/repro/retrieval_tuning_val.json
$run scripts/cache_retrieval.py

$run scripts/train_selector.py --out runs/repro/selector_training.json
$run scripts/train_nli.py --out runs/repro/nli_training.json
$run scripts/tune_verdict.py --model models/deberta-nli --selector models/ms-marco-minilm \
    --config configs/verdict_stock.json --out runs/repro/verdict_tuning_val_stock.json
$run scripts/tune_verdict.py --out runs/repro/verdict_tuning_val.json
$run scripts/calibrate.py --out runs/repro/calibration_calib.json

$run scripts/final_eval.py --split test --confirm-final --results runs/repro
