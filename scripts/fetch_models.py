"""Download pinned model snapshots into models/ and record their provenance.

The licence comes from the Hugging Face model card at the pinned revision rather
than from the registry, so results/provenance.json would expose a mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from claimlens.models import MODELS, local_path

WEIGHT_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "spm.model",
    "added_tokens.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--provenance", type=Path, default=Path("results/provenance.json"))
    args = parser.parse_args()

    api = HfApi()
    provenance = json.loads(args.provenance.read_text()) if args.provenance.exists() else {}
    records = provenance.setdefault("models", {})
    for key in args.only:
        spec = MODELS[key]
        snapshot_download(
            spec.repo,
            revision=spec.revision,
            local_dir=local_path(key),
            allow_patterns=WEIGHT_FILES,
        )
        card = api.model_info(spec.repo, revision=spec.revision).card_data
        licence = card.get("license") if card else None
        if licence != spec.licence:
            raise SystemExit(f"{spec.repo}: card says {licence}, registry says {spec.licence}")
        records[key] = {
            "repo": spec.repo,
            "revision": spec.revision,
            "licence_from_model_card": licence,
            "role": spec.role,
        }
        print(f"{key}: {spec.repo} at {spec.revision[:8]} ({licence})")

    args.provenance.parent.mkdir(exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
