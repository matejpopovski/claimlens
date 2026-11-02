"""Shared fixtures. Models here are random and tiny, built on disk with no download."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizerFast

from claimlens.data import Abstract, Claim, Rationale

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "aspirin",
    "reduces",
    "fever",
    "vitamin",
    "cures",
    "scurvy",
    "the",
    "a",
    "study",
    "of",
    "trial",
    "patients",
    "no",
    "effect",
]


SEED = 0


def _tiny_config(num_labels: int, id2label: dict[int, str] | None = None) -> BertConfig:
    # Weights are random, so an unseeded fixture would make any order-sensitive test flaky.
    torch.manual_seed(SEED)
    return BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()} if id2label else None,
    )


def _write_tokenizer(directory: Path) -> None:
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    BertTokenizerFast(str(vocab_file), do_lower_case=True).save_pretrained(directory)


@pytest.fixture(scope="session")
def tiny_encoder_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tiny-encoder")
    BertModel(_tiny_config(1)).save_pretrained(directory)
    _write_tokenizer(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_reranker_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tiny-reranker")
    BertForSequenceClassification(_tiny_config(1)).save_pretrained(directory)
    _write_tokenizer(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_nli_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tiny-nli")
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    BertForSequenceClassification(_tiny_config(3, id2label)).save_pretrained(directory)
    _write_tokenizer(directory)
    return directory


@pytest.fixture
def corpus() -> dict[int, Abstract]:
    abstracts = [
        Abstract(11, "Aspirin and fever", ("A trial of aspirin.", "Aspirin reduces fever.")),
        Abstract(12, "Vitamin C", ("Vitamin C cures scurvy.", "Patients were studied.")),
        Abstract(13, "Unrelated", ("A study of something else.", "No effect was seen.")),
    ]
    return {a.doc_id: a for a in abstracts}


@pytest.fixture
def claims() -> list[Claim]:
    return [
        Claim(
            1,
            "Aspirin reduces fever",
            (11,),
            {11: (Rationale((1,), "SUPPORTS"),)},
        ),
        Claim(
            2,
            "Vitamin C has no effect on scurvy",
            (12,),
            {12: (Rationale((0,), "REFUTES"),)},
        ),
        Claim(3, "The study of patients", (13,), {}),
    ]
