"""The loop is driven with a real tiny transformer, so resume is tested end to end."""

import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimlens.training import (
    PairExample,
    TrainConfig,
    atomic_torch_save,
    epoch_batches,
    fit,
    linear_warmup_lambda,
    save_best_model,
    to_cpu,
)

EXAMPLES = [
    PairExample("aspirin reduces fever", "aspirin", 0),
    PairExample("vitamin cures scurvy", "vitamin", 2),
    PairExample("a study of patients", "trial", 1),
    PairExample("no effect", "fever", 1),
] * 3


def load(directory):
    torch.manual_seed(0)
    return (
        AutoModelForSequenceClassification.from_pretrained(directory),
        AutoTokenizer.from_pretrained(directory),
    )


def train(directory, run_dir, *, epochs=2, resume=False, checkpoint_every=2, stop_after=None):
    model, tokenizer = load(directory)
    calls = {"n": 0}

    def loss_fn(logits, targets):
        calls["n"] += 1
        if stop_after is not None and calls["n"] > stop_after:
            raise KeyboardInterrupt
        return torch.nn.functional.cross_entropy(logits, targets)

    config = TrainConfig(
        epochs=epochs,
        batch_size=4,
        learning_rate=5e-3,
        checkpoint_every=checkpoint_every,
        max_length=16,
    )
    result = fit(
        model,
        tokenizer,
        EXAMPLES,
        config=config,
        loss_fn=loss_fn,
        run_dir=run_dir,
        device=torch.device("cpu"),
        resume=resume,
        log=lambda _: None,
    )
    return model, result


def test_loss_falls_on_a_small_task(tiny_nli_dir, tmp_path):
    model, tokenizer = load(tiny_nli_dir)
    tokens = tokenizer(
        [e.premise for e in EXAMPLES],
        [e.hypothesis for e in EXAMPLES],
        padding=True,
        return_tensors="pt",
    )
    targets = torch.tensor([e.label for e in EXAMPLES])
    before = torch.nn.functional.cross_entropy(model(**tokens).logits, targets).item()
    trained, _ = train(tiny_nli_dir, tmp_path, epochs=8)
    trained.eval()
    after = torch.nn.functional.cross_entropy(trained(**tokens).logits, targets).item()
    assert after < before


def test_interrupted_run_resumes_to_the_same_weights(tiny_nli_dir, tmp_path):
    straight, _ = train(tiny_nli_dir, tmp_path / "straight", epochs=2)

    run = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        train(tiny_nli_dir, run, epochs=2, checkpoint_every=2, stop_after=4)
    assert torch.load(run / "checkpoint.pt", weights_only=False)["step"] == 4

    resumed, result = train(tiny_nli_dir, run, epochs=2, resume=True, checkpoint_every=2)
    assert result["steps"] == 6
    for name, parameter in straight.state_dict().items():
        assert torch.allclose(parameter, resumed.state_dict()[name], atol=1e-6), name


def test_checkpoint_holds_only_cpu_tensors(tiny_nli_dir, tmp_path):
    train(tiny_nli_dir, tmp_path, epochs=1)
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert all(t.device.type == "cpu" for t in state["model"].values())


def test_best_model_is_loadable(tiny_nli_dir, tmp_path):
    train(tiny_nli_dir, tmp_path, epochs=1)
    assert AutoModelForSequenceClassification.from_pretrained(tmp_path / "best")
    assert not (tmp_path / "best.staging").exists()


def test_progress_file_reports_the_final_step(tiny_nli_dir, tmp_path):
    train(tiny_nli_dir, tmp_path, epochs=1)
    assert json.loads((tmp_path / "progress.json").read_text())["step"] == 3


def test_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    atomic_torch_save({"v": 1}, path)

    def crash(payload, destination):
        destination.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", crash)
    with pytest.raises(OSError, match="disk full"):
        atomic_torch_save({"v": 2}, path)
    monkeypatch.undo()
    assert torch.load(path, weights_only=False) == {"v": 1}


def test_the_final_step_of_a_save_is_a_rename_within_one_directory(tmp_path, monkeypatch):
    renames = []
    real_replace = os.replace

    def recording_replace(source, destination):
        renames.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", recording_replace)
    path = tmp_path / "state.pt"
    atomic_torch_save({"v": 1}, path)
    assert renames == [(tmp_path / "state.pt.tmp", path)]


def test_saving_a_model_twice_replaces_it_cleanly(tiny_nli_dir, tmp_path):
    model, tokenizer = load(tiny_nli_dir)
    save_best_model(model, tokenizer, tmp_path / "best")
    save_best_model(model, tokenizer, tmp_path / "best")
    assert not (tmp_path / "best.previous").exists()
    assert (tmp_path / "best" / "model.safetensors").exists()


def test_to_cpu_walks_nested_containers():
    moved = to_cpu({"a": [torch.ones(1), (torch.zeros(1), 3)], "b": "text"})
    assert moved["a"][1][1] == 3
    assert moved["b"] == "text"
    assert moved["a"][0].device.type == "cpu"


def test_epoch_batches_are_a_pure_function_of_seed_and_epoch():
    first = epoch_batches(10, 3, seed=1, epoch=2)
    second = epoch_batches(10, 3, seed=1, epoch=2)
    assert all((a == b).all() for a, b in zip(first, second, strict=True))
    assert sorted(i for b in first for i in b) == list(range(10))
    assert not all(
        (a == b).all() for a, b in zip(first, epoch_batches(10, 3, seed=1, epoch=3), strict=True)
    )


def test_warmup_rises_then_decays_to_zero():
    scale = linear_warmup_lambda(2, 10)
    assert scale(0) == 0.5
    assert scale(1) == 1.0
    assert scale(2) == 1.0
    assert scale(10) == 0.0
