"""A small fine-tuning loop that survives being interrupted.

Long jobs on a shared machine get killed, so this writes one rolling checkpoint
that holds model, optimiser, scheduler position and RNG state. Three properties
matter and are covered by tests:

* Saves are atomic. A crash during a save leaves the previous checkpoint intact.
* Tensors are moved to CPU first, so a checkpoint resumes on any device.
* Batch order is a pure function of (seed, epoch), so resuming continues the same
  sequence of batches rather than starting a new shuffle.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PairExample:
    premise: str
    hypothesis: str
    label: int | float


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 16
    warmup_fraction: float = 0.1
    max_length: int = 256
    seed: int = 13
    checkpoint_every: int = 200
    max_grad_norm: float = 1.0


def to_cpu(value):
    """Recursively copy tensors to CPU, leaving other values alone."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, Mapping):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(to_cpu(v) for v in value)
    return value


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_best_model(model: nn.Module, tokenizer, directory: Path) -> None:
    """Replace a saved model directory in one rename, so readers never see half of one."""
    staging = directory.with_name(directory.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    model.save_pretrained(staging, state_dict=to_cpu(model.state_dict()))
    tokenizer.save_pretrained(staging)
    previous = directory.with_name(directory.name + ".previous")
    if directory.exists():
        os.replace(directory, previous)
    os.replace(staging, directory)
    if previous.exists():
        shutil.rmtree(previous)


def epoch_batches(n_examples: int, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    order = np.random.default_rng([seed, epoch]).permutation(n_examples)
    return [order[i : i + batch_size] for i in range(0, n_examples, batch_size)]


def linear_warmup_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        remaining = total_steps - step
        return max(0.0, remaining / max(total_steps - warmup_steps, 1))

    return scale


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decayed, undecayed = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            exempt = parameter.ndim < 2 or "LayerNorm" in name or name.endswith("bias")
            (undecayed if exempt else decayed).append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]


def fit(
    model: nn.Module,
    tokenizer,
    examples: Sequence[PairExample],
    *,
    config: TrainConfig,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    run_dir: Path,
    device: torch.device,
    evaluate: Callable[[nn.Module], float] | None = None,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, object]:
    """Train, keep the best epoch under run_dir/best, and return the training history.

    ``evaluate`` returns a score to maximise. Without it the last epoch is kept.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    steps_per_epoch = math.ceil(len(examples) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    model.to(device)
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, config.weight_decay), lr=config.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        linear_warmup_lambda(int(total_steps * config.warmup_fraction), total_steps),
    )

    step = 0
    best_score = -math.inf
    history: list[dict[str, float]] = []
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        best_score = state["best_score"]
        history = state["history"]
        torch.set_rng_state(state["torch_rng"])
        log(f"resumed from step {step}")
    else:
        torch.manual_seed(config.seed)

    def save_checkpoint() -> None:
        atomic_torch_save(
            {
                "model": to_cpu(model.state_dict()),
                "optimizer": to_cpu(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_score": best_score,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "config": asdict(config),
            },
            checkpoint_path,
        )

    def beat(loss: float) -> None:
        _atomic_write_text(
            run_dir / "progress.json",
            json.dumps(
                {"step": step, "total_steps": total_steps, "loss": loss, "time": time.time()}
            ),
        )

    started = time.monotonic()
    running_loss = 0.0
    running_steps = 0
    first_epoch = step // steps_per_epoch
    for epoch in range(first_epoch, config.epochs):
        batches = epoch_batches(len(examples), config.batch_size, config.seed, epoch)
        model.train()
        for batch in batches[step - epoch * steps_per_epoch :]:
            selected = [examples[int(i)] for i in batch]
            tokens = tokenizer(
                [e.premise for e in selected],
                [e.hypothesis for e in selected],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(device)
            targets = torch.tensor([e.label for e in selected], device=device)
            loss = loss_fn(model(**tokens).logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            value = float(loss.detach().cpu())
            running_loss += value
            running_steps += 1
            if step % 10 == 0:
                beat(value)
            if step % 50 == 0:
                rate = step / max(time.monotonic() - started, 1e-6)
                log(
                    f"step {step}/{total_steps} loss {running_loss / running_steps:.4f} "
                    f"({rate:.2f} steps/s since start of this process)"
                )
                running_loss, running_steps = 0.0, 0
            if config.checkpoint_every and step % config.checkpoint_every == 0:
                save_checkpoint()

        score = evaluate(model) if evaluate else float(epoch)
        history.append({"epoch": epoch + 1, "step": step, "score": score})
        log(f"epoch {epoch + 1}: validation score {score:.4f}")
        if score > best_score:
            best_score = score
            save_best_model(model, tokenizer, run_dir / "best")
        save_checkpoint()
        beat(0.0)

    return {"best_score": best_score, "history": history, "steps": step}
