"""Device selection, overridable because shared machines need a way to stay off the GPU."""

from __future__ import annotations

import os

import torch


def pick_device(preferred: str | None = None) -> torch.device:
    name = preferred or os.environ.get("CLAIMLENS_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
