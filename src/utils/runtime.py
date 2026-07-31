"""Device selection and reproducibility controls."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda while failing clearly on unavailable CUDA."""
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    return torch.device(normalized)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    if seed < 0:
        raise ValueError("seed cannot be negative.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
