"""Pick the best available torch device: CUDA (Windows dev box) → MPS (Apple
Silicon demo Mac) → CPU.

Override with the HARVESTAI_DEVICE env var ("cuda", "mps", "cpu"), e.g. to
force CPU on the Mac if an MPS op misbehaves.
"""
from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    forced = os.environ.get("HARVESTAI_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
