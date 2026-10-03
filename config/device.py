"""Pick the best available torch device."""
from __future__ import annotations

import os

import torch


def get_device() -> str:
    """Prefer CUDA, then Apple MPS, else CPU.

    Override with env ``EXTREME_QUANT_DEVICE`` = cuda|mps|cpu|cuda:0 …
    """
    override = os.environ.get("EXTREME_QUANT_DEVICE", "").strip()
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_cuda(device: str | torch.device) -> bool:
    return str(device).startswith("cuda")


def setup_device(device: str | None = None) -> str:
    """Resolve device and apply light CUDA/MPS runtime knobs."""
    device = device or get_device()
    if is_cuda(device):
        torch.backends.cudnn.benchmark = True
        # TF32 is fine for this small classifier and speeds Ampere+ GPUs.
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        idx = torch.cuda.current_device() if device == "cuda" else int(str(device).split(":")[-1])
        name = torch.cuda.get_device_name(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"CUDA device: {name} ({mem:.1f} GiB)")
    return device


def default_num_workers(device: str) -> int:
    """DataLoader workers: CUDA benefits from prefetch; MPS/fork is fragile on macOS."""
    if is_cuda(device):
        return max(int(os.environ.get("EXTREME_QUANT_NUM_WORKERS", "4")), 0)
    return 0
