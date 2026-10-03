"""Trusted PyTorch reference implementations."""

from __future__ import annotations

import torch


def reduction_sum(x: torch.Tensor) -> torch.Tensor:
    """Reference float32 sum for a 1-D CUDA tensor."""
    if x.ndim != 1:
        raise ValueError("reduction_sum expects a 1-D tensor")
    if x.dtype != torch.float32:
        raise TypeError("reduction_sum currently supports float32 only")
    return torch.sum(x, dtype=torch.float32)