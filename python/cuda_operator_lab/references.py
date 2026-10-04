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


def row_softmax(x: torch.Tensor) -> torch.Tensor:
    """Trusted row-wise float32 softmax reference."""
    if x.ndim != 2:
        raise ValueError("row_softmax expects a 2-D tensor")
    if x.dtype != torch.float32:
        raise TypeError("row_softmax currently supports float32 only")
    if x.shape[1] == 0:
        raise ValueError("row_softmax requires cols > 0")
    return torch.softmax(x, dim=-1, dtype=torch.float32)
