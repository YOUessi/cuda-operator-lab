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


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Trusted row-wise float32 RMSNorm reference."""
    if x.ndim != 2:
        raise ValueError("rmsnorm expects a 2-D tensor")
    if x.dtype != torch.float32:
        raise TypeError("rmsnorm currently supports float32 only")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight must be 1-D with length equal to x.shape[1]")
    if weight.dtype != torch.float32:
        raise TypeError("rmsnorm weight must be float32")
    if x.shape[1] == 0:
        raise ValueError("rmsnorm requires cols > 0")
    if eps <= 0:
        raise ValueError("eps must be positive")
    mean_square = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Trusted row-wise float32 LayerNorm reference."""
    if x.ndim != 2:
        raise ValueError("layernorm expects a 2-D tensor")
    if x.dtype != torch.float32:
        raise TypeError("layernorm currently supports float32 only")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight must be 1-D with length equal to x.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("bias must be 1-D with length equal to x.shape[1]")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("layernorm weight and bias must be float32")
    if x.shape[1] == 0:
        raise ValueError("layernorm requires cols > 0")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape=(x.shape[1],),
        weight=weight,
        bias=bias,
        eps=eps,
    )
