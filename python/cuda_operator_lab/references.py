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


def fused_residual_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Trusted float32 residual-add + LayerNorm reference."""
    if x.shape != residual.shape or x.ndim != 2:
        raise ValueError("x and residual must have the same 2-D shape")
    if x.dtype != torch.float32 or residual.dtype != torch.float32:
        raise TypeError("fused residual layernorm currently supports float32 only")
    return torch.nn.functional.layer_norm(
        x + residual,
        normalized_shape=(x.shape[1],),
        weight=weight,
        bias=bias,
        eps=eps,
    )


def fused_bias_gelu(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Trusted float32 bias-add + exact GELU reference."""
    if x.ndim != 2:
        raise ValueError("fused_bias_gelu expects a 2-D tensor")
    if bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("bias must be 1-D with length equal to x.shape[1]")
    if x.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("fused_bias_gelu currently supports float32 only")
    return torch.nn.functional.gelu(x + bias, approximate="none")


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Trusted float32 SwiGLU reference."""
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    if gate.dtype != torch.float32 or up.dtype != torch.float32:
        raise TypeError("swiglu currently supports float32 only")
    return torch.nn.functional.silu(gate) * up


def gemm_bias_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Trusted float32 GEMM + bias + exact GELU reference."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    if any(t.dtype != torch.float32 for t in (x, weight, bias)):
        raise TypeError("gemm_bias_gelu currently supports float32 only")
    return torch.nn.functional.gelu(
        torch.matmul(x, weight.transpose(0, 1)) + bias,
        approximate="none",
    )


def gemm_bias_gelu_tanh(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Trusted float32 GEMM + bias + tanh-approximate GELU reference."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    if any(t.dtype != torch.float32 for t in (x, weight, bias)):
        raise TypeError("gemm_bias_gelu_tanh currently supports float32 only")
    return torch.nn.functional.gelu(
        torch.matmul(x, weight.transpose(0, 1)) + bias,
        approximate="tanh",
    )


def gemm_swiglu(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    """Trusted float32 dual-GEMM SwiGLU reference."""
    if x.ndim != 2 or gate_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("x, gate_weight, and up_weight must be 2-D")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape")
    if x.shape[1] != gate_weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if any(t.dtype != torch.float32 for t in (x, gate_weight, up_weight)):
        raise TypeError("gemm_swiglu currently supports float32 only")
    gate = torch.matmul(x, gate_weight.transpose(0, 1))
    up = torch.matmul(x, up_weight.transpose(0, 1))
    return torch.nn.functional.silu(gate) * up


def gemm_swiglu_packed(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Trusted packed single-GEMM SwiGLU reference."""
    if x.ndim != 2 or packed_weight.ndim != 2:
        raise ValueError("x and packed_weight must be 2-D")
    if x.shape[1] != packed_weight.shape[1]:
        raise ValueError("x.shape[1] must equal packed_weight.shape[1]")
    if packed_weight.shape[0] % 2 != 0:
        raise ValueError("packed_weight.shape[0] must be even")
    if x.dtype != torch.float32 or packed_weight.dtype != torch.float32:
        raise TypeError("gemm_swiglu_packed currently supports float32 only")
    packed = torch.matmul(x, packed_weight.transpose(0, 1))
    n = packed_weight.shape[0] // 2
    gate = packed[:, :n]
    up = packed[:, n:]
    return torch.nn.functional.silu(gate) * up


def gemm_swiglu_bf16_fp32_reference(
    x_bf16: torch.Tensor,
    packed_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    """Reference for BF16 inputs with FP32 matmul/post-processing output."""
    if x_bf16.ndim != 2 or packed_weight_bf16.ndim != 2:
        raise ValueError("x_bf16 and packed_weight_bf16 must be 2-D")
    if x_bf16.shape[1] != packed_weight_bf16.shape[1]:
        raise ValueError("inner dimensions must match")
    if packed_weight_bf16.shape[0] % 2 != 0:
        raise ValueError("packed_weight_bf16.shape[0] must be even")
    if x_bf16.dtype != torch.bfloat16 or packed_weight_bf16.dtype != torch.bfloat16:
        raise TypeError("reference expects BF16 tensors")
    packed = torch.matmul(
        x_bf16.float(),
        packed_weight_bf16.float().transpose(0, 1),
    )
    n = packed_weight_bf16.shape[0] // 2
    gate = packed[:, :n]
    up = packed[:, n:]
    return torch.nn.functional.silu(gate) * up


def gemm_swiglu_bf16_workspace_reference(
    x_bf16: torch.Tensor,
    packed_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    """Reference for BF16 GEMM output/workspace with FP32 SwiGLU output."""
    if x_bf16.ndim != 2 or packed_weight_bf16.ndim != 2:
        raise ValueError("x_bf16 and packed_weight_bf16 must be 2-D")
    if x_bf16.shape[1] != packed_weight_bf16.shape[1]:
        raise ValueError("inner dimensions must match")
    if packed_weight_bf16.shape[0] % 2 != 0:
        raise ValueError("packed_weight_bf16.shape[0] must be even")
    if x_bf16.dtype != torch.bfloat16 or packed_weight_bf16.dtype != torch.bfloat16:
        raise TypeError("reference expects BF16 tensors")
    packed_bf16 = torch.matmul(
        x_bf16,
        packed_weight_bf16.transpose(0, 1),
    )
    n = packed_weight_bf16.shape[0] // 2
    gate = packed_bf16[:, :n].float()
    up = packed_bf16[:, n:].float()
    return torch.nn.functional.silu(gate) * up


def gemm_swiglu_bf16_dual_fp32_reference(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    """Reference for BF16 dual projections with FP32 matmul and SwiGLU output."""
    if x_bf16.ndim != 2 or gate_weight_bf16.ndim != 2 or up_weight_bf16.ndim != 2:
        raise ValueError("x and weights must be 2-D")
    if gate_weight_bf16.shape != up_weight_bf16.shape:
        raise ValueError("gate and up weights must have the same shape")
    if x_bf16.shape[1] != gate_weight_bf16.shape[1]:
        raise ValueError("inner dimensions must match")
    if any(t.dtype != torch.bfloat16 for t in (x_bf16, gate_weight_bf16, up_weight_bf16)):
        raise TypeError("reference expects BF16 tensors")
    gate = torch.matmul(
        x_bf16.float(),
        gate_weight_bf16.float().transpose(0, 1),
    )
    up = torch.matmul(
        x_bf16.float(),
        up_weight_bf16.float().transpose(0, 1),
    )
    return torch.nn.functional.silu(gate) * up
