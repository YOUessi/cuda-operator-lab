"""Thin ctypes bindings for the CUDA operator library.

The CUDA kernels stay independent from PyTorch's C++ ABI. PyTorch tensors provide
CUDA storage and streams, while this module passes their raw device pointers to
the C ABI exported by the shared library.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch


def _default_library_path() -> Path:
    override = os.environ.get("CUDA_OPERATOR_LIB")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "build" / "lib" / "libcuda_operator_lab.so"


class _Library:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise RuntimeError(
                f"CUDA operator library not found at {path}. "
                "Run ./scripts/build.sh first."
            )
        self.handle = ctypes.CDLL(str(path))
        for name in (
            "cuda_operator_reduction_v0",
            "cuda_operator_reduction_v1",
            "cuda_operator_reduction_v2",
            "cuda_operator_reduction_v3",
            "cuda_operator_reduction_v4",
            "cuda_operator_reduction_v5",
        ):
            function = getattr(self.handle, name)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_int

        for name in (
            "cuda_operator_softmax_v0",
            "cuda_operator_softmax_v1",
            "cuda_operator_softmax_v2",
            "cuda_operator_softmax_v3",
            "cuda_operator_softmax_v4",
            "cuda_operator_softmax_v5",
        ):
            function = getattr(self.handle, name)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_int

        for name in (
            "cuda_operator_rmsnorm_v0",
            "cuda_operator_rmsnorm_v1",
            "cuda_operator_rmsnorm_v2",
            "cuda_operator_rmsnorm_v3",
            "cuda_operator_rmsnorm_v4",
        ):
            function = getattr(self.handle, name)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_float,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_int

        for name in (
            "cuda_operator_layernorm_v0",
            "cuda_operator_layernorm_v1",
            "cuda_operator_layernorm_v2",
            "cuda_operator_layernorm_v3",
            "cuda_operator_layernorm_v4",
            "cuda_operator_layernorm_v5",
        ):
            function = getattr(self.handle, name)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_float,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_int

        for name in (
            "cuda_operator_fused_residual_layernorm_v0",
            "cuda_operator_fused_residual_layernorm_v1",
            "cuda_operator_fused_residual_layernorm_v2",
            "cuda_operator_fused_residual_layernorm_v3",
            "cuda_operator_fused_residual_layernorm_v4",
        ):
            function = getattr(self.handle, name)
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_float,
                ctypes.c_void_p,
            ]
            function.restype = ctypes.c_int

        self.handle.cuda_operator_fused_bias_gelu_v0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_fused_bias_gelu_v0.restype = ctypes.c_int

        self.handle.cuda_operator_fused_bias_gelu_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_fused_bias_gelu_v1.restype = ctypes.c_int

        self.handle.cuda_operator_fused_bias_gelu_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_fused_bias_gelu_v2.restype = ctypes.c_int

        self.handle.cuda_operator_swiglu_v0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_swiglu_v0.restype = ctypes.c_int

        self.handle.cuda_operator_swiglu_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_swiglu_v1.restype = ctypes.c_int

        self.handle.cuda_operator_swiglu_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_swiglu_v2.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_bias_gelu_v0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_bias_gelu_v0.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_bias_gelu_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_bias_gelu_v1.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_bias_gelu_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_bias_gelu_v2.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_bias_gelu_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_bias_gelu_v3.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_bias_gelu_v4.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_bias_gelu_v4.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_bias_gelu_v5.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_bias_gelu_v5.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_bias_gelu_v6.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_bias_gelu_v6.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v0.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v1.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v2.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v3.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v4.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v4.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v5.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v5.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v6.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v6.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v7.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v7.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v8.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v9.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v9.restype = ctypes.c_int

        self.handle.cuda_operator_gemm_swiglu_v10.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_gemm_swiglu_v10.restype = ctypes.c_int

        self.handle.cuda_operator_error_string.argtypes = [ctypes.c_int]
        self.handle.cuda_operator_error_string.restype = ctypes.c_char_p

    def check(self, code: int) -> None:
        if code == 0:
            return
        message = self.handle.cuda_operator_error_string(code)
        decoded = (
            message.decode("utf-8", errors="replace")
            if message
            else "unknown CUDA error"
        )
        raise RuntimeError(f"CUDA error {code}: {decoded}")


_LIBRARY: _Library | None = None


def _library() -> _Library:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = _Library(_default_library_path())
    return _LIBRARY


def _validate_reduction_input(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("reduction input must be a CUDA tensor")
    if x.ndim != 1:
        raise ValueError("reduction input must be 1-D")
    if x.dtype != torch.float32:
        raise TypeError("reduction currently supports float32 only")
    if not x.is_contiguous():
        raise ValueError("reduction input must be contiguous")


def _validate_reduction_output(x: torch.Tensor, out: torch.Tensor) -> None:
    if not out.is_cuda or out.device != x.device:
        raise ValueError("output must be a CUDA tensor on the same device")
    if out.dtype != torch.float32 or out.numel() != 1 or not out.is_contiguous():
        raise ValueError("output must be one contiguous float32 CUDA value")


def _reduction_into(
    symbol: str,
    x: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _validate_reduction_input(x)
    _validate_reduction_output(x, out)

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    function = getattr(library.handle, symbol)
    code = function(
        ctypes.c_void_p(x.data_ptr()) if x.numel() else ctypes.c_void_p(),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(x.numel()),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def _allocate_scalar(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(1, device=x.device, dtype=torch.float32)


def reduction_v0_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch the V0 single-thread serial reduction into a preallocated scalar."""
    return _reduction_into("cuda_operator_reduction_v0", x, out)


def reduction_v0(x: torch.Tensor) -> torch.Tensor:
    return reduction_v0_into(x, _allocate_scalar(x))


def reduction_v1_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V1: grid-stride local sums plus one global atomic per thread."""
    return _reduction_into("cuda_operator_reduction_v1", x, out)


def reduction_v1(x: torch.Tensor) -> torch.Tensor:
    return reduction_v1_into(x, _allocate_scalar(x))


def reduction_v2_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V2: shared-memory block reduction plus one atomic per block."""
    return _reduction_into("cuda_operator_reduction_v2", x, out)


def reduction_v2(x: torch.Tensor) -> torch.Tensor:
    return reduction_v2_into(x, _allocate_scalar(x))


def reduction_v3_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V3: warp-shuffle block reduction plus one atomic per block."""
    return _reduction_into("cuda_operator_reduction_v3", x, out)


def reduction_v3(x: torch.Tensor) -> torch.Tensor:
    return reduction_v3_into(x, _allocate_scalar(x))


def reduction_v4_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V4: float4 loads when 16-byte aligned, otherwise fall back to V3."""
    return _reduction_into("cuda_operator_reduction_v4", x, out)


def reduction_v4(x: torch.Tensor) -> torch.Tensor:
    return reduction_v4_into(x, _allocate_scalar(x))


def reduction_v5_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V5: V4 float4 kernel with vector-work-based launch geometry."""
    return _reduction_into("cuda_operator_reduction_v5", x, out)


def reduction_v5(x: torch.Tensor) -> torch.Tensor:
    return reduction_v5_into(x, _allocate_scalar(x))


def _validate_softmax_input(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError("softmax input must be a CUDA tensor")
    if x.ndim != 2:
        raise ValueError("softmax input must be 2-D [rows, cols]")
    if x.dtype != torch.float32:
        raise TypeError("softmax currently supports float32 only")
    if not x.is_contiguous():
        raise ValueError("softmax input must be contiguous")
    if x.shape[1] == 0:
        raise ValueError("softmax requires cols > 0")


def _validate_softmax_output(x: torch.Tensor, out: torch.Tensor) -> None:
    if not out.is_cuda or out.device != x.device:
        raise ValueError("softmax output must be a CUDA tensor on the same device")
    if out.dtype != torch.float32 or out.shape != x.shape or not out.is_contiguous():
        raise ValueError(
            "softmax output must be contiguous float32 with the same shape"
        )


def _softmax_into(
    symbol: str,
    x: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _validate_softmax_input(x)
    _validate_softmax_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    function = getattr(library.handle, symbol)
    code = function(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def softmax_v0_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V0: one CUDA thread serially computes one row."""
    return _softmax_into("cuda_operator_softmax_v0", x, out)


def softmax_v0(x: torch.Tensor) -> torch.Tensor:
    """Return row-wise Softmax V0 output."""
    out = torch.empty_like(x)
    return softmax_v0_into(x, out)


def softmax_v1_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V1: one CUDA block cooperatively computes one row."""
    return _softmax_into("cuda_operator_softmax_v1", x, out)


def softmax_v1(x: torch.Tensor) -> torch.Tensor:
    """Return row-wise Softmax V1 output."""
    out = torch.empty_like(x)
    return softmax_v1_into(x, out)


def softmax_v2_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V2: one block per row with warp-shuffle max/sum reductions."""
    return _softmax_into("cuda_operator_softmax_v2", x, out)


def softmax_v2(x: torch.Tensor) -> torch.Tensor:
    """Return row-wise Softmax V2 output."""
    out = torch.empty_like(x)
    return softmax_v2_into(x, out)


def softmax_v3_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V3: V2 warp reductions with row-width-aware block sizing."""
    return _softmax_into("cuda_operator_softmax_v3", x, out)


def softmax_v3(x: torch.Tensor) -> torch.Tensor:
    """Return row-wise Softmax V3 output."""
    out = torch.empty_like(x)
    return softmax_v3_into(x, out)


def softmax_v4_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V4: pack up to eight narrow rows into one 256-thread block."""
    return _softmax_into("cuda_operator_softmax_v4", x, out)


def softmax_v4(x: torch.Tensor) -> torch.Tensor:
    """Return row-wise Softmax V4 output."""
    out = torch.empty_like(x)
    return softmax_v4_into(x, out)


def softmax_v5_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch V5: empirically dispatch between V3 and V4 by shape."""
    return _softmax_into("cuda_operator_softmax_v5", x, out)


def softmax_v5(x: torch.Tensor) -> torch.Tensor:
    """Return row-wise Softmax V5 output."""
    out = torch.empty_like(x)
    return softmax_v5_into(x, out)


def _validate_rmsnorm_input(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("rmsnorm input and weight must be CUDA tensors")
    if x.ndim != 2:
        raise ValueError("rmsnorm input must be 2-D [rows, cols]")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("rmsnorm weight must be 1-D with length equal to cols")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("rmsnorm currently supports float32 only")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("rmsnorm input and weight must be contiguous")
    if x.shape[1] == 0:
        raise ValueError("rmsnorm requires cols > 0")
    if eps <= 0:
        raise ValueError("rmsnorm eps must be positive")


def _validate_rmsnorm_output(x: torch.Tensor, out: torch.Tensor) -> None:
    if not out.is_cuda or out.device != x.device:
        raise ValueError("rmsnorm output must be a CUDA tensor on the same device")
    if out.dtype != torch.float32 or out.shape != x.shape or not out.is_contiguous():
        raise ValueError(
            "rmsnorm output must be contiguous float32 with the same shape"
        )


def rmsnorm_v0_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V0: one CUDA thread serially computes one RMSNorm row."""
    _validate_rmsnorm_input(x, weight, eps)
    _validate_rmsnorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_rmsnorm_v0(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def rmsnorm_v0(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return row-wise RMSNorm V0 output."""
    out = torch.empty_like(x)
    return rmsnorm_v0_into(x, weight, out, eps)


def rmsnorm_v1_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V1: one block per row with shared-memory sum-square reduction."""
    _validate_rmsnorm_input(x, weight, eps)
    _validate_rmsnorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_rmsnorm_v1(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def rmsnorm_v1(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return row-wise RMSNorm V1 output."""
    out = torch.empty_like(x)
    return rmsnorm_v1_into(x, weight, out, eps)


def rmsnorm_v2_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V2: warp-shuffle sum-square reduction per row."""
    _validate_rmsnorm_input(x, weight, eps)
    _validate_rmsnorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_rmsnorm_v2(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def rmsnorm_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return row-wise RMSNorm V2 output."""
    out = torch.empty_like(x)
    return rmsnorm_v2_into(x, weight, out, eps)


def rmsnorm_v3_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V3: aligned float4 IO with V2 scalar fallback."""
    _validate_rmsnorm_input(x, weight, eps)
    _validate_rmsnorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_rmsnorm_v3(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def rmsnorm_v3(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return row-wise RMSNorm V3 output."""
    out = torch.empty_like(x)
    return rmsnorm_v3_into(x, weight, out, eps)


def rmsnorm_v4_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V4: empirical dispatcher between V2 and V3."""
    _validate_rmsnorm_input(x, weight, eps)
    _validate_rmsnorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_rmsnorm_v4(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def rmsnorm_v4(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return rmsnorm_v4_into(x, weight, out, eps)

def _validate_layernorm_input(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> None:
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("layernorm input, weight, and bias must be CUDA tensors")
    if x.ndim != 2:
        raise ValueError("layernorm input must be 2-D [rows, cols]")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("layernorm weight must be 1-D with length equal to cols")
    if bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("layernorm bias must be 1-D with length equal to cols")
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("layernorm currently supports float32 only")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("layernorm input, weight, and bias must be contiguous")
    if x.shape[1] == 0:
        raise ValueError("layernorm requires cols > 0")
    if eps <= 0:
        raise ValueError("layernorm eps must be positive")


def _validate_layernorm_output(x: torch.Tensor, out: torch.Tensor) -> None:
    if not out.is_cuda or out.device != x.device:
        raise ValueError("layernorm output must be a CUDA tensor on the same device")
    if out.dtype != torch.float32 or out.shape != x.shape or not out.is_contiguous():
        raise ValueError(
            "layernorm output must be contiguous float32 with the same shape"
        )


def layernorm_v0_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V0: one CUDA thread serially computes one LayerNorm row."""
    _validate_layernorm_input(x, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_layernorm_v0(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def layernorm_v0(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return layernorm_v0_into(x, weight, bias, out, eps)


def layernorm_v1_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V1: one block per row with shared-memory mean/variance reductions."""
    _validate_layernorm_input(x, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_layernorm_v1(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def layernorm_v1(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return layernorm_v1_into(x, weight, bias, out, eps)


def layernorm_v2_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V2: warp-shuffle reductions for mean and variance."""
    _validate_layernorm_input(x, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_layernorm_v2(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def layernorm_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return layernorm_v2_into(x, weight, bias, out, eps)


def layernorm_v3_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V3: one-pass Welford statistics with warp-level combination."""
    _validate_layernorm_input(x, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_layernorm_v3(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def layernorm_v3(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return layernorm_v3_into(x, weight, bias, out, eps)


def layernorm_v4_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V4: aligned float4 IO with V2 scalar fallback."""
    _validate_layernorm_input(x, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_layernorm_v4(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def layernorm_v4(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return layernorm_v4_into(x, weight, bias, out, eps)


def layernorm_v5_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Launch V5: profiled dispatcher between V2 and V4."""
    _validate_layernorm_input(x, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_layernorm_v5(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def layernorm_v5(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return layernorm_v5_into(x, weight, bias, out, eps)


def _validate_fused_residual_layernorm_input(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> None:
    if not x.is_cuda or not residual.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("all fused residual layernorm inputs must be CUDA tensors")
    if x.ndim != 2 or residual.shape != x.shape:
        raise ValueError("x and residual must have the same 2-D shape")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight must be 1-D with length equal to cols")
    if bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("bias must be 1-D with length equal to cols")
    if any(t.dtype != torch.float32 for t in (x, residual, weight, bias)):
        raise TypeError("fused residual layernorm currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, residual, weight, bias)):
        raise ValueError("all fused residual layernorm inputs must be contiguous")
    if x.shape[1] == 0:
        raise ValueError("fused residual layernorm requires cols > 0")
    if eps <= 0:
        raise ValueError("eps must be positive")


def fused_residual_layernorm_v0_into(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    _validate_fused_residual_layernorm_input(x, residual, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_fused_residual_layernorm_v0(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def fused_residual_layernorm_v0(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return fused_residual_layernorm_v0_into(x, residual, weight, bias, out, eps)


def fused_residual_layernorm_v1_into(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    _validate_fused_residual_layernorm_input(x, residual, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_fused_residual_layernorm_v1(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def fused_residual_layernorm_v1(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return fused_residual_layernorm_v1_into(x, residual, weight, bias, out, eps)


def fused_residual_layernorm_v2_into(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    _validate_fused_residual_layernorm_input(x, residual, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_fused_residual_layernorm_v2(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def fused_residual_layernorm_v2(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return fused_residual_layernorm_v2_into(x, residual, weight, bias, out, eps)


def fused_residual_layernorm_v3_into(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    _validate_fused_residual_layernorm_input(x, residual, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_fused_residual_layernorm_v3(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def fused_residual_layernorm_v3(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return fused_residual_layernorm_v3_into(x, residual, weight, bias, out, eps)


def fused_residual_layernorm_v4_into(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    _validate_fused_residual_layernorm_input(x, residual, weight, bias, eps)
    _validate_layernorm_output(x, out)

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_fused_residual_layernorm_v4(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(residual.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_float(eps),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def fused_residual_layernorm_v4(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return fused_residual_layernorm_v4_into(x, residual, weight, bias, out, eps)


def fused_bias_gelu_v0_into(
    x: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda or not bias.is_cuda or not out.is_cuda:
        raise ValueError("fused bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("expected x[rows, cols] and bias[cols]")
    if x.dtype != torch.float32 or bias.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("fused bias GELU currently supports float32 only")
    if out.shape != x.shape:
        raise ValueError("output shape must equal input shape")
    if not x.is_contiguous() or not bias.is_contiguous() or not out.is_contiguous():
        raise ValueError("all tensors must be contiguous")
    if x.shape[1] == 0:
        raise ValueError("cols must be > 0")

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_fused_bias_gelu_v0(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def fused_bias_gelu_v0(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return fused_bias_gelu_v0_into(x, bias, out)


def fused_bias_gelu_v1_into(
    x: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda or not bias.is_cuda or not out.is_cuda:
        raise ValueError("fused bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("expected x[rows, cols] and bias[cols]")
    if x.dtype != torch.float32 or bias.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("fused bias GELU currently supports float32 only")
    if out.shape != x.shape:
        raise ValueError("output shape must equal input shape")
    if not x.is_contiguous() or not bias.is_contiguous() or not out.is_contiguous():
        raise ValueError("all tensors must be contiguous")
    if x.shape[1] == 0:
        raise ValueError("cols must be > 0")

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_fused_bias_gelu_v1(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def fused_bias_gelu_v1(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return fused_bias_gelu_v1_into(x, bias, out)


def fused_bias_gelu_v2_into(
    x: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not x.is_cuda or not bias.is_cuda or not out.is_cuda:
        raise ValueError("fused bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError("expected x[rows, cols] and bias[cols]")
    if x.dtype != torch.float32 or bias.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("fused bias GELU currently supports float32 only")
    if out.shape != x.shape:
        raise ValueError("output shape must equal input shape")
    if not x.is_contiguous() or not bias.is_contiguous() or not out.is_contiguous():
        raise ValueError("all tensors must be contiguous")
    if x.shape[1] == 0:
        raise ValueError("cols must be > 0")

    rows, cols = x.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_fused_bias_gelu_v2(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def fused_bias_gelu_v2(
    x: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    return fused_bias_gelu_v2_into(x, bias, out)


def swiglu_v0_into(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not gate.is_cuda or not up.is_cuda or not out.is_cuda:
        raise ValueError("SwiGLU tensors must be CUDA tensors")
    if gate.shape != up.shape or out.shape != gate.shape:
        raise ValueError("gate, up, and output must have the same shape")
    if gate.dtype != torch.float32 or up.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("SwiGLU currently supports float32 only")
    if not gate.is_contiguous() or not up.is_contiguous() or not out.is_contiguous():
        raise ValueError("SwiGLU tensors must be contiguous")

    if gate.numel() == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=gate.device)
    code = library.handle.cuda_operator_swiglu_v0(
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_void_p(up.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(gate.numel()),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def swiglu_v0(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(gate)
    return swiglu_v0_into(gate, up, out)


def swiglu_v1_into(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not gate.is_cuda or not up.is_cuda or not out.is_cuda:
        raise ValueError("SwiGLU tensors must be CUDA tensors")
    if gate.shape != up.shape or out.shape != gate.shape:
        raise ValueError("gate, up, and output must have the same shape")
    if gate.dtype != torch.float32 or up.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("SwiGLU currently supports float32 only")
    if not gate.is_contiguous() or not up.is_contiguous() or not out.is_contiguous():
        raise ValueError("SwiGLU tensors must be contiguous")

    if gate.numel() == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=gate.device)
    code = library.handle.cuda_operator_swiglu_v1(
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_void_p(up.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(gate.numel()),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def swiglu_v1(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(gate)
    return swiglu_v1_into(gate, up, out)


def swiglu_v2_into(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not gate.is_cuda or not up.is_cuda or not out.is_cuda:
        raise ValueError("SwiGLU tensors must be CUDA tensors")
    if gate.shape != up.shape or out.shape != gate.shape:
        raise ValueError("gate, up, and output must have the same shape")
    if gate.dtype != torch.float32 or up.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("SwiGLU currently supports float32 only")
    if not gate.is_contiguous() or not up.is_contiguous() or not out.is_contiguous():
        raise ValueError("SwiGLU tensors must be contiguous")
    if gate.ndim != 2:
        raise ValueError("SwiGLU V2 expects a 2-D tensor")

    rows, cols = gate.shape
    if rows == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=gate.device)
    code = library.handle.cuda_operator_swiglu_v2(
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_void_p(up.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(rows),
        ctypes.c_uint64(cols),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def swiglu_v2(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(gate)
    return swiglu_v2_into(gate, up, out)


def gemm_bias_gelu_v0_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, weight, bias, out)):
        raise ValueError("GEMM Bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    expected_shape = (x.shape[0], weight.shape[0])
    if out.shape != expected_shape:
        raise ValueError(f"output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, weight, bias, out)):
        raise TypeError("GEMM Bias GELU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, weight, bias, out)):
        raise ValueError("all GEMM Bias GELU tensors must be contiguous")

    m, k = x.shape
    n = weight.shape[0]
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_bias_gelu_v0(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_bias_gelu_v0(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], weight.shape[0]),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_bias_gelu_v0_into(x, weight, bias, out)


def gemm_bias_gelu_v1_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, weight, bias, out)):
        raise ValueError("GEMM Bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    expected_shape = (x.shape[0], weight.shape[0])
    if out.shape != expected_shape:
        raise ValueError(f"output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, weight, bias, out)):
        raise TypeError("GEMM Bias GELU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, weight, bias, out)):
        raise ValueError("all GEMM Bias GELU tensors must be contiguous")

    m, k = x.shape
    n = weight.shape[0]
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_bias_gelu_v1(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_bias_gelu_v1(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], weight.shape[0]),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_bias_gelu_v1_into(x, weight, bias, out)


def gemm_bias_gelu_v2_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, weight, bias, out)):
        raise ValueError("GEMM Bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    expected_shape = (x.shape[0], weight.shape[0])
    if out.shape != expected_shape:
        raise ValueError(f"output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, weight, bias, out)):
        raise TypeError("GEMM Bias GELU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, weight, bias, out)):
        raise ValueError("all GEMM Bias GELU tensors must be contiguous")

    m, k = x.shape
    n = weight.shape[0]
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_bias_gelu_v2(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_bias_gelu_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], weight.shape[0]),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_bias_gelu_v2_into(x, weight, bias, out)


def gemm_bias_gelu_v3_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, weight, bias, out)):
        raise ValueError("GEMM Bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    expected_shape = (x.shape[0], weight.shape[0])
    if out.shape != expected_shape:
        raise ValueError(f"output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, weight, bias, out)):
        raise TypeError("GEMM Bias GELU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, weight, bias, out)):
        raise ValueError("all GEMM Bias GELU tensors must be contiguous")

    m, k = x.shape
    n = weight.shape[0]
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_bias_gelu_v3(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_bias_gelu_v3(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], weight.shape[0]),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_bias_gelu_v3_into(x, weight, bias, out)


def gemm_bias_gelu_v4_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, weight, bias, out)):
        raise ValueError("GEMM Bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    expected_shape = (x.shape[0], weight.shape[0])
    if out.shape != expected_shape:
        raise ValueError(f"output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, weight, bias, out)):
        raise TypeError("GEMM Bias GELU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, weight, bias, out)):
        raise ValueError("all GEMM Bias GELU tensors must be contiguous")

    m, k = x.shape
    n = weight.shape[0]
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_bias_gelu_v4(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_bias_gelu_v4(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], weight.shape[0]),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_bias_gelu_v4_into(x, weight, bias, out)


def gemm_bias_gelu_v5_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, weight, bias, out)):
        raise ValueError("GEMM Bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    expected_shape = (x.shape[0], weight.shape[0])
    if out.shape != expected_shape:
        raise ValueError(f"output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, weight, bias, out)):
        raise TypeError("GEMM Bias GELU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, weight, bias, out)):
        raise ValueError("all GEMM Bias GELU tensors must be contiguous")

    m, k = x.shape
    n = weight.shape[0]
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_bias_gelu_v5(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_bias_gelu_v5(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], weight.shape[0]),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_bias_gelu_v5_into(x, weight, bias, out)


def gemm_bias_gelu_v6_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, weight, bias, out)):
        raise ValueError("GEMM Bias GELU tensors must be CUDA tensors")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and weight must be 2-D")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("bias length must equal weight.shape[0]")
    expected_shape = (x.shape[0], weight.shape[0])
    if out.shape != expected_shape:
        raise ValueError(f"output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, weight, bias, out)):
        raise TypeError("GEMM Bias GELU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, weight, bias, out)):
        raise ValueError("all GEMM Bias GELU tensors must be contiguous")

    m, k = x.shape
    n = weight.shape[0]
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_bias_gelu_v6(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_bias_gelu_v6(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0], weight.shape[0]),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_bias_gelu_v6_into(x, weight, bias, out)


def gemm_swiglu_v0_into(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, gate_weight, up_weight, workspace, out)):
        raise ValueError("GEMM SwiGLU tensors must be CUDA tensors")
    if x.ndim != 2 or gate_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("x and weights must be 2-D")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape")
    if x.shape[1] != gate_weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    expected_shape = (x.shape[0], gate_weight.shape[0])
    if workspace.shape != expected_shape or out.shape != expected_shape:
        raise ValueError(f"workspace and output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, gate_weight, up_weight, workspace, out)):
        raise TypeError("GEMM SwiGLU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, gate_weight, up_weight, workspace, out)):
        raise ValueError("all GEMM SwiGLU tensors must be contiguous")

    m, k = x.shape
    n = gate_weight.shape[0]
    if m == 0 or n == 0:
        return out

    if workspace.data_ptr() == out.data_ptr():
        raise ValueError("workspace and output must not alias")

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_swiglu_v0(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(gate_weight.data_ptr()),
        ctypes.c_void_p(up_weight.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v0(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    shape = (x.shape[0], gate_weight.shape[0])
    workspace = torch.empty(shape, device=x.device, dtype=torch.float32)
    out = torch.empty(shape, device=x.device, dtype=torch.float32)
    return gemm_swiglu_v0_into(x, gate_weight, up_weight, workspace, out)


def gemm_swiglu_v1_into(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, gate_weight, up_weight, workspace, out)):
        raise ValueError("GEMM SwiGLU tensors must be CUDA tensors")
    if x.ndim != 2 or gate_weight.ndim != 2 or up_weight.ndim != 2:
        raise ValueError("x and weights must be 2-D")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape")
    if x.shape[1] != gate_weight.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    expected_shape = (x.shape[0], gate_weight.shape[0])
    if workspace.shape != expected_shape or out.shape != expected_shape:
        raise ValueError(f"workspace and output shape must be {expected_shape}")
    if any(t.dtype != torch.float32 for t in (x, gate_weight, up_weight, workspace, out)):
        raise TypeError("GEMM SwiGLU currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, gate_weight, up_weight, workspace, out)):
        raise ValueError("all GEMM SwiGLU tensors must be contiguous")

    m, k = x.shape
    n = gate_weight.shape[0]
    if m == 0 or n == 0:
        return out
    if workspace.data_ptr() == out.data_ptr():
        raise ValueError("workspace and output must not alias")

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_swiglu_v1(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(gate_weight.data_ptr()),
        ctypes.c_void_p(up_weight.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v1(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
) -> torch.Tensor:
    shape = (x.shape[0], gate_weight.shape[0])
    workspace = torch.empty(shape, device=x.device, dtype=torch.float32)
    out = torch.empty(shape, device=x.device, dtype=torch.float32)
    return gemm_swiglu_v1_into(x, gate_weight, up_weight, workspace, out)


def gemm_swiglu_v2_into(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, packed_weight, workspace, out)):
        raise ValueError("GEMM SwiGLU V2 tensors must be CUDA tensors")
    if x.ndim != 2 or packed_weight.ndim != 2:
        raise ValueError("x and packed_weight must be 2-D")
    if x.shape[1] != packed_weight.shape[1]:
        raise ValueError("x.shape[1] must equal packed_weight.shape[1]")
    if packed_weight.shape[0] % 2 != 0:
        raise ValueError("packed_weight.shape[0] must be even")
    n = packed_weight.shape[0] // 2
    expected_workspace = (x.shape[0], 2 * n)
    expected_out = (x.shape[0], n)
    if workspace.shape != expected_workspace:
        raise ValueError(f"workspace shape must be {expected_workspace}")
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if any(t.dtype != torch.float32 for t in (x, packed_weight, workspace, out)):
        raise TypeError("GEMM SwiGLU V2 currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, packed_weight, workspace, out)):
        raise ValueError("all GEMM SwiGLU V2 tensors must be contiguous")

    m, k = x.shape
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_swiglu_v2(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(packed_weight.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v2(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    n = packed_weight.shape[0] // 2
    workspace = torch.empty(
        (x.shape[0], 2 * n),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty(
        (x.shape[0], n),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v2_into(x, packed_weight, workspace, out)


def gemm_swiglu_v3_into(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if not all(t.is_cuda for t in (x, packed_weight, workspace, out)):
        raise ValueError("GEMM SwiGLU V3 tensors must be CUDA tensors")
    if x.ndim != 2 or packed_weight.ndim != 2:
        raise ValueError("x and packed_weight must be 2-D")
    if x.shape[1] != packed_weight.shape[1]:
        raise ValueError("x.shape[1] must equal packed_weight.shape[1]")
    if packed_weight.shape[0] % 2 != 0:
        raise ValueError("packed_weight.shape[0] must be even")
    n = packed_weight.shape[0] // 2
    expected_workspace = (x.shape[0], 2 * n)
    expected_out = (x.shape[0], n)
    if workspace.shape != expected_workspace:
        raise ValueError(f"workspace shape must be {expected_workspace}")
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if any(t.dtype != torch.float32 for t in (x, packed_weight, workspace, out)):
        raise TypeError("GEMM SwiGLU V3 currently supports float32 only")
    if not all(t.is_contiguous() for t in (x, packed_weight, workspace, out)):
        raise ValueError("all GEMM SwiGLU V3 tensors must be contiguous")

    m, k = x.shape
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x.device)
    code = library.handle.cuda_operator_gemm_swiglu_v3(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(packed_weight.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v3(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    n = packed_weight.shape[0] // 2
    workspace = torch.empty(
        (x.shape[0], 2 * n),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty(
        (x.shape[0], n),
        device=x.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v3_into(x, packed_weight, workspace, out)


def gemm_swiglu_v4_into(
    x_bf16: torch.Tensor,
    packed_weight_bf16: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """BF16 Tensor Core packed GEMM with FP32 accumulation/output and SwiGLU."""
    if not all(t.is_cuda for t in (x_bf16, packed_weight_bf16, workspace, out)):
        raise ValueError("GEMM SwiGLU V4 tensors must be CUDA tensors")
    if x_bf16.ndim != 2 or packed_weight_bf16.ndim != 2:
        raise ValueError("x_bf16 and packed_weight_bf16 must be 2-D")
    if x_bf16.shape[1] != packed_weight_bf16.shape[1]:
        raise ValueError("x_bf16.shape[1] must equal packed_weight_bf16.shape[1]")
    if packed_weight_bf16.shape[0] % 2 != 0:
        raise ValueError("packed_weight_bf16.shape[0] must be even")
    if x_bf16.dtype != torch.bfloat16 or packed_weight_bf16.dtype != torch.bfloat16:
        raise TypeError("GEMM SwiGLU V4 expects BF16 input and packed weight")
    if workspace.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("GEMM SwiGLU V4 workspace and output must be float32")
    n = packed_weight_bf16.shape[0] // 2
    expected_workspace = (x_bf16.shape[0], 2 * n)
    expected_out = (x_bf16.shape[0], n)
    if workspace.shape != expected_workspace:
        raise ValueError(f"workspace shape must be {expected_workspace}")
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if not all(t.is_contiguous() for t in (x_bf16, packed_weight_bf16, workspace, out)):
        raise ValueError("all GEMM SwiGLU V4 tensors must be contiguous")

    m, k = x_bf16.shape
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x_bf16.device)
    code = library.handle.cuda_operator_gemm_swiglu_v4(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(packed_weight_bf16.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v4(
    x_bf16: torch.Tensor,
    packed_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    n = packed_weight_bf16.shape[0] // 2
    workspace = torch.empty(
        (x_bf16.shape[0], 2 * n),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    out = torch.empty(
        (x_bf16.shape[0], n),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v4_into(x_bf16, packed_weight_bf16, workspace, out)


def gemm_swiglu_v5_into(
    x_bf16: torch.Tensor,
    packed_weight_bf16: torch.Tensor,
    workspace_bf16: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """BF16 Tensor Core packed GEMM with BF16 workspace and FP32 final output."""
    if not all(t.is_cuda for t in (x_bf16, packed_weight_bf16, workspace_bf16, out)):
        raise ValueError("GEMM SwiGLU V5 tensors must be CUDA tensors")
    if x_bf16.ndim != 2 or packed_weight_bf16.ndim != 2:
        raise ValueError("x_bf16 and packed_weight_bf16 must be 2-D")
    if x_bf16.shape[1] != packed_weight_bf16.shape[1]:
        raise ValueError("x_bf16.shape[1] must equal packed_weight_bf16.shape[1]")
    if packed_weight_bf16.shape[0] % 2 != 0:
        raise ValueError("packed_weight_bf16.shape[0] must be even")
    if x_bf16.dtype != torch.bfloat16 or packed_weight_bf16.dtype != torch.bfloat16:
        raise TypeError("GEMM SwiGLU V5 expects BF16 input and packed weight")
    if workspace_bf16.dtype != torch.bfloat16:
        raise TypeError("GEMM SwiGLU V5 workspace must be BF16")
    if out.dtype != torch.float32:
        raise TypeError("GEMM SwiGLU V5 output must be float32")
    n = packed_weight_bf16.shape[0] // 2
    expected_workspace = (x_bf16.shape[0], 2 * n)
    expected_out = (x_bf16.shape[0], n)
    if workspace_bf16.shape != expected_workspace:
        raise ValueError(f"workspace shape must be {expected_workspace}")
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if not all(t.is_contiguous() for t in (x_bf16, packed_weight_bf16, workspace_bf16, out)):
        raise ValueError("all GEMM SwiGLU V5 tensors must be contiguous")

    m, k = x_bf16.shape
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x_bf16.device)
    code = library.handle.cuda_operator_gemm_swiglu_v5(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(packed_weight_bf16.data_ptr()),
        ctypes.c_void_p(workspace_bf16.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v5(
    x_bf16: torch.Tensor,
    packed_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    n = packed_weight_bf16.shape[0] // 2
    workspace = torch.empty(
        (x_bf16.shape[0], 2 * n),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty(
        (x_bf16.shape[0], n),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v5_into(x_bf16, packed_weight_bf16, workspace, out)


def gemm_swiglu_v6_into(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Custom WMMA fused dual-projection SwiGLU baseline."""
    if not all(t.is_cuda for t in (x_bf16, gate_weight_bf16, up_weight_bf16, out)):
        raise ValueError("GEMM SwiGLU V6 tensors must be CUDA tensors")
    if x_bf16.ndim != 2 or gate_weight_bf16.ndim != 2 or up_weight_bf16.ndim != 2:
        raise ValueError("x and weights must be 2-D")
    if gate_weight_bf16.shape != up_weight_bf16.shape:
        raise ValueError("gate and up weights must have the same shape")
    if x_bf16.shape[1] != gate_weight_bf16.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if any(t.dtype != torch.bfloat16 for t in (x_bf16, gate_weight_bf16, up_weight_bf16)):
        raise TypeError("GEMM SwiGLU V6 expects BF16 input and weights")
    if out.dtype != torch.float32:
        raise TypeError("GEMM SwiGLU V6 output must be float32")
    expected_out = (x_bf16.shape[0], gate_weight_bf16.shape[0])
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if not all(t.is_contiguous() for t in (x_bf16, gate_weight_bf16, up_weight_bf16, out)):
        raise ValueError("all GEMM SwiGLU V6 tensors must be contiguous")

    m, k = x_bf16.shape
    n = gate_weight_bf16.shape[0]
    if m == 0 or n == 0:
        return out
    if (m % 16) != 0 or (k % 16) != 0 or (n % 16) != 0:
        raise ValueError("GEMM SwiGLU V6 requires M, K, and N to be multiples of 16")

    library = _library()
    stream = torch.cuda.current_stream(device=x_bf16.device)
    code = library.handle.cuda_operator_gemm_swiglu_v6(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(gate_weight_bf16.data_ptr()),
        ctypes.c_void_p(up_weight_bf16.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v6(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x_bf16.shape[0], gate_weight_bf16.shape[0]),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v6_into(
        x_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        out,
    )


def gemm_swiglu_v7_into(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """4-warp WMMA fused SwiGLU with shared A-tile reuse."""
    if not all(t.is_cuda for t in (x_bf16, gate_weight_bf16, up_weight_bf16, out)):
        raise ValueError("GEMM SwiGLU V7 tensors must be CUDA tensors")
    if x_bf16.ndim != 2 or gate_weight_bf16.ndim != 2 or up_weight_bf16.ndim != 2:
        raise ValueError("x and weights must be 2-D")
    if gate_weight_bf16.shape != up_weight_bf16.shape:
        raise ValueError("gate and up weights must have the same shape")
    if x_bf16.shape[1] != gate_weight_bf16.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if any(t.dtype != torch.bfloat16 for t in (x_bf16, gate_weight_bf16, up_weight_bf16)):
        raise TypeError("GEMM SwiGLU V7 expects BF16 input and weights")
    if out.dtype != torch.float32:
        raise TypeError("GEMM SwiGLU V7 output must be float32")
    expected_out = (x_bf16.shape[0], gate_weight_bf16.shape[0])
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if not all(t.is_contiguous() for t in (x_bf16, gate_weight_bf16, up_weight_bf16, out)):
        raise ValueError("all GEMM SwiGLU V7 tensors must be contiguous")

    m, k = x_bf16.shape
    n = gate_weight_bf16.shape[0]
    if m == 0 or n == 0:
        return out
    if (m % 16) != 0 or (k % 16) != 0 or (n % 64) != 0:
        raise ValueError(
            "GEMM SwiGLU V7 requires M and K multiples of 16 and N multiple of 64"
        )

    library = _library()
    stream = torch.cuda.current_stream(device=x_bf16.device)
    code = library.handle.cuda_operator_gemm_swiglu_v7(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(gate_weight_bf16.data_ptr()),
        ctypes.c_void_p(up_weight_bf16.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v7(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x_bf16.shape[0], gate_weight_bf16.shape[0]),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v7_into(
        x_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        out,
    )


def gemm_swiglu_v8_into(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """8-warp WMMA fused SwiGLU with shared A/B tile reuse."""
    if not all(t.is_cuda for t in (x_bf16, gate_weight_bf16, up_weight_bf16, out)):
        raise ValueError("GEMM SwiGLU V8 tensors must be CUDA tensors")
    if x_bf16.ndim != 2 or gate_weight_bf16.ndim != 2 or up_weight_bf16.ndim != 2:
        raise ValueError("x and weights must be 2-D")
    if gate_weight_bf16.shape != up_weight_bf16.shape:
        raise ValueError("gate and up weights must have the same shape")
    if x_bf16.shape[1] != gate_weight_bf16.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if any(t.dtype != torch.bfloat16 for t in (x_bf16, gate_weight_bf16, up_weight_bf16)):
        raise TypeError("GEMM SwiGLU V8 expects BF16 input and weights")
    if out.dtype != torch.float32:
        raise TypeError("GEMM SwiGLU V8 output must be float32")
    expected_out = (x_bf16.shape[0], gate_weight_bf16.shape[0])
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if not all(t.is_contiguous() for t in (x_bf16, gate_weight_bf16, up_weight_bf16, out)):
        raise ValueError("all GEMM SwiGLU V8 tensors must be contiguous")

    m, k = x_bf16.shape
    n = gate_weight_bf16.shape[0]
    if m == 0 or n == 0:
        return out
    if (m % 32) != 0 or (k % 16) != 0 or (n % 64) != 0:
        raise ValueError(
            "GEMM SwiGLU V8 requires M multiple of 32, K multiple of 16, and N multiple of 64"
        )

    library = _library()
    stream = torch.cuda.current_stream(device=x_bf16.device)
    code = library.handle.cuda_operator_gemm_swiglu_v8(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(gate_weight_bf16.data_ptr()),
        ctypes.c_void_p(up_weight_bf16.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v8(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x_bf16.shape[0], gate_weight_bf16.shape[0]),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v8_into(
        x_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        out,
    )


def gemm_swiglu_v9_into(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """8-warp WMMA fused SwiGLU with 8-way shared A-tile reuse."""
    if not all(t.is_cuda for t in (x_bf16, gate_weight_bf16, up_weight_bf16, out)):
        raise ValueError("GEMM SwiGLU V9 tensors must be CUDA tensors")
    if x_bf16.ndim != 2 or gate_weight_bf16.ndim != 2 or up_weight_bf16.ndim != 2:
        raise ValueError("x and weights must be 2-D")
    if gate_weight_bf16.shape != up_weight_bf16.shape:
        raise ValueError("gate and up weights must have the same shape")
    if x_bf16.shape[1] != gate_weight_bf16.shape[1]:
        raise ValueError("x.shape[1] must equal weight.shape[1]")
    if any(t.dtype != torch.bfloat16 for t in (x_bf16, gate_weight_bf16, up_weight_bf16)):
        raise TypeError("GEMM SwiGLU V9 expects BF16 input and weights")
    if out.dtype != torch.float32:
        raise TypeError("GEMM SwiGLU V9 output must be float32")
    expected_out = (x_bf16.shape[0], gate_weight_bf16.shape[0])
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if not all(t.is_contiguous() for t in (x_bf16, gate_weight_bf16, up_weight_bf16, out)):
        raise ValueError("all GEMM SwiGLU V9 tensors must be contiguous")

    m, k = x_bf16.shape
    n = gate_weight_bf16.shape[0]
    if m == 0 or n == 0:
        return out
    if (m % 16) != 0 or (k % 16) != 0 or (n % 128) != 0:
        raise ValueError(
            "GEMM SwiGLU V9 requires M,K multiples of 16 and N multiple of 128"
        )

    library = _library()
    stream = torch.cuda.current_stream(device=x_bf16.device)
    code = library.handle.cuda_operator_gemm_swiglu_v9(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(gate_weight_bf16.data_ptr()),
        ctypes.c_void_p(up_weight_bf16.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v9(
    x_bf16: torch.Tensor,
    gate_weight_bf16: torch.Tensor,
    up_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(
        (x_bf16.shape[0], gate_weight_bf16.shape[0]),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v9_into(
        x_bf16,
        gate_weight_bf16,
        up_weight_bf16,
        out,
    )


def gemm_swiglu_v10_into(
    x_bf16: torch.Tensor,
    packed_weight_bf16: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Profile-guided hybrid: custom WMMA for small K, cuBLAS otherwise."""
    if not all(t.is_cuda for t in (x_bf16, packed_weight_bf16, workspace, out)):
        raise ValueError("GEMM SwiGLU V10 tensors must be CUDA tensors")
    if x_bf16.ndim != 2 or packed_weight_bf16.ndim != 2:
        raise ValueError("x_bf16 and packed_weight_bf16 must be 2-D")
    if x_bf16.shape[1] != packed_weight_bf16.shape[1]:
        raise ValueError("x.shape[1] must equal packed_weight.shape[1]")
    if packed_weight_bf16.shape[0] % 2 != 0:
        raise ValueError("packed_weight.shape[0] must be even")
    if x_bf16.dtype != torch.bfloat16 or packed_weight_bf16.dtype != torch.bfloat16:
        raise TypeError("GEMM SwiGLU V10 expects BF16 input and packed weight")
    if workspace.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("GEMM SwiGLU V10 workspace/output must be float32")
    n = packed_weight_bf16.shape[0] // 2
    expected_workspace = (x_bf16.shape[0], 2 * n)
    expected_out = (x_bf16.shape[0], n)
    if workspace.shape != expected_workspace:
        raise ValueError(f"workspace shape must be {expected_workspace}")
    if out.shape != expected_out:
        raise ValueError(f"output shape must be {expected_out}")
    if not all(t.is_contiguous() for t in (x_bf16, packed_weight_bf16, workspace, out)):
        raise ValueError("all GEMM SwiGLU V10 tensors must be contiguous")

    m, k = x_bf16.shape
    if m == 0 or n == 0:
        return out

    library = _library()
    stream = torch.cuda.current_stream(device=x_bf16.device)
    code = library.handle.cuda_operator_gemm_swiglu_v10(
        ctypes.c_void_p(x_bf16.data_ptr()),
        ctypes.c_void_p(packed_weight_bf16.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(m),
        ctypes.c_uint64(k),
        ctypes.c_uint64(n),
        ctypes.c_void_p(stream.cuda_stream),
    )
    library.check(code)
    return out


def gemm_swiglu_v10(
    x_bf16: torch.Tensor,
    packed_weight_bf16: torch.Tensor,
) -> torch.Tensor:
    n = packed_weight_bf16.shape[0] // 2
    workspace = torch.empty(
        (x_bf16.shape[0], 2 * n),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    out = torch.empty(
        (x_bf16.shape[0], n),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    return gemm_swiglu_v10_into(x_bf16, packed_weight_bf16, workspace, out)
