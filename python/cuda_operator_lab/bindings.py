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
