"""Thin ctypes bindings for the CUDA operator library.

The CUDA kernels stay independent from PyTorch's C++ ABI.  PyTorch tensors provide
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
        self.handle.cuda_operator_reduction_v0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.handle.cuda_operator_reduction_v0.restype = ctypes.c_int
        self.handle.cuda_operator_error_string.argtypes = [ctypes.c_int]
        self.handle.cuda_operator_error_string.restype = ctypes.c_char_p

    def check(self, code: int) -> None:
        if code == 0:
            return
        message = self.handle.cuda_operator_error_string(code)
        decoded = message.decode("utf-8", errors="replace") if message else "unknown CUDA error"
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
        raise TypeError("reduction v0 currently supports float32 only")
    if not x.is_contiguous():
        raise ValueError("reduction input must be contiguous")


def reduction_v0_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch the V0 single-thread serial reduction into a preallocated scalar."""
    _validate_reduction_input(x)
    if not out.is_cuda or out.device != x.device:
        raise ValueError("output must be a CUDA tensor on the same device")
    if out.dtype != torch.float32 or out.numel() != 1 or not out.is_contiguous():
        raise ValueError("output must be one contiguous float32 CUDA value")

    stream = torch.cuda.current_stream(device=x.device)
    code = _library().handle.cuda_operator_reduction_v0(
        ctypes.c_void_p(x.data_ptr()) if x.numel() else ctypes.c_void_p(),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_uint64(x.numel()),
        ctypes.c_void_p(stream.cuda_stream),
    )
    _library().check(code)
    return out


def reduction_v0(x: torch.Tensor) -> torch.Tensor:
    """Return the V0 CUDA reduction result as a one-element CUDA tensor."""
    out = torch.empty(1, device=x.device, dtype=torch.float32)
    return reduction_v0_into(x, out)