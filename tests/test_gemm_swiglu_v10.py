from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import gemm_swiglu_v10, gemm_swiglu_v10_into
from cuda_operator_lab.references import (
    gemm_swiglu_bf16_dual_fp32_reference,
    gemm_swiglu_bf16_fp32_reference,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (16, 64, 64),
        (32, 128, 256),
        (64, 128, 1024),
        (128, 64, 2048),
    ],
)
def test_gemm_swiglu_v10_custom_path(
    m: int, k: int, n: int
) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261024 + m * 1000000 + k * 1000 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    gate_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    up_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    packed_w = torch.cat((gate_w, up_w), dim=0).contiguous()

    actual = gemm_swiglu_v10(x, packed_w)
    expected = gemm_swiglu_bf16_dual_fp32_reference(x, gate_w, up_w)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-2)


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (32, 256, 64),
        (128, 512, 512),
        (128, 1024, 4096),
        (512, 1024, 4096),
    ],
)
def test_gemm_swiglu_v10_cublas_fallback(
    m: int, k: int, n: int
) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261024 + m * 1000000 + k * 1000 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    packed_w = torch.randn(
        2 * n, k, device="cuda", dtype=torch.float32, generator=g
    ).to(torch.bfloat16).contiguous()

    actual = gemm_swiglu_v10(x, packed_w)
    expected = gemm_swiglu_bf16_fp32_reference(x, packed_w)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-2)


def test_gemm_swiglu_v10_reuses_buffers() -> None:
    m, k, n = 32, 128, 256
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    packed_w = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)

    returned = gemm_swiglu_v10_into(x, packed_w, workspace, out)
    assert returned.data_ptr() == out.data_ptr()
