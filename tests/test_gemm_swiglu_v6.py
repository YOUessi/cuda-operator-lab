from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import gemm_swiglu_v6, gemm_swiglu_v6_into
from cuda_operator_lab.references import gemm_swiglu_bf16_dual_fp32_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (16, 16, 16),
        (32, 128, 256),
        (64, 256, 512),
        (128, 512, 1024),
        (128, 1024, 4096),
    ],
)
def test_gemm_swiglu_v6_matches_reference(m: int, k: int, n: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261018 + m * 1000000 + k * 1000 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    gate_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    up_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)

    actual = gemm_swiglu_v6(x, gate_w, up_w)
    expected = gemm_swiglu_bf16_dual_fp32_reference(x, gate_w, up_w)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-2)


def test_gemm_swiglu_v6_reuses_output() -> None:
    m, k, n = 32, 64, 96
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    gate_w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    up_w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)

    returned = gemm_swiglu_v6_into(x, gate_w, up_w, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out,
        gemm_swiglu_bf16_dual_fp32_reference(x, gate_w, up_w),
        rtol=5e-3,
        atol=5e-2,
    )


def test_gemm_swiglu_v6_rejects_non_multiple_of_16() -> None:
    x = torch.randn(17, 32, device="cuda", dtype=torch.bfloat16)
    gate_w = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    up_w = torch.randn_like(gate_w)
    with pytest.raises(ValueError, match="multiples of 16"):
        gemm_swiglu_v6(x, gate_w, up_w)
