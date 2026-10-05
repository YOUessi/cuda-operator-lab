from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import gemm_swiglu_v10, gemm_swiglu_v11
from cuda_operator_lab.references import (
    gemm_swiglu_bf16_dual_fp32_reference,
    gemm_swiglu_bf16_fp32_reference,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("m", "k", "n", "custom"),
    [
        (16, 64, 64, True),
        (32, 128, 256, True),
        (64, 128, 1024, True),
        (128, 64, 2048, True),
        (32, 256, 64, False),
        (128, 512, 512, False),
        (128, 1024, 4096, False),
        (512, 1024, 4096, False),
    ],
)
def test_gemm_swiglu_v11_matches_v10_and_reference(
    m: int, k: int, n: int, custom: bool
) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261025 + m * 1000000 + k * 1000 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    gate_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    up_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    packed_w = torch.cat((gate_w, up_w), dim=0).contiguous()

    y10 = gemm_swiglu_v10(x, packed_w)
    y11 = gemm_swiglu_v11(x, packed_w)

    if custom:
        ref = gemm_swiglu_bf16_dual_fp32_reference(x, gate_w, up_w)
    else:
        ref = gemm_swiglu_bf16_fp32_reference(x, packed_w)

    torch.cuda.synchronize()
    torch.testing.assert_close(y11, y10, rtol=0.0, atol=0.0)
    torch.testing.assert_close(y11, ref, rtol=5e-3, atol=5e-2)
