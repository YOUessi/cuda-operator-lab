from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import gemm_swiglu_v4, gemm_swiglu_v4_into
from cuda_operator_lab.references import gemm_swiglu_bf16_fp32_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (0, 32, 64),
        (1, 1, 1),
        (4, 16, 32),
        (17, 31, 63),
        (32, 128, 256),
        (64, 256, 512),
        (128, 512, 1024),
        (128, 1024, 4096),
    ],
)
def test_gemm_swiglu_v4_matches_bf16_fp32_reference(
    m: int, k: int, n: int
) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261006 + m * 1000000 + k * 1000 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g).to(torch.bfloat16)
    packed_weight = torch.randn(
        2 * n, k, device="cuda", dtype=torch.float32, generator=g
    ).to(torch.bfloat16)

    actual = gemm_swiglu_v4(x, packed_weight)
    expected = gemm_swiglu_bf16_fp32_reference(x, packed_weight)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-2)


def test_gemm_swiglu_v4_reuses_workspace_and_output() -> None:
    m, k, n = 17, 64, 96
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    packed_weight = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(m, 2 * n, device="cuda", dtype=torch.float32)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)

    returned = gemm_swiglu_v4_into(x, packed_weight, workspace, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out,
        gemm_swiglu_bf16_fp32_reference(x, packed_weight),
        rtol=5e-3,
        atol=5e-2,
    )
