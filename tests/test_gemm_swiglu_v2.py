from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import gemm_swiglu_v2, gemm_swiglu_v2_into
from cuda_operator_lab.references import gemm_swiglu


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
def test_gemm_swiglu_v2_matches_reference(m: int, k: int, n: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261006 + m * 1000000 + k * 1000 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g)
    gate_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)
    up_w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)
    packed_w = torch.cat((gate_w, up_w), dim=0).contiguous()

    actual = gemm_swiglu_v2(x, packed_w)
    expected = gemm_swiglu(x, gate_w, up_w)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-4, atol=4e-4)


def test_gemm_swiglu_v2_reuses_workspace_and_output() -> None:
    x = torch.randn(17, 64, device="cuda", dtype=torch.float32)
    gate_w = torch.randn(96, 64, device="cuda", dtype=torch.float32)
    up_w = torch.randn(96, 64, device="cuda", dtype=torch.float32)
    packed_w = torch.cat((gate_w, up_w), dim=0).contiguous()
    workspace = torch.empty(17, 192, device="cuda", dtype=torch.float32)
    out = torch.empty(17, 96, device="cuda", dtype=torch.float32)

    returned = gemm_swiglu_v2_into(x, packed_w, workspace, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out,
        gemm_swiglu(x, gate_w, up_w),
        rtol=4e-4,
        atol=4e-4,
    )
