from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import (
    fused_residual_layernorm_v4,
    fused_residual_layernorm_v4_into,
)
from cuda_operator_lab.references import fused_residual_layernorm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (0, 128),
        (128, 4096),
        (256, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (128, 8192),
        (256, 8192),
        (512, 8192),
        (1024, 8192),
        (1024, 1024),
        (1536, 512),
        (2048, 512),
        (17, 4097),
    ],
)
def test_fused_residual_layernorm_v4_matches_reference(rows: int, cols: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    residual = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    weight = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)

    expected = fused_residual_layernorm(x, residual, weight, bias)
    actual = fused_residual_layernorm_v4(x, residual, weight, bias)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=6e-5, atol=6e-6)


def test_fused_residual_layernorm_v4_reuses_output() -> None:
    x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    residual = torch.randn_like(x)
    weight = torch.randn(4096, device="cuda", dtype=torch.float32)
    bias = torch.randn(4096, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    returned = fused_residual_layernorm_v4_into(
        x, residual, weight, bias, out
    )
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out,
        fused_residual_layernorm(x, residual, weight, bias),
        rtol=6e-5,
        atol=6e-6,
    )
