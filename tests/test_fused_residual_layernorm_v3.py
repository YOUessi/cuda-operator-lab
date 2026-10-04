from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import (
    fused_residual_layernorm_v3,
    fused_residual_layernorm_v3_into,
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
        (1, 1),
        (1, 3),
        (1, 4),
        (1, 31),
        (1, 32),
        (1, 33),
        (2, 127),
        (2, 128),
        (2, 255),
        (2, 256),
        (7, 511),
        (7, 512),
        (7, 513),
        (17, 1024),
        (17, 4096),
        (17, 4097),
        (128, 4096),
        (128, 8192),
        (1024, 4096),
    ],
)
def test_fused_residual_layernorm_v3_matches_reference(rows: int, cols: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    residual = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    weight = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)

    expected = fused_residual_layernorm(x, residual, weight, bias)
    actual = fused_residual_layernorm_v3(x, residual, weight, bias)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=6e-5, atol=6e-6)


def test_fused_residual_layernorm_v3_scalar_fallback() -> None:
    x = torch.randn(17, 4097, device="cuda", dtype=torch.float32)
    residual = torch.randn_like(x)
    weight = torch.randn(4097, device="cuda", dtype=torch.float32)
    bias = torch.randn(4097, device="cuda", dtype=torch.float32)
    actual = fused_residual_layernorm_v3(x, residual, weight, bias)
    expected = fused_residual_layernorm(x, residual, weight, bias)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=6e-5, atol=6e-6)


def test_fused_residual_layernorm_v3_reuses_output() -> None:
    x = torch.randn(11, 512, device="cuda", dtype=torch.float32)
    residual = torch.randn_like(x)
    weight = torch.randn(512, device="cuda", dtype=torch.float32)
    bias = torch.randn(512, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    returned = fused_residual_layernorm_v3_into(
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
