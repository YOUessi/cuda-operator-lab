from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import fused_bias_gelu_v0, fused_bias_gelu_v0_into
from cuda_operator_lab.references import fused_bias_gelu


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
        (1, 31),
        (1, 32),
        (1, 33),
        (7, 127),
        (7, 128),
        (128, 512),
        (128, 4096),
        (1024, 4096),
    ],
)
def test_fused_bias_gelu_v0_matches_reference(rows: int, cols: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)

    actual = fused_bias_gelu_v0(x, bias)
    expected = fused_bias_gelu(x, bias)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-6, atol=5e-6)


def test_fused_bias_gelu_v0_reuses_output() -> None:
    x = torch.randn(17, 513, device="cuda", dtype=torch.float32)
    bias = torch.randn(513, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    returned = fused_bias_gelu_v0_into(x, bias, out)
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, fused_bias_gelu(x, bias), rtol=5e-6, atol=5e-6)
