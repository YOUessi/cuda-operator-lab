from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import fused_bias_gelu_v2
from cuda_operator_lab.references import fused_bias_gelu


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (0, 128),
        (128, 128),
        (128, 4096),
        (256, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 512),
        (2048, 1024),
        (2048, 4096),
        (17, 4097),
    ],
)
def test_fused_bias_gelu_v2_matches_reference(rows: int, cols: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)

    actual = fused_bias_gelu_v2(x, bias)
    expected = fused_bias_gelu(x, bias)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-6, atol=5e-6)
