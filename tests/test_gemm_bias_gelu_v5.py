from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import gemm_bias_gelu_v5
from cuda_operator_lab.references import gemm_bias_gelu_tanh


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (4, 16, 32),
        (17, 31, 63),
        (32, 128, 256),
        (64, 256, 512),
        (128, 512, 1024),
        (128, 1024, 4096),
        (512, 1024, 4096),
    ],
)
def test_gemm_bias_gelu_v5_matches_reference(m: int, k: int, n: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261006 + m * 1000000 + k * 1000 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)

    actual = gemm_bias_gelu_v5(x, weight, bias)
    expected = gemm_bias_gelu_tanh(x, weight, bias)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)


def test_gemm_bias_gelu_v5_cached_reuse() -> None:
    x = torch.randn(128, 1024, device="cuda", dtype=torch.float32)
    weight = torch.randn(4096, 1024, device="cuda", dtype=torch.float32)
    bias = torch.randn(4096, device="cuda", dtype=torch.float32)

    first = gemm_bias_gelu_v5(x, weight, bias)
    second = gemm_bias_gelu_v5(x, weight, bias)
    torch.cuda.synchronize()

    ref = gemm_bias_gelu_tanh(x, weight, bias)
    torch.testing.assert_close(first, ref, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(second, ref, rtol=3e-4, atol=3e-4)
