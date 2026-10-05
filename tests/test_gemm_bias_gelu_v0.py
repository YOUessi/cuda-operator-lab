from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import gemm_bias_gelu_v0, gemm_bias_gelu_v0_into
from cuda_operator_lab.references import gemm_bias_gelu


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (0, 32, 64),
        (1, 1, 1),
        (1, 7, 13),
        (4, 16, 32),
        (17, 31, 63),
        (32, 128, 256),
        (64, 256, 512),
        (128, 512, 1024),
    ],
)
def test_gemm_bias_gelu_v0_matches_reference(m: int, k: int, n: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + m * 1000000 + k * 1000 + n)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)

    actual = gemm_bias_gelu_v0(x, weight, bias)
    expected = gemm_bias_gelu(x, weight, bias)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)


def test_gemm_bias_gelu_v0_reuses_output() -> None:
    x = torch.randn(17, 64, device="cuda", dtype=torch.float32)
    weight = torch.randn(96, 64, device="cuda", dtype=torch.float32)
    bias = torch.randn(96, device="cuda", dtype=torch.float32)
    out = torch.empty(17, 96, device="cuda", dtype=torch.float32)

    returned = gemm_bias_gelu_v0_into(x, weight, bias, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out,
        gemm_bias_gelu(x, weight, bias),
        rtol=3e-4,
        atol=3e-4,
    )


def test_gemm_bias_gelu_v0_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(32, 128, device="cuda", dtype=torch.float32)
        weight = torch.randn(256, 128, device="cuda", dtype=torch.float32)
        bias = torch.randn(256, device="cuda", dtype=torch.float32)
        actual = gemm_bias_gelu_v0(x, weight, bias)
        expected = gemm_bias_gelu(x, weight, bias)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
