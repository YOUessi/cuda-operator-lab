from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import layernorm_v5, layernorm_v5_into
from cuda_operator_lab.references import layernorm


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
        (128, 8192),
        (256, 8192),
        (512, 8192),
        (1024, 8192),
        (1024, 512),
        (1536, 512),
        (2048, 512),
        (1536, 1024),
        (2048, 1024),
        (17, 4097),
    ],
)
def test_layernorm_v5_matches_reference(rows: int, cols: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    weight = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)

    expected = layernorm(x, weight, bias)
    actual = layernorm_v5(x, weight, bias)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)


def test_layernorm_v5_reuses_preallocated_output() -> None:
    x = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    weight = torch.randn(4096, device="cuda", dtype=torch.float32)
    bias = torch.randn(4096, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    returned = layernorm_v5_into(x, weight, bias, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, layernorm(x, weight, bias), rtol=5e-5, atol=5e-6)


def test_layernorm_v5_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(512, 8192, device="cuda", dtype=torch.float32)
        weight = torch.randn(8192, device="cuda", dtype=torch.float32)
        bias = torch.randn(8192, device="cuda", dtype=torch.float32)
        actual = layernorm_v5(x, weight, bias)
        expected = layernorm(x, weight, bias)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
