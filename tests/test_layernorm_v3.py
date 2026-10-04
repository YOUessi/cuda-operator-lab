from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import layernorm_v3, layernorm_v3_into
from cuda_operator_lab.references import layernorm


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
        (1, 63),
        (1, 64),
        (1, 65),
        (2, 127),
        (2, 255),
        (2, 256),
        (2, 257),
        (7, 511),
        (7, 512),
        (7, 513),
        (17, 1024),
        (8, 4097),
        (128, 128),
        (128, 4096),
        (1024, 4096),
    ],
)
def test_layernorm_v3_matches_reference(rows: int, cols: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    weight = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)

    expected = layernorm(x, weight, bias)
    actual = layernorm_v3(x, weight, bias)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)


def test_layernorm_v3_large_offset_stability() -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005)
    x = (
        torch.randn(64, 4096, device="cuda", dtype=torch.float32, generator=g)
        * 0.1
        + 1000.0
    )
    weight = torch.randn(4096, device="cuda", dtype=torch.float32, generator=g)
    bias = torch.randn(4096, device="cuda", dtype=torch.float32, generator=g)

    actual = layernorm_v3(x, weight, bias)

    x64 = x.double()
    weight64 = weight.double()
    bias64 = bias.double()
    mean64 = x64.mean(dim=-1, keepdim=True)
    variance64 = (x64 - mean64).square().mean(dim=-1, keepdim=True)
    expected = (
        (x64 - mean64)
        * torch.rsqrt(variance64 + 1e-5)
        * weight64
        + bias64
    ).float()

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1.2e-3)


def test_layernorm_v3_reuses_preallocated_output() -> None:
    x = torch.randn(11, 257, device="cuda", dtype=torch.float32)
    weight = torch.randn(257, device="cuda", dtype=torch.float32)
    bias = torch.randn(257, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    returned = layernorm_v3_into(x, weight, bias, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, layernorm(x, weight, bias), rtol=5e-5, atol=5e-6)


def test_layernorm_v3_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(32, 513, device="cuda", dtype=torch.float32)
        weight = torch.randn(513, device="cuda", dtype=torch.float32)
        bias = torch.randn(513, device="cuda", dtype=torch.float32)
        actual = layernorm_v3(x, weight, bias)
        expected = layernorm(x, weight, bias)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-6)
