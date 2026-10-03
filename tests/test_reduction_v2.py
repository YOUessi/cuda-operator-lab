from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import reduction_v2, reduction_v2_into
from cuda_operator_lab.references import reduction_sum


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    "n",
    [
        0,
        1,
        31,
        32,
        33,
        127,
        255,
        256,
        257,
        511,
        512,
        513,
        1024,
        4097,
        65537,
        1_000_003,
    ],
)
def test_reduction_v2_matches_torch(n: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261004 + n)
    x = torch.rand(n, device="cuda", dtype=torch.float32, generator=generator)

    expected = reduction_sum(x)
    actual = reduction_v2(x)[0]

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-4)


def test_reduction_v2_signed_values() -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261004)
    x = torch.randn(
        1_000_003,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )

    expected = reduction_sum(x)
    actual = reduction_v2(x)[0]

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=5e-3)


def test_reduction_v2_resets_preallocated_output() -> None:
    x = torch.arange(1, 1025, device="cuda", dtype=torch.float32)
    out = torch.full((1,), 12345.0, device="cuda", dtype=torch.float32)

    reduction_v2_into(x, out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out[0], reduction_sum(x), rtol=5e-5, atol=5e-4)


def test_reduction_v2_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(4097, device="cuda", dtype=torch.float32)
        actual = reduction_v2(x)
        expected = reduction_sum(x)

    stream.synchronize()
    torch.testing.assert_close(actual[0], expected, rtol=5e-5, atol=5e-4)
