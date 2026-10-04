from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import reduction_v5, reduction_v5_into
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
        2,
        3,
        4,
        5,
        31,
        32,
        33,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        16384,
        262144,
        1_000_003,
    ],
)
def test_reduction_v5_matches_torch(n: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261004 + n)
    x = torch.rand(n, device="cuda", dtype=torch.float32, generator=generator)

    expected = reduction_sum(x)
    actual = reduction_v5(x)[0]

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-4)


def test_reduction_v5_unaligned_contiguous_input_falls_back_safely() -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261004)
    storage = torch.randn(
        1_000_004,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    x = storage[1:]

    assert x.is_contiguous()
    assert x.data_ptr() % 16 == 4

    expected = reduction_sum(x)
    actual = reduction_v5(x)[0]

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=5e-3)


def test_reduction_v5_resets_preallocated_output() -> None:
    x = torch.arange(1, 1026, device="cuda", dtype=torch.float32)
    out = torch.full((1,), 12345.0, device="cuda", dtype=torch.float32)

    reduction_v5_into(x, out)
    torch.cuda.synchronize()

    torch.testing.assert_close(out[0], reduction_sum(x), rtol=5e-5, atol=5e-4)


def test_reduction_v5_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.arange(4099, device="cuda", dtype=torch.float32)
        actual = reduction_v5(x)
        expected = reduction_sum(x)

    stream.synchronize()
    torch.testing.assert_close(actual[0], expected, rtol=5e-5, atol=5e-4)
