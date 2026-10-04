from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import softmax_v5, softmax_v5_into
from cuda_operator_lab.references import row_softmax


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (0, 32),
        (1, 1),
        (128, 32),
        (2047, 128),
        (2048, 128),
        (2049, 128),
        (4095, 32),
        (4096, 32),
        (4097, 32),
        (4095, 64),
        (4096, 64),
        (4097, 64),
        (1024, 129),
        (1024, 512),
        (1024, 4096),
        (16384, 32),
        (16384, 64),
        (16384, 128),
    ],
)
def test_softmax_v5_matches_torch(rows: int, cols: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(
        rows,
        cols,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )

    expected = row_softmax(x)
    actual = softmax_v5(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_softmax_v5_reuses_preallocated_output() -> None:
    x = torch.randn(4096, 64, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = softmax_v5_into(x, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, row_softmax(x), rtol=4e-5, atol=4e-6)


def test_softmax_v5_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(4096, 128, device="cuda", dtype=torch.float32)
        actual = softmax_v5(x)
        expected = row_softmax(x)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_softmax_v5_row_sums_are_normalized() -> None:
    x = torch.randn(16384, 128, device="cuda", dtype=torch.float32)
    actual = softmax_v5(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(16384, device="cuda"),
        rtol=5e-6,
        atol=5e-6,
    )
