from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import softmax_v4, softmax_v4_into
from cuda_operator_lab.references import row_softmax


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (0, 128),
        (1, 1),
        (1, 31),
        (1, 32),
        (1, 33),
        (7, 64),
        (8, 64),
        (9, 64),
        (15, 65),
        (16, 127),
        (17, 128),
        (33, 128),
        (128, 32),
        (128, 64),
        (128, 128),
        (1024, 32),
        (1024, 64),
        (1024, 128),
        (17, 129),
        (17, 513),
        (17, 4097),
    ],
)
def test_softmax_v4_matches_torch(rows: int, cols: int) -> None:
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
    actual = softmax_v4(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_softmax_v4_large_logits() -> None:
    x = torch.tensor(
        [
            [1000.0, 1001.0, 999.0, 998.0],
            [-1000.0, -999.0, -1001.0, -1002.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    expected = row_softmax(x)
    actual = softmax_v4(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_softmax_v4_reuses_preallocated_output() -> None:
    x = torch.randn(17, 127, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = softmax_v4_into(x, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, row_softmax(x), rtol=3e-5, atol=3e-6)


def test_softmax_v4_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(17, 65, device="cuda", dtype=torch.float32)
        actual = softmax_v4(x)
        expected = row_softmax(x)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_softmax_v4_row_sums_are_normalized() -> None:
    x = torch.randn(1024, 128, device="cuda", dtype=torch.float32)
    actual = softmax_v4(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(1024, device="cuda"),
        rtol=4e-6,
        atol=4e-6,
    )
