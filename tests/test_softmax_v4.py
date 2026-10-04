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
        (0, 32),
        (1, 1),
        (7, 31),
        (8, 32),
        (9, 33),
        (15, 63),
        (16, 64),
        (17, 65),
        (31, 127),
        (32, 128),
        (33, 129),
        (128, 32),
        (1024, 64),
        (4096, 128),
        (1024, 513),
        (1024, 4096),
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
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_softmax_v4_is_numerically_stable_for_large_logits() -> None:
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
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_softmax_v4_reuses_preallocated_output() -> None:
    x = torch.randn(17, 127, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = softmax_v4_into(x, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, row_softmax(x), rtol=4e-5, atol=4e-6)


def test_softmax_v4_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(1024, 64, device="cuda", dtype=torch.float32)
        actual = softmax_v4(x)
        expected = row_softmax(x)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_softmax_v4_row_sums_are_normalized() -> None:
    x = torch.randn(4096, 128, device="cuda", dtype=torch.float32)
    actual = softmax_v4(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(4096, device="cuda"),
        rtol=5e-6,
        atol=5e-6,
    )
