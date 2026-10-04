from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import softmax_v3, softmax_v3_into
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
        (1, 3),
        (1, 31),
        (1, 32),
        (1, 33),
        (1, 63),
        (1, 64),
        (1, 65),
        (2, 127),
        (2, 128),
        (2, 129),
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
        (1024, 128),
        (1024, 4096),
    ],
)
def test_softmax_v3_matches_torch(rows: int, cols: int) -> None:
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
    actual = softmax_v3(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_softmax_v3_large_logits() -> None:
    x = torch.tensor(
        [
            [1000.0, 1001.0, 999.0, 998.0],
            [-1000.0, -999.0, -1001.0, -1002.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    expected = row_softmax(x)
    actual = softmax_v3(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_softmax_v3_signed_wide_rows() -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261005)
    x = torch.randn(
        33,
        8193,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )

    expected = row_softmax(x)
    actual = softmax_v3(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_softmax_v3_reuses_preallocated_output() -> None:
    x = torch.randn(11, 129, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = softmax_v3_into(x, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, row_softmax(x), rtol=3e-5, atol=3e-6)


def test_softmax_v3_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(32, 65, device="cuda", dtype=torch.float32)
        actual = softmax_v3(x)
        expected = row_softmax(x)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_softmax_v3_row_sums_are_normalized() -> None:
    x = torch.randn(128, 4097, device="cuda", dtype=torch.float32)
    actual = softmax_v3(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(128, device="cuda"),
        rtol=4e-6,
        atol=4e-6,
    )
