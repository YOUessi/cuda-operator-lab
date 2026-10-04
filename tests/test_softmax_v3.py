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
        (0, 32),
        (1, 1),
        (2, 31),
        (2, 32),
        (2, 33),
        (2, 63),
        (2, 64),
        (2, 65),
        (4, 127),
        (4, 128),
        (4, 129),
        (7, 255),
        (7, 256),
        (7, 257),
        (17, 511),
        (17, 512),
        (17, 513),
        (8, 4097),
        (128, 128),
        (1024, 128),
        (1024, 512),
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


def test_softmax_v3_is_numerically_stable_for_large_logits() -> None:
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


def test_softmax_v3_reuses_preallocated_output() -> None:
    x = torch.randn(11, 127, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = softmax_v3_into(x, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, row_softmax(x), rtol=3e-5, atol=3e-6)


def test_softmax_v3_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(1024, 128, device="cuda", dtype=torch.float32)
        actual = softmax_v3(x)
        expected = row_softmax(x)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_softmax_v3_row_sums_are_normalized() -> None:
    x = torch.randn(1024, 129, device="cuda", dtype=torch.float32)
    actual = softmax_v3(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(1024, device="cuda"),
        rtol=4e-6,
        atol=4e-6,
    )
