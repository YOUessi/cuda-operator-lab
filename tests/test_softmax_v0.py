from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import softmax_v0, softmax_v0_into
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
        (2, 31),
        (2, 32),
        (2, 33),
        (7, 127),
        (64, 128),
        (33, 512),
        (17, 1024),
        (8, 4097),
    ],
)
def test_softmax_v0_matches_torch(rows: int, cols: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261004 + rows * 10000 + cols)
    x = torch.randn(
        rows,
        cols,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )

    expected = row_softmax(x)
    actual = softmax_v0(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_softmax_v0_is_numerically_stable_for_large_logits() -> None:
    x = torch.tensor(
        [[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]],
        device="cuda",
        dtype=torch.float32,
    )

    expected = row_softmax(x)
    actual = softmax_v0(x)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(2, device="cuda"),
        rtol=1e-6,
        atol=1e-6,
    )


def test_softmax_v0_reuses_preallocated_output() -> None:
    x = torch.randn(11, 257, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = softmax_v0_into(x, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, row_softmax(x), rtol=2e-5, atol=2e-6)


def test_softmax_v0_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(32, 513, device="cuda", dtype=torch.float32)
        actual = softmax_v0(x)
        expected = row_softmax(x)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_softmax_v0_rejects_noncontiguous_input() -> None:
    x = torch.randn(16, 32, device="cuda", dtype=torch.float32).t()
    assert not x.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        softmax_v0(x)
