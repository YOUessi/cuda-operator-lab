from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import rmsnorm_v2, rmsnorm_v2_into
from cuda_operator_lab.references import rmsnorm


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
def test_rmsnorm_v2_matches_reference(rows: int, cols: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(
        rows,
        cols,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    weight = torch.randn(
        cols,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )

    expected = rmsnorm(x, weight)
    actual = rmsnorm_v2(x, weight)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_rmsnorm_v2_reuses_preallocated_output() -> None:
    x = torch.randn(11, 257, device="cuda", dtype=torch.float32)
    weight = torch.randn(257, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = rmsnorm_v2_into(x, weight, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, rmsnorm(x, weight), rtol=4e-5, atol=4e-6)


def test_rmsnorm_v2_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(32, 513, device="cuda", dtype=torch.float32)
        weight = torch.randn(513, device="cuda", dtype=torch.float32)
        actual = rmsnorm_v2(x, weight)
        expected = rmsnorm(x, weight)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)
