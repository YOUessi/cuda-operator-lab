from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import rmsnorm_v3, rmsnorm_v3_into
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
        (1, 4),
        (1, 31),
        (1, 32),
        (1, 33),
        (1, 64),
        (2, 127),
        (2, 128),
        (2, 255),
        (2, 256),
        (7, 511),
        (7, 512),
        (7, 513),
        (17, 1024),
        (17, 4096),
        (17, 4097),
        (128, 4096),
        (128, 8192),
        (1024, 4096),
    ],
)
def test_rmsnorm_v3_matches_reference(rows: int, cols: int) -> None:
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
    actual = rmsnorm_v3(x, weight)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_rmsnorm_v3_reuses_preallocated_output() -> None:
    x = torch.randn(11, 512, device="cuda", dtype=torch.float32)
    weight = torch.randn(512, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = rmsnorm_v3_into(x, weight, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, rmsnorm(x, weight), rtol=4e-5, atol=4e-6)


def test_rmsnorm_v3_scalar_fallback_for_non_multiple_of_four() -> None:
    x = torch.randn(17, 4097, device="cuda", dtype=torch.float32)
    weight = torch.randn(4097, device="cuda", dtype=torch.float32)

    actual = rmsnorm_v3(x, weight)
    expected = rmsnorm(x, weight)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_rmsnorm_v3_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(32, 4096, device="cuda", dtype=torch.float32)
        weight = torch.randn(4096, device="cuda", dtype=torch.float32)
        actual = rmsnorm_v3(x, weight)
        expected = rmsnorm(x, weight)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)