from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import rmsnorm_v0, rmsnorm_v0_into
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
        (2, 31),
        (2, 32),
        (2, 33),
        (7, 127),
        (64, 128),
        (33, 512),
        (17, 1024),
        (8, 4097),
        (32, 4096),
        (128, 4096),
    ],
)
def test_rmsnorm_v0_matches_reference(rows: int, cols: int) -> None:
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
    actual = rmsnorm_v0(x, weight)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_rmsnorm_v0_handles_large_magnitude_values() -> None:
    x = torch.tensor(
        [[1000.0, -1000.0, 500.0, -500.0]],
        device="cuda",
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [1.0, 0.5, -2.0, 3.0],
        device="cuda",
        dtype=torch.float32,
    )

    expected = rmsnorm(x, weight)
    actual = rmsnorm_v0(x, weight)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_rmsnorm_v0_reuses_preallocated_output() -> None:
    x = torch.randn(11, 257, device="cuda", dtype=torch.float32)
    weight = torch.randn(257, device="cuda", dtype=torch.float32)
    out = torch.full_like(x, float("nan"))

    returned = rmsnorm_v0_into(x, weight, out)
    torch.cuda.synchronize()

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, rmsnorm(x, weight), rtol=3e-5, atol=3e-6)


def test_rmsnorm_v0_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(32, 513, device="cuda", dtype=torch.float32)
        weight = torch.randn(513, device="cuda", dtype=torch.float32)
        actual = rmsnorm_v0(x, weight)
        expected = rmsnorm(x, weight)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_rmsnorm_v0_rejects_bad_weight_shape() -> None:
    x = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    weight = torch.randn(63, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="weight"):
        rmsnorm_v0(x, weight)


def test_rmsnorm_v0_rejects_nonpositive_eps() -> None:
    x = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="positive"):
        rmsnorm_v0(x, weight, eps=0.0)
