from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import rmsnorm_v4, rmsnorm_v4_into
from cuda_operator_lab.references import rmsnorm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (0, 128),
        (128, 512),
        (1023, 512),
        (1024, 512),
        (1024, 1024),
        (1536, 1024),
        (128, 4096),
        (1536, 4096),
        (1537, 4096),
        (2048, 4096),
        (128, 8192),
        (1024, 8192),
        (1025, 8192),
        (1536, 8192),
        (17, 4097),
        (64, 2048),
    ],
)
def test_rmsnorm_v4_matches_reference(rows: int, cols: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + rows * 10000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    w = torch.randn(cols, device="cuda", dtype=torch.float32, generator=g)

    actual = rmsnorm_v4(x, w)
    expected = rmsnorm(x, w)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)


def test_rmsnorm_v4_reuses_output() -> None:
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    w = torch.randn(1024, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    returned = rmsnorm_v4_into(x, w, out)
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, rmsnorm(x, w), rtol=4e-5, atol=4e-6)


def test_rmsnorm_v4_uses_current_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(1536, 4096, device="cuda", dtype=torch.float32)
        w = torch.randn(4096, device="cuda", dtype=torch.float32)
        actual = rmsnorm_v4(x, w)
        expected = rmsnorm(x, w)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)