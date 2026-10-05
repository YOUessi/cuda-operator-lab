from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import swiglu_v1, swiglu_v1_into
from cuda_operator_lab.references import swiglu


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
        (7, 127),
        (7, 128),
        (128, 512),
        (128, 4096),
        (1024, 4096),
    ],
)
def test_swiglu_v1_matches_reference(rows: int, cols: int) -> None:
    g = torch.Generator(device="cuda")
    g.manual_seed(20261005 + rows * 10000 + cols)
    gate = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)
    up = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=g)

    actual = swiglu_v1(gate, up)
    expected = swiglu(gate, up)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-6, atol=5e-6)


def test_swiglu_v1_scalar_fallback() -> None:
    gate = torch.randn(17, 4097, device="cuda", dtype=torch.float32)
    up = torch.randn_like(gate)
    actual = swiglu_v1(gate, up)
    expected = swiglu(gate, up)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=5e-6, atol=5e-6)


def test_swiglu_v1_reuses_output() -> None:
    gate = torch.randn(17, 512, device="cuda", dtype=torch.float32)
    up = torch.randn_like(gate)
    out = torch.empty_like(gate)
    returned = swiglu_v1_into(gate, up, out)
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, swiglu(gate, up), rtol=5e-6, atol=5e-6)
