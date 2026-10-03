from __future__ import annotations

import pytest
import torch

from cuda_operator_lab.bindings import reduction_v0
from cuda_operator_lab.references import reduction_sum


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 127, 1024, 4097, 65537])
def test_reduction_v0_matches_torch(n: int) -> None:
    torch.manual_seed(20261004 + n)
    x = torch.rand(n, device="cuda", dtype=torch.float32)
    expected = reduction_sum(x)
    actual = reduction_v0(x)[0]

    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)


def test_reduction_v0_rejects_cpu_tensor() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        reduction_v0(torch.ones(8, dtype=torch.float32))