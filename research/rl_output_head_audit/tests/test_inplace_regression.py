from pathlib import Path
import sys
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Counter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.add_inplace = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) == 'aten.add_.Tensor':
            self.add_inplace += 1
        return func(*args, **(kwargs or {}))


class InplaceRegression(unittest.TestCase):
    def test_two_loss_branches_keep_dz_accumulation_in_place(self):
        from demand_head import head
        g = torch.Generator().manual_seed(7207)
        x = torch.randn(7, 5, generator=g).requires_grad_()
        w = torch.randn(11, 5, generator=g).requires_grad_()
        y = torch.arange(7)
        lp, ent = head(x, w, y, 0.7, 'entropy_loss')
        c = Counter()
        with c:
            torch.autograd.grad((lp, ent), (x, w), (torch.ones(7), torch.ones(7)))
        # One dz entropy add, one dX slice add, one dW chunk add.
        self.assertEqual(c.add_inplace, 3)


if __name__ == '__main__':
    unittest.main()
