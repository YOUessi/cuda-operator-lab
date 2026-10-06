import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load():
    spec = importlib.util.spec_from_file_location('p02_demand', ROOT / 'demand_head.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Calls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.mm = []
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        if func == torch.ops.aten.mm.default:
            self.mm.append((tuple(args[0].shape), tuple(args[1].shape)))
        return func(*args, **(kwargs or {}))


def inputs(need_x=True, need_w=True, dtype=torch.float32):
    g = torch.Generator().manual_seed(7207)
    x = torch.randn(7, 5, generator=g).to(dtype).requires_grad_(need_x)
    w = (torch.randn(11, 5, generator=g) / 5 ** 0.5).to(dtype).requires_grad_(need_w)
    y = torch.tensor([1, 0, 5, 10, 3, 4, 9])
    a = torch.randn(7, generator=g)
    b = torch.randn(7, generator=g)
    return x, w, y, a, b


class DemandTests(unittest.TestCase):
    def test_values_and_vjp_all_modes(self):
        import contract
        m = load()
        for mode in ('logprob_only', 'entropy_logged', 'entropy_loss'):
            for need_x, need_w in ((True, True), (True, False), (False, True)):
                for chunk in (3, 512):
                    with self.subTest(mode=mode, dx=need_x, dw=need_w, chunk=chunk):
                        x, w, y, a, b = inputs(need_x, need_w)
                        lp, ent = m.head(x, w, y, 0.7, mode, chunk)
                        loss = (lp * a).sum()
                        if mode != 'logprob_only':
                            loss = loss + (ent * b).sum()
                        req = tuple(t for t in (x, w) if t.requires_grad)
                        actual = torch.autograd.grad(loss, req)
                        rl, re = contract.outputs(x.detach().double(), w.detach().double(), y, temperature=0.7)
                        dx, dw = contract.vjp(x.detach().double(), w.detach().double(), y, a.double(),
                                             b.double() if mode == 'entropy_loss' else None,
                                             temperature=0.7, need_dw=need_w)
                        torch.testing.assert_close(lp.double(), rl, rtol=2e-5, atol=2e-6)
                        if ent is not None:
                            torch.testing.assert_close(ent.double(), re, rtol=2e-5, atol=2e-6)
                            self.assertEqual(ent.requires_grad, mode == 'entropy_loss')
                        else:
                            self.assertEqual(mode, 'logprob_only')
                        refs = ([dx] if need_x else []) + ([dw] if need_w else [])
                        for got, ref in zip(actual, refs):
                            torch.testing.assert_close(got.double(), ref, rtol=2e-5, atol=2e-6)

    def test_frozen_weight_skips_weight_gemm(self):
        m = load()
        x, w, y, a, b = inputs(True, False)
        lp, _ = m.head(x, w, y, 1.0, 'logprob_only')
        c = Calls()
        with c:
            torch.autograd.grad(lp, x, a)
        self.assertEqual(c.mm, [((7, 5), (5, 11)), ((7, 11), (11, 5))])

    def test_frozen_hidden_skips_hidden_gemm(self):
        m = load()
        x, w, y, a, b = inputs(False, True)
        lp, _ = m.head(x, w, y, 1.0, 'logprob_only')
        c = Calls()
        with c:
            torch.autograd.grad(lp, w, a)
        self.assertEqual(c.mm, [((7, 5), (5, 11)), ((11, 7), (7, 5))])

    def test_logprob_only_does_not_compute_entropy(self):
        m = load()
        x, w, y, _, _ = inputs()
        c = Calls()
        with c:
            lp, ent = m.head(x, w, y, 1.0, 'logprob_only')
        self.assertIsNone(ent)
        self.assertNotIn('aten._softmax.default', c.ops)
        self.assertNotIn('aten.logsumexp.default', c.ops)
        self.assertIn('aten._log_softmax.default', c.ops)

    def test_logged_entropy_is_returned_but_not_differentiable(self):
        m = load()
        x, w, y, a, b = inputs()
        lp, ent = m.head(x, w, y, 1.0, 'entropy_logged')
        self.assertFalse(ent.requires_grad)
        self.assertTrue(lp.requires_grad)
        c = Calls()
        with c:
            torch.autograd.grad((lp * a + ent * b).sum(), (x, w))
        self.assertNotIn('aten._log_softmax.default', c.ops)
        self.assertNotIn('aten.logsumexp.default', c.ops)

    def test_entropy_only_backward(self):
        import contract
        m = load()
        x, w, y, a, b = inputs()
        lp, ent = m.head(x, w, y, 0.7, 'entropy_loss')
        got = torch.autograd.grad(ent, (x, w), b)
        dx, dw = contract.vjp(x.detach().double(), w.detach().double(), y,
                             torch.zeros_like(a).double(), b.double(), temperature=0.7)
        for t, r in zip(got, (dx, dw)):
            torch.testing.assert_close(t.double(), r, rtol=2e-5, atol=2e-6)

    def test_bf16_rounding_and_gradients(self):
        m = load()
        for mode in ('logprob_only', 'entropy_logged', 'entropy_loss'):
            x, w, y, a, b = inputs(dtype=torch.bfloat16)
            xx = x.detach().clone().requires_grad_()
            ww = w.detach().clone().requires_grad_()
            z = ((xx @ ww.T) / 0.7).float()
            ref_lp = z.log_softmax(-1).gather(1, y[:, None]).squeeze(1)
            ref_e = (torch.logsumexp(z, -1) - (z.softmax(-1) * z).sum(-1)).to(x.dtype)
            loss_ref = (a * ref_lp).sum()
            if mode == 'entropy_loss':
                loss_ref += (b * ref_e).sum()
            expected = torch.autograd.grad(loss_ref, (xx, ww))
            lp, ent = m.head(x, w, y, 0.7, mode)
            loss = (a * lp).sum()
            if ent is not None:
                loss += (b * ent).sum()
            got = torch.autograd.grad(loss, (x, w))
            torch.testing.assert_close(lp, ref_lp, rtol=0, atol=0)
            if ent is not None:
                torch.testing.assert_close(ent, ref_e, rtol=0, atol=0)
            for t, r in zip(got, expected):
                torch.testing.assert_close(t, r, rtol=0.03, atol=0.02)

    def test_mask_and_ignore(self):
        m = load()
        x, w, y, a, b = inputs()
        y[1] = -100
        mask = torch.tensor([1, 1, 0, 1, 1, 0, 1], dtype=torch.bool)
        lp, ent = m.checked_head(x, w, y, 0.7, 'entropy_loss', 3, mask)
        valid = mask & (y != -100)
        self.assertTrue(torch.equal(lp[~valid], torch.zeros_like(lp[~valid])))
        self.assertTrue(torch.equal(ent[~valid], torch.zeros_like(ent[~valid])))
        dx, dw = torch.autograd.grad((lp * a + ent * b).sum(), (x, w))
        self.assertTrue(torch.equal(dx[~valid], torch.zeros_like(dx[~valid])))

    def test_bad_metadata_rejected(self):
        m = load()
        x, w, y, _, _ = inputs()
        for tau in (0.0, -1.0, float('inf'), float('nan'), True):
            with self.subTest(tau=tau), self.assertRaises((ValueError, TypeError)):
                m.checked_head(x, w, y, tau)
        for chunk in (0, -1, True, 2.5):
            with self.subTest(chunk=chunk), self.assertRaises((ValueError, TypeError)):
                m.checked_head(x, w, y, chunk_size=chunk)
        with self.assertRaises(ValueError):
            m.checked_head(x, w, y, mode='invalid')
        with self.assertRaises(ValueError):
            m.checked_head(x, w, y + 11)


if __name__ == '__main__':
    unittest.main()
