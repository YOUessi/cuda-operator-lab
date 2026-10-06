"""CPU-only mathematical tests. These are not low-precision kernel benchmarks."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("head_contract", ROOT / "contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors():
    generator = torch.Generator().manual_seed(7107)
    x = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    w = torch.randn(11, 5, generator=generator, dtype=torch.float64) / math.sqrt(5)
    y = torch.tensor([1, 0, 5, 10, 3, 4, 9])
    a = torch.randn(7, generator=generator, dtype=torch.float64)
    b = torch.randn(7, generator=generator, dtype=torch.float64)
    return x, w, y, a, b


class ContractTests(unittest.TestCase):
    def test_values_match_dense_definition(self):
        c = load()
        x, w, y, _, _ = tensors()
        lp, ent = c.outputs(x, w, y, temperature=0.7)
        logall = torch.log_softmax(x @ w.T / 0.7, dim=-1)
        torch.testing.assert_close(lp, logall.gather(1, y[:, None]).squeeze(1))
        torch.testing.assert_close(ent, -(logall.exp() * logall).sum(-1))

    def test_vjp_matches_autograd_three_modes(self):
        c = load()
        for tau in (0.5, 1.0, 1.7):
            for mode in ("logprob_only", "entropy_only", "both"):
                with self.subTest(tau=tau, mode=mode):
                    x, w, y, a, b = tensors()
                    x.requires_grad_(); w.requires_grad_()
                    lp, ent = c.outputs(x, w, y, temperature=tau)
                    ga = None if mode == "entropy_only" else a
                    gb = None if mode == "logprob_only" else b
                    loss = sum(t for t in ((lp * ga).sum() if ga is not None else None,
                                          (ent * gb).sum() if gb is not None else None) if t is not None)
                    dx, dw = torch.autograd.grad(loss, (x, w))
                    ax, aw = c.vjp(x.detach(), w.detach(), y, ga, gb, temperature=tau)
                    torch.testing.assert_close(dx, ax, rtol=1e-10, atol=1e-11)
                    torch.testing.assert_close(dw, aw, rtol=1e-10, atol=1e-11)

    def test_mask_and_ignored_target(self):
        c = load()
        x, w, y, a, b = tensors()
        y[2] = -100
        mask = torch.tensor([True, False, True, True, True, False, True])
        x.requires_grad_(); w.requires_grad_()
        lp, ent = c.outputs(x, w, y, mask=mask)
        self.assertEqual(lp[[1, 2, 5]].tolist(), [0.0] * 3)
        self.assertEqual(ent[[1, 2, 5]].tolist(), [0.0] * 3)
        dx, dw = torch.autograd.grad((lp * a + ent * b).sum(), (x, w))
        ax, aw = c.vjp(x.detach(), w.detach(), y, a, b, mask=mask)
        torch.testing.assert_close(dx, ax, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(dw, aw, rtol=1e-10, atol=1e-11)

    def test_logged_entropy_has_no_gradient(self):
        c = load()
        x, w, y, a, b = tensors()
        x.requires_grad_(); w.requires_grad_()
        lp, ent = c.outputs(x, w, y)
        dx, dw = torch.autograd.grad((a * lp + b * ent.detach()).sum(), (x, w))
        ax, aw = c.vjp(x.detach(), w.detach(), y, a, None)
        torch.testing.assert_close(dx, ax)
        torch.testing.assert_close(dw, aw)

    def test_frozen_weight_keeps_input_gradient(self):
        c = load()
        x, w, y, a, b = tensors()
        x.requires_grad_()
        lp, ent = c.outputs(x, w, y)
        dx, = torch.autograd.grad((a * lp + b * ent).sum(), (x,))
        ax, aw = c.vjp(x.detach(), w, y, a, b, need_dw=False)
        torch.testing.assert_close(dx, ax)
        self.assertIsNone(aw)
        self.assertIsNone(w.grad)

    def test_unrequested_input_gradient_is_none(self):
        c = load()
        x, w, y, a, b = tensors()
        ax, aw = c.vjp(x, w, y, a, b, need_dx=False)
        self.assertIsNone(ax)
        self.assertEqual(aw.shape, w.shape)

    def test_all_masked_has_zero_values_and_gradients(self):
        c = load()
        x, w, y, a, b = tensors()
        mask = torch.zeros(7, dtype=torch.bool)
        lp, ent = c.outputs(x, w, y, mask=mask)
        dx, dw = c.vjp(x, w, y, a, b, mask=mask)
        for value in (lp, ent, dx, dw):
            self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_shift_of_all_logits_preserves_values(self):
        c = load()
        x, w, y, _, _ = tensors()
        lp, ent = c.outputs(x, w, y)
        shifted_w = w + torch.tensor([0.5, -0.2, 0.1, 0.3, -0.4])
        lp2, ent2 = c.outputs(x, shifted_w, y)
        torch.testing.assert_close(lp, lp2)
        torch.testing.assert_close(ent, ent2)

    def test_chunked_outputs_and_gradients_match_including_tail(self):
        c = load()
        for size in (1, 3, 7, 10):
            with self.subTest(chunk=size):
                x, w, y, a, b = tensors()
                x.requires_grad_(); w.requires_grad_()
                lp, ent = c.chunked_outputs(x, w, y, chunk_size=size, temperature=0.8)
                dx, dw = torch.autograd.grad((a * lp + b * ent).sum(), (x, w))
                ref_lp, ref_ent = c.outputs(x.detach(), w.detach(), y, temperature=0.8)
                ax, aw = c.vjp(x.detach(), w.detach(), y, a, b, temperature=0.8)
                torch.testing.assert_close(lp, ref_lp)
                torch.testing.assert_close(ent, ref_ent)
                torch.testing.assert_close(dx, ax, rtol=1e-10, atol=1e-11)
                torch.testing.assert_close(dw, aw, rtol=1e-10, atol=1e-11)

    def test_invalid_temperature_rejected(self):
        c = load()
        x, w, y, _, _ = tensors()
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                c.outputs(x, w, y, temperature=value)

    def test_invalid_shapes_targets_and_dtype_rejected(self):
        c = load()
        x, w, y, _, _ = tensors()
        for xx, ww, yy in ((x, w[:, :4], y), (x, w, y[:-1]), (x, w, y.float()),
                           (x.float(), w, y), (x, w, y + 20)):
            with self.assertRaises((ValueError, TypeError)):
                c.outputs(xx, ww, yy)

    def test_upstream_gradient_is_per_token(self):
        c = load()
        x, w, y, a, _ = tensors()
        with self.assertRaises(ValueError):
            c.vjp(x, w, y, torch.ones(1, dtype=torch.float64), None)
        with self.assertRaises(ValueError):
            c.vjp(x, w, y, None, None)
        with self.assertRaises(ValueError):
            c.chunked_outputs(x, w, y, chunk_size=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
