import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location('gpu_validation', ROOT / 'gpu_validation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def snapshot(line='GPU-test, GPU, 580, 16000, 600, 28, 50', pids=None):
    return {'samples': [{'returncode': 0, 'stdout': line} for _ in range(3)],
            'compute_pids': [] if pids is None else pids, 'allowed': False}


class ValidationTests(unittest.TestCase):
    def test_display_load_allows_diagnostic_but_never_timing(self):
        m = load()
        self.assertTrue(m.diagnostic_allowed(snapshot()))
        from gpu_compare import gate
        self.assertFalse(gate([28, 28, 28], []))

    def test_active_compute_blocks(self):
        self.assertFalse(load().diagnostic_allowed(snapshot(pids=[123])))

    def test_unknown_process_query_blocks(self):
        s = snapshot(); s['compute_pids'] = None
        self.assertFalse(load().diagnostic_allowed(s))

    def test_memory_headroom_and_temperature_required(self):
        m = load()
        for line in ('GPU-test, GPU, 580, 16000, 15000, 0, 50',
                     'GPU-test, GPU, 580, 16000, 600, 0, 81'):
            with self.subTest(line=line):
                self.assertFalse(m.diagnostic_allowed(snapshot(line)))

    def test_missing_malformed_multigpu_nonfinite_fail_closed(self):
        m = load()
        for line in ('', 'bad', 'GPU-test, GPU, 580, 16000, 600, nan, 50',
                     'GPU-test, GPU, 580, 16000, 600, 0, 50\nGPU-2, GPU, 580, 16000, 600, 0, 50'):
            with self.subTest(line=line):
                self.assertFalse(m.diagnostic_allowed(snapshot(line)))
        s = snapshot(); s['samples'][0]['returncode'] = 1
        self.assertFalse(m.diagnostic_allowed(s))
        s = snapshot(); s['samples'] = []
        self.assertFalse(m.diagnostic_allowed(s))

    def test_evidence_cannot_claim_performance(self):
        m = load()
        m.validate_evidence({'performance_measured': False, 'records': []})
        for bad in ({'performance_measured': True},
                    {'performance_measured': False, 'summary_us': {}},
                    {'performance_measured': False, 'records': [{'speedup': 1.5}]}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    m.validate_evidence(bad)

    def test_dense_fp32_reference_handles_frozen_weights(self):
        m = load()
        g = torch.Generator().manual_seed(19)
        x = torch.randn(7, 5, generator=g, requires_grad=True)
        w = torch.randn(11, 5, generator=g)
        y = torch.arange(7)
        lp, ent = m.dense_reference(x, w, y)
        torch.testing.assert_close(lp, (x @ w.t()).log_softmax(-1).gather(-1, y[:, None]).squeeze(-1))
        grad = torch.autograd.grad(lp.sum(), x)[0]
        self.assertEqual(grad.shape, x.shape)
        self.assertIsNone(ent)

    def test_reference_bf16_projection_explicitly_promoted(self):
        m = load()
        x = torch.randn(5, 16).to(torch.bfloat16).requires_grad_()
        w = torch.randn(23, 16).to(torch.bfloat16).requires_grad_()
        y = torch.arange(5)
        lp, _ = m.dense_reference(x, w, y)
        self.assertEqual(lp.dtype, torch.float32)
        expected = (x.float() @ w.float().t()).log_softmax(-1)[torch.arange(5), y]
        torch.testing.assert_close(lp, expected)
        dx, dw = torch.autograd.grad(lp.sum(), (x, w))
        self.assertEqual(dx.dtype, x.dtype)
        self.assertEqual(dw.shape, w.shape)


if __name__ == '__main__':
    unittest.main()
