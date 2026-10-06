import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load():
    spec = importlib.util.spec_from_file_location('compat_test', ROOT / 'compile_compat.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class CompatibilityTests(unittest.TestCase):
    def test_removes_only_two_factory_keywords(self):
        m = load()
        raw = b'lp = torch.zeros(T, dtype=torch.float32, requires_grad=output_requires_grad)\nent = x.new_zeros(T, requires_grad=output_requires_grad)\ndw = dz.t() @ x\n'
        patched = m.compatibility_bytes(raw)
        self.assertEqual(patched, b'lp = torch.zeros(T, dtype=torch.float32)\nent = x.new_zeros(T)\ndw = dz.t() @ x\n')

    def test_unexpected_source_rejected(self):
        m = load()
        for raw in (b'', b'a = torch.zeros(T, requires_grad=output_requires_grad)',
                    b'x, requires_grad=output_requires_grad' * 3):
            with self.assertRaises(ValueError):
                m.compatibility_bytes(raw)

    def test_patch_does_not_prune_unneeded_arithmetic(self):
        m = load()
        raw = (b'a = torch.zeros(T, requires_grad=output_requires_grad)\n'
               b'b = x.new_zeros(T, requires_grad=output_requires_grad)\n'
               b'entropy = logsumexp(logits) - (probs * logits).sum()\n'
               b'dw = dlogits.t() @ hidden\n')
        out = m.compatibility_bytes(raw)
        self.assertIn(b'entropy = logsumexp(logits)', out)
        self.assertIn(b'dw = dlogits.t() @ hidden', out)


if __name__ == '__main__':
    unittest.main()
