import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load():
    spec = importlib.util.spec_from_file_location('gpu_compare_test', ROOT / 'gpu_compare.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class GateTests(unittest.TestCase):
    def test_idle_requires_three_samples(self):
        m = load()
        self.assertFalse(m.gate([], []))
        self.assertFalse(m.gate([0, 0], []))
        self.assertTrue(m.gate([0, 1, 2], []))

    def test_any_active_process_blocks_even_low_utilization(self):
        self.assertFalse(load().gate([0, 0, 0], [123]))

    def test_busy_sample_blocks(self):
        self.assertFalse(load().gate([0, 99, 0], []))
        self.assertFalse(load().gate([10, 10, 10], []))

    def test_unknown_or_nonfinite_blocks(self):
        m = load()
        self.assertFalse(m.gate([0, 0, float('nan')], []))
        self.assertFalse(m.gate([0, 0, 0], None))
        self.assertFalse(m.gate([0, -1, 0], []))

    def test_shape_parser_rejects_invalid(self):
        m = load()
        self.assertEqual(m.parse_shape('256x512x32768'), (256, 512, 32768))
        for value in ('1x2', '0x2x3', '-1x2x3', 'axbxc'):
            with self.assertRaises(ValueError):
                m.parse_shape(value)


if __name__ == '__main__':
    unittest.main()
