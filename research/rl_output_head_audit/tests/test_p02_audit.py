import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load():
    spec = importlib.util.spec_from_file_location('p02_audit_test', ROOT / 'p02_audit.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class AuditTests(unittest.TestCase):
    def test_orders_are_position_balanced(self):
        m = load()
        names = ['original_eager', 'minimal_eager', 'original_compiled', 'minimal_compiled']
        orders = m.balanced_orders(names, 20, 17)
        for pos in range(4):
            self.assertEqual(sorted(order[pos] for order in orders), sorted(names * 5))
        for order in orders:
            self.assertEqual(sorted(order), sorted(names))
        self.assertEqual(orders, m.balanced_orders(names, 20, 17))

    def test_orders_reject_unbalanced_count(self):
        m = load()
        for count in (0, 3, 5):
            with self.assertRaises(ValueError):
                m.balanced_orders(['a', 'b'], count, 1)
        with self.assertRaises(ValueError):
            m.balanced_orders(['a', 'a'], 4, 1)

    def test_errors_do_not_hide_nonfinite(self):
        m = load()
        out = m.error_metrics(torch.tensor([1., 2.]), torch.tensor([1., 2.]))
        self.assertEqual(out['max_abs'], 0)
        with self.assertRaises(ValueError):
            m.error_metrics(torch.tensor([float('nan')]), torch.tensor([0.]))
        with self.assertRaises(ValueError):
            m.error_metrics(torch.ones(2), torch.ones(3))

    def test_graph_recorder_preserves_graph_and_mm_metadata(self):
        m = load()
        gm = torch.fx.symbolic_trace(lambda x, w: torch.mm(x, w.t()))
        for node in gm.graph.nodes:
            if node.op == 'call_function' and node.target == torch.mm:
                node.meta['val'] = torch.empty(7, 11)
        record = m.graph_record(gm)
        self.assertIn('code', record)
        self.assertTrue(record['nodes'])
        self.assertTrue(any(n.get('shape') == [7, 11] for n in record['nodes']))
        self.assertIn('mm', record['code'])

    def test_observation_preserves_logged_entropy(self):
        m = load()
        x, w, y, a, b = m.make_inputs((7, 5, 11), True, torch.float32, 'cpu', 7207)
        import demand_head
        fn = lambda x, w, y: demand_head.head(x, w, y, 0.7, 'entropy_logged')
        result = m.observe(fn, x, w, y, a, b, 'entropy_logged')
        self.assertEqual(set(result), {'logp', 'entropy', 'dX', 'dW'})
        self.assertTrue(all(not t.requires_grad for t in result.values()))

    def test_frozen_weight_observation_has_no_dw(self):
        m = load()
        x, w, y, a, b = m.make_inputs((7, 5, 11), False, torch.float32, 'cpu', 7207)
        import demand_head
        fn = lambda x, w, y: demand_head.head(x, w, y, 0.7, 'logprob_only')
        result = m.observe(fn, x, w, y, a, b, 'logprob_only')
        self.assertEqual(set(result), {'logp', 'dX'})


if __name__ == '__main__':
    unittest.main()
