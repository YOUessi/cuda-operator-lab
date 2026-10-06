"""CPU-only G01 contracts. No torch import, allocations or server launches."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location("graph_kv_preflight", ROOT / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def packages():
    return {"sglang": "0.5.21", "torch": "2.13.0", "transformers": "5.12.1"}


def samples(util=0):
    return [{"gpu_uuid": "GPU-test", "utilization_pct": util} for _ in range(5)]


def snapshot(reserved=200, allocated=100, process=500):
    return dict(gpu_uuid="GPU-test", pid=42, allocated_bytes=allocated,
                reserved_bytes=reserved, process_used_bytes=process)


class PreflightTests(unittest.TestCase):
    def test_clean_metadata_still_requires_runtime_validation(self):
        r = module().assess(samples(), packages(), [])
        self.assertEqual(r["status"], "runtime_validation_required")
        self.assertFalse(r["performance_started"])

    def test_busy_gpu_blocks(self):
        self.assertIn("gpu_busy", module().assess(samples(97), packages(), [123])["blockers"])

    def test_one_busy_sample_blocks(self):
        s = samples(); s[-1]["utilization_pct"] = 90
        self.assertIn("gpu_busy", module().assess(s, packages(), [])["blockers"])

    def test_unknown_process_list_blocks(self):
        self.assertIn("process_inventory_unknown", module().assess(samples(), packages(), None)["blockers"])

    def test_idle_looking_compute_process_blocks(self):
        self.assertIn("active_compute_processes", module().assess(samples(), packages(), [7])["blockers"])

    def test_short_sampling_blocks(self):
        self.assertIn("insufficient_gpu_samples", module().assess(samples()[:1], packages(), [])["blockers"])

    def test_uninstalled_engine_blocks(self):
        p = packages(); p["sglang"] = None
        self.assertIn("sglang_version_mismatch", module().assess(samples(), p, [])["blockers"])

    def test_old_torch_blocks(self):
        p = packages(); p["torch"] = "2.10.0"
        self.assertIn("torch_version_mismatch", module().assess(samples(), p, [])["blockers"])

    def test_pytorch_local_version_suffix(self):
        p = packages(); p["torch"] = "2.13.0+cu130"
        self.assertNotIn("torch_version_mismatch", module().assess(samples(), p, [])["blockers"])

    def test_invalid_utilization_is_rejected(self):
        for value in (math.nan, math.inf, -1, 101, None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module().assess(samples(value), packages(), [])

    def test_mixed_gpus_rejected(self):
        s = samples(); s[1]["gpu_uuid"] = "GPU-other"
        with self.assertRaises(ValueError): module().assess(s, packages(), [])

    def test_reserved_already_includes_allocated(self):
        r = module().ledger_delta(snapshot(), snapshot(300, 150, 650))
        self.assertEqual(r["torch_reserved_delta_bytes"], 100)
        self.assertEqual(r["process_used_delta_bytes"], 150)
        self.assertEqual(r["unclassified_remainder_delta_bytes"], 50)
        self.assertIsNone(r["graph_metadata_bytes"])

    def test_negative_remainder_not_clipped(self):
        r = module().ledger_delta(snapshot(), snapshot(400, 150, 550))
        self.assertEqual(r["unclassified_remainder_delta_bytes"], -150)

    def test_missing_process_memory_is_not_zero(self):
        r = module().ledger_delta(snapshot(process=None), snapshot(process=None))
        self.assertIsNone(r["process_used_delta_bytes"])
        self.assertIsNone(r["unclassified_remainder_delta_bytes"])

    def test_cross_process_delta_rejected(self):
        b = snapshot(); b["pid"] = 9
        with self.assertRaises(ValueError): module().ledger_delta(snapshot(), b)

    def test_cross_gpu_delta_rejected(self):
        b = snapshot(); b["gpu_uuid"] = "GPU-other"
        with self.assertRaises(ValueError): module().ledger_delta(snapshot(), b)

    def test_invalid_allocator_accounting_rejected(self):
        with self.assertRaises(ValueError): module().ledger_delta(snapshot(), snapshot(100, 200))

    def test_no_gpu_runtime_imports(self):
        import ast
        tree = ast.parse((ROOT / "preflight.py").read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(a.name for a in node.names)
            if isinstance(node, ast.ImportFrom): imports.append(node.module or "")
        self.assertFalse(any(n.split('.')[0] in {'torch','sglang','vllm','transformers'} for n in imports))


if __name__ == "__main__":
    unittest.main()
