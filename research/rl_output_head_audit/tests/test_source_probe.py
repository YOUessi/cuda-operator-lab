"""Contracts for the source-fragment probe, not CUDA performance tests."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("head_source_probe", ROOT / "source_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProbeTests(unittest.TestCase):
    def test_blob_identity_uses_git_header(self):
        p = load(); data = b"def f():\n    return 3\n"
        expected = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        self.assertEqual(p.verify_blob(data, expected), expected)

    def test_wrong_blob_rejected(self):
        p = load()
        with self.assertRaises(ValueError):
            p.verify_blob(b"changed", "0" * 40)

    def test_only_selected_definitions_are_loaded(self):
        p = load()
        source = b"raise RuntimeError('top-level must not execute')\ndef f():\n    return 3\n"
        env = p.extract_definitions(source, ["f"], {})
        self.assertEqual(env["f"](), 3)

    def test_missing_definition_rejected(self):
        p = load()
        with self.assertRaises(ValueError):
            p.extract_definitions(b"def f():\n    return 1\n", ["missing"], {})

    def test_duplicate_definition_rejected(self):
        p = load()
        with self.assertRaises(ValueError):
            p.extract_definitions(b"def f():\n    return 1\ndef f():\n    return 2\n", ["f"], {})

    def test_mm_roles_include_recompute_and_unrequested_weight(self):
        p = load()
        self.assertEqual(p.mm_role([7, 5], [5, 11], 7, 5, 11), "projection")
        self.assertEqual(p.mm_role([7, 11], [11, 5], 7, 5, 11), "dX")
        self.assertEqual(p.mm_role([11, 7], [7, 5], 7, 5, 11), "dW")
        self.assertEqual(p.mm_role([2, 3], [3, 2], 7, 5, 11), "other")

    def test_small_probe_dimensions_must_be_distinguishable(self):
        p = load()
        with self.assertRaises(ValueError):
            p.mm_role([5, 5], [5, 5], 5, 5, 5)


if __name__ == "__main__":
    unittest.main()
