import os
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.registry import SourceHashError, compute_source_hash


class SourceHashTest(unittest.TestCase):
    def test_order_is_stable_and_contents_are_significant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "b.py").write_text("b = 2\n")
            (root / "a.py").write_text("a = 1\n")
            first = compute_source_hash(root)
            (root / "a.py").write_text("a = 3\n")
            self.assertNotEqual(compute_source_hash(root), first)
            (root / "a.py").write_text("a = 1\n")
            self.assertEqual(compute_source_hash(root), first)

    def test_caches_build_outputs_and_temporary_files_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "implementation.py").write_text("value = 1\n")
            expected = compute_source_hash(root)
            for directory in ("__pycache__", "build", "dist", "demo.egg-info"):
                child = root / directory
                child.mkdir()
                (child / "ignored.txt").write_text(directory)
            (root / "ignored.pyc").write_bytes(b"bytecode")
            (root / "notes.tmp").write_text("temporary")
            self.assertEqual(compute_source_hash(root), expected)

    def test_candidate_id_is_the_only_canonical_manifest_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "candidate.yaml"
            manifest.write_text(
                "schema_version: 1\ncandidate_id: task__20260716T081500Z__00000000\n"
                "framework: python\nmetadata: {answer: 42}\n"
            )
            first = compute_source_hash(root)
            manifest.write_text(
                "metadata:\n  answer: 42\nframework: python\n"
                "candidate_id: task__20260716T081500Z__ffffffff\n"
                "schema_version: 1\n"
            )
            self.assertEqual(compute_source_hash(root), first)
            manifest.write_text(manifest.read_text().replace("answer: 42", "answer: 43"))
            self.assertNotEqual(compute_source_hash(root), first)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.py"
            target.write_text("value = 1\n")
            try:
                (root / "alias.py").symlink_to(target)
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaises(SourceHashError):
                compute_source_hash(root)


if __name__ == "__main__":
    unittest.main()
