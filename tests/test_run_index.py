from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark_engine.reporting import ArtifactWriter, ResumeReader
from tests.reporting_fixtures import identity, manifest


class RunIndexTests(unittest.TestCase):
    def test_one_run_maps_multiple_mirrored_evaluations_without_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            writer = ArtifactWriter(root)
            first = manifest()
            second_identity = identity(
                operator_id="other_operator",
                candidate_id="task_b__20260716T120000Z__feedface",
                evaluation_id="20260716T121100Z__aaaaaaaaaaaa__0123456789ab",
            )
            second = manifest(identity=second_identity)
            writer.initialize(first)
            writer.initialize(second)

            states = ResumeReader(root).for_run(first.identity.run_id)
            self.assertEqual(len(states), 2)
            self.assertEqual(
                {state.manifest.identity.operator_id for state in states},
                {"test_operator", "other_operator"},
            )
            self.assertTrue(
                (root / "test_operator" / first.identity.candidate_id).is_dir()
            )
            self.assertTrue(
                (root / "other_operator" / second.identity.candidate_id).is_dir()
            )
            self.assertFalse((root / first.identity.run_id).exists())
            index = (root / "run_index.csv").read_text("utf-8")
            self.assertIn("test_operator/", index)
            self.assertIn("other_operator/", index)


if __name__ == "__main__":
    unittest.main()
