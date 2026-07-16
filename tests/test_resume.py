from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark_engine.reporting import (
    ArtifactWriter,
    AtomicCsvTable,
    CsvConflictError,
    EvaluationState,
    ManifestContractError,
    RESULTS_SCHEMA,
    ResumeMismatchError,
    ResumeReader,
)
from tests.reporting_fixtures import manifest, results_row


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "results"
        self.writer = ArtifactWriter(self.root)
        self.item = manifest()
        self.writer.initialize(self.item)

    def test_interrupted_resume_deduplicates_completed_result_ids(self) -> None:
        directory = self.writer.evaluation_dir(self.item.identity)
        table = AtomicCsvTable(directory / "results.csv", RESULTS_SCHEMA)
        completed = results_row()
        table.append(completed)
        self.writer.update_status(self.item.identity, EvaluationState.RUNNING)
        self.writer.update_status(
            self.item.identity,
            EvaluationState.INTERRUPTED,
            terminal_reason="operator interrupted",
        )
        state = ResumeReader(self.root).for_run(self.item.identity.run_id)[0]
        self.assertEqual(state.manifest.status, EvaluationState.INTERRUPTED)
        self.assertEqual(state.completed_result_ids, {completed["result_id"]})
        self.assertEqual(
            state.pending_result_ids(
                (completed["result_id"], "res_ffffffffffffffffffffffffffffffff")
            ),
            ("res_ffffffffffffffffffffffffffffffff",),
        )

        expected = manifest()
        resumed = self.writer.initialize(expected, resume=True)
        self.assertEqual(resumed.completed_result_ids, {completed["result_id"]})
        running = self.writer.update_status(
            self.item.identity, EvaluationState.RUNNING
        )
        self.assertEqual(running.status, EvaluationState.RUNNING)
        self.assertIsNone(running.terminal_reason)
        finished = self.writer.update_status(
            self.item.identity, EvaluationState.COMPLETE
        )
        self.assertEqual(finished.status, EvaluationState.COMPLETE)
        self.assertEqual(table.read_rows(), [table.normalise(completed)])

    def test_resume_rejects_source_environment_and_config_mismatch(self) -> None:
        mismatches = (
            replace(self.item, reference_source_hash="a" * 64),
            replace(self.item, candidate_source_hash="e" * 64),
            replace(self.item, environment_fingerprint="f" * 64),
            replace(
                self.item,
                environment_snapshot={"python": "3.13", "gpu": None},
            ),
            replace(
                self.item,
                resolved_config=replace(
                    self.item.resolved_config,
                    performance_samples=self.item.resolved_config.performance_samples
                    + 1,
                ),
            ),
        )
        for expected in mismatches:
            with self.subTest(expected=expected):
                with self.assertRaises(ResumeMismatchError):
                    self.writer.initialize(expected, resume=True)

    def test_resume_rejects_manifest_schema_mismatch(self) -> None:
        path = self.writer.evaluation_dir(self.item.identity) / "evaluation_manifest.json"
        raw = json.loads(path.read_text("utf-8"))
        raw["schema_version"] = 2
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(ManifestContractError):
            self.writer.initialize(self.item, resume=True)

    def test_completed_result_row_cannot_be_overwritten(self) -> None:
        directory = self.writer.evaluation_dir(self.item.identity)
        table = AtomicCsvTable(directory / "results.csv", RESULTS_SCHEMA)
        row = results_row()
        table.append(row)
        before = (directory / "results.csv").read_bytes()
        changed = dict(row)
        changed["status"] = "failed"
        with self.assertRaises(CsvConflictError):
            table.append(changed)
        self.assertEqual((directory / "results.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
