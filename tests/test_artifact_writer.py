from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from benchmark_engine.reporting import (
    ArtifactWriter,
    EvaluationState,
    ManifestContractError,
    atomic_write_text,
)
from tests.reporting_fixtures import manifest


class ArtifactWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "results"
        self.clock_values = iter(
            (
                datetime(2026, 7, 16, 12, 11, tzinfo=timezone.utc),
                datetime(2026, 7, 16, 12, 12, tzinfo=timezone.utc),
            )
        )
        self.writer = ArtifactWriter(self.root, now=lambda: next(self.clock_values))

    def test_initialization_creates_contract_but_not_complete_indexes(self) -> None:
        item = manifest()
        state = self.writer.initialize(item)
        directory = self.writer.evaluation_dir(item.identity)
        self.assertEqual(state.manifest.status, EvaluationState.PLANNED)
        for relative in (
            "evaluation_manifest.json",
            "summary.md",
            "results.csv",
            "correctness_outputs.csv",
            "performance_samples.csv",
            "model_projection.csv",
            "logs/controller.jsonl",
            "logs/worker.jsonl",
            "logs/stdout.log",
            "logs/stderr.log",
        ):
            self.assertTrue((directory / relative).is_file(), relative)
        self.assertTrue((directory / "diagnostics").is_dir())
        candidate_root = directory.parent
        self.assertFalse((candidate_root / "latest.json").exists())
        self.assertFalse((candidate_root / "history.csv").exists())

    def test_latest_and_history_update_only_after_complete(self) -> None:
        item = manifest()
        self.writer.initialize(item)
        candidate_root = self.writer.evaluation_dir(item.identity).parent
        self.writer.update_status(item.identity, EvaluationState.RUNNING)
        self.assertFalse((candidate_root / "latest.json").exists())
        completed = self.writer.update_status(item.identity, EvaluationState.COMPLETE)
        latest = json.loads((candidate_root / "latest.json").read_text("utf-8"))
        self.assertEqual(latest["status"], "complete")
        self.assertEqual(latest["evaluation_id"], item.identity.evaluation_id)
        self.assertIn("complete", (candidate_root / "history.csv").read_text("utf-8"))
        self.assertEqual(completed.status, EvaluationState.COMPLETE)
        self.writer.update_status(item.identity, EvaluationState.COMPLETE)

    def test_idempotent_older_complete_does_not_move_latest_backwards(self) -> None:
        timestamps = iter(
            datetime(2026, 7, 16, 12, minute, tzinfo=timezone.utc)
            for minute in (11, 12, 13, 14)
        )
        writer = ArtifactWriter(self.root, now=lambda: next(timestamps))
        first = manifest()
        second = manifest(
            identity=type(first.identity)(
                run_id=first.identity.run_id,
                evaluation_id="20260716T121100Z__aaaaaaaaaaaa__0123456789ab",
                operator_id=first.identity.operator_id,
                candidate_id=first.identity.candidate_id,
            )
        )
        writer.initialize(first)
        writer.initialize(second)
        writer.update_status(first.identity, EvaluationState.RUNNING)
        writer.update_status(first.identity, EvaluationState.COMPLETE)
        writer.update_status(second.identity, EvaluationState.RUNNING)
        writer.update_status(second.identity, EvaluationState.COMPLETE)

        candidate_root = writer.evaluation_dir(first.identity).parent
        latest_path = candidate_root / "latest.json"
        self.assertEqual(
            json.loads(latest_path.read_text("utf-8"))["evaluation_id"],
            second.identity.evaluation_id,
        )
        writer.update_status(first.identity, EvaluationState.COMPLETE)
        self.assertEqual(
            json.loads(latest_path.read_text("utf-8"))["evaluation_id"],
            second.identity.evaluation_id,
        )

    def test_equal_completion_time_uses_evaluation_id_tie_break(self) -> None:
        timestamp = datetime(2026, 7, 16, 12, 15, tzinfo=timezone.utc)
        writer = ArtifactWriter(self.root, now=lambda: timestamp)
        first = manifest()
        second = manifest(
            identity=type(first.identity)(
                run_id=first.identity.run_id,
                evaluation_id="20260716T121100Z__aaaaaaaaaaaa__0123456789ab",
                operator_id=first.identity.operator_id,
                candidate_id=first.identity.candidate_id,
            )
        )
        writer.initialize(first)
        writer.initialize(second)
        for item in (first, second):
            writer.update_status(item.identity, EvaluationState.RUNNING)
            writer.update_status(item.identity, EvaluationState.COMPLETE)
        latest = json.loads(
            (writer.evaluation_dir(first.identity).parent / "latest.json").read_text(
                "utf-8"
            )
        )
        self.assertEqual(latest["evaluation_id"], second.identity.evaluation_id)

    def test_failed_and_interrupted_are_not_published(self) -> None:
        for terminal in (EvaluationState.FAILED, EvaluationState.INTERRUPTED):
            with self.subTest(terminal=terminal):
                temporary = tempfile.TemporaryDirectory()
                self.addCleanup(temporary.cleanup)
                writer = ArtifactWriter(Path(temporary.name) / "results")
                item = manifest()
                writer.initialize(item)
                writer.update_status(item.identity, EvaluationState.RUNNING)
                writer.update_status(item.identity, terminal, terminal_reason="boom")
                candidate_root = writer.evaluation_dir(item.identity).parent
                self.assertFalse((candidate_root / "latest.json").exists())
                self.assertFalse((candidate_root / "history.csv").exists())

    def test_illegal_state_transition_is_rejected(self) -> None:
        item = manifest()
        self.writer.initialize(item)
        with self.assertRaisesRegex(ManifestContractError, "planned -> complete"):
            self.writer.update_status(item.identity, EvaluationState.COMPLETE)

    def test_manifest_identity_mismatch_is_rejected(self) -> None:
        item = manifest()
        self.writer.initialize(item)
        path = self.writer.evaluation_dir(item.identity) / "evaluation_manifest.json"
        raw = json.loads(path.read_text("utf-8"))
        raw["identity"]["operator_id"] = "other_operator"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ManifestContractError, "identity"):
            self.writer.read_manifest(item.identity)

    def test_fault_before_replace_keeps_old_json_readable(self) -> None:
        path = self.root / "stable.json"
        atomic_write_text(path, '{"version": 1}\n')

        def fail_replace(source: object, target: object) -> None:
            raise OSError("injected replace failure")

        with self.assertRaisesRegex(OSError, "injected"):
            atomic_write_text(path, '{"version": 2}\n', replace=fail_replace)
        self.assertEqual(json.loads(path.read_text("utf-8")), {"version": 1})
        self.assertEqual(list(path.parent.glob(".stable.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
