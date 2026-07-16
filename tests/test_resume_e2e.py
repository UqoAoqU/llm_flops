from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark_engine.engine import (
    build_execution_plan,
    build_resume_plan,
    execute_plan,
)
from benchmark_engine.execution import WorkerController
from benchmark_engine.reporting import ArtifactWriter, EvaluationState
from benchmark_engine.selectors import Selectors
from tests.test_correctness_cli_e2e import OPERATOR, PASS, ROOT


class InterruptAfterOne:
    def __init__(self, writer: ArtifactWriter):
        self.real = WorkerController(artifact_writer=writer)
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        return self.real.run(*args, **kwargs)


class InterruptImmediately:
    def run(self, *args, **kwargs):
        raise KeyboardInterrupt


class ResumeE2ETests(unittest.TestCase):
    def test_interrupted_run_resumes_only_pending_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            plan, snapshot, environment = build_execution_plan(
                ROOT,
                "smoke",
                Selectors(operators=(OPERATOR,), candidates=(PASS,)),
                output_root=output,
                mode="correctness",
            )
            writer = ArtifactWriter(output)
            first = execute_plan(
                plan,
                snapshot,
                environment,
                output_root=output,
                original_command=("bench", "run", "--suite", "smoke"),
                controller=InterruptAfterOne(writer),
            )
            self.assertEqual(first.exit_code, 130)
            identity = plan.jobs[0].identity
            self.assertIs(writer.read_manifest(identity).status, EvaluationState.INTERRUPTED)
            log = writer.evaluation_dir(identity) / "logs" / "controller.jsonl"
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 1)

            resumed, current_snapshot, current_environment = build_resume_plan(
                ROOT, plan.run_id, output_root=output
            )
            second = execute_plan(
                resumed,
                current_snapshot,
                current_environment,
                output_root=output,
                original_command=("bench", "run", "--resume", plan.run_id),
                resume=True,
            )
            self.assertEqual(second.exit_code, 0)
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 2)
            self.assertIs(writer.read_manifest(identity).status, EvaluationState.COMPLETE)

            again, snap, env = build_resume_plan(ROOT, plan.run_id, output_root=output)
            third = execute_plan(
                again,
                snap,
                env,
                output_root=output,
                original_command=("bench", "run", "--resume", plan.run_id),
                resume=True,
            )
            self.assertEqual(third.exit_code, 0)
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 2)

    def test_resume_preserves_original_case_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            plan, snapshot, environment = build_execution_plan(
                ROOT,
                "smoke",
                Selectors(
                    operators=(OPERATOR,), candidates=(PASS,), cases=("tiny",)
                ),
                output_root=output,
                mode="correctness",
            )
            outcome = execute_plan(
                plan,
                snapshot,
                environment,
                output_root=output,
                original_command=(
                    "bench", "run", "--suite", "smoke", "--operator", OPERATOR,
                    "--candidate", PASS, "--case", "tiny",
                ),
                controller=InterruptImmediately(),
            )
            self.assertEqual(outcome.exit_code, 130)
            resumed, _, _ = build_resume_plan(ROOT, plan.run_id, output_root=output)
            self.assertEqual(len(resumed.jobs), 1)
            self.assertEqual(resumed.jobs[0].case.case_id, "tiny")


if __name__ == "__main__":
    unittest.main()
