from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark_engine.config import ResolvedEvaluationConfig
from benchmark_engine.execution.protocol import (
    EventName,
    ProtocolError,
    StageTimeouts,
    WorkerEvent,
    WorkerOutcome,
    WorkerRequest,
    WorkerResponse,
    WorkerStage,
    validate_event_sequence,
)
from benchmark_engine.models import CaseSpec, EvaluationIdentity


def request() -> WorkerRequest:
    root = Path(__file__).resolve().parent
    identity = EvaluationIdentity(
        run_id="run_0123456789abcdef",
        evaluation_id="20260716T120000Z__abcdef123456__0123456789ab",
        operator_id="test_operator",
        candidate_id="task__20260716T120000Z__abcdef12",
    )
    return WorkerRequest(
        identity=identity,
        result_id="res_0123456789abcdef0123456789abcdef",
        reference_root=root,
        reference_entrypoint="fixture:operator",
        spec_entrypoint="fixture:spec",
        candidate_root=root,
        candidate_entrypoint="fixture:operator",
        artifact_root=root,
        case=CaseSpec("small", {"m": 2}, 7, frozenset({"smoke"})),
        mode="all",
        resolved_config=ResolvedEvaluationConfig(),
        build_argv=("python", "-c", "pass"),
        timeouts=StageTimeouts(1, 2, 3, 4),
    )


class WorkerProtocolTests(unittest.TestCase):
    def test_request_json_round_trip_is_lossless(self) -> None:
        value = request()
        self.assertEqual(WorkerRequest.from_json(value.to_json()), value)
        json.dumps(value.to_dict(), allow_nan=False)

    def test_response_json_round_trip_is_lossless(self) -> None:
        value = WorkerResponse(
            identity=request().identity,
            result_id=request().result_id,
            outcome=WorkerOutcome.ERROR,
            stage=WorkerStage.IMPORT,
            worker_pid=123,
            started_at_utc="2026-07-16T12:00:00Z",
            finished_at_utc="2026-07-16T12:00:01Z",
            elapsed_s=1.0,
            stage_elapsed_s={"import": 0.75},
            diagnostic_path="diagnostics/import.txt",
            error_type="RuntimeError",
            error_message="short message",
            exit_code=1,
        )
        self.assertEqual(WorkerResponse.from_json(value.to_json()), value)

        base = value.to_dict()
        invalid_values = []
        for diagnostic in ("/tmp/trace.txt", "../trace.txt", "diagnostics/../trace.txt"):
            invalid = dict(base)
            invalid["diagnostic_path"] = diagnostic
            invalid_values.append(invalid)
        long_error = dict(base)
        long_error["error_message"] = "é" * 257
        invalid_values.append(long_error)
        wrong_result = dict(base)
        wrong_result["result_id"] = "not-a-result"
        invalid_values.append(wrong_result)
        unknown = dict(base)
        unknown["traceback"] = "forbidden"
        invalid_values.append(unknown)
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaises((ProtocolError, TypeError, ValueError)):
                    WorkerResponse.from_dict(invalid)

    def test_request_rejects_missing_unknown_type_and_version(self) -> None:
        base = request().to_dict()
        variants = []
        missing = dict(base)
        missing.pop("mode")
        variants.append(missing)
        unknown = dict(base)
        unknown["tensor"] = "not allowed"
        variants.append(unknown)
        wrong_type = dict(base)
        wrong_type["build_argv"] = "python -c pass"
        variants.append(wrong_type)
        version = dict(base)
        version["schema_version"] = 2
        variants.append(version)
        relative = dict(base)
        relative["candidate_root"] = "relative/path"
        variants.append(relative)
        bad_result = dict(base)
        bad_result["result_id"] = "result-1"
        variants.append(bad_result)
        bad_identity = dict(base)
        bad_identity["identity"] = {
            **base["identity"],
            "run_id": "run_12345678",
        }
        variants.append(bad_identity)
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises((ProtocolError, TypeError, ValueError)):
                    WorkerRequest.from_dict(value)

    def test_event_schema_and_complete_sequence(self) -> None:
        identity = request().identity
        names = (
            (EventName.DISCOVERED, WorkerStage.STARTUP, "discovered"),
            (EventName.WORKER_STARTED, WorkerStage.STARTUP, "started"),
            (EventName.IMPORT_STARTED, WorkerStage.IMPORT, "started"),
            (EventName.HEARTBEAT, WorkerStage.IMPORT, "heartbeat"),
            (EventName.IMPORT_FINISHED, WorkerStage.IMPORT, "success"),
            (EventName.BUILD_STARTED, WorkerStage.BUILD, "started"),
            (EventName.BUILD_FINISHED, WorkerStage.BUILD, "skipped"),
            (EventName.CORRECTNESS_STARTED, WorkerStage.CORRECTNESS, "started"),
            (EventName.CORRECTNESS_FINISHED, WorkerStage.CORRECTNESS, "skipped"),
            (EventName.WARMUP_STARTED, WorkerStage.WARMUP, "started"),
            (EventName.WARMUP_FINISHED, WorkerStage.WARMUP, "skipped"),
            (EventName.SAMPLING_STARTED, WorkerStage.SAMPLING, "started"),
            (EventName.SAMPLING_FINISHED, WorkerStage.SAMPLING, "skipped"),
            (EventName.REPORT_WRITTEN, WorkerStage.REPORT, "finished"),
            (EventName.WORKER_EXITED, WorkerStage.COMPLETE, "success"),
        )
        events = tuple(
            WorkerEvent(
                sequence=index,
                event=name,
                timestamp_utc="2026-07-16T12:00:00Z",
                pid=123,
                identity=identity,
                result_id=request().result_id,
                stage=stage,
                status=status,
                elapsed_s=index / 10,
                child_pids=(456,) if name is EventName.HEARTBEAT else (),
                log_tail="ptxas compiling" if name is EventName.HEARTBEAT else "",
            )
            for index, (name, stage, status) in enumerate(names)
        )
        validate_event_sequence(events)
        self.assertEqual(WorkerEvent.from_json(events[3].to_json()), events[3])
        bad = replace(events[3], sequence=99)
        with self.assertRaises(ProtocolError):
            validate_event_sequence(events[:3] + (bad,) + events[4:])

    def test_event_rejects_unbounded_log_tail(self) -> None:
        with self.assertRaises(ProtocolError):
            WorkerEvent(
                sequence=0,
                event=EventName.HEARTBEAT,
                timestamp_utc="2026-07-16T12:00:00Z",
                pid=123,
                identity=request().identity,
                result_id=request().result_id,
                stage=WorkerStage.BUILD,
                status="heartbeat",
                elapsed_s=1,
                log_tail="x" * (16 * 1024 + 1),
            )
        with self.assertRaises(ProtocolError):
            WorkerEvent(
                sequence=0,
                event=EventName.HEARTBEAT,
                timestamp_utc="2026-07-16T12:00:00Z",
                pid=123,
                identity=request().identity,
                result_id=request().result_id,
                stage=WorkerStage.BUILD,
                status="heartbeat",
                elapsed_s=1,
                message="é" * 257,
            )


if __name__ == "__main__":
    unittest.main()
