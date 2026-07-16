"""Controller-side managed worker lifecycle.

The controller performs static request validation and process supervision only;
it never imports reference or candidate implementation modules.
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from benchmark_engine.models import EvaluationJob
from benchmark_engine.registry.validation import validate_entrypoint
from benchmark_engine.reporting import ArtifactWriter
from benchmark_engine.reporting.csv_writer import atomic_write_text

from .event_log import EventLog
from .isolation import (
    BoundedLogDrain,
    bounded_append_text,
    cleanup_exited_process_group,
    terminate_process_group,
    validated_artifact_root,
    validated_root,
)
from .protocol import (
    EventName,
    ProtocolError,
    StageTimeouts,
    WorkerEvent,
    WorkerOutcome,
    WorkerRequest,
    WorkerResponse,
    WorkerStage,
    response_outcome_from_text,
    utc_timestamp,
    validate_event_sequence,
)


class ControllerError(RuntimeError):
    pass


_PHASES = (
    (EventName.IMPORT_STARTED, EventName.IMPORT_FINISHED, WorkerStage.IMPORT),
    (EventName.BUILD_STARTED, EventName.BUILD_FINISHED, WorkerStage.BUILD),
    (
        EventName.CORRECTNESS_STARTED,
        EventName.CORRECTNESS_FINISHED,
        WorkerStage.CORRECTNESS,
    ),
    (EventName.WARMUP_STARTED, EventName.WARMUP_FINISHED, WorkerStage.WARMUP),
    (
        EventName.SAMPLING_STARTED,
        EventName.SAMPLING_FINISHED,
        WorkerStage.SAMPLING,
    ),
)


def build_worker_request(
    job: EvaluationJob,
    *,
    spec_entrypoint: str,
    artifact_root: Path | None = None,
    build_argv: tuple[str, ...] = (),
    timeouts: StageTimeouts | None = None,
) -> WorkerRequest:
    """Construct and statically validate a JSON-only worker request."""

    reference_root = validated_root(job.reference.root, "reference_root")
    candidate_root = validated_root(job.candidate.root, "candidate_root")
    output = validated_artifact_root(artifact_root or job.output_dir)
    validate_entrypoint(
        reference_root,
        spec_entrypoint,
        reference_root / "operator.yaml",
        "spec_entrypoint",
    )
    validate_entrypoint(
        reference_root,
        job.reference.entrypoint,
        reference_root / "operator.yaml",
        "reference_entrypoint",
    )
    validate_entrypoint(
        candidate_root,
        job.candidate.entrypoint,
        candidate_root / "candidate.yaml",
        "candidate_entrypoint",
    )
    return WorkerRequest(
        identity=job.identity,
        result_id=job.result_id,
        reference_root=reference_root,
        reference_entrypoint=job.reference.entrypoint,
        spec_entrypoint=spec_entrypoint,
        candidate_root=candidate_root,
        candidate_entrypoint=job.candidate.entrypoint,
        artifact_root=output,
        case=job.case,
        mode=job.mode,
        resolved_config=job.resolved_config,
        build_argv=tuple(build_argv),
        timeouts=timeouts or StageTimeouts(
            correctness_s=float(job.case.timeout_s or 300),
            performance_s=float(job.resolved_config.performance_timeout_s),
        ),
    )


def _active_stage(events: tuple[WorkerEvent, ...]) -> WorkerStage | None:
    active: WorkerStage | None = None
    for event in events:
        if event.event.value.endswith("_STARTED"):
            active = event.stage
        elif event.event.value.endswith("_FINISHED") and event.stage is active:
            active = None
    return active


def _short_message(text: str) -> str:
    data = " ".join(text.strip().split()).encode("utf-8", errors="replace")[:512]
    return data.decode("utf-8", errors="ignore")


def _synthetic_response(
    request: WorkerRequest,
    *,
    pid: int,
    outcome: WorkerOutcome,
    stage: WorkerStage,
    started_at: str,
    elapsed_s: float,
    diagnostic_path: str | None,
    error_type: str,
    message: str,
    exit_code: int | None,
) -> WorkerResponse:
    return WorkerResponse(
        identity=request.identity,
        result_id=request.result_id,
        outcome=outcome,
        stage=stage,
        worker_pid=pid,
        started_at_utc=started_at,
        finished_at_utc=utc_timestamp(),
        elapsed_s=max(0.0, elapsed_s),
        stage_elapsed_s={},
        diagnostic_path=diagnostic_path,
        error_type=error_type,
        error_message=_short_message(message),
        exit_code=exit_code,
    )


def _append_synthetic_lifecycle(
    event_log: EventLog,
    events: tuple[WorkerEvent, ...],
    request: WorkerRequest,
    *,
    pid: int,
    outcome: WorkerOutcome,
    stage: WorkerStage,
    elapsed_s: float,
    message: str,
) -> tuple[WorkerEvent, ...]:
    """Close an event transcript after a signal, timeout, or abrupt exit."""

    current = list(events)
    significant = [event.event for event in current if event.event is not EventName.HEARTBEAT]
    sequence = 0 if not current else current[-1].sequence + 1

    def append(event: EventName, event_stage: WorkerStage, status: str, note: str | None = None) -> None:
        nonlocal sequence
        value = WorkerEvent(
            sequence=sequence,
            event=event,
            timestamp_utc=utc_timestamp(),
            pid=pid,
            identity=request.identity,
            result_id=request.result_id,
            stage=event_stage,
            status=status,
            elapsed_s=elapsed_s,
            message=note,
        )
        event_log.append(value)
        current.append(value)
        significant.append(event)
        sequence += 1

    if EventName.WORKER_STARTED not in significant:
        append(EventName.WORKER_STARTED, WorkerStage.STARTUP, "started")
    for started, finished, event_stage in _PHASES:
        has_started = started in significant
        has_finished = finished in significant
        if has_finished:
            continue
        if not has_started:
            append(started, event_stage, "started")
            append(finished, event_stage, "skipped", f"skipped after {stage.value} failure")
        else:
            status = outcome.value if event_stage is stage else "skipped"
            append(finished, event_stage, status, message)
    if EventName.REPORT_WRITTEN not in significant:
        append(EventName.REPORT_WRITTEN, WorkerStage.REPORT, outcome.value, message)
    if EventName.WORKER_EXITED not in significant:
        append(EventName.WORKER_EXITED, WorkerStage.COMPLETE, outcome.value, message)
    return tuple(current)


class WorkerController:
    def __init__(
        self,
        *,
        python_executable: str | Path = sys.executable,
        artifact_writer: ArtifactWriter | None = None,
        poll_interval_s: float = 0.02,
        terminate_grace_s: float = 0.25,
        log_limit_bytes: int = 8 * 1024 * 1024,
        summary_limit_bytes: int = 16 * 1024,
        poll_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.python_executable = str(python_executable)
        self.artifact_writer = artifact_writer
        self.poll_interval_s = poll_interval_s
        self.terminate_grace_s = terminate_grace_s
        self.log_limit_bytes = log_limit_bytes
        self.summary_limit_bytes = summary_limit_bytes
        self.poll_sleep = poll_sleep

    def run(
        self,
        job: EvaluationJob,
        *,
        spec_entrypoint: str,
        build_argv: tuple[str, ...] = (),
        timeouts: StageTimeouts | None = None,
    ) -> WorkerResponse:
        artifact_root = Path(job.output_dir).resolve()
        if self.artifact_writer is not None:
            expected = self.artifact_writer.evaluation_dir(job.identity)
            if artifact_root != expected:
                raise ControllerError(
                    f"job output_dir {artifact_root} does not match ArtifactWriter {expected}"
                )
        artifact_root.mkdir(parents=True, exist_ok=True)
        logs = artifact_root / "logs"
        diagnostics = artifact_root / "diagnostics"
        logs.mkdir(exist_ok=True)
        diagnostics.mkdir(exist_ok=True)
        request = build_worker_request(
            job,
            spec_entrypoint=spec_entrypoint,
            artifact_root=artifact_root,
            build_argv=build_argv,
            timeouts=timeouts,
        )
        result_token = hashlib.sha256(job.result_id.encode("utf-8")).hexdigest()[:16]
        event_path = logs / "worker.jsonl"
        stdout_path = logs / "stdout.log"
        stderr_path = logs / "stderr.log"
        event_log = EventLog(event_path)
        try:
            existing_events = event_log.read()
            if existing_events:
                validate_event_sequence(existing_events)
        except ProtocolError as error:
            raise ControllerError(f"existing worker event log is invalid: {error}") from error
        first_sequence = (
            0 if not existing_events else existing_events[-1].sequence + 1
        )
        attempt_number = 1 + sum(
            event.event is EventName.DISCOVERED
            and event.identity == request.identity
            and event.result_id == request.result_id
            for event in existing_events
        )
        attempt_token = f"{result_token}.attempt-{attempt_number:04d}"
        request_path = diagnostics / f"{attempt_token}.request.json"
        response_path = diagnostics / f"{attempt_token}.response.json"
        atomic_write_text(request_path, request.to_json())
        event_log.append(
            WorkerEvent(
                sequence=first_sequence,
                event=EventName.DISCOVERED,
                timestamp_utc=utc_timestamp(),
                pid=os.getpid(),
                identity=request.identity,
                result_id=request.result_id,
                stage=WorkerStage.STARTUP,
                status="discovered",
                elapsed_s=0.0,
            )
        )
        started = time.monotonic()
        started_at = utc_timestamp()
        process = subprocess.Popen(
            (
                self.python_executable,
                "-m",
                "benchmark_engine.execution.worker",
                "--request",
                str(request_path),
                "--response",
                str(response_path),
                "--events",
                str(event_path),
                "--initial-sequence",
                str(first_sequence + 1),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout = BoundedLogDrain(
            process.stdout,
            stdout_path,
            file_limit_bytes=self.log_limit_bytes,
            summary_limit_bytes=self.summary_limit_bytes,
        )
        stderr = BoundedLogDrain(
            process.stderr,
            stderr_path,
            file_limit_bytes=self.log_limit_bytes,
            summary_limit_bytes=self.summary_limit_bytes,
        )
        stdout.start()
        stderr.start()
        timed_out = False
        interrupted = False
        stage = WorkerStage.IMPORT
        stage_seen_at = started
        last_stage: WorkerStage | None = None
        try:
            while process.poll() is None:
                events = event_log.for_result(request)
                active = _active_stage(events)
                if active is not None and active is not last_stage:
                    last_stage = active
                    stage = active
                    stage_seen_at = time.monotonic()
                timeout = request.timeouts.for_stage(stage)
                if time.monotonic() - stage_seen_at > timeout:
                    timed_out = True
                    terminate_process_group(process, grace_s=self.terminate_grace_s)
                    break
                self.poll_sleep(self.poll_interval_s)
        except KeyboardInterrupt:
            interrupted = True
            terminate_process_group(process, grace_s=self.terminate_grace_s)
        finally:
            if process.poll() is None:
                terminate_process_group(process, grace_s=self.terminate_grace_s)
            else:
                process.wait()
                cleanup_exited_process_group(process.pid)
            drain_errors: list[BaseException] = []
            for drain in (stdout, stderr):
                try:
                    drain.join(2)
                except BaseException as error:
                    drain_errors.append(error)
                    try:
                        drain.source.close()
                    except OSError:
                        pass
                if not drain.source.closed:
                    drain.source.close()
            if drain_errors:
                raise ControllerError(
                    "worker log drain cleanup failed: "
                    + "; ".join(str(error) for error in drain_errors)
                )

        elapsed = time.monotonic() - started
        events = event_log.for_result(request)
        diagnostic_path: str | None = None
        message = ""
        if interrupted:
            outcome = WorkerOutcome.INTERRUPTED
            message = "worker interrupted by user"
            error_type = "KeyboardInterrupt"
            response = None
        elif timed_out:
            outcome = WorkerOutcome.TIMEOUT
            message = f"{stage.value} stage exceeded {request.timeouts.for_stage(stage):g}s"
            error_type = "StageTimeout"
            response = None
        else:
            response = None
            if response_path.is_file():
                try:
                    response = WorkerResponse.from_json(
                        response_path.read_text(encoding="utf-8")
                    )
                    if response.identity != request.identity or response.result_id != request.result_id:
                        raise ProtocolError("worker response identity mismatch")
                except (OSError, ProtocolError, TypeError, ValueError) as error:
                    message = f"invalid worker response: {error}"
                    error_type = "ProtocolError"
                    response = None
            if response is not None:
                outcome = response.outcome
                error_type = response.error_type or ""
                message = response.error_message or ""
            else:
                combined = stdout.tail() + "\n" + stderr.tail()
                classified = response_outcome_from_text(combined)
                outcome = classified or WorkerOutcome.CRASHED
                error_type = "WorkerCrash"
                if not message:
                    message = (
                        f"worker exited with status {process.returncode} without a valid response"
                    )

        if response is None:
            diagnostic = diagnostics / f"{attempt_token}.controller.txt"
            atomic_write_text(
                diagnostic,
                json.dumps(
                    {
                        "outcome": outcome.value,
                        "stage": stage.value,
                        "returncode": process.returncode,
                        "message": message,
                        "stdout_tail": stdout.tail(),
                        "stderr_tail": stderr.tail(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            diagnostic_path = diagnostic.relative_to(artifact_root).as_posix()
            response = _synthetic_response(
                request,
                pid=process.pid,
                outcome=outcome,
                stage=stage,
                started_at=started_at,
                elapsed_s=elapsed,
                diagnostic_path=diagnostic_path,
                error_type=error_type,
                message=message,
                exit_code=130 if interrupted else process.returncode,
            )
            atomic_write_text(response_path, response.to_json())
            events = _append_synthetic_lifecycle(
                event_log,
                events,
                request,
                pid=process.pid,
                outcome=outcome,
                stage=stage,
                elapsed_s=elapsed,
                message=_short_message(message),
            )
        try:
            validate_event_sequence(event_log.read())
        except ProtocolError as error:
            if response.outcome is WorkerOutcome.SUCCESS:
                raise ControllerError(f"invalid worker event transcript: {error}") from error
        controller_record = (
            json.dumps(
                {
                    "schema_version": 1,
                    "timestamp_utc": utc_timestamp(),
                    "pid": os.getpid(),
                    "worker_pid": process.pid,
                    "result_id": request.result_id,
                    "attempt": attempt_number,
                    "outcome": response.outcome.value,
                    "stage": response.stage.value,
                    "elapsed_s": elapsed,
                    "stdout_truncated": stdout.truncated,
                    "stderr_truncated": stderr.truncated,
                },
                sort_keys=True,
            )
            + "\n"
        )
        if not bounded_append_text(
            logs / "controller.jsonl",
            controller_record,
            limit_bytes=self.log_limit_bytes,
        ):
            raise ControllerError("controller.jsonl reached its configured byte limit")
        return response


__all__ = ["ControllerError", "WorkerController", "build_worker_request"]
