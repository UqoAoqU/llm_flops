"""Managed worker process for isolated import, build, and correctness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import threading
import time
import traceback
import json
from collections.abc import Sequence
from pathlib import Path

from benchmark_engine.registry.validation import validate_entrypoint
from benchmark_engine.reporting.csv_writer import atomic_write_text
from benchmark_engine.correctness import CorrectnessEvaluator
from benchmark_engine.models import CaseSpec

from .event_log import EventEmitter, EventLog
from .isolation import ensure_beneath, validated_artifact_root, validated_root
from .protocol import (
    EventName,
    WorkerOutcome,
    WorkerRequest,
    WorkerResponse,
    WorkerStage,
    utc_timestamp,
)


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


def _bounded_message(value: object, limit: int = 512) -> str:
    data = str(value).replace("\n", " ").encode("utf-8", errors="replace")[:limit]
    return data.decode("utf-8", errors="ignore")


def _disable_core_dumps() -> None:
    """Avoid a large core dump delaying signal observation by the controller."""

    if os.name != "posix":
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError) as error:
        print(
            f"benchmark-engine warning: could not disable core dumps: {error}",
            file=sys.stderr,
            flush=True,
        )


def _attribute(value: object, dotted: str) -> object:
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def _load_entrypoint(root: Path, entrypoint: str, role: str, token: str) -> object:
    module_name, attribute = entrypoint.split(":", 1)
    module_path = validate_entrypoint(
        root, entrypoint, root / "operator.yaml", f"{role}_entrypoint"
    )
    module_key = f"_benchmark_engine_worker_{role}_{token}_{module_name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {role} module {module_name!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    try:
        spec.loader.exec_module(module)
        return _attribute(module, attribute)
    except BaseException:
        sys.modules.pop(module_key, None)
        raise


def _classify(error: BaseException, extra_text: str = "") -> WorkerOutcome:
    text = f"{type(error).__name__}: {error}\n{extra_text}".lower()
    if isinstance(error, (MemoryError,)) or "out of memory" in text or "cuda oom" in text:
        return WorkerOutcome.OOM
    if isinstance(error, NotImplementedError) or "unsupported" in text:
        return WorkerOutcome.UNSUPPORTED
    if isinstance(error, KeyboardInterrupt):
        return WorkerOutcome.INTERRUPTED
    return WorkerOutcome.ERROR


def _diagnostic_path(artifact_root: Path, result_id: str, stage: WorkerStage) -> Path:
    token = hashlib.sha256(result_id.encode("utf-8")).hexdigest()[:16]
    return artifact_root / "diagnostics" / f"{token}-{stage.value}.txt"


def _tail(path: Path, limit: int = 8 * 1024) -> bytes:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read(limit)
    except OSError:
        return b""


def _recent_logs(artifact_root: Path) -> str:
    data = _tail(artifact_root / "logs" / "stdout.log") + _tail(
        artifact_root / "logs" / "stderr.log"
    )
    return data[-16 * 1024 :].decode("utf-8", errors="replace")


def _heartbeat_loop(
    emitter: EventEmitter,
    artifact_root: Path,
    stop: threading.Event,
    interval_s: float,
) -> None:
    while not stop.wait(interval_s):
        emitter.heartbeat(_recent_logs(artifact_root))


def _skip_from(emitter: EventEmitter, start_index: int, message: str) -> None:
    for started, finished, stage in _PHASES[start_index:]:
        emitter.emit(started, stage, "started")
        emitter.emit(finished, stage, "skipped", message=message)


def _write_response(path: Path, response: WorkerResponse) -> None:
    atomic_write_text(path, response.to_json())


_OPERATOR_SPEC_MEMBERS = (
    "operator_id",
    "cases",
    "make_inputs",
    "clone_inputs",
    "normalize_output",
    "comparator",
    "cost_model",
)


def _validate_operator_spec(runtime_spec: object, request: WorkerRequest) -> bool:
    """Validate a formal Phase-7 spec; return false for a Phase-5 fixture.

    Phase-5 worker fixtures deliberately expose an empty object and remain a
    supported import/build-only compatibility path.  Once an entrypoint
    declares any OperatorSpec member, however, it is a formal spec and partial
    implementations must fail with a stable correctness-stage diagnostic.
    """

    declared = tuple(
        name for name in _OPERATOR_SPEC_MEMBERS if hasattr(runtime_spec, name)
    )
    if not declared:
        return False
    missing = tuple(
        name for name in _OPERATOR_SPEC_MEMBERS if not hasattr(runtime_spec, name)
    )
    if missing:
        raise TypeError(
            "OperatorSpec contract is incomplete; missing: " + ", ".join(missing)
        )
    operator_id = getattr(runtime_spec, "operator_id")
    if not isinstance(operator_id, str) or not operator_id:
        raise TypeError("OperatorSpec.operator_id must be a non-empty string")
    if operator_id != request.identity.operator_id:
        raise ValueError(
            f"OperatorSpec.operator_id {operator_id!r} does not match request "
            f"identity {request.identity.operator_id!r}"
        )
    for name in _OPERATOR_SPEC_MEMBERS[1:]:
        if not callable(getattr(runtime_spec, name)):
            raise TypeError(f"OperatorSpec.{name} must be callable")
    try:
        cases = tuple(runtime_spec.cases())
    except TypeError as error:
        raise TypeError("OperatorSpec.cases() must return CaseSpec values") from error
    if not cases or not all(isinstance(case, CaseSpec) for case in cases):
        raise TypeError("OperatorSpec.cases() must return non-empty CaseSpec values")
    case_ids = tuple(case.case_id for case in cases)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("OperatorSpec.cases() returned duplicate case_id values")
    declared_case = next(
        (case for case in cases if case.case_id == request.case.case_id), None
    )
    if declared_case is None:
        raise ValueError(
            f"request case {request.case.case_id!r} is not declared by OperatorSpec.cases()"
        )
    # CLI seed overrides are allowed, while the shape/tag identity used to
    # construct the plan must still agree with the trusted declaration.
    if (
        dict(declared_case.symbols) != dict(request.case.symbols)
        or declared_case.tags != request.case.tags
    ):
        raise ValueError(
            f"request case {request.case.case_id!r} does not match its OperatorSpec declaration"
        )
    return True


def execute(
    request: WorkerRequest,
    response_path: Path,
    event_path: Path,
    *,
    initial_sequence: int = 1,
) -> WorkerResponse:
    started_clock = time.monotonic()
    started_at = utc_timestamp()
    artifact_root = validated_artifact_root(request.artifact_root)
    response_path = Path(response_path).resolve()
    event_path = Path(event_path).resolve()
    # Response and event locations are controller-owned paths beneath the
    # already validated evaluation artifact root.
    for path, field in ((response_path, "response"), (event_path, "events")):
        try:
            path.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(f"{field} path escapes artifact_root") from error
    reference_root = validated_root(request.reference_root, "reference_root")
    candidate_root = validated_root(request.candidate_root, "candidate_root")
    emitter = EventEmitter(
        EventLog(event_path),
        request,
        initial_sequence=initial_sequence,
        started=started_clock,
    )
    emitter.emit(EventName.WORKER_STARTED, WorkerStage.STARTUP, "started")
    _disable_core_dumps()
    stage_elapsed: dict[str, float] = {}
    stop = threading.Event()
    interval = max(
        0.05,
        min(5.0, float(os.environ.get("BENCHMARK_ENGINE_HEARTBEAT_INTERVAL_S", "1.0"))),
    )
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(emitter, artifact_root, stop, interval),
        daemon=True,
    )
    heartbeat.start()
    outcome = WorkerOutcome.SUCCESS
    failed_stage = WorkerStage.COMPLETE
    diagnostic: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    exit_code = 0
    result_payload: dict[str, object] | None = None
    try:
        stage_started = time.monotonic()
        emitter.emit(EventName.IMPORT_STARTED, WorkerStage.IMPORT, "started")
        try:
            token = hashlib.sha256(request.result_id.encode("utf-8")).hexdigest()[:12]
            runtime_spec = _load_entrypoint(
                reference_root, request.spec_entrypoint, "reference_spec", token
            )
            reference_callable = _load_entrypoint(
                reference_root,
                request.reference_entrypoint,
                "reference_implementation",
                token,
            )
            candidate_callable = _load_entrypoint(
                candidate_root,
                request.candidate_entrypoint,
                "candidate_implementation",
                token,
            )
        except BaseException as error:
            stage_elapsed[WorkerStage.IMPORT.value] = time.monotonic() - stage_started
            outcome = _classify(error)
            failed_stage = WorkerStage.IMPORT
            error_type = type(error).__name__
            error_message = _bounded_message(error)
            path = _diagnostic_path(artifact_root, request.result_id, failed_stage)
            atomic_write_text(path, traceback.format_exc()[-1024 * 1024 :])
            diagnostic = path.relative_to(artifact_root).as_posix()
            emitter.emit(
                EventName.IMPORT_FINISHED,
                WorkerStage.IMPORT,
                outcome.value,
                message=error_message,
            )
            _skip_from(emitter, 1, "skipped after import failure")
        else:
            stage_elapsed[WorkerStage.IMPORT.value] = time.monotonic() - stage_started
            emitter.emit(
                EventName.IMPORT_FINISHED, WorkerStage.IMPORT, "success"
            )

            stage_started = time.monotonic()
            emitter.emit(EventName.BUILD_STARTED, WorkerStage.BUILD, "started")
            try:
                if request.build_argv:
                    completed = subprocess.run(
                        request.build_argv,
                        cwd=candidate_root,
                        stdin=subprocess.DEVNULL,
                        check=False,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"build command exited with status {completed.returncode}"
                        )
            except BaseException as error:
                stage_elapsed[WorkerStage.BUILD.value] = time.monotonic() - stage_started
                outcome = _classify(error, _recent_logs(artifact_root))
                failed_stage = WorkerStage.BUILD
                error_type = type(error).__name__
                error_message = _bounded_message(error)
                path = _diagnostic_path(artifact_root, request.result_id, failed_stage)
                atomic_write_text(path, traceback.format_exc()[-1024 * 1024 :])
                diagnostic = path.relative_to(artifact_root).as_posix()
                emitter.emit(
                    EventName.BUILD_FINISHED,
                    WorkerStage.BUILD,
                    outcome.value,
                    message=error_message,
                )
                _skip_from(emitter, 2, "skipped after build failure")
            else:
                stage_elapsed[WorkerStage.BUILD.value] = time.monotonic() - stage_started
                emitter.emit(
                    EventName.BUILD_FINISHED,
                    WorkerStage.BUILD,
                    "success" if request.build_argv else "skipped",
                    message=None if request.build_argv else "no build argv declared",
                )
                stage_started = time.monotonic()
                emitter.emit(
                    EventName.CORRECTNESS_STARTED,
                    WorkerStage.CORRECTNESS,
                    "started",
                )
                try:
                    formal_spec = _validate_operator_spec(runtime_spec, request)
                    if not formal_spec:
                        emitter.emit(
                            EventName.CORRECTNESS_FINISHED,
                            WorkerStage.CORRECTNESS,
                            "skipped",
                            message="Phase-5 import/build-only fixture",
                        )
                    else:
                        if not callable(reference_callable) or not callable(candidate_callable):
                            raise TypeError("implementation entrypoints must be callable")
                        result = CorrectnessEvaluator(
                            determinism_repeats=request.resolved_config.correctness_determinism_repeats
                        ).evaluate(
                            spec=runtime_spec,
                            reference=reference_callable,
                            candidate=candidate_callable,
                            case=request.case,
                            operator_id=request.identity.operator_id,
                            candidate_id=request.identity.candidate_id,
                        )
                        result_payload = result.to_dict()
                        if result.diagnostic is not None:
                            path = _diagnostic_path(
                                artifact_root, request.result_id, WorkerStage.CORRECTNESS
                            ).with_suffix(".json")
                            atomic_write_text(
                                path,
                                json.dumps(
                                    result_payload,
                                    ensure_ascii=False,
                                    indent=2,
                                    sort_keys=True,
                                    allow_nan=False,
                                )
                                + "\n",
                            )
                            diagnostic = path.relative_to(artifact_root).as_posix()
                except BaseException as error:
                    stage_elapsed[WorkerStage.CORRECTNESS.value] = time.monotonic() - stage_started
                    outcome = _classify(error)
                    failed_stage = WorkerStage.CORRECTNESS
                    error_type = type(error).__name__
                    error_message = _bounded_message(error)
                    path = _diagnostic_path(artifact_root, request.result_id, failed_stage)
                    atomic_write_text(path, traceback.format_exc()[-1024 * 1024 :])
                    diagnostic = path.relative_to(artifact_root).as_posix()
                    emitter.emit(
                        EventName.CORRECTNESS_FINISHED,
                        WorkerStage.CORRECTNESS,
                        outcome.value,
                        message=error_message,
                    )
                else:
                    if formal_spec:
                        emitter.emit(
                            EventName.CORRECTNESS_FINISHED,
                            WorkerStage.CORRECTNESS,
                            "success",
                            message=None if result_payload is None else str(result_payload["status"]),
                        )
                stage_elapsed[WorkerStage.CORRECTNESS.value] = time.monotonic() - stage_started
                _skip_from(
                    emitter,
                    3,
                    "performance_not_implemented",
                )
    finally:
        stop.set()
        heartbeat.join(timeout=max(interval * 2, 0.2))

    response = WorkerResponse(
        identity=request.identity,
        result_id=request.result_id,
        outcome=outcome,
        stage=failed_stage,
        worker_pid=os.getpid(),
        started_at_utc=started_at,
        finished_at_utc=utc_timestamp(),
        elapsed_s=max(0.0, time.monotonic() - started_clock),
        stage_elapsed_s=stage_elapsed,
        diagnostic_path=diagnostic,
        error_type=error_type,
        error_message=error_message,
        exit_code=exit_code,
        result_payload=result_payload,
    )
    _write_response(response_path, response)
    emitter.emit(
        EventName.REPORT_WRITTEN,
        WorkerStage.REPORT,
        "finished" if outcome is WorkerOutcome.SUCCESS else outcome.value,
        message=diagnostic,
    )
    emitter.emit(
        EventName.WORKER_EXITED,
        WorkerStage.COMPLETE,
        outcome.value,
        message=error_message,
    )
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-engine-worker")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--initial-sequence", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        request = WorkerRequest.from_json(arguments.request.read_text(encoding="utf-8"))
        if arguments.initial_sequence < 0:
            raise ValueError("--initial-sequence must be non-negative")
        execute(
            request,
            arguments.response,
            arguments.events,
            initial_sequence=arguments.initial_sequence,
        )
    except KeyboardInterrupt:
        return 130
    except BaseException:
        traceback.print_exc()
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["execute", "main"]
