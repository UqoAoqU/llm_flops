"""Strict JSON-only controller/worker protocol contracts.

The protocol deliberately carries metadata and validated filesystem locations
only.  Runtime Python objects, callables, tensors and pickle payloads never
cross the process boundary.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath

from benchmark_engine.config import ResolvedEvaluationConfig
from benchmark_engine.ids import (
    IdentifierError,
    validate_candidate_id,
    validate_evaluation_id,
    validate_operator_id,
    validate_result_id,
    validate_run_id,
)
from benchmark_engine.models import CaseSpec, EvaluationIdentity


PROTOCOL_SCHEMA_VERSION = 1


class ProtocolError(ValueError):
    """A malformed or incompatible protocol record."""


class WorkerOutcome(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"
    CRASHED = "crashed"
    UNSUPPORTED = "unsupported"
    INTERRUPTED = "interrupted"


class WorkerStage(str, Enum):
    STARTUP = "startup"
    IMPORT = "import"
    BUILD = "build"
    CORRECTNESS = "correctness"
    WARMUP = "warmup"
    SAMPLING = "sampling"
    REPORT = "report"
    COMPLETE = "complete"


class EventName(str, Enum):
    DISCOVERED = "DISCOVERED"
    WORKER_STARTED = "WORKER_STARTED"
    IMPORT_STARTED = "IMPORT_STARTED"
    IMPORT_FINISHED = "IMPORT_FINISHED"
    BUILD_STARTED = "BUILD_STARTED"
    BUILD_FINISHED = "BUILD_FINISHED"
    CORRECTNESS_STARTED = "CORRECTNESS_STARTED"
    CORRECTNESS_FINISHED = "CORRECTNESS_FINISHED"
    WARMUP_STARTED = "WARMUP_STARTED"
    WARMUP_FINISHED = "WARMUP_FINISHED"
    SAMPLING_STARTED = "SAMPLING_STARTED"
    SAMPLING_FINISHED = "SAMPLING_FINISHED"
    HEARTBEAT = "HEARTBEAT"
    REPORT_WRITTEN = "REPORT_WRITTEN"
    WORKER_EXITED = "WORKER_EXITED"


_PAIR_NAMES = (
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


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProtocolError(f"{field} must be an object")
    return value


def _fields(
    value: Mapping[str, object], *, required: frozenset[str], field: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required.difference(value)
    extra = set(value).difference(required | optional)
    if missing:
        raise ProtocolError(f"{field} missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ProtocolError(f"{field} unknown fields: {', '.join(sorted(extra))}")


def _string(value: object, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ProtocolError(f"{field} must be a string")
    return value


def _integer(value: object, field: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{field} must be an integer")
    if nonnegative and value < 0:
        raise ProtocolError(f"{field} must be non-negative")
    return value


def _number(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ProtocolError(f"{field} must be a finite positive number")
    return number


def _version(value: object, field: str) -> int:
    version = _integer(value, field)
    if version != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolError(
            f"{field} unsupported schema version {version}; "
            f"expected {PROTOCOL_SCHEMA_VERSION}"
        )
    return version


def _absolute_path(value: object, field: str) -> Path:
    raw = _string(value, field)
    assert isinstance(raw, str)
    path = Path(raw)
    if not path.is_absolute():
        raise ProtocolError(f"{field} must be absolute")
    return path


def _validate_identity(identity: EvaluationIdentity, result_id: str, field: str) -> None:
    if not isinstance(identity, EvaluationIdentity):
        raise ProtocolError(f"{field}.identity must be EvaluationIdentity")
    try:
        validate_run_id(identity.run_id)
        validate_evaluation_id(identity.evaluation_id)
        validate_operator_id(identity.operator_id)
        validate_candidate_id(identity.candidate_id)
        validate_result_id(result_id)
    except IdentifierError as error:
        raise ProtocolError(f"{field} identity is invalid: {error}") from error


def _relative_diagnostic(value: str | None) -> None:
    if value is None:
        return
    if not value or "\\" in value or len(value.encode("utf-8")) > 512:
        raise ProtocolError("diagnostic_path must be a bounded relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ProtocolError("diagnostic_path must be a bounded relative POSIX path")


def _bounded_text(value: str | None, field: str, limit: int) -> None:
    if value is not None and (
        not isinstance(value, str) or len(value.encode("utf-8")) > limit
    ):
        raise ProtocolError(f"{field} exceeds {limit} UTF-8 bytes")


@dataclass(frozen=True)
class StageTimeouts:
    import_s: float = 60.0
    build_s: float = 600.0
    correctness_s: float = 300.0
    performance_s: float = 600.0

    def __post_init__(self) -> None:
        for name in ("import_s", "build_s", "correctness_s", "performance_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "import_s": float(self.import_s),
            "build_s": float(self.build_s),
            "correctness_s": float(self.correctness_s),
            "performance_s": float(self.performance_s),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StageTimeouts":
        data = _object(value, "timeouts")
        required = frozenset(
            {"import_s", "build_s", "correctness_s", "performance_s"}
        )
        _fields(data, required=required, field="timeouts")
        return cls(
            import_s=_number(data["import_s"], "timeouts.import_s", positive=True),
            build_s=_number(data["build_s"], "timeouts.build_s", positive=True),
            correctness_s=_number(
                data["correctness_s"], "timeouts.correctness_s", positive=True
            ),
            performance_s=_number(
                data["performance_s"], "timeouts.performance_s", positive=True
            ),
        )

    def for_stage(self, stage: WorkerStage) -> float:
        if stage is WorkerStage.IMPORT:
            return float(self.import_s)
        if stage is WorkerStage.BUILD:
            return float(self.build_s)
        if stage is WorkerStage.CORRECTNESS:
            return float(self.correctness_s)
        if stage in {WorkerStage.WARMUP, WorkerStage.SAMPLING}:
            return float(self.performance_s)
        return float(self.import_s)


@dataclass(frozen=True)
class WorkerRequest:
    identity: EvaluationIdentity
    result_id: str
    reference_root: Path
    reference_entrypoint: str
    spec_entrypoint: str
    candidate_root: Path
    candidate_entrypoint: str
    artifact_root: Path
    case: CaseSpec
    mode: str
    resolved_config: ResolvedEvaluationConfig
    build_argv: tuple[str, ...] = ()
    timeouts: StageTimeouts = StageTimeouts()
    schema_version: int = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolError("WorkerRequest schema_version is incompatible")
        _validate_identity(self.identity, self.result_id, "WorkerRequest")
        if self.mode not in {"all", "correctness", "performance"}:
            raise ProtocolError("mode must be all, correctness, or performance")
        if self.resolved_config.mode != self.mode:
            raise ProtocolError("mode and resolved_config.mode disagree")
        for name in ("reference_root", "candidate_root", "artifact_root"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ProtocolError(f"{name} must be an absolute pathlib.Path")
        if not isinstance(self.build_argv, tuple) or not all(
            isinstance(part, str) and part for part in self.build_argv
        ):
            raise ProtocolError("build_argv must be a tuple of non-empty strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "result_id": self.result_id,
            "reference_root": str(self.reference_root),
            "reference_entrypoint": self.reference_entrypoint,
            "spec_entrypoint": self.spec_entrypoint,
            "candidate_root": str(self.candidate_root),
            "candidate_entrypoint": self.candidate_entrypoint,
            "artifact_root": str(self.artifact_root),
            "case": self.case.to_dict(),
            "mode": self.mode,
            "resolved_config": self.resolved_config.to_dict(),
            "build_argv": list(self.build_argv),
            "timeouts": self.timeouts.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorkerRequest":
        data = _object(value, "WorkerRequest")
        required = frozenset(
            {
                "schema_version",
                "identity",
                "result_id",
                "reference_root",
                "reference_entrypoint",
                "spec_entrypoint",
                "candidate_root",
                "candidate_entrypoint",
                "artifact_root",
                "case",
                "mode",
                "resolved_config",
                "build_argv",
                "timeouts",
            }
        )
        _fields(data, required=required, field="WorkerRequest")
        argv = data["build_argv"]
        if not isinstance(argv, list) or not all(
            isinstance(part, str) and part for part in argv
        ):
            raise ProtocolError("WorkerRequest.build_argv must be an array of strings")
        return cls(
            schema_version=_version(data["schema_version"], "WorkerRequest.schema_version"),
            identity=EvaluationIdentity.from_dict(
                _object(data["identity"], "WorkerRequest.identity")
            ),
            result_id=str(_string(data["result_id"], "WorkerRequest.result_id")),
            reference_root=_absolute_path(
                data["reference_root"], "WorkerRequest.reference_root"
            ),
            reference_entrypoint=str(
                _string(
                    data["reference_entrypoint"],
                    "WorkerRequest.reference_entrypoint",
                )
            ),
            spec_entrypoint=str(
                _string(data["spec_entrypoint"], "WorkerRequest.spec_entrypoint")
            ),
            candidate_root=_absolute_path(
                data["candidate_root"], "WorkerRequest.candidate_root"
            ),
            candidate_entrypoint=str(
                _string(
                    data["candidate_entrypoint"],
                    "WorkerRequest.candidate_entrypoint",
                )
            ),
            artifact_root=_absolute_path(
                data["artifact_root"], "WorkerRequest.artifact_root"
            ),
            case=CaseSpec.from_dict(_object(data["case"], "WorkerRequest.case")),
            mode=str(_string(data["mode"], "WorkerRequest.mode")),
            resolved_config=ResolvedEvaluationConfig.from_dict(
                _object(data["resolved_config"], "WorkerRequest.resolved_config")
            ),
            build_argv=tuple(argv),
            timeouts=StageTimeouts.from_dict(
                _object(data["timeouts"], "WorkerRequest.timeouts")
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, value: str) -> "WorkerRequest":
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"invalid WorkerRequest JSON: {error}") from error
        return cls.from_dict(_object(raw, "WorkerRequest"))


@dataclass(frozen=True)
class WorkerResponse:
    identity: EvaluationIdentity
    result_id: str
    outcome: WorkerOutcome
    stage: WorkerStage
    worker_pid: int
    started_at_utc: str
    finished_at_utc: str
    elapsed_s: float
    stage_elapsed_s: Mapping[str, float]
    diagnostic_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    exit_code: int | None = None
    result_payload: Mapping[str, object] | None = None
    schema_version: int = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolError("WorkerResponse schema_version is incompatible")
        _validate_identity(self.identity, self.result_id, "WorkerResponse")
        if isinstance(self.worker_pid, bool) or self.worker_pid <= 0:
            raise ProtocolError("worker_pid must be positive")
        if not isinstance(self.outcome, WorkerOutcome) or not isinstance(
            self.stage, WorkerStage
        ):
            raise ProtocolError("outcome/stage must be stable protocol enums")
        if not math.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ProtocolError("elapsed_s must be finite and non-negative")
        if not isinstance(self.stage_elapsed_s, Mapping):
            raise ProtocolError("stage_elapsed_s must be an object")
        for key, value in self.stage_elapsed_s.items():
            if key not in {stage.value for stage in WorkerStage}:
                raise ProtocolError(f"unknown stage duration {key!r}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProtocolError(f"stage duration {key!r} must be a number")
            if not math.isfinite(float(value)) or value < 0:
                raise ProtocolError(f"stage duration {key!r} is invalid")
        if self.exit_code is not None and (
            isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int)
        ):
            raise ProtocolError("exit_code must be an integer or null")
        _relative_diagnostic(self.diagnostic_path)
        _bounded_text(self.error_type, "error_type", 128)
        _bounded_text(self.error_message, "error_message", 512)
        if self.result_payload is not None:
            if not isinstance(self.result_payload, Mapping) or not all(
                isinstance(key, str) for key in self.result_payload
            ):
                raise ProtocolError("result_payload must be a string-keyed object or null")
            try:
                encoded = json.dumps(self.result_payload, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as error:
                raise ProtocolError("result_payload must be strictly JSON-safe") from error
            if len(encoded) > 1024 * 1024:
                raise ProtocolError("result_payload exceeds 1 MiB")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "result_id": self.result_id,
            "outcome": self.outcome.value,
            "stage": self.stage.value,
            "worker_pid": self.worker_pid,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "elapsed_s": self.elapsed_s,
            "stage_elapsed_s": dict(self.stage_elapsed_s),
            "diagnostic_path": self.diagnostic_path,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "exit_code": self.exit_code,
            "result_payload": None if self.result_payload is None else dict(self.result_payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorkerResponse":
        data = _object(value, "WorkerResponse")
        required = frozenset(
            {
                "schema_version",
                "identity",
                "result_id",
                "outcome",
                "stage",
                "worker_pid",
                "started_at_utc",
                "finished_at_utc",
                "elapsed_s",
                "stage_elapsed_s",
                "diagnostic_path",
                "error_type",
                "error_message",
                "exit_code",
            }
        )
        _fields(
            data,
            required=required,
            optional=frozenset({"result_payload"}),
            field="WorkerResponse",
        )
        durations = _object(data["stage_elapsed_s"], "WorkerResponse.stage_elapsed_s")
        try:
            outcome = WorkerOutcome(data["outcome"])
            stage = WorkerStage(data["stage"])
        except (TypeError, ValueError) as error:
            raise ProtocolError("WorkerResponse outcome/stage is invalid") from error
        return cls(
            schema_version=_version(data["schema_version"], "WorkerResponse.schema_version"),
            identity=EvaluationIdentity.from_dict(
                _object(data["identity"], "WorkerResponse.identity")
            ),
            result_id=str(_string(data["result_id"], "WorkerResponse.result_id")),
            outcome=outcome,
            stage=stage,
            worker_pid=_integer(data["worker_pid"], "WorkerResponse.worker_pid"),
            started_at_utc=str(
                _string(data["started_at_utc"], "WorkerResponse.started_at_utc")
            ),
            finished_at_utc=str(
                _string(data["finished_at_utc"], "WorkerResponse.finished_at_utc")
            ),
            elapsed_s=_number(data["elapsed_s"], "WorkerResponse.elapsed_s"),
            stage_elapsed_s={
                key: _number(item, f"WorkerResponse.stage_elapsed_s.{key}")
                for key, item in durations.items()
            },
            diagnostic_path=_string(
                data["diagnostic_path"],
                "WorkerResponse.diagnostic_path",
                nullable=True,
            ),
            error_type=_string(
                data["error_type"], "WorkerResponse.error_type", nullable=True
            ),
            error_message=_string(
                data["error_message"],
                "WorkerResponse.error_message",
                nullable=True,
            ),
            exit_code=(
                None
                if data["exit_code"] is None
                else _integer(data["exit_code"], "WorkerResponse.exit_code")
            ),
            result_payload=(
                None
                if data.get("result_payload") is None
                else _object(data["result_payload"], "WorkerResponse.result_payload")
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, value: str) -> "WorkerResponse":
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"invalid WorkerResponse JSON: {error}") from error
        return cls.from_dict(_object(raw, "WorkerResponse"))


@dataclass(frozen=True)
class WorkerEvent:
    sequence: int
    event: EventName
    timestamp_utc: str
    pid: int
    identity: EvaluationIdentity
    result_id: str
    stage: WorkerStage
    status: str
    elapsed_s: float
    child_pids: tuple[int, ...] = ()
    log_tail: str = ""
    message: str | None = None
    schema_version: int = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolError("WorkerEvent schema_version is incompatible")
        _validate_identity(self.identity, self.result_id, "WorkerEvent")
        if self.sequence < 0 or self.pid <= 0:
            raise ProtocolError("WorkerEvent sequence/pid is invalid")
        if not isinstance(self.event, EventName) or not isinstance(self.stage, WorkerStage):
            raise ProtocolError("WorkerEvent event/stage is invalid")
        if self.status not in {
            "discovered",
            "started",
            "finished",
            "success",
            "skipped",
            "error",
            "timeout",
            "oom",
            "crashed",
            "unsupported",
            "interrupted",
            "heartbeat",
        }:
            raise ProtocolError("WorkerEvent status is invalid")
        if not math.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ProtocolError("WorkerEvent elapsed_s is invalid")
        if len(self.log_tail.encode("utf-8")) > 16 * 1024:
            raise ProtocolError("WorkerEvent log_tail exceeds 16 KiB")
        if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in self.child_pids):
            raise ProtocolError("WorkerEvent child_pids is invalid")
        _bounded_text(self.message, "WorkerEvent.message", 512)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "event": self.event.value,
            "timestamp_utc": self.timestamp_utc,
            "pid": self.pid,
            "identity": self.identity.to_dict(),
            "result_id": self.result_id,
            "stage": self.stage.value,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "child_pids": list(self.child_pids),
            "log_tail": self.log_tail,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "WorkerEvent":
        data = _object(value, "WorkerEvent")
        required = frozenset(
            {
                "schema_version",
                "sequence",
                "event",
                "timestamp_utc",
                "pid",
                "identity",
                "result_id",
                "stage",
                "status",
                "elapsed_s",
                "child_pids",
                "log_tail",
                "message",
            }
        )
        _fields(data, required=required, field="WorkerEvent")
        raw_pids = data["child_pids"]
        if not isinstance(raw_pids, list):
            raise ProtocolError("WorkerEvent.child_pids must be an array")
        try:
            event = EventName(data["event"])
            stage = WorkerStage(data["stage"])
        except (TypeError, ValueError) as error:
            raise ProtocolError("WorkerEvent event/stage is invalid") from error
        return cls(
            schema_version=_version(data["schema_version"], "WorkerEvent.schema_version"),
            sequence=_integer(
                data["sequence"], "WorkerEvent.sequence", nonnegative=True
            ),
            event=event,
            timestamp_utc=str(
                _string(data["timestamp_utc"], "WorkerEvent.timestamp_utc")
            ),
            pid=_integer(data["pid"], "WorkerEvent.pid"),
            identity=EvaluationIdentity.from_dict(
                _object(data["identity"], "WorkerEvent.identity")
            ),
            result_id=str(_string(data["result_id"], "WorkerEvent.result_id")),
            stage=stage,
            status=str(_string(data["status"], "WorkerEvent.status")),
            elapsed_s=_number(data["elapsed_s"], "WorkerEvent.elapsed_s"),
            child_pids=tuple(
                _integer(pid, "WorkerEvent.child_pids[]") for pid in raw_pids
            ),
            log_tail=str(_string(data["log_tail"], "WorkerEvent.log_tail")),
            message=_string(data["message"], "WorkerEvent.message", nullable=True),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, value: str) -> "WorkerEvent":
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"invalid WorkerEvent JSON: {error}") from error
        return cls.from_dict(_object(raw, "WorkerEvent"))


def _validate_transcript(events: tuple[WorkerEvent, ...]) -> None:
    if not events:
        raise ProtocolError("event transcript is empty")
    base_sequence = events[0].sequence
    for index, event in enumerate(events):
        if event.sequence != base_sequence + index:
            raise ProtocolError("one result transcript must be contiguous")
        if event.identity != events[0].identity or event.result_id != events[0].result_id:
            raise ProtocolError("event transcript identities must not change")
    significant = tuple(event for event in events if event.event is not EventName.HEARTBEAT)
    expected = [EventName.DISCOVERED, EventName.WORKER_STARTED]
    for started, finished, _ in _PAIR_NAMES:
        expected.extend((started, finished))
    expected.extend((EventName.REPORT_WRITTEN, EventName.WORKER_EXITED))
    if [event.event for event in significant] != expected:
        raise ProtocolError("event lifecycle order is invalid")
    for event in events:
        if event.event is not EventName.HEARTBEAT:
            continue
        prior = significant[0]
        for candidate in significant:
            if candidate.sequence > event.sequence:
                break
            prior = candidate
        if prior.event not in {item[0] for item in _PAIR_NAMES}:
            raise ProtocolError("heartbeat is outside an active stage")
        if event.stage is not prior.stage or event.status != "heartbeat":
            raise ProtocolError("heartbeat stage/status is invalid")


def split_event_attempts(
    events: tuple[WorkerEvent, ...],
) -> tuple[tuple[WorkerEvent, ...], ...]:
    """Split a serial event log on each explicit attempt boundary."""

    attempts: list[tuple[WorkerEvent, ...]] = []
    current: list[WorkerEvent] = []
    for event in events:
        if event.event is EventName.DISCOVERED:
            if current:
                attempts.append(tuple(current))
            current = [event]
            continue
        if not current:
            raise ProtocolError("event appears before a DISCOVERED attempt boundary")
        current.append(event)
    if current:
        attempts.append(tuple(current))
    return tuple(attempts)


def validate_event_sequence(events: tuple[WorkerEvent, ...]) -> None:
    """Validate one transcript or a serial evaluation-level event log."""

    if not events:
        raise ProtocolError("event sequence is empty")
    base_sequence = events[0].sequence
    for index, event in enumerate(events):
        if event.sequence != base_sequence + index:
            raise ProtocolError("event sequence numbers must be globally contiguous")
    attempts = split_event_attempts(events)
    if not attempts:
        raise ProtocolError("event sequence contains no attempts")
    for attempt in attempts:
        _validate_transcript(attempt)


def response_outcome_from_text(text: str) -> WorkerOutcome | None:
    lowered = text.lower()
    if "memoryerror" in lowered or "out of memory" in lowered or "cuda oom" in lowered:
        return WorkerOutcome.OOM
    if "notimplementederror" in lowered or "unsupported" in lowered:
        return WorkerOutcome.UNSUPPORTED
    return None


__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "EventName",
    "ProtocolError",
    "StageTimeouts",
    "WorkerEvent",
    "WorkerOutcome",
    "WorkerRequest",
    "WorkerResponse",
    "WorkerStage",
    "response_outcome_from_text",
    "split_event_attempts",
    "utc_timestamp",
    "validate_event_sequence",
]
