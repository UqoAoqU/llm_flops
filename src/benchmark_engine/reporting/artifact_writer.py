"""Crash-safe evaluation artifacts, mirrored indexes, and resume discovery."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from benchmark_engine.config import ResolvedEvaluationConfig
from benchmark_engine.ids import (
    candidate_result_path,
    evaluation_result_path,
    validate_run_id,
)
from benchmark_engine.models import EvaluationIdentity

from .csv_writer import (
    CSV_SCHEMA_VERSION,
    AtomicCsvTable,
    CORRECTNESS_OUTPUTS_SCHEMA,
    CsvColumn,
    CsvContractError,
    CsvSchema,
    PERFORMANCE_SAMPLES_SCHEMA,
    RESULTS_SCHEMA,
    atomic_write_text,
)
from .summary import render_summary


MANIFEST_SCHEMA_VERSION = 1


class ArtifactError(RuntimeError):
    """Base class for artifact lifecycle failures."""


class ArtifactExistsError(ArtifactError):
    """Raised when a new evaluation would overwrite an existing directory."""


class ManifestContractError(ArtifactError):
    """Raised for malformed manifests, identities, or lifecycle transitions."""


class ResumeMismatchError(ArtifactError):
    """Raised when on-disk identity/configuration cannot safely be resumed."""


class EvaluationState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


_ALLOWED_TRANSITIONS = {
    EvaluationState.PLANNED: frozenset({EvaluationState.RUNNING}),
    EvaluationState.RUNNING: frozenset(
        {
            EvaluationState.COMPLETE,
            EvaluationState.FAILED,
            EvaluationState.INTERRUPTED,
        }
    ),
    EvaluationState.COMPLETE: frozenset(),
    EvaluationState.FAILED: frozenset(),
    EvaluationState.INTERRUPTED: frozenset({EvaluationState.RUNNING}),
}


RUN_INDEX_SCHEMA = CsvSchema(
    "run_index.csv",
    tuple(
        CsvColumn(
            name,
            "integer" if name == "schema_version" else "string",
            False,
        )
        for name in (
            "schema_version",
            "run_id",
            "operator_id",
            "candidate_id",
            "evaluation_id",
            "relative_path",
        )
    ),
    ("run_id", "operator_id", "candidate_id", "evaluation_id"),
)

HISTORY_SCHEMA = CsvSchema(
    "history.csv",
    tuple(
        CsvColumn(
            name,
            "integer" if name == "schema_version" else "string",
            False,
            frozenset({EvaluationState.COMPLETE.value}) if name == "status" else None,
        )
        for name in (
            "schema_version",
            "evaluation_id",
            "run_id",
            "operator_id",
            "candidate_id",
            "status",
            "completed_at_utc",
            "relative_path",
            "suite_id",
            "mode",
            "reference_source_hash",
            "candidate_source_hash",
            "environment_fingerprint",
        )
    ),
    ("evaluation_id",),
)


def _json_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ManifestContractError(f"{field} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ManifestContractError(f"{field} must be JSON-safe") from error
    return dict(value)


def _utc_now(now: Callable[[], datetime]) -> str:
    value = now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class EvaluationManifest:
    """Schema-v1 durable identity and state for one evaluation directory."""

    identity: EvaluationIdentity
    status: EvaluationState
    original_command: tuple[str, ...]
    resolved_config: ResolvedEvaluationConfig
    reference_source_hash: str
    candidate_source_hash: str
    environment_snapshot: Mapping[str, object]
    environment_fingerprint: str
    suite_id: str
    mode: str
    created_at_utc: str
    updated_at_utc: str
    terminal_reason: str | None = None
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != MANIFEST_SCHEMA_VERSION
        ):
            raise ManifestContractError(
                f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}"
            )
        if not isinstance(self.identity, EvaluationIdentity):
            raise ManifestContractError("identity must be EvaluationIdentity")
        validate_run_id(self.identity.run_id)
        if not isinstance(self.status, EvaluationState):
            raise ManifestContractError("status must be EvaluationState")
        if not isinstance(self.original_command, tuple) or not all(
            isinstance(part, str) for part in self.original_command
        ):
            raise ManifestContractError("original_command must contain strings")
        if not isinstance(self.resolved_config, ResolvedEvaluationConfig):
            raise ManifestContractError(
                "resolved_config must be ResolvedEvaluationConfig"
            )
        for name in (
            "reference_source_hash",
            "candidate_source_hash",
            "environment_fingerprint",
            "suite_id",
            "mode",
            "created_at_utc",
            "updated_at_utc",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ManifestContractError(f"{name} must be a non-empty string")
        if self.mode not in {"all", "correctness", "performance"}:
            raise ManifestContractError("mode is invalid")
        if self.mode != self.resolved_config.mode:
            raise ManifestContractError("mode and resolved_config.mode disagree")
        _json_mapping(self.environment_snapshot, "environment_snapshot")
        if self.terminal_reason is not None and not isinstance(
            self.terminal_reason, str
        ):
            raise ManifestContractError("terminal_reason must be a string or null")

    @classmethod
    def create(
        cls,
        *,
        identity: EvaluationIdentity,
        original_command: Sequence[str],
        resolved_config: ResolvedEvaluationConfig,
        reference_source_hash: str,
        candidate_source_hash: str,
        environment_snapshot: Mapping[str, object],
        environment_fingerprint: str,
        suite_id: str,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> "EvaluationManifest":
        timestamp = _utc_now(now)
        return cls(
            identity=identity,
            status=EvaluationState.PLANNED,
            original_command=tuple(original_command),
            resolved_config=resolved_config,
            reference_source_hash=reference_source_hash,
            candidate_source_hash=candidate_source_hash,
            environment_snapshot=dict(environment_snapshot),
            environment_fingerprint=environment_fingerprint,
            suite_id=suite_id,
            mode=resolved_config.mode,
            created_at_utc=timestamp,
            updated_at_utc=timestamp,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "status": self.status.value,
            "original_command": list(self.original_command),
            "resolved_config": self.resolved_config.to_dict(),
            "reference": {"source_hash": self.reference_source_hash},
            "candidate": {"source_hash": self.candidate_source_hash},
            "environment": {
                "snapshot": dict(self.environment_snapshot),
                "fingerprint": self.environment_fingerprint,
            },
            "suite_id": self.suite_id,
            "mode": self.mode,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "terminal_reason": self.terminal_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EvaluationManifest":
        if not isinstance(value, dict):
            raise ManifestContractError("manifest must be an object")
        expected = {
            "schema_version",
            "identity",
            "status",
            "original_command",
            "resolved_config",
            "reference",
            "candidate",
            "environment",
            "suite_id",
            "mode",
            "created_at_utc",
            "updated_at_utc",
            "terminal_reason",
        }
        if set(value) != expected:
            raise ManifestContractError("manifest has missing or unknown fields")

        def object_field(name: str) -> dict[str, object]:
            raw = value[name]
            if not isinstance(raw, dict):
                raise ManifestContractError(f"{name} must be an object")
            return raw

        reference = object_field("reference")
        candidate = object_field("candidate")
        environment = object_field("environment")
        if set(reference) != {"source_hash"} or set(candidate) != {"source_hash"}:
            raise ManifestContractError("source objects have incompatible fields")
        if set(environment) != {"snapshot", "fingerprint"}:
            raise ManifestContractError("environment has incompatible fields")
        command = value["original_command"]
        if not isinstance(command, list):
            raise ManifestContractError("original_command must be an array")
        try:
            status = EvaluationState(value["status"])
        except (TypeError, ValueError) as error:
            raise ManifestContractError("manifest status is invalid") from error

        def string(name: str, raw: object) -> str:
            if not isinstance(raw, str):
                raise ManifestContractError(f"{name} must be a string")
            return raw

        terminal_reason = value["terminal_reason"]
        if terminal_reason is not None and not isinstance(terminal_reason, str):
            raise ManifestContractError("terminal_reason must be a string or null")
        return cls(
            schema_version=value["schema_version"],
            identity=EvaluationIdentity.from_dict(object_field("identity")),
            status=status,
            original_command=tuple(
                string("original_command[]", part) for part in command
            ),
            resolved_config=ResolvedEvaluationConfig.from_dict(
                object_field("resolved_config")
            ),
            reference_source_hash=string(
                "reference.source_hash", reference["source_hash"]
            ),
            candidate_source_hash=string(
                "candidate.source_hash", candidate["source_hash"]
            ),
            environment_snapshot=_json_mapping(
                environment["snapshot"], "environment.snapshot"
            ),
            environment_fingerprint=string(
                "environment.fingerprint", environment["fingerprint"]
            ),
            suite_id=string("suite_id", value["suite_id"]),
            mode=string("mode", value["mode"]),
            created_at_utc=string("created_at_utc", value["created_at_utc"]),
            updated_at_utc=string("updated_at_utc", value["updated_at_utc"]),
            terminal_reason=terminal_reason,
        )


@dataclass(frozen=True)
class ResumeState:
    manifest: EvaluationManifest
    completed_result_ids: frozenset[str]

    @property
    def is_complete(self) -> bool:
        return self.manifest.status is EvaluationState.COMPLETE

    def pending_result_ids(self, result_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            result_id
            for result_id in result_ids
            if result_id not in self.completed_result_ids
        )


def _manifest_json(manifest: EvaluationManifest) -> str:
    return json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


class ArtifactWriter:
    """Own all formal writes beneath one ``results`` root."""

    def __init__(
        self,
        output_root: Path,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        replace: Callable[[os.PathLike[str], os.PathLike[str]], None] = os.replace,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self._now = now
        self._replace = replace

    def evaluation_dir(self, identity: EvaluationIdentity) -> Path:
        return evaluation_result_path(
            self.output_root,
            identity.operator_id,
            identity.candidate_id,
            identity.evaluation_id,
        )

    def initialize(
        self, manifest: EvaluationManifest, *, resume: bool = False
    ) -> ResumeState:
        if manifest.status is not EvaluationState.PLANNED:
            raise ManifestContractError(
                "a new or expected resume manifest must have planned status"
            )
        directory = self.evaluation_dir(manifest.identity)
        if directory.exists():
            if not resume:
                raise ArtifactExistsError(f"evaluation directory exists: {directory}")
            existing = self.read_manifest(manifest.identity)
            validate_resume_compatibility(existing, manifest)
            return self.resume_state(existing)
        if resume:
            raise ResumeMismatchError(
                f"cannot resume missing evaluation directory: {directory}"
            )

        candidate_root = candidate_result_path(
            self.output_root,
            manifest.identity.operator_id,
            manifest.identity.candidate_id,
        )
        candidate_root.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError as error:
            raise ArtifactExistsError(
                f"evaluation directory was created concurrently: {directory}"
            ) from error
        (directory / "diagnostics").mkdir()
        (directory / "logs").mkdir()

        self._write_manifest(manifest)
        atomic_write_text(
            directory / "summary.md",
            render_summary(manifest.to_dict()),
            replace=self._replace,
        )
        for filename in (
            "controller.jsonl",
            "worker.jsonl",
            "stdout.log",
            "stderr.log",
        ):
            atomic_write_text(directory / "logs" / filename, "", replace=self._replace)
        for schema in (
            RESULTS_SCHEMA,
            CORRECTNESS_OUTPUTS_SCHEMA,
            PERFORMANCE_SAMPLES_SCHEMA,
        ):
            AtomicCsvTable(
                directory / schema.filename, schema, replace=self._replace
            ).initialise()
        self._register_run(manifest)
        return ResumeState(manifest, frozenset())

    def read_manifest(self, identity: EvaluationIdentity) -> EvaluationManifest:
        path = self.evaluation_dir(identity) / "evaluation_manifest.json"
        try:
            with path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
            manifest = EvaluationManifest.from_dict(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestContractError(f"cannot read manifest {path}: {error}") from error
        except ManifestContractError:
            raise
        except (TypeError, ValueError) as error:
            raise ManifestContractError(
                f"manifest contract is invalid at {path}: {error}"
            ) from error
        if manifest.identity != identity:
            raise ManifestContractError(
                f"manifest identity does not match directory: {path}"
            )
        return manifest

    def update_status(
        self,
        identity: EvaluationIdentity,
        status: EvaluationState,
        *,
        terminal_reason: str | None = None,
    ) -> EvaluationManifest:
        current = self.read_manifest(identity)
        if not isinstance(status, EvaluationState):
            raise ManifestContractError("status must be EvaluationState")
        if current.status is status:
            if terminal_reason not in {None, current.terminal_reason}:
                raise ManifestContractError(
                    "idempotent status update cannot change terminal_reason"
                )
            if status is EvaluationState.COMPLETE:
                atomic_write_text(
                    self.evaluation_dir(identity) / "summary.md",
                    render_summary(current.to_dict()),
                    replace=self._replace,
                )
                self._publish_complete(current)
            return current
        if status not in _ALLOWED_TRANSITIONS[current.status]:
            raise ManifestContractError(
                f"illegal evaluation transition: {current.status.value} -> {status.value}"
            )
        if status is EvaluationState.COMPLETE and terminal_reason is not None:
            raise ManifestContractError("complete evaluation cannot have terminal_reason")
        updated = dataclass_replace(
            current,
            status=status,
            updated_at_utc=_utc_now(self._now),
            terminal_reason=terminal_reason,
        )
        self._write_manifest(updated)
        atomic_write_text(
            self.evaluation_dir(identity) / "summary.md",
            render_summary(updated.to_dict()),
            replace=self._replace,
        )
        if status is EvaluationState.COMPLETE:
            self._publish_complete(updated)
        return updated

    def resume_state(self, manifest: EvaluationManifest) -> ResumeState:
        path = self.evaluation_dir(manifest.identity) / RESULTS_SCHEMA.filename
        if not path.is_file():
            raise ResumeMismatchError(f"resume results table is missing: {path}")
        try:
            rows = AtomicCsvTable(
                path,
                RESULTS_SCHEMA,
                replace=self._replace,
            ).read_rows()
        except CsvContractError as error:
            raise ResumeMismatchError(
                f"resume results table is incompatible: {error}"
            ) from error
        return ResumeState(
            manifest,
            frozenset(row["result_id"] for row in rows),
        )

    def _write_manifest(self, manifest: EvaluationManifest) -> None:
        atomic_write_text(
            self.evaluation_dir(manifest.identity) / "evaluation_manifest.json",
            _manifest_json(manifest),
            replace=self._replace,
        )

    def _relative_path(self, identity: EvaluationIdentity) -> str:
        return self.evaluation_dir(identity).relative_to(self.output_root).as_posix()

    def _register_run(self, manifest: EvaluationManifest) -> None:
        AtomicCsvTable(
            self.output_root / RUN_INDEX_SCHEMA.filename,
            RUN_INDEX_SCHEMA,
            replace=self._replace,
        ).append(
            {
                "schema_version": CSV_SCHEMA_VERSION,
                **manifest.identity.to_dict(),
                "relative_path": self._relative_path(manifest.identity),
            }
        )

    def _publish_complete(self, manifest: EvaluationManifest) -> None:
        if manifest.status is not EvaluationState.COMPLETE:
            raise ManifestContractError("only complete evaluations can be published")
        identity = manifest.identity
        relative_path = self._relative_path(identity)
        candidate_root = candidate_result_path(
            self.output_root, identity.operator_id, identity.candidate_id
        )
        history = AtomicCsvTable(
            candidate_root / HISTORY_SCHEMA.filename,
            HISTORY_SCHEMA,
            replace=self._replace,
        )
        history.append(
            {
                "schema_version": CSV_SCHEMA_VERSION,
                "evaluation_id": identity.evaluation_id,
                "run_id": identity.run_id,
                "operator_id": identity.operator_id,
                "candidate_id": identity.candidate_id,
                "status": manifest.status.value,
                "completed_at_utc": manifest.updated_at_utc,
                "relative_path": relative_path,
                "suite_id": manifest.suite_id,
                "mode": manifest.mode,
                "reference_source_hash": manifest.reference_source_hash,
                "candidate_source_hash": manifest.candidate_source_hash,
                "environment_fingerprint": manifest.environment_fingerprint,
            }
        )
        # Rebuild latest from durable, complete history rather than from the
        # evaluation whose idempotent publication happened to run last.  ISO
        # UTC timestamps sort chronologically; evaluation_id is the stable
        # tie-break for equal completion instants.
        latest_row = max(
            history.read_rows(),
            key=lambda row: (row["completed_at_utc"], row["evaluation_id"]),
        )
        latest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": latest_row["run_id"],
            "operator_id": latest_row["operator_id"],
            "candidate_id": latest_row["candidate_id"],
            "evaluation_id": latest_row["evaluation_id"],
            "status": EvaluationState.COMPLETE.value,
            "completed_at_utc": latest_row["completed_at_utc"],
            "relative_path": latest_row["relative_path"],
        }
        atomic_write_text(
            candidate_root / "latest.json",
            json.dumps(
                latest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            replace=self._replace,
        )


class ResumeReader:
    """Resolve a run ID to its mirrored evaluations without a run directory."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root).resolve()
        self._writer = ArtifactWriter(self.output_root)

    def for_run(self, run_id: str) -> tuple[ResumeState, ...]:
        validate_run_id(run_id)
        try:
            rows = AtomicCsvTable(
                self.output_root / RUN_INDEX_SCHEMA.filename, RUN_INDEX_SCHEMA
            ).read_rows()
        except CsvContractError as error:
            raise ResumeMismatchError(str(error)) from error
        selected = [row for row in rows if row["run_id"] == run_id]
        if not selected:
            raise ResumeMismatchError(f"run_id not found in run_index.csv: {run_id}")
        states: list[ResumeState] = []
        for row in selected:
            identity = EvaluationIdentity(
                run_id=row["run_id"],
                operator_id=row["operator_id"],
                candidate_id=row["candidate_id"],
                evaluation_id=row["evaluation_id"],
            )
            expected_path = self._writer.evaluation_dir(identity).relative_to(
                self.output_root
            ).as_posix()
            if row["relative_path"] != expected_path:
                raise ResumeMismatchError(
                    f"run index path disagrees with identity: {row['relative_path']}"
                )
            manifest = self._writer.read_manifest(identity)
            states.append(self._writer.resume_state(manifest))
        states.sort(
            key=lambda state: (
                state.manifest.identity.operator_id,
                state.manifest.identity.candidate_id,
                state.manifest.identity.evaluation_id,
            )
        )
        return tuple(states)


def validate_resume_compatibility(
    existing: EvaluationManifest, expected: EvaluationManifest
) -> None:
    """Reject every identity, schema, source, config, or environment drift."""

    checks = {
        "schema_version": (existing.schema_version, expected.schema_version),
        "identity": (existing.identity, expected.identity),
        "reference source": (
            existing.reference_source_hash,
            expected.reference_source_hash,
        ),
        "candidate source": (
            existing.candidate_source_hash,
            expected.candidate_source_hash,
        ),
        "environment fingerprint": (
            existing.environment_fingerprint,
            expected.environment_fingerprint,
        ),
        "environment snapshot": (
            dict(existing.environment_snapshot),
            dict(expected.environment_snapshot),
        ),
        "resolved config": (existing.resolved_config, expected.resolved_config),
        "suite_id": (existing.suite_id, expected.suite_id),
        "mode": (existing.mode, expected.mode),
    }
    mismatches = [name for name, (actual, wanted) in checks.items() if actual != wanted]
    if mismatches:
        raise ResumeMismatchError(
            "resume is incompatible: " + ", ".join(mismatches)
        )


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ArtifactError",
    "ArtifactExistsError",
    "ArtifactWriter",
    "EvaluationManifest",
    "EvaluationState",
    "HISTORY_SCHEMA",
    "ManifestContractError",
    "RUN_INDEX_SCHEMA",
    "ResumeMismatchError",
    "ResumeReader",
    "ResumeState",
    "validate_resume_compatibility",
]
