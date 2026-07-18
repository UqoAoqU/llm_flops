"""Strict, non-rankable import of legacy DeepSeek V4 summary CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path

from benchmark_engine.ids import (
    evaluation_result_path,
    generate_result_id,
    validate_candidate_id,
    validate_evaluation_id,
    validate_operator_id,
    validate_run_id,
)
from benchmark_engine.projection import DEEPSEEK_V4_PROJECTION
from benchmark_engine.projection.base import ProjectionMapping

from .csv_writer import (
    AtomicCsvTable,
    CORRECTNESS_OUTPUTS_SCHEMA,
    MODEL_PROJECTION_SCHEMA,
    PERFORMANCE_SAMPLES_SCHEMA,
    RESULTS_SCHEMA,
    atomic_write_text,
)


LEGACY_IMPORT_MANIFEST_SCHEMA_VERSION = 1
LEGACY_IMPORT_OPERATOR_ID = "deepseek_v4_legacy_projection"
LEGACY_HEADER = (
    "phase",
    "quant_profile",
    "environment_fingerprint",
    "m",
    "context",
    "operator",
    "backend",
    "instances",
    "call_ms",
    "model_ms",
    "pct",
    "status",
    "input_shape",
    "output_shape",
    "error",
)


class LegacyImportError(ValueError):
    """Raised when legacy data cannot be mapped without inventing facts."""


class LegacyImportConflictError(LegacyImportError):
    """Raised when an import would overwrite a different durable artifact."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_identities(source_sha256: str, candidate_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"legacy-deepseek-v4\0{source_sha256}\0{candidate_id}".encode("utf-8")
    ).hexdigest()
    run_id = validate_run_id(f"run_{digest[:32]}")
    evaluation_id = validate_evaluation_id(
        f"19700101T000000Z__{source_sha256[:12]}__{digest[:12]}"
    )
    return run_id, evaluation_id


def _finite_number(value: str, field: str, row_number: int) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise LegacyImportError(
            f"legacy CSV row {row_number}: {field} must be a finite number"
        ) from error
    if not math.isfinite(number):
        raise LegacyImportError(
            f"legacy CSV row {row_number}: {field} must be a finite number"
        )
    return number


def _positive_integer(value: str, field: str, row_number: int) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise LegacyImportError(
            f"legacy CSV row {row_number}: {field} must be a positive integer"
        ) from error
    if number <= 0 or str(number) != value:
        raise LegacyImportError(
            f"legacy CSV row {row_number}: {field} must be a positive integer"
        )
    return number


def _shape_text(*dimensions: int) -> str:
    return "(" + ",".join(str(dimension) for dimension in dimensions) + ")"


def _legacy_shapes(mapping: ProjectionMapping, model_input: int, context: int) -> tuple[str, str]:
    kind = mapping.kind
    shape = mapping.shape
    if kind in {"fp8", "bf16"} and shape is not None:
        k, n = shape
        return (
            f"x={_shape_text(model_input, k)}; weight={_shape_text(n, k)}",
            f"y={_shape_text(model_input, n)}",
        )
    if kind == "grouped_bf16" and shape is not None:
        groups, k, n = shape
        return (
            f"x={_shape_text(model_input, groups, k)}; weight={_shape_text(groups, k, n)}",
            f"y={_shape_text(model_input, groups, n)}",
        )
    if kind.startswith("moe_") and shape is not None:
        experts, hidden, intermediate = shape
        local_pairs = model_input * 6 * 16 // 384
        return (
            f"x={_shape_text(model_input, hidden)}; "
            f"topk_ids/weights={_shape_text(model_input, 6)}; "
            f"local_pairs={local_pairs}; "
            f"w13={_shape_text(experts, 2 * intermediate, hidden)}; "
            f"w2={_shape_text(experts, hidden, intermediate)}",
            f"y={_shape_text(model_input, hidden)}",
        )
    if kind in {"prefill_attention", "dense_prefill_attention"}:
        selected = 1024 if kind == "prefill_attention" else min(128, context)
        return (
            f"q={_shape_text(model_input, 128, 512)}; "
            f"kv={_shape_text(context, 1, 512)}; "
            f"indices={_shape_text(model_input, 1, selected)}",
            f"y={_shape_text(model_input, 128, 512)}",
        )
    if kind.startswith("decode_attention") or kind == "dense_decode_attention":
        return (
            f"q={_shape_text(model_input, 1, 128, 512)}; context={context}",
            f"y={_shape_text(model_input, 1, 128, 512)}",
        )
    if kind == "fp8_quant":
        return (
            f"q={_shape_text(model_input, 64, 128)}",
            f"q_fp8={_shape_text(model_input, 64, 128)}",
        )
    if kind == "fp8_logits":
        compressed = (context + 3) // 4
        return (
            f"q={_shape_text(model_input, 1, 64, 128)}; "
            f"raw_kv={context}; c4_kv={compressed}",
            f"logits={_shape_text(model_input, compressed)}",
        )
    if kind == "topk":
        compressed = (context + 3) // 4
        return (
            f"scores={_shape_text(model_input, compressed)}; raw_kv={context}",
            f"indices={_shape_text(model_input, 1024)}",
        )
    raise LegacyImportError(f"legacy shape mapping is not defined for {mapping.adapter_id}")


@dataclass(frozen=True)
class ParsedLegacyRow:
    row_number: int
    mapping: ProjectionMapping
    phase: str
    quant_profile: str
    environment_fingerprint: str
    model_input: int
    context: int
    call_ms: float | None
    model_ms: float | None
    status: str
    input_shape: str
    output_shape: str
    error: str

    @property
    def operator_id(self) -> str:
        return self.mapping.operator_id or LEGACY_IMPORT_OPERATOR_ID


@dataclass(frozen=True)
class LegacyImportManifest:
    identity: Mapping[str, str]
    source_csv_sha256: str
    phase: str
    quant_profile: str
    environment_fingerprint: str
    model_input: int
    context: int
    imported_at_utc: str
    row_count: int
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = LEGACY_IMPORT_MANIFEST_SCHEMA_VERSION
    artifact_kind: str = "legacy_import"
    imported_legacy: bool = True
    reference_source_hash: None = None
    candidate_source_hash: None = None
    correctness_available: bool = False
    raw_samples_available: bool = False
    ranking_eligible: bool = False
    resumable: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != LEGACY_IMPORT_MANIFEST_SCHEMA_VERSION:
            raise LegacyImportError("legacy manifest schema_version must be 1")
        if self.artifact_kind != "legacy_import" or self.imported_legacy is not True:
            raise LegacyImportError("legacy manifest kind/import marker is invalid")
        if any(
            value is not False
            for value in (
                self.correctness_available,
                self.raw_samples_available,
                self.ranking_eligible,
                self.resumable,
            )
        ):
            raise LegacyImportError("legacy imports cannot claim formal evaluation data")
        if self.reference_source_hash is not None or self.candidate_source_hash is not None:
            raise LegacyImportError("legacy source hashes must be null")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_csv_sha256):
            raise LegacyImportError("source_csv_sha256 must be lowercase SHA-256")
        required_identity = {"run_id", "evaluation_id", "operator_id", "candidate_id"}
        if set(self.identity) != required_identity:
            raise LegacyImportError("legacy manifest identity fields are invalid")
        validate_run_id(self.identity["run_id"])
        validate_evaluation_id(self.identity["evaluation_id"])
        validate_operator_id(self.identity["operator_id"])
        validate_candidate_id(self.identity["candidate_id"])
        if self.phase not in {"prefill", "decode"}:
            raise LegacyImportError("legacy manifest phase is invalid")
        if self.quant_profile != "fp8_mxfp8":
            raise LegacyImportError("legacy manifest quant_profile is unsupported")
        if self.model_input <= 0 or self.context <= 0 or self.row_count <= 0:
            raise LegacyImportError("legacy manifest dimensions/count must be positive")
        artifact_names = {
            RESULTS_SCHEMA.filename,
            CORRECTNESS_OUTPUTS_SCHEMA.filename,
            PERFORMANCE_SAMPLES_SCHEMA.filename,
            MODEL_PROJECTION_SCHEMA.filename,
            "summary.md",
        }
        if self.artifact_sha256 and (
            set(self.artifact_sha256) != artifact_names
            or not all(
                isinstance(name, str)
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest)
                for name, digest in self.artifact_sha256.items()
            )
        ):
            raise LegacyImportError("legacy artifact SHA-256 inventory is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "imported_legacy": self.imported_legacy,
            "identity": dict(self.identity),
            "source_csv": {"sha256": self.source_csv_sha256},
            "legacy_case": {
                "phase": self.phase,
                "quant_profile": self.quant_profile,
                "environment_fingerprint": self.environment_fingerprint,
                "model_input": self.model_input,
                "context": self.context,
            },
            "availability": {
                "reference_source_hash": self.reference_source_hash,
                "candidate_source_hash": self.candidate_source_hash,
                "correctness": self.correctness_available,
                "raw_samples": self.raw_samples_available,
                "ranking_eligible": self.ranking_eligible,
                "resumable": self.resumable,
            },
            "artifacts": {
                name: {"sha256": digest}
                for name, digest in sorted(self.artifact_sha256.items())
            },
            "imported_at_utc": self.imported_at_utc,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "LegacyImportManifest":
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "artifact_kind", "imported_legacy", "identity",
            "source_csv", "legacy_case", "availability", "artifacts",
            "imported_at_utc", "row_count",
        }:
            raise LegacyImportError("legacy import manifest has missing or unknown fields")
        identity = value["identity"]
        source = value["source_csv"]
        case = value["legacy_case"]
        availability = value["availability"]
        artifacts = value["artifacts"]
        if not all(isinstance(item, dict) for item in (identity, source, case, availability, artifacts)):
            raise LegacyImportError("legacy import manifest nested fields must be objects")
        if set(source) != {"sha256"} or set(case) != {
            "phase", "quant_profile", "environment_fingerprint", "model_input", "context",
        } or set(availability) != {
            "reference_source_hash", "candidate_source_hash", "correctness", "raw_samples",
            "ranking_eligible", "resumable",
        }:
            raise LegacyImportError("legacy import manifest nested contract is invalid")
        try:
            artifact_sha256 = {}
            for name, contract in artifacts.items():
                if not isinstance(name, str) or not isinstance(contract, dict) \
                        or set(contract) != {"sha256"}:
                    raise LegacyImportError("legacy artifact hash contract is invalid")
                artifact_sha256[name] = contract["sha256"]
            return cls(
                schema_version=value["schema_version"],
                artifact_kind=value["artifact_kind"],
                imported_legacy=value["imported_legacy"],
                identity=dict(identity),
                source_csv_sha256=source["sha256"],
                phase=case["phase"],
                quant_profile=case["quant_profile"],
                environment_fingerprint=case["environment_fingerprint"],
                model_input=case["model_input"],
                context=case["context"],
                artifact_sha256=artifact_sha256,
                reference_source_hash=availability["reference_source_hash"],
                candidate_source_hash=availability["candidate_source_hash"],
                correctness_available=availability["correctness"],
                raw_samples_available=availability["raw_samples"],
                ranking_eligible=availability["ranking_eligible"],
                resumable=availability["resumable"],
                imported_at_utc=value["imported_at_utc"],
                row_count=value["row_count"],
            )
        except (KeyError, TypeError) as error:
            raise LegacyImportError("legacy import manifest field types are invalid") from error

    def semantic_identity(self) -> tuple[object, ...]:
        return (
            tuple(sorted(self.identity.items())), self.source_csv_sha256, self.phase,
            self.quant_profile, self.environment_fingerprint, self.model_input,
            self.context, self.row_count,
        )


@dataclass(frozen=True)
class LegacyImportReport:
    run_id: str
    evaluation_id: str
    directories: tuple[str, ...]
    created: int
    reused: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "imported_legacy": True,
            "run_id": self.run_id,
            "evaluation_id": self.evaluation_id,
            "directories": list(self.directories),
            "created": self.created,
            "reused": self.reused,
        }


def _parse_csv(path: Path) -> tuple[str, tuple[ParsedLegacyRow, ...]]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise LegacyImportError(f"cannot read legacy CSV {path}: {error}") from error
    digest = hashlib.sha256(content).hexdigest()
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != LEGACY_HEADER:
                raise LegacyImportError(
                    "legacy CSV header is incompatible; expected: " + ",".join(LEGACY_HEADER)
                )
            raw_rows = list(reader)
    except UnicodeDecodeError as error:
        raise LegacyImportError("legacy CSV must be UTF-8") from error
    if not raw_rows:
        raise LegacyImportError("legacy CSV contains no data rows")

    parsed: list[ParsedLegacyRow] = []
    seen: set[str] = set()
    case_identity: tuple[str, str, str, int, int] | None = None
    for row_number, row in enumerate(raw_rows, start=2):
        if None in row or set(row) != set(LEGACY_HEADER):
            raise LegacyImportError(f"legacy CSV row {row_number} is malformed")
        phase = row["phase"]
        profile = row["quant_profile"]
        if phase not in {"prefill", "decode"}:
            raise LegacyImportError(f"legacy CSV row {row_number}: unsupported phase {phase!r}")
        if profile != "fp8_mxfp8":
            raise LegacyImportError(
                f"legacy CSV row {row_number}: unsupported quant_profile {profile!r}"
            )
        fingerprint = row["environment_fingerprint"]
        if re.fullmatch(r"[0-9a-f]{8,64}", fingerprint) is None:
            raise LegacyImportError(
                f"legacy CSV row {row_number}: environment_fingerprint must be lowercase hex"
            )
        model_input = _positive_integer(row["m"], "m", row_number)
        context = _positive_integer(row["context"], "context", row_number)
        identity = (phase, profile, fingerprint, model_input, context)
        if case_identity is None:
            case_identity = identity
        elif identity != case_identity:
            raise LegacyImportError("legacy CSV mixes phase/profile/environment/shape identities")

        mappings = DEEPSEEK_V4_PROJECTION.mappings(phase, profile)
        by_name = {mapping.display_name: mapping for mapping in mappings}
        operator_name = row["operator"]
        if operator_name not in by_name:
            raise LegacyImportError(
                f"legacy CSV row {row_number}: unknown operator mapping {operator_name!r}"
            )
        if operator_name in seen:
            raise LegacyImportError(
                f"legacy CSV row {row_number}: duplicate operator mapping {operator_name!r}"
            )
        seen.add(operator_name)
        mapping = by_name[operator_name]
        if row["backend"] != mapping.backend:
            raise LegacyImportError(
                f"legacy CSV row {row_number}: backend mismatch for {operator_name}: "
                f"expected {mapping.backend!r}, got {row['backend']!r}"
            )
        instances = _positive_integer(row["instances"], "instances", row_number)
        if instances != mapping.instances:
            raise LegacyImportError(
                f"legacy CSV row {row_number}: instances mismatch for {operator_name}: "
                f"expected {mapping.instances}, got {instances}"
            )
        expected_input, expected_output = _legacy_shapes(mapping, model_input, context)
        if row["input_shape"] != expected_input or row["output_shape"] != expected_output:
            raise LegacyImportError(
                f"legacy CSV row {row_number}: shape mismatch for {operator_name}"
            )
        status = row["status"]
        if status not in {"executed", "unavailable"}:
            raise LegacyImportError(
                f"legacy CSV row {row_number}: status must be executed or unavailable"
            )
        if status == "executed":
            call_ms = _finite_number(row["call_ms"], "call_ms", row_number)
            model_ms = _finite_number(row["model_ms"], "model_ms", row_number)
            _finite_number(row["pct"], "pct", row_number)
            if call_ms < 0 or model_ms < 0:
                raise LegacyImportError(
                    f"legacy CSV row {row_number}: timing values must be non-negative"
                )
            if not math.isclose(model_ms, call_ms * instances, rel_tol=0.0, abs_tol=1e-6):
                raise LegacyImportError(
                    f"legacy CSV row {row_number}: model_ms is not call_ms * instances"
                )
        else:
            if row["call_ms"] or row["model_ms"]:
                raise LegacyImportError(
                    f"legacy CSV row {row_number}: unavailable timing must be empty"
                )
            call_ms = model_ms = None
        parsed.append(
            ParsedLegacyRow(
                row_number, mapping, phase, profile, fingerprint, model_input, context,
                call_ms, model_ms, status, row["input_shape"], row["output_shape"], row["error"],
            )
        )
    assert case_identity is not None
    expected_names = {mapping.display_name for mapping in DEEPSEEK_V4_PROJECTION.mappings(case_identity[0], case_identity[1])}
    missing = sorted(expected_names.difference(seen))
    if missing:
        raise LegacyImportError("legacy CSV is missing operator mappings: " + ", ".join(missing))
    return digest, tuple(parsed)


def _case_id(row: ParsedLegacyRow) -> str:
    return (
        f"legacy_{row.phase}_{row.mapping.adapter_id}_"
        f"m{row.model_input}_ctx{row.context}"
    )


def _result_row(
    row: ParsedLegacyRow,
    *,
    run_id: str,
    evaluation_id: str,
    candidate_id: str,
    result_id: str,
) -> dict[str, object]:
    executed = row.status == "executed"
    return {
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "timestamp_utc": "",
        "suite_id": "legacy_import",
        "mode": "performance",
        "result_id": result_id,
        "operator_id": row.operator_id,
        "contract_version": 1,
        "candidate_id": candidate_id,
        "reference_id": "",
        "candidate_source_hash": "",
        "reference_source_hash": "",
        "imported_legacy": True,
        "environment_fingerprint": row.environment_fingerprint,
        "device": "cuda",
        "case_id": _case_id(row),
        "case_hash": "",
        "seed": 0,
        "tags": json.dumps(["imported_legacy", row.phase], separators=(",", ":")),
        "input_summary": json.dumps(
            {"input_shape": row.input_shape, "output_shape": row.output_shape},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "status": "skipped",
        "correctness_status": "skipped",
        "performance_status": "skipped",
        "skip_reason": "imported_legacy_without_correctness_or_raw_samples",
        "correctness_pass": None,
        "timer": "legacy_cuda_graph",
        "requested_timer": "legacy_cuda_graph",
        "effective_timer": "legacy_cuda_graph",
        "legacy_graph_ms": row.call_ms,
        "performance_formal": False,
        "ranking_eligible": False,
        "performance_gate_status": "skipped",
        "performance_gate_reasons": '["imported_legacy_non_rankable"]',
        "perf_on_correctness_fail": False,
        "gate_unsupported_policy": "fail",
        "error_type": None if executed else "legacy_unavailable",
        "error_message": row.error or None,
    }


def _projection_row(
    row: ParsedLegacyRow,
    *,
    run_id: str,
    evaluation_id: str,
    candidate_id: str,
    result_id: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "result_id": result_id,
        "suite_id": "legacy_import",
        "projection_id": "deepseek_v4_pro",
        "phase": row.phase,
        "quant_profile": row.quant_profile,
        "model_input": row.model_input,
        "raw_context": row.context,
        "operator_id": row.operator_id,
        "candidate_id": candidate_id,
        "case_id": _case_id(row),
        "adapter_id": row.mapping.adapter_id,
        "display_name": row.mapping.display_name,
        "backend": row.mapping.backend,
        "kind": row.mapping.kind,
        "legacy_shape": json.dumps(
            {
                "input_shape": row.input_shape,
                "output_shape": row.output_shape,
                "logical_shape": (
                    list(row.mapping.shape) if row.mapping.shape is not None else None
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "instances": row.mapping.instances,
        "implementation_role": "candidate",
        "per_call_ms": row.call_ms,
        "projected_model_ms": row.model_ms,
        "status": "measured" if row.status == "executed" else "unavailable",
        "reason": "imported_legacy_nonformal" if row.status == "executed" else row.error,
    }


def _manifest_text(manifest: LegacyImportManifest) -> str:
    return json.dumps(
        manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def read_legacy_manifest(path: Path) -> LegacyImportManifest:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyImportError(f"cannot read legacy import manifest {path}: {error}") from error
    return LegacyImportManifest.from_dict(raw)


def is_legacy_import_directory(path: Path) -> bool:
    return (Path(path) / "legacy_import_manifest.json").is_file()


def _casefold_safe(parent: Path, component: str, label: str) -> None:
    if not parent.is_dir():
        return
    for child in parent.iterdir():
        if child.is_dir() and child.name.casefold() == component.casefold() \
                and child.name != component:
            raise LegacyImportConflictError(
                f"{label} case-insensitively collides with {child.name!r} under {parent}"
            )


def _validate_directory(path: Path) -> LegacyImportManifest:
    manifest = read_legacy_manifest(path / "legacy_import_manifest.json")
    if not manifest.artifact_sha256:
        raise LegacyImportError("legacy manifest is missing artifact SHA-256 inventory")
    for filename, expected in manifest.artifact_sha256.items():
        artifact = path / filename
        try:
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except OSError as error:
            raise LegacyImportError(f"cannot hash imported artifact {artifact}: {error}") from error
        if actual != expected:
            raise LegacyImportConflictError(
                f"imported artifact content hash mismatch: {artifact}"
            )
    for schema in (
        RESULTS_SCHEMA,
        CORRECTNESS_OUTPUTS_SCHEMA,
        PERFORMANCE_SAMPLES_SCHEMA,
        MODEL_PROJECTION_SCHEMA,
    ):
        AtomicCsvTable(path / schema.filename, schema).read_rows()
    result_rows = AtomicCsvTable(path / RESULTS_SCHEMA.filename, RESULTS_SCHEMA).read_rows()
    projection_rows = AtomicCsvTable(path / MODEL_PROJECTION_SCHEMA.filename, MODEL_PROJECTION_SCHEMA).read_rows()
    identity = manifest.identity
    if len(result_rows) != manifest.row_count or len(projection_rows) != manifest.row_count:
        raise LegacyImportConflictError("legacy artifact row count disagrees with manifest")
    result_ids: set[str] = set()
    for row in result_rows:
        if any(
            row[field] != identity[field]
            for field in ("run_id", "evaluation_id", "operator_id", "candidate_id")
        ) or row["imported_legacy"] != "true":
            raise LegacyImportConflictError("legacy result identity/import marker mismatch")
        expected_result = generate_result_id(
            identity["operator_id"], identity["candidate_id"], identity["evaluation_id"],
            row["case_id"], int(row["seed"]),
        )
        if row["result_id"] != expected_result:
            raise LegacyImportConflictError("legacy result_id is not derived from its identity")
        result_ids.add(row["result_id"])
    for row in projection_rows:
        if any(
            row[field] != identity[field]
            for field in ("run_id", "evaluation_id", "operator_id", "candidate_id")
        ) or row["result_id"] not in result_ids:
            raise LegacyImportConflictError("legacy projection identity/result_id mismatch")
    if AtomicCsvTable(path / CORRECTNESS_OUTPUTS_SCHEMA.filename, CORRECTNESS_OUTPUTS_SCHEMA).read_rows():
        raise LegacyImportError("legacy correctness_outputs.csv must contain no rows")
    if AtomicCsvTable(path / PERFORMANCE_SAMPLES_SCHEMA.filename, PERFORMANCE_SAMPLES_SCHEMA).read_rows():
        raise LegacyImportError("legacy performance_samples.csv must contain no rows")
    return manifest


def _write_directory(
    path: Path,
    manifest: LegacyImportManifest,
    rows: Sequence[ParsedLegacyRow],
    *,
    replace_file: Callable[[os.PathLike[str], os.PathLike[str]], None] = os.replace,
) -> LegacyImportManifest:
    path.mkdir()
    (path / "diagnostics").mkdir()
    result_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    identity = manifest.identity
    for row in rows:
        case_id = _case_id(row)
        result_id = generate_result_id(
            identity["operator_id"], identity["candidate_id"], identity["evaluation_id"], case_id, 0
        )
        result_rows.append(
            _result_row(
                row, run_id=identity["run_id"], evaluation_id=identity["evaluation_id"],
                candidate_id=identity["candidate_id"], result_id=result_id,
            )
        )
        projection_rows.append(
            _projection_row(
                row, run_id=identity["run_id"], evaluation_id=identity["evaluation_id"],
                candidate_id=identity["candidate_id"], result_id=result_id,
            )
        )
    AtomicCsvTable(path / RESULTS_SCHEMA.filename, RESULTS_SCHEMA, replace=replace_file).append_many(result_rows)
    AtomicCsvTable(path / CORRECTNESS_OUTPUTS_SCHEMA.filename, CORRECTNESS_OUTPUTS_SCHEMA, replace=replace_file).initialise()
    AtomicCsvTable(path / PERFORMANCE_SAMPLES_SCHEMA.filename, PERFORMANCE_SAMPLES_SCHEMA, replace=replace_file).initialise()
    AtomicCsvTable(path / MODEL_PROJECTION_SCHEMA.filename, MODEL_PROJECTION_SCHEMA, replace=replace_file).append_many(projection_rows)
    summary = (
        "# Imported legacy DeepSeek V4 result\n\n"
        "This directory contains legacy aggregate CUDA Graph latency only. "
        "It has no source hashes, correctness outputs, raw timing samples, CV, "
        "speedup, resume state, or ranking eligibility.\n"
    )
    atomic_write_text(path / "summary.md", summary, replace=replace_file)
    hashes = {
        filename: hashlib.sha256((path / filename).read_bytes()).hexdigest()
        for filename in (
            RESULTS_SCHEMA.filename,
            CORRECTNESS_OUTPUTS_SCHEMA.filename,
            PERFORMANCE_SAMPLES_SCHEMA.filename,
            MODEL_PROJECTION_SCHEMA.filename,
            "summary.md",
        )
    }
    final_manifest = dataclass_replace(manifest, artifact_sha256=hashes)
    atomic_write_text(
        path / "legacy_import_manifest.json",
        _manifest_text(final_manifest),
        replace=replace_file,
    )
    return final_manifest


def convert_legacy_csv(
    source: Path,
    output_root: Path,
    candidate_id: str,
    *,
    repository_root: Path | None = None,
    replace_directory: Callable[[os.PathLike[str], os.PathLike[str]], None] = os.replace,
) -> LegacyImportReport:
    """Validate and atomically publish one legacy CSV as mirrored artifacts."""

    candidate = validate_candidate_id(candidate_id)
    digest, rows = _parse_csv(Path(source))
    run_id, evaluation_id = _stable_identities(digest, candidate)
    grouped: dict[str, list[ParsedLegacyRow]] = defaultdict(list)
    for row in rows:
        grouped[validate_operator_id(row.operator_id)].append(row)

    root = Path(output_root).resolve()
    targets: list[tuple[Path, LegacyImportManifest, tuple[ParsedLegacyRow, ...]]] = []
    reused = 0
    for operator_id in sorted(grouped):
        candidate_parent = root / operator_id
        _casefold_safe(candidate_parent, candidate, "candidate_id")
        if repository_root is not None:
            _casefold_safe(
                Path(repository_root).resolve() / "operators" / "candidates" / operator_id,
                candidate,
                "candidate_id",
            )
        target = evaluation_result_path(root, operator_id, candidate, evaluation_id)
        _casefold_safe(target.parent, evaluation_id, "evaluation_id")
        selected = tuple(grouped[operator_id])
        first = selected[0]
        manifest = LegacyImportManifest(
            identity={
                "run_id": run_id,
                "evaluation_id": evaluation_id,
                "operator_id": operator_id,
                "candidate_id": candidate,
            },
            source_csv_sha256=digest,
            phase=first.phase,
            quant_profile=first.quant_profile,
            environment_fingerprint=first.environment_fingerprint,
            model_input=first.model_input,
            context=first.context,
            imported_at_utc=_utc_now(),
            row_count=len(selected),
        )
        if target.exists():
            existing = _validate_directory(target)
            if existing.semantic_identity() != manifest.semantic_identity():
                raise LegacyImportConflictError(
                    f"legacy import target contains different facts: {target}"
                )
            expected_directory = Path(
                tempfile.mkdtemp(prefix=f".{evaluation_id}.verify.", dir=target.parent)
            )
            expected_directory.rmdir()
            try:
                expected = _write_directory(expected_directory, manifest, selected)
                _validate_directory(expected_directory)
                if dict(existing.artifact_sha256) != dict(expected.artifact_sha256):
                    raise LegacyImportConflictError(
                        f"legacy import target artifacts differ from source CSV: {target}"
                    )
            finally:
                shutil.rmtree(expected_directory, ignore_errors=True)
            reused += 1
            continue
        targets.append((target, manifest, selected))

    staged: list[tuple[Path, Path]] = []
    try:
        for target, manifest, selected in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{evaluation_id}.", dir=target.parent))
            temporary.rmdir()
            final_manifest = _write_directory(temporary, manifest, selected)
            validated = _validate_directory(temporary)
            if validated.semantic_identity() != final_manifest.semantic_identity():
                raise LegacyImportError("staged legacy manifest changed during validation")
            staged.append((temporary, target))
        for temporary, target in staged:
            replace_directory(temporary, target)
    except BaseException:
        for temporary, _ in staged:
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    directories = tuple(
        evaluation_result_path(root, operator_id, candidate, evaluation_id)
        .relative_to(root)
        .as_posix()
        for operator_id in sorted(grouped)
    )
    return LegacyImportReport(run_id, evaluation_id, directories, len(targets), reused)


__all__ = [
    "LEGACY_HEADER",
    "LEGACY_IMPORT_MANIFEST_SCHEMA_VERSION",
    "LegacyImportConflictError",
    "LegacyImportError",
    "LegacyImportManifest",
    "LegacyImportReport",
    "convert_legacy_csv",
    "is_legacy_import_directory",
    "read_legacy_manifest",
]
