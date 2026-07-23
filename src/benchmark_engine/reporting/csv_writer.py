"""Versioned RFC 4180 tables with crash-safe replacement semantics.

The controller is the sole writer of reporting tables.  Every mutation reads
the last complete table and rewrites it through a temporary file in the same
directory; readers therefore see either the old table or the new table, never
a partial append.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchmark_engine.models import (
    CorrectnessStatus,
    PerformanceStatus,
    ResultStatus,
)


CSV_SCHEMA_VERSION = 1
ColumnKind = Literal["string", "integer", "number", "boolean"]


class CsvContractError(ValueError):
    """Raised when a table or row violates its versioned contract."""


class CsvConflictError(CsvContractError):
    """Raised when a primary key already exists with different content."""


@dataclass(frozen=True)
class CsvColumn:
    name: str
    kind: ColumnKind = "string"
    nullable: bool = True
    allowed_values: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.allowed_values is not None:
            if not isinstance(self.allowed_values, frozenset):
                raise ValueError("allowed_values must be immutable (frozenset)")
            if self.kind != "string":
                raise ValueError("allowed_values is only valid for string columns")
            if not self.allowed_values or not all(
                isinstance(value, str) and value for value in self.allowed_values
            ):
                raise ValueError("allowed_values must be a non-empty frozenset of strings")


@dataclass(frozen=True)
class CsvSchema:
    filename: str
    columns: tuple[CsvColumn, ...]
    primary_key: tuple[str, ...]
    version: int = CSV_SCHEMA_VERSION
    compatible_previous: tuple["CsvSchema", ...] = ()
    migrate_previous: Callable[[Mapping[str, object], int], Mapping[str, object]] | None = None
    validate_row: Callable[[Mapping[str, str]], None] | None = None

    def __post_init__(self) -> None:
        names = self.fieldnames
        if not names or names[0] != "schema_version":
            raise ValueError("schema_version must be the first CSV column")
        if len(set(names)) != len(names):
            raise ValueError("CSV column names must be unique")
        if not self.primary_key or not set(self.primary_key).issubset(names):
            raise ValueError("CSV primary key must name schema columns")
        if self.compatible_previous and self.migrate_previous is None:
            raise ValueError("compatible_previous requires migrate_previous")
        if any(previous.filename != self.filename for previous in self.compatible_previous):
            raise ValueError("compatible schemas must use the same filename")
        if any(previous.version >= self.version for previous in self.compatible_previous):
            raise ValueError("compatible schemas must have an older version")

    @property
    def fieldnames(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)


def _columns(
    names: tuple[str, ...],
    *,
    required: frozenset[str],
    integers: frozenset[str] = frozenset(),
    numbers: frozenset[str] = frozenset(),
    booleans: frozenset[str] = frozenset(),
    allowed_values: Mapping[str, frozenset[str]] | None = None,
) -> tuple[CsvColumn, ...]:
    if allowed_values is not None:
        unknown = set(allowed_values).difference(names)
        if unknown:
            raise ValueError(
                "allowed_values names unknown columns: "
                + ", ".join(sorted(unknown))
            )
    result: list[CsvColumn] = []
    for name in names:
        kind: ColumnKind = "string"
        if name in integers:
            kind = "integer"
        elif name in numbers:
            kind = "number"
        elif name in booleans:
            kind = "boolean"
        result.append(
            CsvColumn(
                name,
                kind,
                name not in required,
                None if allowed_values is None else allowed_values.get(name),
            )
        )
    return tuple(result)


RESULTS_FIELDNAMES_V4 = (
    "schema_version",
    "run_id",
    "evaluation_id",
    "timestamp_utc",
    "suite_id",
    "mode",
    "result_id",
    "operator_id",
    "contract_version",
    "candidate_id",
    "reference_id",
    "candidate_source_hash",
    "reference_source_hash",
    "imported_legacy",
    "environment_fingerprint",
    "device",
    "gpu_name",
    "gpu_uuid",
    "logical_device",
    "visible_device",
    "cuda_visible_devices",
    "driver_version",
    "other_compute_processes_detected",
    "telemetry_error",
    "cuda_version",
    "torch_version",
    "case_id",
    "case_hash",
    "seed",
    "tags",
    "input_summary",
    "status",
    "correctness_status",
    "performance_status",
    "skip_reason",
    "correctness_pass",
    "failed_output_count",
    "max_abs_error",
    "max_rel_error",
    "rmse",
    "rel_l2",
    "cosine_similarity",
    "mismatch_count",
    "mismatch_rate",
    "timer",
    "requested_timer",
    "effective_timer",
    "timer_fallback_reason",
    "reference_requested_timer",
    "reference_effective_timer",
    "reference_timer_fallback_reason",
    "import_ms",
    "build_ms",
    "first_call_ms",
    "warmup_ms",
    "graph_capture_ms",
    "steady_state_ms",
    "reference_first_call_ms",
    "reference_warmup_ms",
    "reference_graph_capture_ms",
    "reference_steady_state_ms",
    "reference_mean_ms",
    "reference_median_ms",
    "reference_min_ms",
    "reference_max_ms",
    "reference_stddev_ms",
    "reference_cv",
    "reference_p50_ms",
    "reference_p90_ms",
    "reference_p95_ms",
    "reference_p99_ms",
    "reference_unstable",
    "legacy_graph_ms",
    "candidate_mean_ms",
    "candidate_median_ms",
    "candidate_min_ms",
    "candidate_max_ms",
    "candidate_p50_ms",
    "candidate_p90_ms",
    "candidate_p95_ms",
    "candidate_p99_ms",
    "candidate_stddev_ms",
    "candidate_cv",
    "candidate_unstable",
    "instability_reason",
    "speedup",
    "slowdown_pct",
    "latency_delta_ms",
    "performance_formal",
    "ranking_eligible",
    "performance_gate_status",
    "performance_gate_reasons",
    "perf_on_correctness_fail",
    "gate_max_slowdown_pct",
    "gate_min_speedup",
    "gate_max_candidate_median_ms",
    "gate_max_cv",
    "gate_max_memory_bytes",
    "gate_unsupported_policy",
    "tflops",
    "effective_bandwidth_gbps",
    "flops",
    "estimated_bytes",
    "arithmetic_intensity",
    "throughput",
    "peak_memory_bytes",
    "peak_reserved_memory_bytes",
    "workspace_bytes",
    "error_type",
    "error_message",
    "diagnostic_path",
    "stdout_path",
    "stderr_path",
    "profile_path",
)

_ACCELERATOR_FIELDS = (
    "visible_devices",
    "accelerator_backend",
    "accelerator_runtime_version",
    "gpu_arch",
    "gpu_identity_resolution",
)
_ACCELERATOR_INSERT_AT = RESULTS_FIELDNAMES_V4.index("cuda_visible_devices")
RESULTS_FIELDNAMES = (
    *RESULTS_FIELDNAMES_V4[:_ACCELERATOR_INSERT_AT],
    *_ACCELERATOR_FIELDS,
    *RESULTS_FIELDNAMES_V4[_ACCELERATOR_INSERT_AT:],
)

_RESULTS_V2_FIELDNAMES = tuple(name for name in RESULTS_FIELDNAMES_V4 if name not in {
    "imported_legacy", "legacy_graph_ms",
    "gpu_uuid", "logical_device", "visible_device", "cuda_visible_devices", "driver_version",
    "other_compute_processes_detected", "telemetry_error", "reference_requested_timer",
    "reference_effective_timer", "reference_timer_fallback_reason", "latency_delta_ms",
    "performance_formal", "ranking_eligible", "performance_gate_status",
    "performance_gate_reasons", "perf_on_correctness_fail", "gate_max_slowdown_pct",
    "gate_min_speedup", "gate_max_candidate_median_ms", "gate_max_cv",
    "gate_max_memory_bytes", "gate_unsupported_policy", "peak_reserved_memory_bytes",
})

CORRECTNESS_OUTPUT_FIELDNAMES = (
    "schema_version",
    "result_id",
    "output_path",
    "comparator",
    "reference_dtype",
    "candidate_dtype",
    "reference_shape",
    "candidate_shape",
    "rtol",
    "atol",
    "passed",
    "max_abs_error",
    "mean_abs_error",
    "p95_abs_error",
    "max_rel_error",
    "rmse",
    "rel_l2",
    "cosine_similarity",
    "mismatch_count",
    "mismatch_rate",
    "reference_nan_count",
    "candidate_nan_count",
    "diagnostic_path",
)

PERFORMANCE_SAMPLE_FIELDNAMES = (
    "schema_version",
    "result_id",
    "implementation_role",
    "reference_dtype",
    "candidate_dtype",
    "reference_shape",
    "candidate_shape",
    "sample_index",
    "inner_iterations",
    "elapsed_ms",
    "per_call_ms",
    "order_index",
    "requested_timer",
    "effective_timer",
    "fallback_reason",
    "gpu_clock_mhz",
    "memory_clock_mhz",
    "temperature_c",
    "power_w",
)

MODEL_PROJECTION_FIELDNAMES = (
    "schema_version", "run_id", "evaluation_id", "result_id", "suite_id",
    "projection_id", "phase", "quant_profile", "model_input", "raw_context",
    "operator_id", "candidate_id", "case_id", "adapter_id", "display_name",
    "backend", "kind", "legacy_shape", "instances", "implementation_role", "per_call_ms",
    "projected_model_ms", "status", "reason",
)

_RESULTS_V1_FIELDNAMES = tuple(
    name
    for name in _RESULTS_V2_FIELDNAMES
    if name
    not in {
        "requested_timer",
        "effective_timer",
        "timer_fallback_reason",
        "steady_state_ms",
        "reference_first_call_ms",
        "reference_warmup_ms",
        "reference_graph_capture_ms",
        "reference_steady_state_ms",
        "reference_mean_ms",
        "reference_min_ms",
        "reference_max_ms",
        "reference_stddev_ms",
        "reference_cv",
        "reference_p50_ms",
        "reference_p90_ms",
        "reference_p95_ms",
        "reference_p99_ms",
        "reference_unstable",
        "candidate_mean_ms",
        "candidate_min_ms",
        "candidate_max_ms",
        "candidate_p50_ms",
        "candidate_p90_ms",
        "candidate_p99_ms",
        "candidate_unstable",
        "instability_reason",
        "flops",
        "estimated_bytes",
        "arithmetic_intensity",
    }
)


def _results_columns(names: tuple[str, ...]) -> tuple[CsvColumn, ...]:
    return _columns(
        names,
        required=frozenset(
            {
                "schema_version",
                "run_id",
                "evaluation_id",
                "suite_id",
                "mode",
                "result_id",
                "operator_id",
                "contract_version",
                "candidate_id",
                "imported_legacy",
                "environment_fingerprint",
                "case_id",
                "seed",
                "status",
                "correctness_status",
                "performance_status",
            }
        ),
        integers=frozenset(
            {
                "schema_version",
                "contract_version",
                "seed",
                "failed_output_count",
                "mismatch_count",
                "peak_memory_bytes",
                "workspace_bytes",
                "peak_reserved_memory_bytes",
                "gate_max_memory_bytes",
            }
        ),
        numbers=frozenset(
            {
                "max_abs_error",
                "max_rel_error",
                "rmse",
                "rel_l2",
                "cosine_similarity",
                "mismatch_rate",
                "import_ms",
                "build_ms",
                "first_call_ms",
                "warmup_ms",
                "graph_capture_ms",
                "steady_state_ms",
                "reference_first_call_ms",
                "reference_warmup_ms",
                "reference_graph_capture_ms",
                "reference_steady_state_ms",
                "reference_mean_ms",
                "reference_median_ms",
                "reference_min_ms",
                "reference_max_ms",
                "reference_stddev_ms",
                "reference_cv",
                "reference_p50_ms",
                "reference_p90_ms",
                "reference_p95_ms",
                "reference_p99_ms",
                "legacy_graph_ms",
                "candidate_mean_ms",
                "candidate_median_ms",
                "candidate_min_ms",
                "candidate_max_ms",
                "candidate_p50_ms",
                "candidate_p90_ms",
                "candidate_p95_ms",
                "candidate_p99_ms",
                "candidate_stddev_ms",
                "candidate_cv",
                "speedup",
                "slowdown_pct",
                "latency_delta_ms",
                "gate_max_slowdown_pct",
                "gate_min_speedup",
                "gate_max_candidate_median_ms",
                "gate_max_cv",
                "tflops",
                "effective_bandwidth_gbps",
                "flops",
                "estimated_bytes",
                "arithmetic_intensity",
                "throughput",
            }
        ),
        booleans=frozenset(
            {"correctness_pass", "reference_unstable", "candidate_unstable",
             "performance_formal", "ranking_eligible", "perf_on_correctness_fail",
             "other_compute_processes_detected", "imported_legacy"}
        ),
        allowed_values={key: value for key, value in {
            "mode": frozenset({"all", "correctness", "performance"}),
            "status": frozenset(status.value for status in ResultStatus),
            "correctness_status": frozenset(
                status.value for status in CorrectnessStatus
            ),
            "performance_status": frozenset(
                status.value for status in PerformanceStatus
            ),
            "performance_gate_status": frozenset({"passed", "failed", "skipped"}),
            "gate_unsupported_policy": frozenset({"fail", "allow"}),
            "accelerator_backend": frozenset({"rocm", "cuda"}),
            "gpu_identity_resolution": frozenset({"kfd", "torch", "unresolved"}),
        }.items() if key in names},
    )


RESULTS_SCHEMA_V1 = CsvSchema(
    "results.csv",
    _results_columns(_RESULTS_V1_FIELDNAMES),
    ("result_id",),
    version=1,
)


def _migrate_results_v1(row: Mapping[str, object], version: int) -> Mapping[str, object]:
    if version not in {1, 2}:
        raise CsvContractError(f"unsupported results.csv migration from v{version}")
    migrated: dict[str, object] = dict(row)
    migrated["schema_version"] = 3
    timer = row.get("timer") or ""
    if version == 1:
        migrated["requested_timer"] = timer
        migrated["effective_timer"] = timer
        migrated["timer_fallback_reason"] = "not_recorded_in_schema_v1" if timer else None
    migrated.update({
        "reference_requested_timer": None,
        "reference_effective_timer": None,
        "reference_timer_fallback_reason": "not_recorded_before_schema_v3",
        "performance_formal": False,
        "ranking_eligible": False,
        "performance_gate_status": "skipped",
        "performance_gate_reasons": '["legacy_schema_not_rankable"]',
        "perf_on_correctness_fail": False,
        "gate_max_slowdown_pct": None,
        "gate_min_speedup": None,
        "gate_max_candidate_median_ms": None,
        "gate_max_cv": None,
        "gate_max_memory_bytes": None,
        "gate_unsupported_policy": "fail",
    })
    return migrated


RESULTS_SCHEMA_V2 = CsvSchema(
    "results.csv", _results_columns(_RESULTS_V2_FIELDNAMES), ("result_id",), version=2,
)

RESULTS_SCHEMA_V3 = CsvSchema(
    "results.csv",
    _results_columns(
        tuple(
            name
            for name in RESULTS_FIELDNAMES_V4
            if name not in {"imported_legacy", "legacy_graph_ms"}
        )
    ),
    ("result_id",),
    version=3,
    compatible_previous=(RESULTS_SCHEMA_V1, RESULTS_SCHEMA_V2),
    migrate_previous=_migrate_results_v1,
)


def _validate_results_v4_row(row: Mapping[str, str]) -> None:
    """Enforce provenance semantics beyond nullable column mechanics."""

    visible_devices = row.get("visible_devices")
    if visible_devices:
        try:
            mapping = json.loads(visible_devices)
        except (TypeError, json.JSONDecodeError) as error:
            raise CsvContractError(
                "visible_devices must be a JSON object"
            ) from error
        allowed = {
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        }
        if (
            not isinstance(mapping, dict)
            or any(key not in allowed for key in mapping)
            or any(not isinstance(value, str) for value in mapping.values())
        ):
            raise CsvContractError(
                "visible_devices must map known visibility variables to strings"
            )
    if (
        row.get("gpu_identity_resolution") == "torch"
        and "gpu_identity_torch_fallback" not in row.get("telemetry_error", "")
    ):
        raise CsvContractError(
            "torch GPU identity fallback must be explicit in telemetry_error"
        )

    imported = row.get("imported_legacy") == "true"
    provenance = (
        "timestamp_utc",
        "reference_id",
        "candidate_source_hash",
        "reference_source_hash",
        "case_hash",
    )
    if not imported:
        missing = [name for name in provenance if not row.get(name)]
        if missing:
            raise CsvContractError(
                "non-legacy results require provenance: " + ", ".join(missing)
            )
        if row.get("legacy_graph_ms"):
            raise CsvContractError(
                "non-legacy results must leave legacy_graph_ms empty"
            )
        return

    allowed_populated = {
        "schema_version", "run_id", "evaluation_id", "suite_id", "mode",
        "result_id", "operator_id", "contract_version", "candidate_id",
        "imported_legacy", "environment_fingerprint", "device", "case_id",
        "seed", "tags", "input_summary", "status", "correctness_status",
        "performance_status", "skip_reason", "timer", "requested_timer",
        "effective_timer", "legacy_graph_ms", "performance_formal",
        "ranking_eligible", "performance_gate_status", "performance_gate_reasons",
        "perf_on_correctness_fail", "gate_unsupported_policy", "error_type",
        "error_message",
    }
    unexpected = sorted(
        name for name, value in row.items()
        if value not in {None, ""} and name not in allowed_populated
    )
    if unexpected:
        raise CsvContractError(
            "legacy imports cannot populate unavailable fields: "
            + ", ".join(unexpected)
        )
    expected = {
        "suite_id": "legacy_import",
        "mode": "performance",
        "status": "skipped",
        "correctness_status": "skipped",
        "performance_status": "skipped",
        "skip_reason": "imported_legacy_without_correctness_or_raw_samples",
        "timer": "legacy_cuda_graph",
        "requested_timer": "legacy_cuda_graph",
        "effective_timer": "legacy_cuda_graph",
        "performance_formal": "false",
        "ranking_eligible": "false",
        "performance_gate_status": "skipped",
        "performance_gate_reasons": '["imported_legacy_non_rankable"]',
        "perf_on_correctness_fail": "false",
        "gate_unsupported_policy": "fail",
    }
    mismatches = [
        f"{name}={row.get(name)!r}"
        for name, value in expected.items()
        if row.get(name) != value
    ]
    if mismatches:
        raise CsvContractError(
            "legacy imports are non-correctness and non-rankable: "
            + ", ".join(mismatches)
        )
    graph_ms = row.get("legacy_graph_ms", "")
    error_type = row.get("error_type", "")
    if error_type not in {"", "legacy_unavailable"}:
        raise CsvContractError(
            "legacy error_type must be empty or legacy_unavailable"
        )
    unavailable = error_type == "legacy_unavailable"
    if unavailable:
        if graph_ms:
            raise CsvContractError("unavailable legacy rows cannot contain legacy_graph_ms")
    else:
        if row.get("error_message"):
            raise CsvContractError("executed legacy rows cannot contain error_message")
        try:
            graph_value = float(graph_ms)
        except (TypeError, ValueError) as error:
            raise CsvContractError("executed legacy rows require legacy_graph_ms") from error
        if not math.isfinite(graph_value) or graph_value < 0:
            raise CsvContractError("legacy_graph_ms must be finite and non-negative")


def _migrate_results_to_v4(
    row: Mapping[str, object], version: int
) -> Mapping[str, object]:
    if version in {1, 2}:
        migrated = dict(_migrate_results_v1(row, version))
    elif version == 3:
        migrated = dict(row)
    else:
        raise CsvContractError(f"unsupported results.csv migration from v{version}")
    migrated["schema_version"] = 4
    migrated["imported_legacy"] = False
    migrated["legacy_graph_ms"] = None
    return migrated


RESULTS_SCHEMA_V4 = CsvSchema(
    "results.csv",
    _results_columns(RESULTS_FIELDNAMES_V4),
    ("result_id",),
    version=4,
    compatible_previous=(RESULTS_SCHEMA_V1, RESULTS_SCHEMA_V2, RESULTS_SCHEMA_V3),
    migrate_previous=_migrate_results_to_v4,
    validate_row=_validate_results_v4_row,
)


def _migrate_results_to_v5(
    row: Mapping[str, object], version: int
) -> Mapping[str, object]:
    if version in {1, 2, 3}:
        migrated = dict(_migrate_results_to_v4(row, version))
    elif version == 4:
        migrated = dict(row)
    else:
        raise CsvContractError(f"unsupported results.csv migration from v{version}")
    migrated["schema_version"] = 5
    for name in _ACCELERATOR_FIELDS:
        migrated[name] = None
    return migrated


RESULTS_SCHEMA = CsvSchema(
    "results.csv",
    _results_columns(RESULTS_FIELDNAMES),
    ("result_id",),
    version=5,
    compatible_previous=(
        RESULTS_SCHEMA_V1,
        RESULTS_SCHEMA_V2,
        RESULTS_SCHEMA_V3,
        RESULTS_SCHEMA_V4,
    ),
    migrate_previous=_migrate_results_to_v5,
    validate_row=_validate_results_v4_row,
)

CORRECTNESS_OUTPUTS_SCHEMA = CsvSchema(
    "correctness_outputs.csv",
    _columns(
        CORRECTNESS_OUTPUT_FIELDNAMES,
        required=frozenset(
            {
                "schema_version",
                "result_id",
                "output_path",
                "comparator",
                "reference_dtype",
                "candidate_dtype",
                "reference_shape",
                "candidate_shape",
                "passed",
            }
        ),
        integers=frozenset(
            {
                "schema_version",
                "mismatch_count",
                "reference_nan_count",
                "candidate_nan_count",
            }
        ),
        numbers=frozenset(
            {
                "rtol",
                "atol",
                "max_abs_error",
                "mean_abs_error",
                "p95_abs_error",
                "max_rel_error",
                "rmse",
                "rel_l2",
                "cosine_similarity",
                "mismatch_rate",
            }
        ),
        booleans=frozenset({"passed"}),
    ),
    ("result_id", "output_path"),
)

_PERFORMANCE_SAMPLE_V2_FIELDNAMES = tuple(
    name
    for name in PERFORMANCE_SAMPLE_FIELDNAMES
    if name
    not in {
        "reference_dtype",
        "candidate_dtype",
        "reference_shape",
        "candidate_shape",
    }
)

_PERFORMANCE_SAMPLE_V1_FIELDNAMES = tuple(
    name
    for name in _PERFORMANCE_SAMPLE_V2_FIELDNAMES
    if name not in {"requested_timer", "effective_timer", "fallback_reason"}
)


def _performance_sample_columns(names: tuple[str, ...]) -> tuple[CsvColumn, ...]:
    required = {
        "schema_version",
        "result_id",
        "implementation_role",
        "sample_index",
        "inner_iterations",
        "elapsed_ms",
        "per_call_ms",
        "order_index",
    }
    if "requested_timer" in names:
        required.update({"requested_timer", "effective_timer"})
    return _columns(
        names,
        required=frozenset(required),
        integers=frozenset(
            {"schema_version", "sample_index", "inner_iterations", "order_index"}
        ),
        numbers=frozenset(
            {
                "elapsed_ms",
                "per_call_ms",
                "gpu_clock_mhz",
                "memory_clock_mhz",
                "temperature_c",
                "power_w",
            }
        ),
        allowed_values={
            "implementation_role": frozenset({"reference", "candidate"})
        },
    )


PERFORMANCE_SAMPLES_SCHEMA_V1 = CsvSchema(
    "performance_samples.csv",
    _performance_sample_columns(_PERFORMANCE_SAMPLE_V1_FIELDNAMES),
    ("result_id", "implementation_role", "sample_index"),
    version=1,
)


PERFORMANCE_SAMPLES_SCHEMA_V2 = CsvSchema(
    "performance_samples.csv",
    _performance_sample_columns(_PERFORMANCE_SAMPLE_V2_FIELDNAMES),
    ("result_id", "implementation_role", "sample_index"),
    version=2,
)


def _migrate_performance_samples(
    row: Mapping[str, object], version: int
) -> Mapping[str, object]:
    if version not in {1, 2}:
        raise CsvContractError(
            f"unsupported performance_samples.csv migration from v{version}"
        )
    migrated: dict[str, object] = dict(row)
    migrated["schema_version"] = 3
    if version == 1:
        migrated.update(
            {
                "requested_timer": "legacy_unknown",
                "effective_timer": "legacy_unknown",
                "fallback_reason": "not_recorded_in_schema_v1",
            }
        )
    migrated.update(
        {
            "reference_dtype": None,
            "candidate_dtype": None,
            "reference_shape": None,
            "candidate_shape": None,
        }
    )
    return migrated


PERFORMANCE_SAMPLES_SCHEMA = CsvSchema(
    "performance_samples.csv",
    _performance_sample_columns(PERFORMANCE_SAMPLE_FIELDNAMES),
    ("result_id", "implementation_role", "sample_index"),
    version=3,
    compatible_previous=(
        PERFORMANCE_SAMPLES_SCHEMA_V1,
        PERFORMANCE_SAMPLES_SCHEMA_V2,
    ),
    migrate_previous=_migrate_performance_samples,
)

MODEL_PROJECTION_SCHEMA = CsvSchema(
    "model_projection.csv",
    _columns(
        MODEL_PROJECTION_FIELDNAMES,
        required=frozenset({
            "schema_version", "run_id", "evaluation_id", "result_id", "suite_id",
            "projection_id", "phase", "quant_profile", "model_input", "raw_context",
            "operator_id", "candidate_id", "case_id", "adapter_id", "display_name",
            "backend", "kind", "legacy_shape", "instances", "implementation_role", "status",
        }),
        integers=frozenset({"schema_version", "model_input", "raw_context", "instances"}),
        numbers=frozenset({"per_call_ms", "projected_model_ms"}),
        allowed_values={
            "implementation_role": frozenset({"reference", "candidate"}),
            "status": frozenset({"measured", "correctness_failed", "unsupported",
                                 "unavailable", "not_measured"}),
        },
    ),
    ("result_id", "adapter_id", "implementation_role"),
)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync (not available on every platform)."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    replace: Callable[[str | bytes | os.PathLike[str] | os.PathLike[bytes], str | bytes | os.PathLike[str] | os.PathLike[bytes]], None] = os.replace,
) -> None:
    """Atomically replace *path* with UTF-8 text after flushing it to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _normalise_value(column: CsvColumn, value: object) -> str:
    if value is None or value == "":
        if column.nullable:
            return ""
        raise CsvContractError(f"{column.name} must not be empty")
    if column.kind == "string":
        if not isinstance(value, str):
            raise CsvContractError(f"{column.name} must be a string")
        if column.allowed_values is not None and value not in column.allowed_values:
            raise CsvContractError(
                f"{column.name} must be one of: "
                + ", ".join(sorted(column.allowed_values))
            )
        return value
    if column.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CsvContractError(f"{column.name} must be an integer")
        return str(value)
    if column.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CsvContractError(f"{column.name} must be a finite number")
        if not math.isfinite(float(value)):
            raise CsvContractError(f"{column.name} must be a finite number")
        return str(value)
    if not isinstance(value, bool):
        raise CsvContractError(f"{column.name} must be a boolean")
    return "true" if value else "false"


def _validate_stored_value(column: CsvColumn, value: str, location: str) -> None:
    if value == "":
        if column.nullable:
            return
        raise CsvContractError(f"{location}.{column.name} must not be empty")
    try:
        if column.allowed_values is not None and value not in column.allowed_values:
            raise ValueError
        if column.kind == "integer":
            parsed = int(value)
            if str(parsed) != value:
                raise ValueError
        elif column.kind == "number":
            if not math.isfinite(float(value)):
                raise ValueError
        elif column.kind == "boolean" and value not in {"true", "false"}:
            raise ValueError
    except ValueError as error:
        raise CsvContractError(
            f"{location}.{column.name} is not a valid {column.kind}"
        ) from error


def _parse_stored_row(
    schema: CsvSchema, row: Mapping[str, str]
) -> dict[str, object]:
    """Convert an already validated stored row to typed migration input."""

    result: dict[str, object] = {}
    for column in schema.columns:
        value = row[column.name]
        if value == "":
            result[column.name] = None
        elif column.kind == "integer":
            result[column.name] = int(value)
        elif column.kind == "number":
            result[column.name] = float(value)
        elif column.kind == "boolean":
            result[column.name] = value == "true"
        else:
            result[column.name] = value
    return result


class AtomicCsvTable:
    """A schema-bound, deduplicating, atomically replaced CSV table."""

    def __init__(
        self,
        path: Path,
        schema: CsvSchema,
        *,
        replace: Callable[[str | bytes | os.PathLike[str] | os.PathLike[bytes], str | bytes | os.PathLike[str] | os.PathLike[bytes]], None] = os.replace,
    ) -> None:
        self.path = Path(path)
        self.schema = schema
        self._replace = replace
        if self.path.name != schema.filename:
            raise CsvContractError(
                f"{self.path.name} does not match schema file {schema.filename}"
            )

    def initialise(self) -> None:
        if not self.path.exists():
            self._write_rows(())
        else:
            self.read_rows()

    def read_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            source_schema = self.schema
            if reader.fieldnames != list(self.schema.fieldnames):
                source_schema = next(
                    (
                        previous
                        for previous in self.schema.compatible_previous
                        if reader.fieldnames == list(previous.fieldnames)
                    ),
                    None,
                )
            if source_schema is None:
                raise CsvContractError(
                    f"{self.path} has an incompatible header/schema"
                )
            rows = list(reader)
        seen: set[tuple[str, ...]] = set()
        for number, row in enumerate(rows, start=2):
            if None in row or set(row) != set(source_schema.fieldnames):
                raise CsvContractError(f"{self.path}:{number} is malformed")
            if row["schema_version"] != str(source_schema.version):
                raise CsvContractError(
                    f"{self.path}:{number} has incompatible schema_version"
                )
            for column in source_schema.columns:
                _validate_stored_value(
                    column, row[column.name], f"{self.path}:{number}"
                )
            if source_schema.validate_row is not None:
                source_schema.validate_row(row)
            key = tuple(row[name] for name in source_schema.primary_key)
            if key in seen:
                raise CsvContractError(f"{self.path}:{number} duplicates key {key}")
            seen.add(key)
        if source_schema is self.schema:
            return rows
        assert self.schema.migrate_previous is not None
        migrated = [
            self.normalise(
                self.schema.migrate_previous(
                    _parse_stored_row(source_schema, row), source_schema.version
                )
            )
            for row in rows
        ]
        return migrated

    def append(self, row: Mapping[str, object]) -> bool:
        """Persist one row; return ``False`` for an identical existing row."""

        normalised = self.normalise(row)
        rows = self.read_rows()
        new_key = self._key(normalised)
        for existing in rows:
            if self._key(existing) != new_key:
                continue
            if existing == normalised:
                return False
            raise CsvConflictError(
                f"{self.path} already contains key {new_key} with different data"
            )
        self._write_rows((*rows, normalised))
        return True

    def append_many(self, rows: Iterable[Mapping[str, object]]) -> int:
        """Atomically persist a batch after validating every key and row."""

        current = self.read_rows()
        by_key = {self._key(row): row for row in current}
        inserted = 0
        for row in rows:
            normalised = self.normalise(row)
            key = self._key(normalised)
            existing = by_key.get(key)
            if existing is not None:
                if existing != normalised:
                    raise CsvConflictError(
                        f"{self.path} already contains key {key} with different data"
                    )
                continue
            current.append(normalised)
            by_key[key] = normalised
            inserted += 1
        if inserted:
            self._write_rows(current)
        return inserted

    def replace_partitions(
        self,
        rows: Iterable[Mapping[str, object]],
        *,
        partition_fields: tuple[str, ...],
    ) -> int:
        """Atomically replace derived rows for one or more incomplete results.

        This is intended for artifacts written before a separate completion
        marker. A resumed result may produce different timings, so stale rows
        must be replaced rather than treated as a key conflict.
        """

        if not partition_fields or not set(partition_fields).issubset(self.schema.fieldnames):
            raise CsvContractError("partition_fields must name schema columns")
        replacements = [self.normalise(row) for row in rows]
        if not replacements:
            return 0
        partitions = {tuple(row[field] for field in partition_fields) for row in replacements}
        replacement_keys = [self._key(row) for row in replacements]
        if len(replacement_keys) != len(set(replacement_keys)):
            raise CsvConflictError("replacement rows contain duplicate primary keys")
        current = [
            row for row in self.read_rows()
            if tuple(row[field] for field in partition_fields) not in partitions
        ]
        current.extend(replacements)
        self._write_rows(current)
        return len(replacements)

    def normalise(self, row: Mapping[str, object]) -> dict[str, str]:
        unknown = set(row).difference(self.schema.fieldnames)
        if unknown:
            raise CsvContractError(
                "unknown CSV fields: " + ", ".join(sorted(unknown))
            )
        candidate = dict(row)
        candidate.setdefault("schema_version", self.schema.version)
        if candidate["schema_version"] != self.schema.version:
            raise CsvContractError(
                f"schema_version must be {self.schema.version}"
            )
        normalised = {
            column.name: _normalise_value(column, candidate.get(column.name))
            for column in self.schema.columns
        }
        if self.schema.validate_row is not None:
            self.schema.validate_row(normalised)
        return normalised

    def _key(self, row: Mapping[str, str]) -> tuple[str, ...]:
        return tuple(row[name] for name in self.schema.primary_key)

    def _write_rows(self, rows: Iterable[Mapping[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=self.schema.fieldnames,
                    dialect="excel",
                    lineterminator="\r\n",
                    extrasaction="raise",
                )
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            self._replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise


__all__ = [
    "CSV_SCHEMA_VERSION",
    "AtomicCsvTable",
    "CORRECTNESS_OUTPUTS_SCHEMA",
    "CORRECTNESS_OUTPUT_FIELDNAMES",
    "CsvColumn",
    "CsvConflictError",
    "CsvContractError",
    "CsvSchema",
    "MODEL_PROJECTION_FIELDNAMES",
    "MODEL_PROJECTION_SCHEMA",
    "PERFORMANCE_SAMPLES_SCHEMA",
    "PERFORMANCE_SAMPLES_SCHEMA_V1",
    "PERFORMANCE_SAMPLES_SCHEMA_V2",
    "PERFORMANCE_SAMPLE_FIELDNAMES",
    "RESULTS_FIELDNAMES",
    "RESULTS_SCHEMA",
    "RESULTS_SCHEMA_V1",
    "RESULTS_SCHEMA_V2",
    "RESULTS_SCHEMA_V3",
    "RESULTS_SCHEMA_V4",
    "atomic_write_text",
]
