"""Versioned RFC 4180 tables with crash-safe replacement semantics.

The controller is the sole writer of reporting tables.  Every mutation reads
the last complete table and rewrites it through a temporary file in the same
directory; readers therefore see either the old table or the new table, never
a partial append.
"""

from __future__ import annotations

import csv
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

    def __post_init__(self) -> None:
        names = self.fieldnames
        if not names or names[0] != "schema_version":
            raise ValueError("schema_version must be the first CSV column")
        if len(set(names)) != len(names):
            raise ValueError("CSV column names must be unique")
        if not self.primary_key or not set(self.primary_key).issubset(names):
            raise ValueError("CSV primary key must name schema columns")

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


RESULTS_FIELDNAMES = (
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
    "environment_fingerprint",
    "device",
    "gpu_name",
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
    "import_ms",
    "build_ms",
    "first_call_ms",
    "warmup_ms",
    "graph_capture_ms",
    "reference_median_ms",
    "candidate_median_ms",
    "candidate_p95_ms",
    "candidate_stddev_ms",
    "candidate_cv",
    "speedup",
    "slowdown_pct",
    "tflops",
    "effective_bandwidth_gbps",
    "throughput",
    "peak_memory_bytes",
    "workspace_bytes",
    "error_type",
    "error_message",
    "diagnostic_path",
    "stdout_path",
    "stderr_path",
    "profile_path",
)

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
    "sample_index",
    "inner_iterations",
    "elapsed_ms",
    "per_call_ms",
    "order_index",
    "gpu_clock_mhz",
    "memory_clock_mhz",
    "temperature_c",
    "power_w",
)

RESULTS_SCHEMA = CsvSchema(
    "results.csv",
    _columns(
        RESULTS_FIELDNAMES,
        required=frozenset(
            {
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
                "environment_fingerprint",
                "case_id",
                "case_hash",
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
                "reference_median_ms",
                "candidate_median_ms",
                "candidate_p95_ms",
                "candidate_stddev_ms",
                "candidate_cv",
                "speedup",
                "slowdown_pct",
                "tflops",
                "effective_bandwidth_gbps",
                "throughput",
            }
        ),
        booleans=frozenset({"correctness_pass"}),
        allowed_values={
            "mode": frozenset({"all", "correctness", "performance"}),
            "status": frozenset(status.value for status in ResultStatus),
            "correctness_status": frozenset(
                status.value for status in CorrectnessStatus
            ),
            "performance_status": frozenset(
                status.value for status in PerformanceStatus
            ),
        },
    ),
    ("result_id",),
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

PERFORMANCE_SAMPLES_SCHEMA = CsvSchema(
    "performance_samples.csv",
    _columns(
        PERFORMANCE_SAMPLE_FIELDNAMES,
        required=frozenset(
            {
                "schema_version",
                "result_id",
                "implementation_role",
                "sample_index",
                "inner_iterations",
                "elapsed_ms",
                "per_call_ms",
                "order_index",
            }
        ),
        integers=frozenset(
            {
                "schema_version",
                "sample_index",
                "inner_iterations",
                "order_index",
            }
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
    ),
    ("result_id", "implementation_role", "sample_index"),
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
            if reader.fieldnames != list(self.schema.fieldnames):
                raise CsvContractError(
                    f"{self.path} has an incompatible header/schema"
                )
            rows = list(reader)
        seen: set[tuple[str, ...]] = set()
        for number, row in enumerate(rows, start=2):
            if None in row or set(row) != set(self.schema.fieldnames):
                raise CsvContractError(f"{self.path}:{number} is malformed")
            if row["schema_version"] != str(self.schema.version):
                raise CsvContractError(
                    f"{self.path}:{number} has incompatible schema_version"
                )
            for column in self.schema.columns:
                _validate_stored_value(
                    column, row[column.name], f"{self.path}:{number}"
                )
            key = self._key(row)
            if key in seen:
                raise CsvContractError(f"{self.path}:{number} duplicates key {key}")
            seen.add(key)
        return rows

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
        return {
            column.name: _normalise_value(column, candidate.get(column.name))
            for column in self.schema.columns
        }

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
    "PERFORMANCE_SAMPLES_SCHEMA",
    "PERFORMANCE_SAMPLE_FIELDNAMES",
    "RESULTS_FIELDNAMES",
    "RESULTS_SCHEMA",
    "atomic_write_text",
]
