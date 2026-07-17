"""Strict manifest, entrypoint, and source-tree validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from benchmark_engine.ids import (
    IdentifierError,
    candidate_hash_suffix,
    validate_candidate_id,
    validate_operator_id,
)

from .base import (
    BuildManifest,
    CandidateManifest,
    CorrectnessManifest,
    JsonValue,
    OperatorManifest,
    PerformanceManifest,
    RegistryIssue,
)


_ENTRYPOINT_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPERATOR_FIELDS = frozenset(
    {
        "schema_version",
        "operator_id",
        "contract_version",
        "description",
        "reference_entrypoint",
        "spec_entrypoint",
        "device_types",
        "tags",
        "correctness",
        "performance",
        "cost_model",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "operator_id",
        "candidate_id",
        "entrypoint",
        "framework",
        "build",
        "metadata",
    }
)
_CORRECTNESS_FIELDS = frozenset(
    {"default_comparator", "rtol", "atol", "equal_nan", "determinism_repeats"}
)
_PERFORMANCE_FIELDS = frozenset(
    {
        "timer",
        "graph_mode",
        "warmup",
        "samples",
        "inner_iterations",
        "timeout_s",
        "regression_threshold_pct",
        "perf_on_correctness_fail", "min_speedup", "max_candidate_median_ms",
        "max_cv", "max_memory_bytes", "unsupported_policy", "gpu_lock_timeout_s",
    }
)
_BUILD_FIELDS = frozenset({"command", "timeout_s"})


class ManifestValidationError(ValueError):
    """One stable manifest validation failure."""

    def __init__(self, code: str, path: Path, field: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.field = field
        self.message = message

    def as_issue(self) -> RegistryIssue:
        return RegistryIssue(self.code, self.path, self.field, self.message)


class SourceHashError(ValueError):
    """A source tree is unsafe or cannot be hashed canonically."""

    def __init__(self, code: str, path: Path, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


def _fail(code: str, path: Path, field: str, message: str) -> None:
    raise ManifestValidationError(code, path, field, message)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        _fail("manifest.missing", path, "$", "manifest file is missing")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        _fail("manifest.yaml", path, "$", f"invalid YAML: {error}")
    if not isinstance(value, dict):
        _fail("manifest.type", path, "$", "manifest root must be a mapping")
    if not all(isinstance(key, str) for key in value):
        _fail("manifest.type", path, "$", "manifest keys must be strings")
    return value


def _check_fields(
    value: Mapping[str, Any],
    *,
    path: Path,
    field: str,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        name = unknown[0]
        _fail(
            "manifest.unknown_field",
            path,
            f"{field}.{name}" if field != "$" else name,
            f"unknown field {name!r}",
        )
    missing = sorted(required.difference(value))
    if missing:
        name = missing[0]
        _fail(
            "manifest.missing_field",
            path,
            f"{field}.{name}" if field != "$" else name,
            f"required field {name!r} is missing",
        )


def _mapping(value: object, path: Path, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail("manifest.type", path, field, "must be a mapping with string keys")
    return value


def _string(value: object, path: Path, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail("manifest.type", path, field, "must be a string")
    if nonempty and not value.strip():
        _fail("manifest.value", path, field, "must not be empty")
    return value


def _integer(
    value: object,
    path: Path,
    field: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("manifest.type", path, field, "must be an integer")
    if positive and value <= 0:
        _fail("manifest.value", path, field, "must be a positive integer")
    if nonnegative and value < 0:
        _fail("manifest.value", path, field, "must be a non-negative integer")
    return value


def _number(
    value: object, path: Path, field: str, *, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("manifest.type", path, field, "must be a number")
    number = float(value)
    if not math.isfinite(number):
        _fail("manifest.value", path, field, "must be finite")
    if nonnegative and number < 0:
        _fail("manifest.value", path, field, "must be non-negative")
    return number


def _boolean(value: object, path: Path, field: str) -> bool:
    if not isinstance(value, bool):
        _fail("manifest.type", path, field, "must be a boolean")
    return value


def _string_tuple(
    value: object, path: Path, field: str, *, unique: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail("manifest.type", path, field, "must be an array of strings")
    result = tuple(_string(item, path, f"{field}[{index}]") for index, item in enumerate(value))
    if unique and len(set(result)) != len(result):
        _fail("manifest.value", path, field, "must not contain duplicate values")
    return result


def _schema_version(value: object, path: Path) -> int:
    version = _integer(value, path, "schema_version")
    if version != 1:
        _fail(
            "manifest.schema_version",
            path,
            "schema_version",
            f"unsupported schema version {version}; expected 1",
        )
    return version


def _correctness(value: object, path: Path) -> CorrectnessManifest:
    data = _mapping(value, path, "correctness")
    _check_fields(
        data, path=path, field="correctness", allowed=_CORRECTNESS_FIELDS
    )
    defaults = CorrectnessManifest()
    return CorrectnessManifest(
        default_comparator=_string(
            data.get("default_comparator", defaults.default_comparator),
            path,
            "correctness.default_comparator",
        ),
        rtol=_number(
            data.get("rtol", defaults.rtol),
            path,
            "correctness.rtol",
            nonnegative=True,
        ),
        atol=_number(
            data.get("atol", defaults.atol),
            path,
            "correctness.atol",
            nonnegative=True,
        ),
        equal_nan=_boolean(
            data.get("equal_nan", defaults.equal_nan),
            path,
            "correctness.equal_nan",
        ),
        determinism_repeats=_integer(
            data.get("determinism_repeats", defaults.determinism_repeats),
            path,
            "correctness.determinism_repeats",
            positive=True,
        ),
    )


def _performance(value: object, path: Path) -> PerformanceManifest:
    data = _mapping(value, path, "performance")
    _check_fields(
        data, path=path, field="performance", allowed=_PERFORMANCE_FIELDS
    )
    defaults = PerformanceManifest()
    timer = _string(data.get("timer", defaults.timer), path, "performance.timer")
    if timer not in {"auto", "cuda_event", "cuda_graph", "wall_clock"}:
        _fail(
            "value",
            path,
            "performance.timer",
            "must be auto, cuda_event, cuda_graph, or wall_clock",
        )
    graph_mode = _string(
            data.get("graph_mode", defaults.graph_mode),
            path,
            "performance.graph_mode",
        )
    if graph_mode not in {"auto", "enabled", "disabled"}:
        _fail(
            "value",
            path,
            "performance.graph_mode",
            "must be auto, enabled, or disabled",
        )
    unsupported_policy = _string(data.get("unsupported_policy", defaults.unsupported_policy), path, "performance.unsupported_policy")
    if unsupported_policy not in {"fail", "allow"}:
        _fail("manifest.value", path, "performance.unsupported_policy", "must be fail or allow")
    min_speedup = data.get("min_speedup", defaults.min_speedup)
    max_candidate_median_ms = data.get(
        "max_candidate_median_ms", defaults.max_candidate_median_ms
    )
    max_cv = data.get("max_cv", defaults.max_cv)
    max_memory_bytes = data.get("max_memory_bytes", defaults.max_memory_bytes)
    return PerformanceManifest(
        timer=timer,
        graph_mode=graph_mode,
        warmup=_integer(
            data.get("warmup", defaults.warmup),
            path,
            "performance.warmup",
            nonnegative=True,
        ),
        samples=_integer(
            data.get("samples", defaults.samples),
            path,
            "performance.samples",
            positive=True,
        ),
        inner_iterations=_integer(
            data.get("inner_iterations", defaults.inner_iterations),
            path,
            "performance.inner_iterations",
            positive=True,
        ),
        timeout_s=_integer(
            data.get("timeout_s", defaults.timeout_s),
            path,
            "performance.timeout_s",
            positive=True,
        ),
        regression_threshold_pct=_number(
            data.get(
                "regression_threshold_pct", defaults.regression_threshold_pct
            ),
            path,
            "performance.regression_threshold_pct",
            nonnegative=True,
        ),
        perf_on_correctness_fail=_boolean(data.get("perf_on_correctness_fail", defaults.perf_on_correctness_fail), path, "performance.perf_on_correctness_fail"),
        min_speedup=(None if min_speedup is None else _number(min_speedup, path, "performance.min_speedup", nonnegative=True)),
        max_candidate_median_ms=(None if max_candidate_median_ms is None else _number(max_candidate_median_ms, path, "performance.max_candidate_median_ms", nonnegative=True)),
        max_cv=(None if max_cv is None else _number(max_cv, path, "performance.max_cv", nonnegative=True)),
        max_memory_bytes=(None if max_memory_bytes is None else _integer(max_memory_bytes, path, "performance.max_memory_bytes", nonnegative=True)),
        unsupported_policy=unsupported_policy,
        gpu_lock_timeout_s=_number(data.get("gpu_lock_timeout_s", defaults.gpu_lock_timeout_s), path, "performance.gpu_lock_timeout_s", nonnegative=True),
    )


def parse_operator_manifest(path: Path) -> OperatorManifest:
    """Parse ``operator.yaml`` without importing any Python module."""

    path = Path(path)
    data = _load_yaml(path)
    if "schema_version" not in data:
        _fail(
            "manifest.missing_field",
            path,
            "schema_version",
            "required field 'schema_version' is missing",
        )
    schema_version = _schema_version(data["schema_version"], path)
    _check_fields(
        data,
        path=path,
        field="$",
        allowed=_OPERATOR_FIELDS,
        required=_OPERATOR_FIELDS,
    )
    operator_id = _string(data["operator_id"], path, "operator_id")
    try:
        validate_operator_id(operator_id)
    except IdentifierError as error:
        _fail("manifest.value", path, "operator_id", str(error))
    device_types = _string_tuple(data["device_types"], path, "device_types")
    if not device_types:
        _fail("manifest.value", path, "device_types", "must not be empty")
    return OperatorManifest(
        schema_version=schema_version,
        operator_id=operator_id,
        contract_version=_integer(
            data["contract_version"], path, "contract_version", positive=True
        ),
        description=_string(data["description"], path, "description"),
        reference_entrypoint=_string(
            data["reference_entrypoint"], path, "reference_entrypoint"
        ),
        spec_entrypoint=_string(data["spec_entrypoint"], path, "spec_entrypoint"),
        device_types=device_types,
        tags=_string_tuple(data["tags"], path, "tags"),
        correctness=_correctness(data["correctness"], path),
        performance=_performance(data["performance"], path),
        cost_model=_string(data["cost_model"], path, "cost_model"),
    )


def _freeze_json(value: object, path: Path, field: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("manifest.value", path, field, "must contain finite JSON numbers")
        return value
    if isinstance(value, list):
        return tuple(
            _freeze_json(item, path, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            _fail("manifest.type", path, field, "must contain only string keys")
        return MappingProxyType(
            {
                key: _freeze_json(item, path, f"{field}.{key}")
                for key, item in value.items()
            }
        )
    _fail(
        "manifest.type",
        path,
        field,
        f"must be JSON-safe, not {type(value).__name__}",
    )


def _build(value: object, path: Path) -> BuildManifest:
    data = _mapping(value, path, "build")
    _check_fields(
        data,
        path=path,
        field="build",
        allowed=_BUILD_FIELDS,
        required=frozenset({"command"}),
    )
    command = _string_tuple(
        data["command"], path, "build.command", unique=False
    )
    if not command:
        _fail("manifest.value", path, "build.command", "must not be empty")
    return BuildManifest(
        command=command,
        timeout_s=_integer(
            data.get("timeout_s", 600), path, "build.timeout_s", positive=True
        ),
    )


def parse_candidate_manifest(path: Path) -> CandidateManifest:
    """Parse the optional strict schema-v1 ``candidate.yaml``."""

    path = Path(path)
    data = _load_yaml(path)
    if "schema_version" not in data:
        _fail(
            "manifest.missing_field",
            path,
            "schema_version",
            "required field 'schema_version' is missing",
        )
    schema_version = _schema_version(data["schema_version"], path)
    _check_fields(
        data,
        path=path,
        field="$",
        allowed=_CANDIDATE_FIELDS,
        required=frozenset({"schema_version", "candidate_id"}),
    )
    metadata = _mapping(data.get("metadata", {}), path, "metadata")
    frozen_metadata = _freeze_json(metadata, path, "metadata")
    assert isinstance(frozen_metadata, Mapping)
    operator_id = data.get("operator_id")
    candidate_id = _string(data["candidate_id"], path, "candidate_id")
    try:
        validate_candidate_id(candidate_id)
    except IdentifierError as error:
        _fail("manifest.value", path, "candidate_id", str(error))
    if operator_id is not None:
        operator_id = _string(operator_id, path, "operator_id")
        try:
            validate_operator_id(operator_id)
        except IdentifierError as error:
            _fail("manifest.value", path, "operator_id", str(error))
    return CandidateManifest(
        schema_version=schema_version,
        operator_id=operator_id,
        candidate_id=candidate_id,
        entrypoint=_string(
            data.get("entrypoint", "implementation:operator"),
            path,
            "entrypoint",
        ),
        framework=_string(data.get("framework", "python"), path, "framework"),
        build=None if "build" not in data else _build(data["build"], path),
        metadata=frozen_metadata,
    )


def validate_entrypoint(root: Path, entrypoint: str, manifest: Path, field: str) -> Path:
    """Resolve an entrypoint to a Python file without importing it."""

    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
        _fail(
            "entrypoint.format",
            manifest,
            field,
            "must have the form <module>:<attribute>",
        )
    module, attribute = entrypoint.split(":", 1)
    module_parts = module.split(".")
    attribute_parts = attribute.split(".")
    if (
        not module_parts
        or not attribute_parts
        or any(_ENTRYPOINT_PART.fullmatch(part) is None for part in module_parts)
        or any(_ENTRYPOINT_PART.fullmatch(part) is None for part in attribute_parts)
    ):
        _fail(
            "entrypoint.format",
            manifest,
            field,
            "module and attribute must be dotted Python identifiers",
        )
    resolved_root = Path(root).resolve()
    module_path = resolved_root.joinpath(*module_parts).with_suffix(".py").resolve()
    package_path = resolved_root.joinpath(*module_parts, "__init__.py").resolve()
    for candidate in (module_path, package_path):
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    _fail(
        "entrypoint.missing",
        manifest,
        field,
        f"Python module {module!r} does not exist below {resolved_root}",
    )


def _excluded_source_path(relative: Path) -> bool:
    parts = relative.parts
    if any(
        part in {"__pycache__", "build", "dist"} or part.endswith(".egg-info")
        for part in parts
    ):
        return True
    name = relative.name
    return (
        name.endswith((".pyc", ".pyo", ".tmp", ".temp", ".swp", ".swo", ".bak", "~"))
        or name == ".DS_Store"
        or name.startswith(".#")
        or (name.startswith("#") and name.endswith("#"))
    )


def _canonical_candidate_manifest(path: Path) -> bytes:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SourceHashError(
            "source_hash.manifest", path, f"candidate.yaml cannot be canonicalized: {error}"
        ) from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SourceHashError(
            "source_hash.manifest", path, "candidate.yaml root must be a string-keyed mapping"
        )
    canonical = dict(value)
    canonical.pop("candidate_id", None)
    try:
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SourceHashError(
            "source_hash.manifest",
            path,
            f"candidate.yaml is not canonical JSON metadata: {error}",
        ) from error


def _hash_frame(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def compute_source_hash(root: Path) -> str:
    """Hash relative POSIX paths, normalized file modes, and file contents.

    ``candidate.yaml`` participates in the hash, but its ``candidate_id`` field
    is removed and the remaining YAML value is encoded as sorted canonical JSON.
    This deliberately breaks the otherwise circular dependency between the ID
    hash suffix and the manifest containing that ID.  Every other manifest field
    remains hash-significant.
    """

    root = Path(root)
    if root.is_symlink():
        raise SourceHashError("source_hash.symlink", root, "source root is a symlink")
    if not root.is_dir():
        raise SourceHashError("source_hash.root", root, "source root is not a directory")

    files: list[tuple[str, Path]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(directory_names):
            child = directory_path / name
            relative = child.relative_to(root)
            if child.is_symlink():
                raise SourceHashError(
                    "source_hash.symlink", child, "source tree contains a symlink"
                )
            if _excluded_source_path(relative):
                directory_names.remove(name)
        for name in file_names:
            child = directory_path / name
            relative = child.relative_to(root)
            if child.is_symlink():
                raise SourceHashError(
                    "source_hash.symlink", child, "source tree contains a symlink"
                )
            if not _excluded_source_path(relative):
                files.append((relative.as_posix(), child))

    digest = hashlib.sha256()
    digest.update(b"benchmark-engine-source-v1\0")
    for relative, path in sorted(files, key=lambda item: item[0]):
        file_stat = path.stat()
        mode = b"100755" if file_stat.st_mode & stat.S_IXUSR else b"100644"
        content = (
            _canonical_candidate_manifest(path)
            if relative == "candidate.yaml"
            else path.read_bytes()
        )
        _hash_frame(digest, relative.encode("utf-8"))
        _hash_frame(digest, mode)
        _hash_frame(digest, content)
    return digest.hexdigest()


def validate_candidate_hash(candidate_id: str, source_hash: str) -> None:
    """Require the ID's lowercase hex suffix to prefix the full source hash."""

    suffix = candidate_hash_suffix(candidate_id)
    if not source_hash.startswith(suffix):
        raise ValueError(
            f"candidate hash suffix {suffix!r} does not match source hash "
            f"{source_hash!r}"
        )
