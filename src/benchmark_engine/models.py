"""Immutable, JSON-serializable benchmark-engine data contracts.

The controller and future workers exchange only these explicit metadata
representations. Runtime objects such as tensors and candidate callables are
intentionally rejected.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


_TYPE_KEY = "__benchmark_engine_type__"


class CorrectnessStatus(str, Enum):
    """Stable correctness outcome values."""

    PLANNED = "planned"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"
    CRASHED = "crashed"


class PerformanceStatus(str, Enum):
    """Stable performance outcome values."""

    PLANNED = "planned"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    UNSTABLE = "unstable"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"
    CRASHED = "crashed"


class ResultStatus(str, Enum):
    """Stable overall evaluation lifecycle and outcome values."""

    PLANNED = "planned"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"
    CRASHED = "crashed"


_ENUM_TYPES: dict[str, type[Enum]] = {
    enum_type.__name__: enum_type
    for enum_type in (CorrectnessStatus, PerformanceStatus, ResultStatus)
}


def _encode_metadata(value: object, path: str = "value") -> object:
    """Return a JSON-safe representation or reject a runtime object."""

    if isinstance(value, Enum):
        enum_type = type(value)
        if enum_type.__name__ not in _ENUM_TYPES:
            raise TypeError(f"{path} contains an unsupported enum {enum_type.__name__}")
        return {
            _TYPE_KEY: "enum",
            "enum": enum_type.__name__,
            "value": value.value,
        }
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Path):
        return {_TYPE_KEY: "path", "value": str(value)}
    if isinstance(value, tuple):
        return {
            _TYPE_KEY: "tuple",
            "items": [
                _encode_metadata(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, frozenset):
        encoded = [_encode_metadata(item, f"{path}[]") for item in value]
        try:
            encoded.sort(key=lambda item: repr(item))
        except TypeError:
            pass
        return {_TYPE_KEY: "frozenset", "items": encoded}
    if isinstance(value, list):
        return [
            _encode_metadata(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        encoded_mapping: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string mapping key {key!r}")
            encoded_mapping[key] = _encode_metadata(item, f"{path}.{key}")
        return encoded_mapping
    raise TypeError(
        f"{path} contains non-JSON object of type {type(value).__name__}"
    )


def _decode_metadata(value: object, path: str = "value") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        return [
            _decode_metadata(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not JSON metadata")

    tag = value.get(_TYPE_KEY)
    if tag == "path" and set(value) == {_TYPE_KEY, "value"}:
        if not isinstance(value["value"], str):
            raise TypeError(f"{path}.value must be a string")
        return Path(value["value"])
    if tag == "tuple" and set(value) == {_TYPE_KEY, "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise TypeError(f"{path}.items must be a list")
        return tuple(
            _decode_metadata(item, f"{path}[{index}]")
            for index, item in enumerate(items)
        )
    if tag == "frozenset" and set(value) == {_TYPE_KEY, "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise TypeError(f"{path}.items must be a list")
        return frozenset(
            _decode_metadata(item, f"{path}[]") for item in items
        )
    if tag == "enum" and set(value) == {_TYPE_KEY, "enum", "value"}:
        enum_name = value["enum"]
        if not isinstance(enum_name, str) or enum_name not in _ENUM_TYPES:
            raise ValueError(f"{path} names an unsupported enum")
        return _ENUM_TYPES[enum_name](value["value"])
    return {
        key: _decode_metadata(item, f"{path}.{key}")
        for key, item in value.items()
    }


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _optional_integer(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _check_keys(
    data: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required.difference(data)
    extra = set(data).difference(required | optional)
    if missing:
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"unknown fields: {', '.join(sorted(extra))}")


@dataclass(frozen=True)
class ImplementationSpec:
    operator_id: str
    implementation_id: str
    role: Literal["reference", "candidate"]
    root: Path
    entrypoint: str
    source_hash: str
    manifest_version: int

    def __post_init__(self) -> None:
        if self.role not in {"reference", "candidate"}:
            raise ValueError("role must be 'reference' or 'candidate'")
        if not isinstance(self.root, Path):
            raise TypeError("root must be a pathlib.Path")

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "implementation_id": self.implementation_id,
            "role": self.role,
            "root": str(self.root),
            "entrypoint": self.entrypoint,
            "source_hash": self.source_hash,
            "manifest_version": self.manifest_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ImplementationSpec":
        data = _mapping(value, "ImplementationSpec")
        _check_keys(
            data,
            required=frozenset(
                {
                    "operator_id",
                    "implementation_id",
                    "role",
                    "root",
                    "entrypoint",
                    "source_hash",
                    "manifest_version",
                }
            ),
        )
        role = _string(data["role"], "ImplementationSpec.role")
        if role not in {"reference", "candidate"}:
            raise ValueError("ImplementationSpec.role is invalid")
        return cls(
            operator_id=_string(data["operator_id"], "ImplementationSpec.operator_id"),
            implementation_id=_string(
                data["implementation_id"], "ImplementationSpec.implementation_id"
            ),
            role=role,
            root=Path(_string(data["root"], "ImplementationSpec.root")),
            entrypoint=_string(data["entrypoint"], "ImplementationSpec.entrypoint"),
            source_hash=_string(data["source_hash"], "ImplementationSpec.source_hash"),
            manifest_version=_integer(
                data["manifest_version"], "ImplementationSpec.manifest_version"
            ),
        )


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    symbols: Mapping[str, object]
    seed: int
    tags: frozenset[str]
    timeout_s: int | None = None

    def __post_init__(self) -> None:
        _encode_metadata(self.symbols, "CaseSpec.symbols")
        if not isinstance(self.tags, frozenset) or not all(
            isinstance(tag, str) for tag in self.tags
        ):
            raise TypeError("tags must be a frozenset of strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "symbols": _encode_metadata(self.symbols, "CaseSpec.symbols"),
            "seed": self.seed,
            "tags": sorted(self.tags),
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CaseSpec":
        data = _mapping(value, "CaseSpec")
        _check_keys(
            data,
            required=frozenset({"case_id", "symbols", "seed", "tags"}),
            optional=frozenset({"timeout_s"}),
        )
        symbols = _decode_metadata(
            _mapping(data["symbols"], "CaseSpec.symbols"), "CaseSpec.symbols"
        )
        if not isinstance(symbols, dict):
            raise TypeError("CaseSpec.symbols must decode to a mapping")
        tags = _list(data["tags"], "CaseSpec.tags")
        if not all(isinstance(tag, str) for tag in tags):
            raise TypeError("CaseSpec.tags must contain strings")
        return cls(
            case_id=_string(data["case_id"], "CaseSpec.case_id"),
            symbols=symbols,
            seed=_integer(data["seed"], "CaseSpec.seed"),
            tags=frozenset(tags),
            timeout_s=_optional_integer(
                data.get("timeout_s"), "CaseSpec.timeout_s"
            ),
        )


@dataclass(frozen=True)
class InputBundle:
    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] = field(default_factory=dict)
    observed_state: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            raise TypeError("args must be a tuple")
        _encode_metadata(self.args, "InputBundle.args")
        _encode_metadata(self.kwargs, "InputBundle.kwargs")
        _encode_metadata(self.observed_state, "InputBundle.observed_state")

    def to_dict(self) -> dict[str, object]:
        return {
            "args": [
                _encode_metadata(item, f"InputBundle.args[{index}]")
                for index, item in enumerate(self.args)
            ],
            "kwargs": _encode_metadata(self.kwargs, "InputBundle.kwargs"),
            "observed_state": _encode_metadata(
                self.observed_state, "InputBundle.observed_state"
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "InputBundle":
        data = _mapping(value, "InputBundle")
        _check_keys(
            data,
            required=frozenset({"args", "kwargs", "observed_state"}),
        )
        args = tuple(
            _decode_metadata(item, f"InputBundle.args[{index}]")
            for index, item in enumerate(_list(data["args"], "InputBundle.args"))
        )
        kwargs = _decode_metadata(
            _mapping(data["kwargs"], "InputBundle.kwargs"), "InputBundle.kwargs"
        )
        observed_state = _decode_metadata(
            _mapping(data["observed_state"], "InputBundle.observed_state"),
            "InputBundle.observed_state",
        )
        if not isinstance(kwargs, dict) or not isinstance(observed_state, dict):
            raise TypeError("InputBundle mappings decoded to invalid values")
        return cls(args=args, kwargs=kwargs, observed_state=observed_state)


@dataclass(frozen=True)
class OutputLeaf:
    path: str
    dtype: str
    shape: tuple[int, ...]
    device: str
    stride: tuple[int, ...] | None = None
    layout: str | None = None
    comparator: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or not all(
            isinstance(size, int) and not isinstance(size, bool)
            for size in self.shape
        ):
            raise TypeError("shape must be a tuple of integers")
        if self.stride is not None and (
            not isinstance(self.stride, tuple)
            or not all(
                isinstance(size, int) and not isinstance(size, bool)
                for size in self.stride
            )
        ):
            raise TypeError("stride must be a tuple of integers or None")
        _encode_metadata(self.metadata, "OutputLeaf.metadata")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "device": self.device,
            "stride": None if self.stride is None else list(self.stride),
            "layout": self.layout,
            "comparator": self.comparator,
            "metadata": _encode_metadata(self.metadata, "OutputLeaf.metadata"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "OutputLeaf":
        data = _mapping(value, "OutputLeaf")
        _check_keys(
            data,
            required=frozenset({"path", "dtype", "shape", "device"}),
            optional=frozenset({"stride", "layout", "comparator", "metadata"}),
        )
        shape = tuple(
            _integer(item, "OutputLeaf.shape[]")
            for item in _list(data["shape"], "OutputLeaf.shape")
        )
        raw_stride = data.get("stride")
        stride = (
            None
            if raw_stride is None
            else tuple(
                _integer(item, "OutputLeaf.stride[]")
                for item in _list(raw_stride, "OutputLeaf.stride")
            )
        )
        metadata = _decode_metadata(
            _mapping(data.get("metadata", {}), "OutputLeaf.metadata"),
            "OutputLeaf.metadata",
        )
        if not isinstance(metadata, dict):
            raise TypeError("OutputLeaf.metadata must decode to a mapping")

        def optional_string(name: str) -> str | None:
            raw = data.get(name)
            return None if raw is None else _string(raw, f"OutputLeaf.{name}")

        return cls(
            path=_string(data["path"], "OutputLeaf.path"),
            dtype=_string(data["dtype"], "OutputLeaf.dtype"),
            shape=shape,
            device=_string(data["device"], "OutputLeaf.device"),
            stride=stride,
            layout=optional_string("layout"),
            comparator=optional_string("comparator"),
            metadata=metadata,
        )


@dataclass(frozen=True)
class OutputBundle:
    leaves: tuple[OutputLeaf, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.leaves, tuple) or not all(
            isinstance(leaf, OutputLeaf) for leaf in self.leaves
        ):
            raise TypeError("leaves must be a tuple of OutputLeaf values")
        _encode_metadata(self.metadata, "OutputBundle.metadata")

    def to_dict(self) -> dict[str, object]:
        return {
            "leaves": [leaf.to_dict() for leaf in self.leaves],
            "metadata": _encode_metadata(self.metadata, "OutputBundle.metadata"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "OutputBundle":
        data = _mapping(value, "OutputBundle")
        _check_keys(
            data,
            required=frozenset({"leaves"}),
            optional=frozenset({"metadata"}),
        )
        metadata = _decode_metadata(
            _mapping(data.get("metadata", {}), "OutputBundle.metadata"),
            "OutputBundle.metadata",
        )
        if not isinstance(metadata, dict):
            raise TypeError("OutputBundle.metadata must decode to a mapping")
        return cls(
            leaves=tuple(
                OutputLeaf.from_dict(_mapping(item, "OutputBundle.leaves[]"))
                for item in _list(data["leaves"], "OutputBundle.leaves")
            ),
            metadata=metadata,
        )


@dataclass(frozen=True)
class EvaluationIdentity:
    run_id: str
    evaluation_id: str
    operator_id: str
    candidate_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "evaluation_id": self.evaluation_id,
            "operator_id": self.operator_id,
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EvaluationIdentity":
        data = _mapping(value, "EvaluationIdentity")
        _check_keys(
            data,
            required=frozenset(
                {"run_id", "evaluation_id", "operator_id", "candidate_id"}
            ),
        )
        return cls(
            run_id=_string(data["run_id"], "EvaluationIdentity.run_id"),
            evaluation_id=_string(
                data["evaluation_id"], "EvaluationIdentity.evaluation_id"
            ),
            operator_id=_string(
                data["operator_id"], "EvaluationIdentity.operator_id"
            ),
            candidate_id=_string(
                data["candidate_id"], "EvaluationIdentity.candidate_id"
            ),
        )


@dataclass(frozen=True)
class EvaluationJob:
    identity: EvaluationIdentity
    reference: ImplementationSpec
    candidate: ImplementationSpec
    case: CaseSpec
    mode: Literal["all", "correctness", "performance"]
    output_dir: Path

    def __post_init__(self) -> None:
        if self.mode not in {"all", "correctness", "performance"}:
            raise ValueError("mode must be all, correctness, or performance")
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "case": self.case.to_dict(),
            "mode": self.mode,
            "output_dir": str(self.output_dir),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EvaluationJob":
        data = _mapping(value, "EvaluationJob")
        _check_keys(
            data,
            required=frozenset(
                {
                    "identity",
                    "reference",
                    "candidate",
                    "case",
                    "mode",
                    "output_dir",
                }
            ),
        )
        mode = _string(data["mode"], "EvaluationJob.mode")
        if mode not in {"all", "correctness", "performance"}:
            raise ValueError("EvaluationJob.mode is invalid")
        return cls(
            identity=EvaluationIdentity.from_dict(
                _mapping(data["identity"], "EvaluationJob.identity")
            ),
            reference=ImplementationSpec.from_dict(
                _mapping(data["reference"], "EvaluationJob.reference")
            ),
            candidate=ImplementationSpec.from_dict(
                _mapping(data["candidate"], "EvaluationJob.candidate")
            ),
            case=CaseSpec.from_dict(_mapping(data["case"], "EvaluationJob.case")),
            mode=mode,
            output_dir=Path(
                _string(data["output_dir"], "EvaluationJob.output_dir")
            ),
        )


@dataclass(frozen=True)
class EvaluationPlan:
    run_id: str
    mode: Literal["all", "correctness", "performance"]
    jobs: tuple[EvaluationJob, ...]
    environment_fingerprint: str

    def __post_init__(self) -> None:
        if self.mode not in {"all", "correctness", "performance"}:
            raise ValueError("mode must be all, correctness, or performance")
        if not isinstance(self.jobs, tuple) or not all(
            isinstance(job, EvaluationJob) for job in self.jobs
        ):
            raise TypeError("jobs must be a tuple of EvaluationJob values")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "jobs": [job.to_dict() for job in self.jobs],
            "environment_fingerprint": self.environment_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EvaluationPlan":
        data = _mapping(value, "EvaluationPlan")
        _check_keys(
            data,
            required=frozenset(
                {"run_id", "mode", "jobs", "environment_fingerprint"}
            ),
        )
        mode = _string(data["mode"], "EvaluationPlan.mode")
        if mode not in {"all", "correctness", "performance"}:
            raise ValueError("EvaluationPlan.mode is invalid")
        return cls(
            run_id=_string(data["run_id"], "EvaluationPlan.run_id"),
            mode=mode,
            jobs=tuple(
                EvaluationJob.from_dict(_mapping(item, "EvaluationPlan.jobs[]"))
                for item in _list(data["jobs"], "EvaluationPlan.jobs")
            ),
            environment_fingerprint=_string(
                data["environment_fingerprint"],
                "EvaluationPlan.environment_fingerprint",
            ),
        )
