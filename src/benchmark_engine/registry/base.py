"""Immutable registry contracts used by the controller.

The objects in this module contain paths and declarative metadata only.  In
particular, an :class:`OperatorRegistry` implementation must not import an
operator or candidate while satisfying this interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeAlias

from benchmark_engine.models import ImplementationSpec


JsonValue: TypeAlias = (
    type(None)
    | bool
    | int
    | float
    | str
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)


def _immutable_mapping(value: Mapping) -> Mapping:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class CorrectnessManifest:
    """Declarative correctness defaults from ``operator.yaml``."""

    default_comparator: str = "floating"
    rtol: float = 0.0
    atol: float = 0.0
    equal_nan: bool = False
    determinism_repeats: int = 1


@dataclass(frozen=True)
class PerformanceManifest:
    """Declarative performance defaults from ``operator.yaml``."""

    timer: str = "auto"
    graph_mode: str = "auto"
    warmup: int = 5
    samples: int = 30
    inner_iterations: int = 20
    timeout_s: int = 600
    regression_threshold_pct: float = 5.0


@dataclass(frozen=True)
class BuildManifest:
    """A build command represented as argv, never as a shell command."""

    command: tuple[str, ...]
    timeout_s: int = 600


@dataclass(frozen=True)
class OperatorManifest:
    """Strict schema-v1 representation of an operator manifest."""

    schema_version: int
    operator_id: str
    contract_version: int
    description: str
    reference_entrypoint: str
    spec_entrypoint: str
    device_types: tuple[str, ...]
    tags: tuple[str, ...]
    correctness: CorrectnessManifest
    performance: PerformanceManifest
    cost_model: str


@dataclass(frozen=True)
class CandidateManifest:
    """Strict schema-v1 representation of an optional candidate manifest."""

    schema_version: int
    candidate_id: str
    operator_id: str | None = None
    entrypoint: str = "implementation:operator"
    framework: str = "python"
    build: BuildManifest | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class OperatorSpecMetadata:
    """Static location of an operator spec; importing it belongs to a worker."""

    operator_id: str
    root: Path
    entrypoint: str
    source_hash: str
    manifest: OperatorManifest


@dataclass(frozen=True, order=True)
class RegistryIssue:
    """A deterministic, machine-readable registry validation error."""

    code: str
    path: Path
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.path} [{self.field}]: {self.message}"


@dataclass(frozen=True)
class RegistrySnapshot:
    """An immutable point-in-time view of registry metadata."""

    references: Mapping[str, ImplementationSpec] = field(default_factory=dict)
    candidates: Mapping[str, tuple[ImplementationSpec, ...]] = field(
        default_factory=dict
    )
    operator_manifests: Mapping[str, OperatorManifest] = field(default_factory=dict)
    candidate_manifests: Mapping[tuple[str, str], CandidateManifest] = field(
        default_factory=dict
    )
    operator_specs: Mapping[str, OperatorSpecMetadata] = field(default_factory=dict)
    discovered_operator_ids: tuple[str, ...] = ()
    issues: tuple[RegistryIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", _immutable_mapping(self.references))
        object.__setattr__(
            self,
            "candidates",
            _immutable_mapping(
                {key: tuple(value) for key, value in self.candidates.items()}
            ),
        )
        object.__setattr__(
            self, "operator_manifests", _immutable_mapping(self.operator_manifests)
        )
        object.__setattr__(
            self, "candidate_manifests", _immutable_mapping(self.candidate_manifests)
        )
        object.__setattr__(
            self, "operator_specs", _immutable_mapping(self.operator_specs)
        )
        object.__setattr__(
            self, "discovered_operator_ids", tuple(self.discovered_operator_ids)
        )
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def is_valid(self) -> bool:
        return not self.issues


class OperatorRegistry(Protocol):
    """Controller-side registry interface."""

    def discover(self) -> RegistrySnapshot: ...

    def get_reference(self, operator_id: str) -> ImplementationSpec: ...

    def get_candidates(self, operator_id: str) -> Sequence[ImplementationSpec]: ...

    def get_operator_spec(self, operator_id: str) -> OperatorSpecMetadata: ...

    def validate(self) -> Sequence[RegistryIssue]: ...
