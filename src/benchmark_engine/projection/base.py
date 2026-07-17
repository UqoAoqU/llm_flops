"""Model-level projections derived from immutable per-call benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from benchmark_engine.models import CaseSpec


@dataclass(frozen=True)
class ProjectionMapping:
    adapter_id: str
    phase: str
    display_name: str
    backend: str
    instances: int
    operator_id: str | None
    kind: str
    shape: tuple[int, ...] | None = None


class ModelProjection(Protocol):
    """A projection may multiply timing fields, but never alter evaluation data."""

    projection_id: str

    def mappings(self, phase: str, quant_profile: str) -> tuple[ProjectionMapping, ...]: ...

    def mapping_for_case(self, operator_id: str, case: CaseSpec) -> ProjectionMapping | None: ...


__all__ = ["ModelProjection", "ProjectionMapping"]
