"""Reference-owned correctness extension protocols."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from benchmark_engine.models import CaseSpec

from .models import InputBundle, OutputBundle


@runtime_checkable
class Comparator(Protocol):
    def compare(self, reference: OutputBundle, candidate: OutputBundle, **kwargs: object): ...


@runtime_checkable
class OperatorSpec(Protocol):
    """Stable contract implemented by a trusted reference ``spec.py``."""

    operator_id: str

    def cases(self) -> Iterable[CaseSpec]: ...
    def make_inputs(self, case: CaseSpec, context: object) -> InputBundle: ...
    def clone_inputs(self, inputs: InputBundle) -> InputBundle: ...
    def normalize_output(self, output: object) -> OutputBundle | object: ...
    def comparator(self, case: CaseSpec) -> Comparator: ...
    def cost_model(self, case: CaseSpec) -> object | None: ...
