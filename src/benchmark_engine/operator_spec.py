"""Trusted controller-side loading of pure operator case metadata."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from .models import CaseSpec
from .registry import (
    OperatorSpecMetadata,
    RegistrySnapshot,
    selected_registry_issues,
)
from .registry.validation import validate_entrypoint


class OperatorSpecError(ValueError):
    pass


def _attribute(value: object, dotted: str) -> object:
    for part in dotted.split("."):
        try:
            value = getattr(value, part)
        except AttributeError as error:
            raise OperatorSpecError(f"spec entrypoint has no attribute {dotted!r}") from error
    return value


def load_operator_cases(
    snapshot: RegistrySnapshot, operator_id: str
) -> tuple[CaseSpec, ...]:
    """Import only the trusted reference spec and return declarative cases."""

    issues = selected_registry_issues(
        snapshot, (operator_id,), candidate_patterns=()
    )
    if issues:
        raise OperatorSpecError(
            "registry is invalid for operator spec: "
            + "; ".join(str(issue) for issue in issues)
        )
    try:
        metadata: OperatorSpecMetadata = snapshot.operator_specs[operator_id]
    except KeyError as error:
        raise OperatorSpecError(f"operator has no spec metadata: {operator_id}") from error
    module_name, attribute_name = metadata.entrypoint.split(":", 1)
    module_path = validate_entrypoint(
        metadata.root,
        metadata.entrypoint,
        metadata.root / "operator.yaml",
        "spec_entrypoint",
    )
    import_spec = importlib.util.spec_from_file_location(
        f"_benchmark_engine_spec_{operator_id}_{metadata.source_hash[:12]}",
        Path(module_path),
    )
    if import_spec is None or import_spec.loader is None:
        raise OperatorSpecError(f"cannot load spec module {module_name!r}")
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise OperatorSpecError(f"cannot parse trusted spec module: {error}") from error
    for node in ast.walk(tree):
        imported = ()
        if isinstance(node, ast.Import):
            imported = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported = (node.module,)
        if any(name == "torch" or name.startswith("torch.") for name in imported):
            raise OperatorSpecError("pure metadata spec must not import torch")
    module = importlib.util.module_from_spec(import_spec)
    try:
        import_spec.loader.exec_module(module)
    except Exception as error:
        raise OperatorSpecError(f"trusted spec import failed: {error}") from error
    spec_object = _attribute(module, attribute_name)
    cases_method = getattr(spec_object, "cases", None)
    if not callable(cases_method):
        raise OperatorSpecError("spec entrypoint must provide cases()")
    values = cases_method()
    if not isinstance(values, (tuple, list)) or not all(
        isinstance(case, CaseSpec) for case in values
    ):
        raise OperatorSpecError("spec cases() must return CaseSpec values")
    cases = tuple(values)
    if not cases:
        raise OperatorSpecError("spec cases() must not be empty")
    ids = tuple(case.case_id for case in cases)
    if len(ids) != len(set(ids)):
        raise OperatorSpecError("spec cases() returned duplicate case_id values")
    return tuple(sorted(cases, key=lambda case: case.case_id))


__all__ = ["OperatorSpecError", "load_operator_cases"]
