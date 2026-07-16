"""Adapter over the legacy benchmark environment collector."""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Any


def _legacy() -> Any:
    # One implementation owns dependency/GPU version semantics.
    return importlib.import_module("benchmark_environment")


def load_lock(path: Path | None = None) -> dict[str, Any]:
    legacy = _legacy()
    return legacy.load_lock() if path is None else legacy.load_lock(Path(path))


def collect_environment(
    lock: dict[str, Any], *, include_cuda: bool = True
) -> dict[str, Any]:
    # Some optional dependency probes import third-party packages which still
    # emit deprecation warnings at import time.  Environment introspection is
    # a CLI implementation detail, so keep those warnings from leaking into a
    # successful command's stderr.  The filter is deliberately scoped to this
    # legacy probe; application warnings and probe exceptions remain visible.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _legacy().collect_environment(lock, include_cuda=include_cuda)


def validate_environment(
    lock: dict[str, Any], observed: dict[str, Any]
) -> list[str]:
    return _legacy().validate_environment(lock, observed)


def collect_report(lock_path: Path) -> dict[str, object]:
    from .fingerprint import environment_fingerprint

    lock = load_lock(lock_path)
    observed = collect_environment(lock)
    errors = validate_environment(lock, observed)
    return {
        "fingerprint": environment_fingerprint(observed),
        "fingerprint_kind": "runtime",
        "valid": not errors,
        "errors": errors,
        "environment": observed,
    }


__all__ = [
    "collect_environment",
    "collect_report",
    "load_lock",
    "validate_environment",
]
