"""Bounded, JSON-safe correctness diagnostics."""

from __future__ import annotations

import math
import shlex
import traceback
from typing import Mapping

MAX_MESSAGE = 512
MAX_TRACEBACK = 64 * 1024
MAX_WORST = 16


def _safe(value: object, depth: int = 0) -> object:
    if depth > 8:
        return "<depth-limit>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 64:
                result["_truncated"] = True
                break
            result[str(key)[:128]] = _safe(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth + 1) for item in value[:64]]
    return repr(value)[:256]


def reproduction_command(operator_id: str, candidate_id: str, case_id: str, seed: int) -> str:
    parts = ["bench", "run", "--mode", "correctness", "--operator", operator_id, "--candidate", candidate_id, "--case", case_id, "--seed", str(seed)]
    return " ".join(shlex.quote(part) for part in parts)


def exception_diagnostic(
    error: BaseException,
    *,
    stage: str,
    reproduction: str,
) -> dict[str, object]:
    full = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return {
        "kind": "exception",
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error)[:MAX_MESSAGE],
        "traceback": full[:MAX_TRACEBACK],
        "traceback_truncated": len(full) > MAX_TRACEBACK,
        "reproduction_command": reproduction,
    }


def comparison_diagnostic(comparison: object, *, reproduction: str) -> dict[str, object]:
    diagnostics = tuple(getattr(comparison, "diagnostics", ()))[:MAX_WORST]
    return {
        "kind": "comparison",
        "comparator": str(getattr(comparison, "comparator", "unknown"))[:128],
        "failed_output_path": getattr(comparison, "failed_path", None),
        "metrics": _safe(getattr(comparison, "metrics", {})),
        "worst_mismatches": _safe(diagnostics),
        "reproduction_command": reproduction,
    }


def json_safe_diagnostic(value: Mapping[str, object]) -> dict[str, object]:
    return _safe(value)  # type: ignore[return-value]
