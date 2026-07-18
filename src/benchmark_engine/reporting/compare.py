"""Strict, read-only comparison of mirrored evaluation artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from .artifact_writer import RUN_INDEX_SCHEMA
from .csv_writer import AtomicCsvTable, RESULTS_SCHEMA
from .legacy_import import is_legacy_import_directory


class CompareCompatibilityError(ValueError):
    pass


def _paths_for(output_root: Path, selector: str, *, run: bool) -> tuple[Path, ...]:
    root = Path(output_root).resolve()
    index = AtomicCsvTable(root / RUN_INDEX_SCHEMA.filename, RUN_INDEX_SCHEMA).read_rows()
    field = "run_id" if run else "evaluation_id"
    rows = [row for row in index if row[field] == selector]
    if not rows:
        raise CompareCompatibilityError(f"{field} not found: {selector}")
    paths = tuple(root / row["relative_path"] for row in rows)
    if not run and len(paths) != 1:
        raise CompareCompatibilityError(
            f"evaluation_id is ambiguous; pass its mirrored directory: {selector}"
        )
    return paths


def _evaluation_paths(output_root: Path, selector: str, *, run: bool) -> tuple[Path, ...]:
    path = Path(selector)
    if not run and (path.is_absolute() or "/" in selector or "\\" in selector):
        if not path.is_absolute():
            under_output = Path(output_root) / path
            path = under_output if under_output.is_dir() else Path(output_root).parent / path
        if not path.is_dir(): raise CompareCompatibilityError(f"evaluation path not found: {path}")
        return (path.resolve(),)
    return _paths_for(output_root, selector, run=run)


def _read(paths: tuple[Path, ...], *, include_candidate: bool) -> dict[tuple[str, ...], Mapping[str, str]]:
    result = {}
    for path in paths:
        if is_legacy_import_directory(path):
            raise CompareCompatibilityError(
                f"legacy import is non-rankable and cannot be compared: {path}"
            )
        rows = AtomicCsvTable(
            path / RESULTS_SCHEMA.filename, RESULTS_SCHEMA
        ).read_rows()
        if any(row.get("imported_legacy") == "true" for row in rows):
            raise CompareCompatibilityError(
                f"legacy import rows are non-rankable and cannot be compared: {path}"
            )
        for row in rows:
            key = (row["operator_id"], row["case_id"], row["seed"])
            if include_candidate:
                key = (row["operator_id"], row["candidate_id"], row["case_id"], row["seed"])
            if key in result: raise CompareCompatibilityError(f"duplicate compare case: {key}")
            result[key] = row
    return result


def compare_artifacts(output_root: Path, current: str, baseline: str, *, run: bool = False) -> str:
    current_rows = _read(_evaluation_paths(output_root, current, run=run), include_candidate=run)
    baseline_rows = _read(_evaluation_paths(output_root, baseline, run=run), include_candidate=run)
    if set(current_rows) != set(baseline_rows):
        raise CompareCompatibilityError("current and baseline case identities differ")
    items = []
    compatibility = (
        "operator_id", "contract_version", "case_id", "case_hash", "seed",
        "environment_fingerprint", "requested_timer", "effective_timer",
        "reference_effective_timer", "reference_source_hash",
    )
    for key in sorted(current_rows):
        now, old = current_rows[key], baseline_rows[key]
        mismatches = [name for name in compatibility if now.get(name) != old.get(name)]
        if mismatches:
            raise CompareCompatibilityError(
                f"incompatible case {key}: " + ", ".join(mismatches)
            )
        current_median = _finite(now.get("candidate_median_ms"))
        baseline_median = _finite(old.get("candidate_median_ms"))
        eligible = all(row.get("ranking_eligible") == "true" and
                       row.get("performance_gate_status") == "passed"
                       for row in (now, old))
        speedup = delta = None
        reasons: list[str] = []
        if not eligible: reasons.append("non_rankable_input")
        if current_median is None or baseline_median is None or current_median <= 0 or baseline_median <= 0:
            reasons.append("invalid_median")
        elif eligible:
            speedup = baseline_median / current_median
            delta = current_median - baseline_median
        items.append({"operator_id": key[0], "baseline_candidate_id": old["candidate_id"],
                      "current_candidate_id": now["candidate_id"],
                      "case_id": now["case_id"], "seed": int(now["seed"]),
                      "baseline_median_ms": baseline_median,
                      "current_median_ms": current_median,
                      "speedup": speedup, "latency_delta_ms": delta,
                      "ranking_eligible": eligible and not reasons,
                      "reasons": reasons})
    return json.dumps({"schema_version": 1, "kind": "run" if run else "evaluation",
                       "baseline": baseline, "current": current, "cases": items},
                      indent=2, sort_keys=True, allow_nan=False) + "\n"


def _finite(value: object) -> float | None:
    try: number = float(value)
    except (TypeError, ValueError): return None
    return number if math.isfinite(number) else None


__all__ = ["CompareCompatibilityError", "compare_artifacts"]
