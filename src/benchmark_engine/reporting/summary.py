"""Human-readable summaries derived only from durable artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from .csv_writer import AtomicCsvTable, RESULTS_SCHEMA


def render_summary(manifest: Mapping[str, object]) -> str:
    """Render the small stable summary available before evaluators exist."""

    identity = manifest["identity"]
    if not isinstance(identity, Mapping):
        raise TypeError("manifest identity must be an object")
    environment = manifest["environment"]
    if not isinstance(environment, Mapping):
        raise TypeError("manifest environment must be an object")
    reason = manifest.get("terminal_reason")
    lines = [
        "# Benchmark evaluation",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Run: `{identity['run_id']}`",
        f"- Operator: `{identity['operator_id']}`",
        f"- Candidate: `{identity['candidate_id']}`",
        f"- Evaluation: `{identity['evaluation_id']}`",
        f"- Suite: `{manifest['suite_id']}`",
        f"- Mode: `{manifest['mode']}`",
        f"- Environment: `{environment['fingerprint']}`",
    ]
    if reason:
        lines.append(f"- Terminal reason: {reason}")
    lines.extend(
        (
            "",
            "This file is generated from `evaluation_manifest.json`; do not edit it.",
            "",
        )
    )
    return "\n".join(lines)


def summarize_evaluation(path: Path) -> str:
    """Render a correctness summary without importing or executing code."""

    target = Path(path)
    if target.is_dir():
        target = target / RESULTS_SCHEMA.filename
    rows = AtomicCsvTable(target, RESULTS_SCHEMA).read_rows()
    if not rows:
        raise ValueError(f"results table contains no rows: {target}")
    counts: dict[str, int] = {}
    for row in rows:
        status = row["correctness_status"]
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# Correctness summary",
        "",
        f"- Total: {len(rows)}",
        f"- Passed: {counts.get('passed', 0)}",
        f"- Failed: {len(rows) - counts.get('passed', 0)}",
        f"- Performance: skipped (`performance_not_implemented`)",
        "",
        "## Status counts",
        "",
    ]
    lines.extend(f"- {name}: {counts[name]}" for name in sorted(counts))
    failures = [row for row in rows if row["correctness_status"] != "passed"]
    if failures:
        lines.extend(("", "## Failures", ""))
        for row in failures:
            command = (
                "./bench.sh run --mode correctness "
                f"--operator {row['operator_id']} --candidate {row['candidate_id']} "
                f"--case {row['case_id']} --seed {row['seed']}"
            )
            lines.extend(
                (
                    f"### {row['case_id']} (seed {row['seed']})",
                    "",
                    f"- Status: `{row['correctness_status']}`",
                    f"- Diagnostic: `{row['diagnostic_path'] or 'not available'}`",
                    f"- Error: {row['error_message'] or 'comparison mismatch'}",
                    f"- Reproduce: `{command}`",
                    "",
                )
            )
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_summary", "summarize_evaluation"]
