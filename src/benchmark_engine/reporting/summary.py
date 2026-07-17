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
    performance_counts: dict[str, int] = {}
    gate_counts: dict[str, int] = {}
    for row in rows:
        status = row["correctness_status"]
        counts[status] = counts.get(status, 0) + 1
        performance = row["performance_status"]
        performance_counts[performance] = performance_counts.get(performance, 0) + 1
        gate = row.get("performance_gate_status") or "legacy/unavailable"
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    lines = [
        "# Correctness summary",
        "",
        f"- Total: {len(rows)}",
        f"- Passed: {counts.get('passed', 0)}",
        f"- Failed: {len(rows) - counts.get('passed', 0)}",
        f"- Performance measured: {sum(value for key, value in performance_counts.items() if key not in {'skipped', 'planned'})}",
        f"- Ranking eligible: {sum(row.get('ranking_eligible') == 'true' for row in rows)}",
        "",
        "## Status counts",
        "",
    ]
    if set(performance_counts) == {"skipped"}:
        lines.insert(7, "- Legacy performance note: `performance_not_implemented` or correctness-only")
    lines.extend(f"- {name}: {counts[name]}" for name in sorted(counts))
    lines.extend(("", "## Performance status counts", ""))
    lines.extend(f"- {name}: {performance_counts[name]}" for name in sorted(performance_counts))
    lines.extend(("", "## Performance gate counts", ""))
    lines.extend(f"- {name}: {gate_counts[name]}" for name in sorted(gate_counts))
    lines.extend(("", "## Model projection", "", "- Status: not available for this operator/suite", ""))
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
