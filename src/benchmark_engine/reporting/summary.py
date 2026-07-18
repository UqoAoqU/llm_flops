"""Human-readable summaries derived only from durable artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from benchmark_engine.ids import validate_run_id
from benchmark_engine.projection import projection_for_id

from .csv_writer import AtomicCsvTable, MODEL_PROJECTION_SCHEMA, RESULTS_SCHEMA


def _projection_lines(rows: list[dict[str, str]], *, complete_model: bool) -> list[str]:
    if not rows:
        return ["## Model projection", "", "- Status: unavailable (no projection artifact)", ""]
    lines = []
    projection_ids = sorted({row.get("projection_id") or "deepseek_v4_pro" for row in rows})
    for projection_id in projection_ids:
        projection = projection_for_id(projection_id)
        projection_rows = [row for row in rows
                           if (row.get("projection_id") or "deepseek_v4_pro") == projection_id]
        if projection is None:
            lines.extend((f"## {projection_id} model projection", "",
                          f"- Status: unavailable (unknown projection_id `{projection_id}`)", ""))
            continue
        display_name = getattr(projection, "display_name", projection_id)
        lines.extend((f"## {display_name} model projection", ""))
        groups = sorted({(row["phase"], row["quant_profile"], int(row["model_input"]),
                          int(row["raw_context"]), row.get("_seed", "unknown"),
                          row.get("evaluation_id", "unknown"))
                         for row in projection_rows})
        for phase, profile, model_input, context, seed, evaluation_id in groups:
            selected = [row for row in projection_rows
                        if (row["phase"], row["quant_profile"], int(row["model_input"]),
                            int(row["raw_context"]), row.get("_seed", "unknown"),
                            row.get("evaluation_id", "unknown"))
                        == (phase, profile, model_input, context, seed, evaluation_id)]
            expected = projection.mappings(phase, profile)
            candidates = {mapping.adapter_id: sorted(
                (row for row in selected if row["implementation_role"] == "candidate"
                 and row["adapter_id"] == mapping.adapter_id),
                key=lambda row: (row["candidate_id"], row.get("result_id", "")),
            ) for mapping in expected}
            references = {mapping.adapter_id: sorted(
                (row for row in selected if row["implementation_role"] == "reference"
                 and row["adapter_id"] == mapping.adapter_id),
                key=lambda row: (row["candidate_id"], row.get("result_id", "")),
            ) for mapping in expected}
            lines.extend((f"### {phase} / {profile} / input={model_input} / context={context} / seed={seed} / evaluation={evaluation_id}", "",
                          "| Operator | Candidate | Backend | Instances | Candidate ms/call | Candidate model-ms | Status |",
                          "|---|---|---|---:|---:|---:|---|"))
            candidate_total = 0.0
            reference_total = 0.0
            missing = []
            unavailable = []
            ambiguous = []
            for mapping in expected:
                candidate_rows = candidates[mapping.adapter_id]
                reference_rows = references[mapping.adapter_id]
                candidate_ids = sorted({row["candidate_id"] for row in candidate_rows})
                candidate_label = ", ".join(candidate_ids) if candidate_ids else "-"
                if not candidate_rows:
                    status = "missing" if complete_model else "not in this evaluation"
                    call_ms = model_ms = "-"
                    missing.append(mapping.display_name)
                elif len(candidate_rows) != 1:
                    status = "ambiguous"
                    call_ms = model_ms = "-"
                    ambiguous.append(f"{mapping.display_name} ({candidate_label})")
                else:
                    row = candidate_rows[0]
                    status = row["status"] + ((f": {row['reason']}") if row.get("reason") else "")
                    call_ms = row.get("per_call_ms") or "-"
                    model_ms = row.get("projected_model_ms") or "-"
                    if row["status"] == "measured" and row.get("projected_model_ms"):
                        candidate_total += float(row["projected_model_ms"])
                    else:
                        unavailable.append(f"{mapping.display_name} ({row['status']})")
                    # Reference rows are paired with candidate evaluations. Count
                    # only the unique pair; ambiguous candidates must not select
                    # or duplicate a reference measurement.
                    if len(reference_rows) == 1 and reference_rows[0]["status"] == "measured" \
                            and reference_rows[0].get("projected_model_ms"):
                        reference_total += float(reference_rows[0]["projected_model_ms"])
                lines.append(f"| {mapping.display_name} | {candidate_label} | {mapping.backend} | {mapping.instances} | {call_ms} | {model_ms} | {status} |")
            lines.extend(("", f"- Candidate measured partial total: {candidate_total:.6f} ms/model",
                          f"- Reference measured partial total: {reference_total:.6f} ms/model",
                          f"- Missing operators: {', '.join(missing) if missing else 'none'}",
                          f"- Unavailable/unsupported: {', '.join(unavailable) if unavailable else 'none'}",
                          f"- Ambiguous candidates: {', '.join(ambiguous) if ambiguous else 'none'}", ""))
    return lines


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
    legacy_root = target if target.is_dir() else target.parent
    legacy_import = (legacy_root / "legacy_import_manifest.json").is_file()
    if target.is_dir():
        target = target / RESULTS_SCHEMA.filename
    rows = AtomicCsvTable(target, RESULTS_SCHEMA).read_rows()
    if not rows:
        raise ValueError(f"results table contains no rows: {target}")
    import_markers = {row.get("imported_legacy") == "true" for row in rows}
    if len(import_markers) != 1:
        raise ValueError(f"results table mixes normal and legacy rows: {target}")
    legacy_import = legacy_import or True in import_markers
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
        "# Imported legacy summary" if legacy_import else "# Correctness summary",
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
    if legacy_import:
        lines[2:2] = [
            "- Imported legacy: true",
            "- Correctness/source hashes/raw samples/CV/speedup: unavailable",
            "- Ranking/resume/run-level discovery: disabled",
        ]
    if set(performance_counts) == {"skipped"}:
        lines.insert(7, "- Legacy performance note: `performance_not_implemented` or correctness-only")
    lines.extend(f"- {name}: {counts[name]}" for name in sorted(counts))
    lines.extend(("", "## Performance status counts", ""))
    lines.extend(f"- {name}: {performance_counts[name]}" for name in sorted(performance_counts))
    lines.extend(("", "## Performance gate counts", ""))
    lines.extend(f"- {name}: {gate_counts[name]}" for name in sorted(gate_counts))
    projection_rows = AtomicCsvTable(
        target.parent / MODEL_PROJECTION_SCHEMA.filename, MODEL_PROJECTION_SCHEMA
    ).read_rows()
    seeds = {row["result_id"]: row["seed"] for row in rows}
    projection_rows = [dict(row, _seed=seeds.get(row["result_id"], "unknown"))
                       for row in projection_rows]
    lines.extend(("", *_projection_lines(projection_rows, complete_model=False)))
    failures = [
        row for row in rows
        if row["correctness_status"] != "passed"
        and row.get("imported_legacy") != "true"
    ]
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


def summarize_run(output_root: Path, run_id: str) -> str:
    """Aggregate every mirrored evaluation registered for one run."""

    from .artifact_writer import RUN_INDEX_SCHEMA

    validate_run_id(run_id)
    root = Path(output_root).resolve()
    index_rows = AtomicCsvTable(root / RUN_INDEX_SCHEMA.filename, RUN_INDEX_SCHEMA).read_rows()
    selected = [row for row in index_rows if row["run_id"] == run_id]
    if not selected:
        raise ValueError(f"run_id not found in run_index.csv: {run_id}")
    projection_rows: list[dict[str, str]] = []
    result_count = 0
    for index_row in sorted(selected, key=lambda row: (row["operator_id"], row["candidate_id"], row["evaluation_id"])):
        directory = (root / index_row["relative_path"]).resolve()
        if root not in directory.parents:
            raise ValueError("run index contains a path outside output root")
        result_rows = AtomicCsvTable(directory / RESULTS_SCHEMA.filename, RESULTS_SCHEMA).read_rows()
        result_count += len(result_rows)
        seeds = {row["result_id"]: row["seed"] for row in result_rows}
        projection_rows.extend(
            dict(row, _seed=seeds.get(row["result_id"], "unknown"))
            for row in AtomicCsvTable(directory / MODEL_PROJECTION_SCHEMA.filename,
                                      MODEL_PROJECTION_SCHEMA).read_rows()
        )
    lines = ["# Benchmark run summary", "", f"- Run: `{run_id}`",
             f"- Evaluations: {len(selected)}", f"- Results: {result_count}", ""]
    lines.extend(_projection_lines(projection_rows, complete_model=True))
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_summary", "summarize_evaluation", "summarize_run"]
