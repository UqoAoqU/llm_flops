#!/usr/bin/env python3
"""Run and compare legacy graph_ms with engine CUDA Graph medians on one idle B200."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import statistics
from pathlib import Path

from benchmark_engine.projection import DEEPSEEK_V4_PROJECTION
from benchmark_engine.reporting import (
    AtomicCsvTable,
    MODEL_PROJECTION_SCHEMA,
    PERFORMANCE_SAMPLES_SCHEMA,
    RESULTS_SCHEMA,
    atomic_write_text,
)


ROOT = Path(__file__).resolve().parents[1]
RELATIVE_THRESHOLD = 0.10
ABSOLUTE_THRESHOLD_MS = 0.02
DEFAULT_INPUTS = {"prefill": (1024, 2048, 4096), "decode": (16, 32)}


class RegressionError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the predeclared DeepSeek V4 legacy/new B200 latency regression."
    )
    parser.add_argument("--phase", required=True, choices=("prefill", "decode"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="reuse legacy/ and engine/ artifacts already present in work-dir",
    )
    return parser


def _commands(phase: str, work: Path, warmup: int, runs: int, context: int) -> tuple[list[str], list[str]]:
    inputs = ",".join(str(value) for value in DEFAULT_INPUTS[phase])
    legacy_base = work / "legacy" / f"deepseek_v4_{phase}.csv"
    legacy = [
        str(ROOT / "run.sh"), phase, "--quant-profile", "fp8_mxfp8",
        "--m", inputs, "--context", str(context), "--warmup", str(warmup),
        "--runs", str(runs), "--csv", str(legacy_base),
    ]
    engine = [
        str(ROOT / "bench.sh"), "run", "--suite", f"deepseek_v4_{phase}",
        "--timer", "cuda_graph", "--warmup", str(warmup), "--samples", str(runs),
        "--inner-iterations", "1", "--output-root", str(work / "engine"),
    ]
    return legacy, engine


def _gpu0_identity() -> dict[str, str]:
    command = [
        "nvidia-smi", "--id=0", "--query-gpu=uuid,name", "--format=csv,noheader"
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RegressionError(f"cannot resolve physical GPU 0: {result.stderr.strip()}")
    parts = [part.strip() for part in result.stdout.strip().split(",", 1)]
    if len(parts) != 2 or not parts[0].startswith("GPU-"):
        raise RegressionError(f"unexpected physical GPU 0 identity: {result.stdout.strip()}")
    return {"physical_index": "0", "uuid": parts[0], "name": parts[1]}


def _gpu_idle_evidence(gpu_uuid: str, checkpoint: str) -> dict[str, object]:
    command = [
        "nvidia-smi", f"--id={gpu_uuid}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RegressionError(f"cannot prove GPU idle: {result.stderr.strip()}")
    evidence = result.stdout.strip()
    if evidence:
        raise RegressionError(f"GPU has active compute processes: {evidence}")
    return {
        "checkpoint": checkpoint,
        "gpu_uuid": gpu_uuid,
        "query": command,
        "compute_processes": [],
    }


def _legacy_paths(work: Path, phase: str) -> tuple[Path, ...]:
    base = work / "legacy" / f"deepseek_v4_{phase}"
    return tuple(base.with_name(f"{base.name}_m{value}.csv") for value in DEFAULT_INPUTS[phase])


def _read_legacy(work: Path, phase: str) -> dict[tuple[int, str], float]:
    rows: dict[tuple[int, str], float] = {}
    mapped = {
        mapping.display_name
        for mapping in DEEPSEEK_V4_PROJECTION.mappings(phase, "fp8_mxfp8")
        if mapping.operator_id is not None
    }
    for path in _legacy_paths(work, phase):
        if not path.is_file():
            raise RegressionError(f"missing legacy regression CSV: {path}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["operator"] not in mapped or row["status"] != "executed":
                    continue
                key = (int(row["m"]), row["operator"])
                if key in rows:
                    raise RegressionError(f"duplicate legacy regression row: {key}")
                try:
                    timing = float(row["call_ms"])
                except ValueError as error:
                    raise RegressionError(f"invalid legacy timing for {key}") from error
                if not math.isfinite(timing) or timing < 0:
                    raise RegressionError(f"legacy timing must be finite/non-negative: {key}")
                rows[key] = timing
    if not rows:
        raise RegressionError("legacy regression produced no comparable rows")
    return rows


def _read_engine(work: Path, phase: str, runs: int) -> dict[tuple[int, str], float]:
    rows: dict[tuple[int, str], float] = {}
    candidates_by_operator: dict[str, str] = {}
    for path in sorted((work / "engine").glob("*/*/*/model_projection.csv")):
        evaluation = path.parent
        result_rows = {
            row["result_id"]: row
            for row in AtomicCsvTable(
                evaluation / RESULTS_SCHEMA.filename, RESULTS_SCHEMA
            ).read_rows()
        }
        sample_rows = AtomicCsvTable(
            evaluation / PERFORMANCE_SAMPLES_SCHEMA.filename,
            PERFORMANCE_SAMPLES_SCHEMA,
        ).read_rows()
        projection_rows = AtomicCsvTable(
            path, MODEL_PROJECTION_SCHEMA
        ).read_rows()
        for row in projection_rows:
            if row["phase"] != phase or row["implementation_role"] != "candidate":
                continue
            if row["status"] != "measured" or not row["per_call_ms"]:
                continue
            result = result_rows.get(row["result_id"])
            if result is None:
                raise RegressionError(f"projection has no result row: {row['result_id']}")
            required = {
                "imported_legacy": "false",
                "correctness_status": "passed",
                "performance_status": "passed",
                "performance_formal": "true",
                "ranking_eligible": "true",
                "performance_gate_status": "passed",
                "requested_timer": "cuda_graph",
                "effective_timer": "cuda_graph",
                "reference_requested_timer": "cuda_graph",
                "reference_effective_timer": "cuda_graph",
                "other_compute_processes_detected": "false",
                "telemetry_error": "",
            }
            mismatches = [
                f"{name}={result.get(name)!r}"
                for name, expected in required.items()
                if result.get(name) != expected
            ]
            if mismatches:
                raise RegressionError(
                    f"engine row is not a formal cuda_graph result {row['result_id']}: "
                    + ", ".join(mismatches)
                )
            if not result.get("candidate_source_hash") or not result.get("reference_source_hash"):
                raise RegressionError("engine regression row is missing source identity")
            candidate = row["candidate_id"]
            operator = row["operator_id"]
            previous = candidates_by_operator.setdefault(operator, candidate)
            if previous != candidate:
                raise RegressionError(
                    f"multiple candidate identities selected for operator {operator}"
                )
            result_samples = [
                sample for sample in sample_rows
                if sample["result_id"] == row["result_id"]
            ]
            if len(result_samples) != 2 * runs or sorted(
                int(sample["order_index"]) for sample in result_samples
            ) != list(range(2 * runs)):
                raise RegressionError(
                    f"engine row does not have a complete interleaved sample order: {row['result_id']}"
                )
            raw_values: list[float] = []
            for role in ("reference", "candidate"):
                selected_samples = [
                    sample for sample in result_samples
                    if sample["implementation_role"] == role
                ]
                if len(selected_samples) != runs or sorted(
                    int(sample["sample_index"]) for sample in selected_samples
                ) != list(range(runs)):
                    raise RegressionError(
                        f"engine row does not have exactly {runs} {role} samples: {row['result_id']}"
                    )
                for sample in selected_samples:
                    if sample["effective_timer"] != "cuda_graph" \
                            or sample["requested_timer"] != "cuda_graph":
                        raise RegressionError("engine raw sample timer is not cuda_graph")
                    if not all(
                        sample.get(field)
                        for field in (
                            "reference_dtype", "candidate_dtype",
                            "reference_shape", "candidate_shape",
                        )
                    ):
                        raise RegressionError("engine raw sample is missing dtype/shape contract")
                    value = float(sample["per_call_ms"])
                    if not math.isfinite(value) or value < 0:
                        raise RegressionError("engine raw timing must be finite/non-negative")
                    if role == "candidate":
                        raw_values.append(value)
            projection_value = float(row["per_call_ms"])
            result_median = float(result["candidate_median_ms"])
            recomputed = statistics.median(raw_values)
            if not all(math.isfinite(value) and value >= 0 for value in (
                projection_value, result_median, recomputed
            )) or not math.isclose(projection_value, result_median, rel_tol=1e-9, abs_tol=1e-12) \
                    or not math.isclose(result_median, recomputed, rel_tol=1e-9, abs_tol=1e-12):
                raise RegressionError("engine projection/summary/raw median disagree")
            key = (int(row["model_input"]), row["display_name"])
            if key in rows:
                raise RegressionError(f"ambiguous engine regression row: {key}")
            rows[key] = projection_value
    if not rows:
        raise RegressionError("engine regression produced no comparable rows")
    return rows


def _compare(legacy: dict[tuple[int, str], float], engine: dict[tuple[int, str], float]) -> list[dict[str, object]]:
    if not legacy or not engine:
        raise RegressionError("legacy/new regression inputs must both be non-empty")
    if set(legacy) != set(engine):
        missing = sorted(set(legacy).difference(engine))
        extra = sorted(set(engine).difference(legacy))
        raise RegressionError(f"legacy/new case identities differ; missing={missing}, extra={extra}")
    report: list[dict[str, object]] = []
    for model_input, operator in sorted(legacy):
        old = legacy[(model_input, operator)]
        new = engine[(model_input, operator)]
        absolute = abs(new - old)
        relative = absolute / old if old > 0 else math.inf
        passed = absolute <= ABSOLUTE_THRESHOLD_MS or relative <= RELATIVE_THRESHOLD
        report.append(
            {
                "model_input": model_input,
                "operator": operator,
                "legacy_graph_ms": old,
                "engine_median_ms": new,
                "absolute_delta_ms": absolute,
                "relative_delta": relative,
                "passed": passed,
            }
        )
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.warmup <= 0 or arguments.runs < 5 or arguments.context != 65536:
        print("regression error: warmup must be positive, runs >= 5, and context must be 65536", file=sys.stderr)
        return 2
    work = arguments.work_dir.resolve()
    legacy_command, engine_command = _commands(
        arguments.phase, work, arguments.warmup, arguments.runs, arguments.context
    )
    plan = {
        "schema_version": 1,
        "phase": arguments.phase,
        "inputs": list(DEFAULT_INPUTS[arguments.phase]),
        "context": arguments.context,
        "warmup": arguments.warmup,
        "runs": arguments.runs,
        "timer_pair": {"legacy": "graph_ms", "engine": "cuda_graph median"},
        "threshold": {
            "relative": RELATIVE_THRESHOLD,
            "absolute_ms": ABSOLUTE_THRESHOLD_MS,
            "rule": "absolute <= absolute_ms OR relative <= relative",
        },
        "legacy_command": legacy_command,
        "engine_command": engine_command,
    }
    if arguments.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        evidence: list[dict[str, object]] = []
        gpu_identity: dict[str, str] | None = None
        if not arguments.compare_only:
            if work.exists() and any(work.iterdir()):
                raise RegressionError(f"work-dir must be empty for execution: {work}")
            (work / "legacy").mkdir(parents=True, exist_ok=True)
            (work / "engine").mkdir(parents=True, exist_ok=True)
            gpu_identity = _gpu0_identity()
            evidence.append(_gpu_idle_evidence(gpu_identity["uuid"], "before_legacy"))
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            subprocess.run(legacy_command, cwd=ROOT, env=environment, check=True)
            # Re-check between frameworks so overlap cannot become a valid comparison.
            evidence.append(_gpu_idle_evidence(gpu_identity["uuid"], "between_frameworks"))
            subprocess.run(engine_command, cwd=ROOT, env=environment, check=True)
            evidence.append(_gpu_idle_evidence(gpu_identity["uuid"], "after_engine"))
        comparisons = _compare(
            _read_legacy(work, arguments.phase),
            _read_engine(work, arguments.phase, arguments.runs),
        )
        output = dict(plan)
        output["gpu"] = gpu_identity
        output["gpu_idle_evidence"] = evidence
        output["compare_only"] = arguments.compare_only
        output["comparisons"] = comparisons
        output["comparison_passed"] = all(row["passed"] for row in comparisons)
        output["evidence_complete"] = not arguments.compare_only and len(evidence) == 3
        output["passed"] = output["comparison_passed"] and output["evidence_complete"]
        atomic_write_text(
            work / "regression_report.json",
            json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0 if output["passed"] else 1
    except (OSError, ValueError, RegressionError, subprocess.CalledProcessError) as error:
        print(f"regression error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
