"""Small schema-valid Phase 4 reporting fixtures."""

from __future__ import annotations

from datetime import datetime, timezone

from benchmark_engine.config import ResolvedEvaluationConfig
from benchmark_engine.models import EvaluationIdentity
from benchmark_engine.reporting import EvaluationManifest, RESULTS_SCHEMA


RUN_ID = "run_0123456789abcdef0123456789abcdef"
CANDIDATE_ID = "task_a__20260716T120000Z__deadbeef"
ENVIRONMENT = "a" * 64


def identity(
    *,
    operator_id: str = "test_operator",
    candidate_id: str = CANDIDATE_ID,
    evaluation_id: str = "20260716T121000Z__aaaaaaaaaaaa__0123456789ab",
) -> EvaluationIdentity:
    return EvaluationIdentity(
        run_id=RUN_ID,
        evaluation_id=evaluation_id,
        operator_id=operator_id,
        candidate_id=candidate_id,
    )


def manifest(**overrides: object) -> EvaluationManifest:
    values: dict[str, object] = {
        "identity": identity(),
        "original_command": ("bench", "run", "--suite", "smoke"),
        "resolved_config": ResolvedEvaluationConfig(mode="correctness"),
        "reference_source_hash": "b" * 64,
        "candidate_source_hash": "c" * 64,
        "environment_snapshot": {"python": "3.12", "gpu": None},
        "environment_fingerprint": ENVIRONMENT,
        "suite_id": "smoke",
        "now": lambda: datetime(2026, 7, 16, 12, 10, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return EvaluationManifest.create(**values)


def results_row(result_id: str = "res_0123456789abcdef0123456789abcdef") -> dict[str, object]:
    item = identity()
    return {
        "schema_version": RESULTS_SCHEMA.version,
        "run_id": item.run_id,
        "evaluation_id": item.evaluation_id,
        "timestamp_utc": "2026-07-16T12:11:00Z",
        "suite_id": "smoke",
        "mode": "correctness",
        "result_id": result_id,
        "operator_id": item.operator_id,
        "contract_version": 1,
        "candidate_id": item.candidate_id,
        "reference_id": "reference",
        "candidate_source_hash": "c" * 64,
        "reference_source_hash": "b" * 64,
        "environment_fingerprint": ENVIRONMENT,
        "case_id": "case_0",
        "case_hash": "d" * 64,
        "seed": 0,
        "status": "passed",
        "correctness_status": "passed",
        "performance_status": "skipped",
    }
