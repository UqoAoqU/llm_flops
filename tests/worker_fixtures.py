from __future__ import annotations

from pathlib import Path

from benchmark_engine.config import ResolvedEvaluationConfig
from benchmark_engine.models import (
    CaseSpec,
    EvaluationIdentity,
    EvaluationJob,
    ImplementationSpec,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "workers"


def make_job(output_dir: Path, candidate_fixture: str = "normal") -> EvaluationJob:
    identity = EvaluationIdentity(
        run_id="run_0123456789abcdef",
        evaluation_id="20260716T120000Z__abcdef123456__0123456789ab",
        operator_id="test_operator",
        candidate_id="task__20260716T120000Z__abcdef12",
    )
    return EvaluationJob(
        identity=identity,
        reference=ImplementationSpec(
            operator_id=identity.operator_id,
            implementation_id="reference",
            role="reference",
            root=FIXTURE_ROOT / "reference",
            entrypoint="implementation:operator",
            source_hash="a" * 64,
            manifest_version=1,
        ),
        candidate=ImplementationSpec(
            operator_id=identity.operator_id,
            implementation_id=identity.candidate_id,
            role="candidate",
            root=FIXTURE_ROOT / candidate_fixture,
            entrypoint="implementation:operator",
            source_hash="b" * 64,
            manifest_version=1,
        ),
        case=CaseSpec(
            case_id="small",
            symbols={"size": 2},
            seed=0,
            tags=frozenset({"smoke"}),
        ),
        mode="all",
        output_dir=Path(output_dir),
        result_id="res_0123456789abcdef0123456789abcdef",
        resolved_config=ResolvedEvaluationConfig(),
    )


__all__ = ["FIXTURE_ROOT", "make_job"]
