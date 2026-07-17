"""Deterministic expansion of registry metadata into an evaluation plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import resolve_evaluation_config
from .ids import (
    evaluation_result_path,
    generate_evaluation_id,
    generate_result_id,
    generate_run_id,
    validate_evaluation_id,
)
from .models import EvaluationIdentity, EvaluationJob, EvaluationPlan
from .operator_spec import load_operator_cases
from .registry import RegistrySnapshot, selected_registry_issues
from .selectors import (
    Selectors,
    select_candidates,
    select_cases,
    select_operators,
)
from .suite import SuiteConfig


class PlanningError(ValueError):
    pass


class PlanBuilder:
    def __init__(
        self,
        *,
        now: Callable[[], datetime] | None = None,
        random_hex: Callable[[int], str] | None = None,
    ) -> None:
        self._now = now
        self._random_hex = random_hex

    def build(
        self,
        snapshot: RegistrySnapshot,
        suite: SuiteConfig,
        selectors: Selectors,
        *,
        environment_fingerprint: str,
        output_root: Path,
        mode: str | None = None,
        seeds: tuple[int, ...] = (),
        evaluation_id: str | None = None,
        resume: bool = False,
        performance_timer: str | None = None,
        performance_warmup: int | None = None,
        performance_samples: int | None = None,
        performance_inner_iterations: int | None = None,
        performance_max_slowdown_pct: float | None = None,
        perf_on_correctness_fail: bool | None = None,
        performance_min_speedup: float | None = None,
        performance_max_candidate_median_ms: float | None = None,
        performance_max_cv: float | None = None,
        performance_max_memory_bytes: int | None = None,
        performance_unsupported_policy: str | None = None,
        gpu_lock_timeout_s: float | None = None,
    ) -> EvaluationPlan:
        operator_ids = select_operators(snapshot, suite, selectors)
        issues = selected_registry_issues(
            snapshot,
            operator_ids,
            candidate_patterns=(
                selectors.candidates if selectors.candidates else None
            ),
        )
        if issues:
            raise PlanningError(
                "registry is invalid for selected scope: "
                + "; ".join(str(issue) for issue in issues)
            )
        candidates = select_candidates(snapshot, operator_ids, selectors, suite)
        pairs = tuple(
            (operator_id, candidate)
            for operator_id in operator_ids
            for candidate in candidates[operator_id]
        )
        if evaluation_id is not None:
            validate_evaluation_id(evaluation_id)
            if len(pairs) != 1:
                raise PlanningError("--evaluation-id requires exactly one candidate")

        run_id = generate_run_id(self._random_hex)
        evaluations: dict[tuple[str, str], str] = {}
        for operator_id, candidate in pairs:
            evaluations[(operator_id, candidate.implementation_id)] = (
                evaluation_id
                or generate_evaluation_id(
                    environment_fingerprint,
                    run_id,
                    now=self._now,
                )
            )

        jobs: list[EvaluationJob] = []
        result_ids: set[str] = set()
        checked_paths: set[Path] = set()
        for operator_id, candidate in pairs:
            resolved = resolve_evaluation_config(
                snapshot.operator_manifests[operator_id],
                suite,
                mode=mode,
                seeds=seeds,
                performance_timer=performance_timer,
                performance_warmup=performance_warmup,
                performance_samples=performance_samples,
                performance_inner_iterations=performance_inner_iterations,
                performance_max_slowdown_pct=performance_max_slowdown_pct,
                perf_on_correctness_fail=perf_on_correctness_fail,
                performance_min_speedup=performance_min_speedup,
                performance_max_candidate_median_ms=performance_max_candidate_median_ms,
                performance_max_cv=performance_max_cv,
                performance_max_memory_bytes=performance_max_memory_bytes,
                performance_unsupported_policy=performance_unsupported_policy,
                gpu_lock_timeout_s=gpu_lock_timeout_s,
            )
            has_cuda = any(device.lower().startswith("cuda") for device in snapshot.operator_manifests[operator_id].device_types)
            if resolved.mode in {"all", "performance"} and not has_cuda and resolved.performance_timer != "wall_clock":
                raise PlanningError(
                    f"CPU-only operator {operator_id!r} requires wall_clock; "
                    f"{resolved.performance_timer!r} CUDA timing would be misleading"
                )
            selected_cases = select_cases(
                load_operator_cases(snapshot, operator_id), suite, selectors
            )
            current_evaluation_id = evaluations[
                (operator_id, candidate.implementation_id)
            ]
            output_dir = evaluation_result_path(
                output_root,
                operator_id,
                candidate.implementation_id,
                current_evaluation_id,
            )
            if output_dir not in checked_paths:
                if output_dir.exists() and not resume:
                    raise PlanningError(
                        f"evaluation path already exists (use --resume): {output_dir}"
                    )
                if output_dir.exists() and not output_dir.is_dir():
                    raise PlanningError(
                        f"evaluation path is not a directory: {output_dir}"
                    )
                checked_paths.add(output_dir)
            identity = EvaluationIdentity(
                run_id=run_id,
                evaluation_id=current_evaluation_id,
                operator_id=operator_id,
                candidate_id=candidate.implementation_id,
            )
            for case in selected_cases:
                for seed in resolved.correctness_seeds:
                    job_case = replace(case, seed=seed)
                    result_id = generate_result_id(
                        operator_id,
                        candidate.implementation_id,
                        current_evaluation_id,
                        case.case_id,
                        seed,
                    )
                    if result_id in result_ids:
                        raise PlanningError(f"duplicate result_id: {result_id}")
                    result_ids.add(result_id)
                    jobs.append(
                        EvaluationJob(
                            identity=identity,
                            reference=snapshot.references[operator_id],
                            candidate=candidate,
                            case=job_case,
                            mode=resolved.mode,
                            output_dir=output_dir,
                            result_id=result_id,
                            resolved_config=resolved,
                        )
                    )
        jobs.sort(
            key=lambda job: (
                job.identity.operator_id,
                job.identity.candidate_id,
                job.case.case_id,
                job.case.seed,
            )
        )
        if not jobs:
            raise PlanningError("selector matched no evaluation jobs")
        resolved_mode = jobs[0].mode
        return EvaluationPlan(
            run_id=run_id,
            mode=resolved_mode,
            jobs=tuple(jobs),
            environment_fingerprint=environment_fingerprint,
            suite_id=suite.suite_id,
            fingerprint_kind="provisional/dry-run",
        )


__all__ = ["PlanBuilder", "PlanningError"]
