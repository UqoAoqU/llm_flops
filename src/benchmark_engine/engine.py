"""Discovery-to-artifact orchestration for the correctness MVP."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .environment import collect_environment, environment_fingerprint, load_lock, planning_fingerprint
from .execution import StageTimeouts, WorkerController, WorkerOutcome, WorkerStage
from .models import CorrectnessStatus, EvaluationJob, EvaluationPlan, PerformanceStatus, ResultStatus
from .operator_spec import load_operator_cases
from .planning import PlanBuilder
from .registry import FilesystemRegistry, RegistrySnapshot
from .reporting import (
    ArtifactWriter,
    EvaluationManifest,
    EvaluationState,
    ResumeMismatchError,
    ResumeReader,
)
from .reporting.csv_writer import (
    AtomicCsvTable,
    CORRECTNESS_OUTPUTS_SCHEMA,
    RESULTS_SCHEMA,
    atomic_write_text,
)
from .selectors import Selectors, select_cases
from .suite import load_suite
from .ids import generate_result_id


PERFORMANCE_SKIP_REASON = "performance_not_implemented"


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    evaluation_paths: tuple[Path, ...]
    passed: int
    failed: int
    infrastructure_failures: int
    interrupted: bool = False

    @property
    def exit_code(self) -> int:
        if self.interrupted:
            return 130
        if self.infrastructure_failures:
            return 3
        return 1 if self.failed else 0


def _suite(repository_root: Path, suite_id: str):
    if re.fullmatch(r"[a-z][a-z0-9_-]{1,79}", suite_id) is None:
        raise ValueError("suite ID is invalid")
    suite = load_suite(repository_root / "suites" / f"{suite_id}.yaml")
    if suite.suite_id != suite_id:
        raise ValueError(
            f"suite_id {suite.suite_id!r} does not match requested {suite_id!r}"
        )
    return suite


def build_dry_run_plan(
    repository_root: Path,
    suite_id: str,
    selectors: Selectors,
    *,
    output_root: Path,
    mode: str | None = None,
    seeds: tuple[int, ...] = (),
    evaluation_id: str | None = None,
    resume: bool = False,
    plan_builder: PlanBuilder | None = None,
):
    """Build an import-free-of-candidates, artifact-free plan."""

    root = Path(repository_root).resolve()
    suite = _suite(root, suite_id)
    snapshot = FilesystemRegistry(root).discover()
    lock = load_lock(root / "requirements" / "benchmark-lock.json")
    fingerprint = planning_fingerprint(lock)
    return (plan_builder or PlanBuilder()).build(
        snapshot,
        suite,
        selectors,
        environment_fingerprint=fingerprint,
        output_root=output_root,
        mode=mode,
        seeds=seeds,
        evaluation_id=evaluation_id,
        resume=resume,
    )


def _runtime_environment(root: Path) -> tuple[dict[str, object], str]:
    lock = load_lock(root / "requirements" / "benchmark-lock.json")
    # Phase 7 executes correctness only.  CPU operators must not initialise
    # CUDA merely to construct an identity; CUDA collection begins with the
    # performance timer phase.
    observed = collect_environment(lock, include_cuda=False)
    return observed, environment_fingerprint(observed)


def build_execution_plan(
    repository_root: Path,
    suite_id: str,
    selectors: Selectors,
    *,
    output_root: Path,
    mode: str | None = None,
    seeds: tuple[int, ...] = (),
    evaluation_id: str | None = None,
) -> tuple[EvaluationPlan, RegistrySnapshot, Mapping[str, object]]:
    root = Path(repository_root).resolve()
    suite = _suite(root, suite_id)
    snapshot = FilesystemRegistry(root).discover()
    environment, fingerprint = _runtime_environment(root)
    plan = PlanBuilder().build(
        snapshot,
        suite,
        selectors,
        environment_fingerprint=fingerprint,
        output_root=output_root,
        mode=mode,
        seeds=seeds,
        evaluation_id=evaluation_id,
    )
    return replace(plan, fingerprint_kind="runtime"), snapshot, environment


def _candidate(snapshot: RegistrySnapshot, operator_id: str, candidate_id: str):
    for candidate in snapshot.candidates.get(operator_id, ()):
        if candidate.implementation_id == candidate_id:
            return candidate
    raise ResumeMismatchError(
        f"resume candidate is not present in registry: {operator_id}/{candidate_id}"
    )


def _resume_selectors(command: Sequence[str]) -> Selectors:
    """Recover case/tag narrowing from the immutable original command."""

    cases: list[str] = []
    tags: list[str] = []
    tokens = tuple(command[1:] if command and command[0] == "bench" else command)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        destination = None
        if token in {"--case", "--tag"} and index + 1 < len(tokens):
            destination = cases if token == "--case" else tags
            destination.append(tokens[index + 1])
            index += 2
            continue
        for option, destination in (("--case=", cases), ("--tag=", tags)):
            if token.startswith(option):
                destination.append(token[len(option) :])
                break
        index += 1
    return Selectors(cases=tuple(cases), tags=tuple(tags))


def build_resume_plan(
    repository_root: Path, run_id: str, *, output_root: Path
) -> tuple[EvaluationPlan, RegistrySnapshot, Mapping[str, object]]:
    root = Path(repository_root).resolve()
    snapshot = FilesystemRegistry(root).discover()
    environment, fingerprint = _runtime_environment(root)
    states = ResumeReader(output_root).for_run(run_id)
    jobs: list[EvaluationJob] = []
    suite_ids = {state.manifest.suite_id for state in states}
    modes = {state.manifest.mode for state in states}
    if len(suite_ids) != 1 or len(modes) != 1:
        raise ResumeMismatchError("run contains incompatible suite or mode values")
    suite_id = next(iter(suite_ids))
    suite = _suite(root, suite_id)
    for state in states:
        manifest = state.manifest
        identity = manifest.identity
        reference = snapshot.references.get(identity.operator_id)
        if reference is None:
            raise ResumeMismatchError(
                f"resume reference is not present: {identity.operator_id}"
            )
        candidate = _candidate(snapshot, identity.operator_id, identity.candidate_id)
        mismatches = []
        if reference.source_hash != manifest.reference_source_hash:
            mismatches.append("reference source")
        if candidate.source_hash != manifest.candidate_source_hash:
            mismatches.append("candidate source")
        if fingerprint != manifest.environment_fingerprint:
            mismatches.append("environment fingerprint")
        if mismatches:
            raise ResumeMismatchError("resume is incompatible: " + ", ".join(mismatches))
        cases = select_cases(
            load_operator_cases(snapshot, identity.operator_id),
            suite,
            _resume_selectors(manifest.original_command),
        )
        for case in cases:
            for seed in manifest.resolved_config.correctness_seeds:
                job_case = replace(case, seed=seed)
                jobs.append(
                    EvaluationJob(
                        identity=identity,
                        reference=reference,
                        candidate=candidate,
                        case=job_case,
                        mode=manifest.mode,
                        output_dir=Path(output_root).resolve()
                        / identity.operator_id
                        / identity.candidate_id
                        / identity.evaluation_id,
                        result_id=generate_result_id(
                            identity.operator_id,
                            identity.candidate_id,
                            identity.evaluation_id,
                            case.case_id,
                            seed,
                        ),
                        resolved_config=manifest.resolved_config,
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
        raise ResumeMismatchError("resume run has no expected jobs")
    return (
        EvaluationPlan(
            run_id=run_id,
            mode=next(iter(modes)),
            jobs=tuple(jobs),
            environment_fingerprint=fingerprint,
            suite_id=suite_id,
            fingerprint_kind="runtime",
        ),
        snapshot,
        environment,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _metric(metrics: Mapping[str, object], name: str) -> object | None:
    direct = metrics.get(name)
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return direct
    values = [
        item[name]
        for item in metrics.values()
        if isinstance(item, Mapping)
        and isinstance(item.get(name), (int, float))
        and not isinstance(item.get(name), bool)
    ]
    return max(values) if values else None


def _correctness_status(raw: str) -> CorrectnessStatus:
    return {
        "pass": CorrectnessStatus.PASSED,
        "fail": CorrectnessStatus.FAILED,
        "nondeterministic": CorrectnessStatus.FAILED,
        "error": CorrectnessStatus.ERROR,
        "timeout": CorrectnessStatus.TIMEOUT,
        "oom": CorrectnessStatus.OOM,
        "unsupported": CorrectnessStatus.UNSUPPORTED,
        "crashed": CorrectnessStatus.CRASHED,
    }[raw]


def _result_status(status: CorrectnessStatus) -> ResultStatus:
    return {
        CorrectnessStatus.PASSED: ResultStatus.PASSED,
        CorrectnessStatus.FAILED: ResultStatus.FAILED,
        CorrectnessStatus.ERROR: ResultStatus.ERROR,
        CorrectnessStatus.TIMEOUT: ResultStatus.TIMEOUT,
        CorrectnessStatus.OOM: ResultStatus.OOM,
        CorrectnessStatus.UNSUPPORTED: ResultStatus.UNSUPPORTED,
        CorrectnessStatus.CRASHED: ResultStatus.CRASHED,
    }[status]


def _payload_for_worker_response(response) -> tuple[dict[str, object] | None, bool]:
    if response.result_payload is not None:
        return dict(response.result_payload), False
    if response.stage is WorkerStage.CORRECTNESS and response.outcome in {
        WorkerOutcome.TIMEOUT,
        WorkerOutcome.OOM,
        WorkerOutcome.UNSUPPORTED,
    }:
        return {
            "status": response.outcome.value,
            "case_hash": "unavailable",
            "input_summary": {},
            "comparison": None,
            "diagnostic": {
                "kind": response.outcome.value,
                "message": response.error_message,
            },
        }, False
    return None, True


def _result_error_fields(
    raw_status: str,
    diagnostic: Mapping[str, object] | None,
    response,
) -> tuple[str | None, str | None]:
    """Project bounded, stable error columns from a structured diagnostic."""

    if response.error_type or response.error_message:
        return response.error_type, response.error_message
    if raw_status in {"pass", "fail", "nondeterministic"}:
        return None, None
    error_type: str | None = raw_status
    error_message: str | None = None
    if diagnostic is not None:
        exception_type = diagnostic.get("exception_type")
        message = diagnostic.get("message")
        if isinstance(exception_type, str) and exception_type:
            error_type = exception_type
        if isinstance(message, str) and message:
            error_message = message
    return (
        None if error_type is None else error_type[:128],
        None if error_message is None else error_message[:512],
    )


def _append_result(
    job: EvaluationJob,
    snapshot: RegistrySnapshot,
    plan: EvaluationPlan,
    response,
) -> tuple[bool, bool]:
    payload, infrastructure = _payload_for_worker_response(response)
    if infrastructure or payload is None:
        return False, True
    raw_status = payload.get("status")
    if not isinstance(raw_status, str) or raw_status not in {
        "pass", "fail", "nondeterministic", "error", "timeout", "oom", "unsupported", "crashed"
    }:
        return False, True
    correctness = _correctness_status(raw_status)
    comparison = payload.get("comparison")
    comparison = comparison if isinstance(comparison, Mapping) else {}
    metrics = comparison.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    raw_diagnostic = payload.get("diagnostic")
    diagnostic = raw_diagnostic if isinstance(raw_diagnostic, Mapping) else None
    error_type, error_message = _result_error_fields(
        raw_status, diagnostic, response
    )
    manifest = snapshot.operator_manifests[job.identity.operator_id]
    output_contracts = payload.get("output_contracts")
    output_contracts = output_contracts if isinstance(output_contracts, Mapping) else {}
    output_rows: list[dict[str, object]] = []
    for output_path, raw_contract in sorted(output_contracts.items()):
        if not isinstance(output_path, str) or not isinstance(raw_contract, Mapping):
            continue
        path_metrics = metrics.get(output_path)
        path_metrics = path_metrics if isinstance(path_metrics, Mapping) else metrics
        mismatch_count = _metric(path_metrics, "mismatch_count")
        path_passed = correctness is CorrectnessStatus.PASSED or mismatch_count == 0
        output_rows.append(
            {
                "result_id": job.result_id,
                "output_path": output_path,
                "comparator": str(comparison.get("comparator") or "unknown"),
                "reference_dtype": str(raw_contract.get("reference_dtype") or "unknown"),
                "candidate_dtype": str(raw_contract.get("candidate_dtype") or "unknown"),
                "reference_shape": json.dumps(raw_contract.get("reference_shape", [])),
                "candidate_shape": json.dumps(raw_contract.get("candidate_shape", [])),
                "passed": path_passed,
                "rtol": _metric(path_metrics, "rtol"),
                "atol": _metric(path_metrics, "atol"),
                "max_abs_error": _metric(path_metrics, "max_abs_error"),
                "mean_abs_error": _metric(path_metrics, "mean_abs_error"),
                "p95_abs_error": _metric(path_metrics, "p95_abs_error"),
                "max_rel_error": _metric(path_metrics, "max_rel_error"),
                "rmse": _metric(path_metrics, "rmse"),
                "rel_l2": _metric(path_metrics, "relative_l2"),
                "cosine_similarity": _metric(path_metrics, "cosine_similarity"),
                "mismatch_count": mismatch_count,
                "mismatch_rate": _metric(path_metrics, "mismatch_rate"),
                "reference_nan_count": _metric(path_metrics, "reference_nan_count"),
                "candidate_nan_count": _metric(path_metrics, "candidate_nan_count"),
                "diagnostic_path": response.diagnostic_path,
            }
        )
    failed_output_count = sum(not bool(output["passed"]) for output in output_rows)
    if correctness is not CorrectnessStatus.PASSED and not failed_output_count:
        failed_output_count = 1
    row = {
        "run_id": job.identity.run_id,
        "evaluation_id": job.identity.evaluation_id,
        "timestamp_utc": _utc_now(),
        "suite_id": plan.suite_id,
        "mode": job.mode,
        "result_id": job.result_id,
        "operator_id": job.identity.operator_id,
        "contract_version": manifest.contract_version,
        "candidate_id": job.identity.candidate_id,
        "reference_id": job.reference.implementation_id,
        "candidate_source_hash": job.candidate.source_hash,
        "reference_source_hash": job.reference.source_hash,
        "environment_fingerprint": plan.environment_fingerprint,
        "device": "cpu",
        "case_id": job.case.case_id,
        "case_hash": str(payload.get("case_hash") or "unavailable"),
        "seed": job.case.seed,
        "tags": json.dumps(sorted(job.case.tags)),
        "input_summary": json.dumps(payload.get("input_summary", {}), sort_keys=True),
        "status": _result_status(correctness).value,
        "correctness_status": correctness.value,
        "performance_status": PerformanceStatus.SKIPPED.value,
        "skip_reason": PERFORMANCE_SKIP_REASON,
        "correctness_pass": correctness is CorrectnessStatus.PASSED,
        "failed_output_count": failed_output_count,
        "max_abs_error": _metric(metrics, "max_abs_error"),
        "max_rel_error": _metric(metrics, "max_rel_error"),
        "rmse": _metric(metrics, "rmse"),
        "rel_l2": _metric(metrics, "relative_l2"),
        "cosine_similarity": _metric(metrics, "cosine_similarity"),
        "mismatch_count": _metric(metrics, "mismatch_count"),
        "mismatch_rate": _metric(metrics, "mismatch_rate"),
        "error_type": error_type,
        "error_message": error_message,
        "diagnostic_path": response.diagnostic_path,
        "stdout_path": "logs/stdout.log",
        "stderr_path": "logs/stderr.log",
    }
    # Per-output details are durable before results.csv acts as the completion
    # marker consumed by resume.
    if output_rows:
        AtomicCsvTable(
            job.output_dir / CORRECTNESS_OUTPUTS_SCHEMA.filename,
            CORRECTNESS_OUTPUTS_SCHEMA,
        ).append_many(output_rows)
    AtomicCsvTable(job.output_dir / RESULTS_SCHEMA.filename, RESULTS_SCHEMA).append(row)
    return correctness is CorrectnessStatus.PASSED, False


def execute_plan(
    plan: EvaluationPlan,
    snapshot: RegistrySnapshot,
    environment_snapshot: Mapping[str, object],
    *,
    output_root: Path,
    original_command: Sequence[str],
    resume: bool = False,
    fail_fast: bool = False,
    timeout_s: float | None = None,
    controller: WorkerController | None = None,
) -> RunOutcome:
    if plan.mode == "performance":
        raise ValueError("performance mode is not implemented in Phase 7")
    writer = ArtifactWriter(output_root)
    grouped: dict[object, list[EvaluationJob]] = {}
    for job in plan.jobs:
        grouped.setdefault(job.identity, []).append(job)
    paths: list[Path] = []
    completed: dict[object, frozenset[str]] = {}
    active: set[object] = set()
    for identity, jobs in grouped.items():
        paths.append(writer.evaluation_dir(identity))
        if resume:
            existing = writer.read_manifest(identity)
            state = writer.resume_state(existing)
            if existing.status is EvaluationState.INTERRUPTED:
                writer.update_status(identity, EvaluationState.RUNNING)
                active.add(identity)
            elif existing.status is EvaluationState.RUNNING:
                active.add(identity)
            elif existing.status is EvaluationState.COMPLETE:
                pass
            else:
                raise ResumeMismatchError(
                    f"evaluation status cannot be resumed: {existing.status.value}"
                )
        else:
            first = jobs[0]
            manifest = EvaluationManifest.create(
                identity=identity,
                original_command=original_command,
                resolved_config=first.resolved_config,
                reference_source_hash=first.reference.source_hash,
                candidate_source_hash=first.candidate.source_hash,
                environment_snapshot=environment_snapshot,
                environment_fingerprint=plan.environment_fingerprint,
                suite_id=plan.suite_id,
            )
            state = writer.initialize(manifest)
            writer.update_status(identity, EvaluationState.RUNNING)
            active.add(identity)
        completed[identity] = state.completed_result_ids

    worker_controller = controller or WorkerController(artifact_writer=writer)
    passed = failed = infrastructure = 0
    if resume:
        for identity in grouped:
            for existing_row in AtomicCsvTable(
                writer.evaluation_dir(identity) / RESULTS_SCHEMA.filename,
                RESULTS_SCHEMA,
            ).read_rows():
                if existing_row["correctness_status"] == CorrectnessStatus.PASSED.value:
                    passed += 1
                else:
                    failed += 1
    infrastructure_identities: set[object] = set()
    interrupted = False
    stopped = False
    for job in plan.jobs:
        if stopped or job.result_id in completed[job.identity]:
            continue
        metadata = snapshot.operator_specs[job.identity.operator_id]
        candidate_manifest = snapshot.candidate_manifests.get(
            (job.identity.operator_id, job.identity.candidate_id)
        )
        build = None if candidate_manifest is None else candidate_manifest.build
        limits = StageTimeouts(
            import_s=60,
            build_s=float(build.timeout_s if build is not None else 600),
            correctness_s=float(timeout_s or job.case.timeout_s or 300),
            performance_s=float(job.resolved_config.performance_timeout_s),
        )
        try:
            response = worker_controller.run(
                job,
                spec_entrypoint=metadata.entrypoint,
                build_argv=() if build is None else build.command,
                timeouts=limits,
            )
        except KeyboardInterrupt:
            interrupted = True
            stopped = True
            break
        except Exception as error:
            infrastructure += 1
            infrastructure_identities.add(job.identity)
            stopped = True
            atomic_write_text(
                job.output_dir / "diagnostics" / "engine-error.txt",
                f"{type(error).__name__}: {error}\n",
            )
            break
        if response.outcome is WorkerOutcome.INTERRUPTED:
            interrupted = True
            stopped = True
            break
        try:
            did_pass, infra = _append_result(job, snapshot, plan, response)
        except Exception as error:
            infrastructure += 1
            infrastructure_identities.add(job.identity)
            stopped = True
            atomic_write_text(
                job.output_dir / "diagnostics" / "engine-error.txt",
                f"{type(error).__name__}: {error}\n",
            )
            continue
        if infra:
            infrastructure += 1
            infrastructure_identities.add(job.identity)
            stopped = True
        elif did_pass:
            passed += 1
        else:
            failed += 1
            if fail_fast:
                stopped = True

    for identity, jobs in grouped.items():
        if identity not in active:
            continue
        current_ids = {
            row["result_id"]
            for row in AtomicCsvTable(
                writer.evaluation_dir(identity) / RESULTS_SCHEMA.filename,
                RESULTS_SCHEMA,
            ).read_rows()
        }
        expected = {job.result_id for job in jobs}
        if current_ids == expected:
            writer.update_status(identity, EvaluationState.COMPLETE)
        elif interrupted:
            writer.update_status(
                identity, EvaluationState.INTERRUPTED, terminal_reason="user_interrupt"
            )
        elif identity in infrastructure_identities:
            writer.update_status(
                identity, EvaluationState.FAILED, terminal_reason="infrastructure_failure"
            )
        elif infrastructure:
            writer.update_status(
                identity,
                EvaluationState.INTERRUPTED,
                terminal_reason="stopped_after_other_infrastructure_failure",
            )
        else:
            writer.update_status(
                identity, EvaluationState.FAILED, terminal_reason="fail_fast"
            )
    return RunOutcome(
        plan.run_id,
        tuple(paths),
        passed,
        failed,
        infrastructure,
        interrupted,
    )


__all__ = [
    "PERFORMANCE_SKIP_REASON",
    "RunOutcome",
    "build_dry_run_plan",
    "build_execution_plan",
    "build_resume_plan",
    "execute_plan",
]
