"""Phase-3 discovery-to-plan orchestration; execution is intentionally absent."""

from __future__ import annotations

import re
from pathlib import Path

from .environment import load_lock, planning_fingerprint
from .planning import PlanBuilder
from .registry import FilesystemRegistry
from .selectors import Selectors
from .suite import load_suite


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
    if re.fullmatch(r"[a-z][a-z0-9_-]{1,79}", suite_id) is None:
        raise ValueError("suite ID is invalid")
    suite = load_suite(root / "suites" / f"{suite_id}.yaml")
    if suite.suite_id != suite_id:
        raise ValueError(
            f"suite_id {suite.suite_id!r} does not match requested {suite_id!r}"
        )
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


__all__ = ["build_dry_run_plan"]
