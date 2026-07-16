from __future__ import annotations

import shutil
from pathlib import Path

from benchmark_engine.registry.validation import compute_source_hash


OPERATOR_ID = "fixture_cpu_add"


def write_reference(repository: Path, operator_id: str = OPERATOR_ID) -> Path:
    root = repository / "operators" / "references" / operator_id
    root.mkdir(parents=True)
    (root / "implementation.py").write_text(
        "def operator(left, right):\n    return left + right\n", encoding="utf-8"
    )
    (root / "spec.py").write_text(
        f'class Spec:\n    operator_id = "{operator_id}"\n'
        "SPEC = Spec()\ndef cost_model(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (root / "operator.yaml").write_text(
        f"""schema_version: 1
operator_id: {operator_id}
contract_version: 1
description: CPU fixture
reference_entrypoint: implementation:operator
spec_entrypoint: spec:SPEC
device_types: [cpu]
tags: [fixture, cpu]
correctness:
  default_comparator: exact
  rtol: 0.0
  atol: 0.0
  equal_nan: false
  determinism_repeats: 1
performance:
  timer: wall_clock
  graph_mode: disabled
  warmup: 0
  samples: 2
  inner_iterations: 1
  timeout_s: 10
  regression_threshold_pct: 5.0
cost_model: spec:cost_model
""",
        encoding="utf-8",
    )
    return root


def write_candidate(
    repository: Path,
    task: str,
    timestamp: str,
    *,
    with_manifest: bool,
    import_marker: Path | None = None,
) -> tuple[Path, str]:
    parent = repository / "operators" / "candidates" / OPERATOR_ID
    temporary = parent / f"building-{task}"
    temporary.mkdir(parents=True)
    marker_statement = (
        f'raise RuntimeError("candidate imported: {import_marker}")\n'
        if import_marker is not None
        else ""
    )
    (temporary / "implementation.py").write_text(
        marker_statement
        + "def operator(left, right):\n    return left + right\n",
        encoding="utf-8",
    )
    if with_manifest:
        (temporary / "candidate.yaml").write_text(
            f"""schema_version: 1
operator_id: {OPERATOR_ID}
candidate_id: {task}__{timestamp}__00000000
entrypoint: implementation:operator
framework: python
build:
  command: [python, -m, build]
  timeout_s: 30
metadata:
  task_id: {task}
  enabled: true
""",
            encoding="utf-8",
        )
    source_hash = compute_source_hash(temporary)
    candidate_id = f"{task}__{timestamp}__{source_hash[:8]}"
    if with_manifest:
        manifest = temporary / "candidate.yaml"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace(
                f"{task}__{timestamp}__00000000", candidate_id
            ),
            encoding="utf-8",
        )
        assert compute_source_hash(temporary) == source_hash
    destination = parent / candidate_id
    temporary.rename(destination)
    return destination, candidate_id


def write_registry(repository: Path) -> tuple[str, str]:
    write_reference(repository)
    _, first = write_candidate(
        repository,
        "task_one",
        "20260716T081500Z",
        with_manifest=False,
    )
    _, second = write_candidate(
        repository,
        "task_two",
        "20260716T083000Z",
        with_manifest=True,
    )
    return first, second


def copy_example(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)
