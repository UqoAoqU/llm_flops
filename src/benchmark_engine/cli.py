"""Registry, environment, and Phase-3 dry-run planning commands."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import DEFAULT_OUTPUT_ROOT
from .engine import build_dry_run_plan
from .environment import collect_report
from .registry import (
    FilesystemRegistry,
    RegistryIssue,
    RegistrySnapshot,
    selected_registry_issues,
)
from .selectors import Selectors


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 3 CLI parser."""

    parser = argparse.ArgumentParser(
        prog="bench",
        description="Reproducible operator benchmark engine.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command")

    list_parser = commands.add_parser(
        "list", help="list statically discovered operators and candidates"
    )
    list_parser.add_argument(
        "--operator", default="*", metavar="GLOB", help="operator ID glob"
    )
    list_parser.add_argument(
        "--candidate", default="*", metavar="GLOB", help="candidate ID glob"
    )

    validate_parser = commands.add_parser(
        "validate", help="validate operator and candidate manifests"
    )
    validate_parser.add_argument(
        "--operator", default="*", metavar="GLOB", help="operator ID glob"
    )

    env_parser = commands.add_parser("env", help="collect the benchmark environment")
    env_parser.add_argument("--json", action="store_true", help="emit stable JSON")

    run_parser = commands.add_parser("run", help="plan a suite evaluation")
    run_parser.add_argument("--suite", required=True, metavar="ID")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--operator", action="append", default=[], metavar="GLOB")
    run_parser.add_argument("--candidate", action="append", default=[], metavar="GLOB")
    run_parser.add_argument("--case", action="append", default=[], metavar="GLOB")
    run_parser.add_argument("--tag", action="append", default=[], metavar="TAG")
    run_parser.add_argument(
        "--exclude-operator", action="append", default=[], metavar="GLOB"
    )
    run_parser.add_argument("--seed", action="append", default=[], type=int)
    run_parser.add_argument(
        "--mode", choices=("all", "correctness", "performance")
    )
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--evaluation-id")
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="allow an existing evaluation path (recovery starts in Phase 4)",
    )
    return parser


def _print_issues(issues: Sequence[RegistryIssue]) -> None:
    for issue in issues:
        print(issue, file=sys.stderr)


def _selected_operators(snapshot: RegistrySnapshot, pattern: str) -> tuple[str, ...]:
    return tuple(
        operator_id
        for operator_id in snapshot.discovered_operator_ids
        if fnmatch.fnmatchcase(operator_id, pattern)
    )


def _run_list(
    snapshot: RegistrySnapshot,
    repository_root: Path,
    operator_glob: str,
    candidate_glob: str,
) -> int:
    selected = _selected_operators(snapshot, operator_glob)
    if not selected:
        print(f"selector matched no operators: {operator_glob}", file=sys.stderr)
        return 2
    issues = selected_registry_issues(
        snapshot,
        selected,
        candidate_patterns=(candidate_glob,),
        repository_root=repository_root,
    )
    if issues:
        _print_issues(issues)
        return 2

    rows: list[tuple[str, str, str]] = []
    for operator_id in selected:
        matching = tuple(
            candidate
            for candidate in snapshot.candidates.get(operator_id, ())
            if fnmatch.fnmatchcase(candidate.implementation_id, candidate_glob)
        )
        rows.extend(
            (operator_id, candidate.implementation_id, candidate.source_hash)
            for candidate in matching
        )
        if candidate_glob == "*" and not matching:
            rows.append((operator_id, "-", "-"))
    if not rows:
        print(f"selector matched no candidates: {candidate_glob}", file=sys.stderr)
        return 2

    print("operator_id\tcandidate_id\tsource_hash")
    for row in sorted(rows):
        print("\t".join(row))
    return 0


def _run_validate(
    snapshot: RegistrySnapshot, repository_root: Path, operator_glob: str
) -> int:
    selected = _selected_operators(snapshot, operator_glob)
    if not selected:
        print(f"selector matched no operators: {operator_glob}", file=sys.stderr)
        return 2
    issues = selected_registry_issues(
        snapshot, selected, repository_root=repository_root
    )
    if issues:
        _print_issues(issues)
        return 2
    candidate_count = sum(len(snapshot.candidates.get(item, ())) for item in selected)
    print(
        f"registry valid: {len(selected)} operator(s), "
        f"{candidate_count} candidate(s)"
    )
    return 0


def main(
    argv: Sequence[str] | None = None, *, repository_root: Path | None = None
) -> int:
    """Run an import-free registry command."""

    arguments = build_parser().parse_args(argv)
    if arguments.command is None:
        return 0
    root = Path(repository_root or Path.cwd()).resolve()
    if arguments.command == "env":
        try:
            report = collect_report(root / "requirements" / "benchmark-lock.json")
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"environment error: {error}", file=sys.stderr)
            return 2
        if arguments.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            observed = report["environment"]
            assert isinstance(observed, dict)
            gpu = observed.get("gpu", {})
            assert isinstance(gpu, dict)
            print(f"Benchmark environment: {report['fingerprint']}")
            print(f"Python {observed.get('python')}  CUDA {observed.get('cuda')}")
            print(f"GPU {gpu.get('name')}  capability={gpu.get('capability')}")
            for error in report["errors"]:
                print(f"ERROR: {error}")
        return 0
    if arguments.command == "run":
        if not arguments.dry_run:
            print(
                "execution not available until later phase; use --dry-run",
                file=sys.stderr,
            )
            return 2
        output_root = arguments.output_root
        if not output_root.is_absolute():
            output_root = root / output_root
        try:
            plan = build_dry_run_plan(
                root,
                arguments.suite,
                Selectors(
                    operators=tuple(arguments.operator),
                    candidates=tuple(arguments.candidate),
                    cases=tuple(arguments.case),
                    tags=tuple(arguments.tag),
                    exclude_operators=tuple(arguments.exclude_operator),
                ),
                output_root=output_root,
                mode=arguments.mode,
                seeds=tuple(arguments.seed),
                evaluation_id=arguments.evaluation_id,
                resume=arguments.resume,
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0

    registry = FilesystemRegistry(root)
    snapshot = registry.discover()
    if arguments.command == "list":
        return _run_list(
            snapshot,
            registry.repository_root,
            arguments.operator,
            arguments.candidate,
        )
    if arguments.command == "validate":
        return _run_validate(snapshot, registry.repository_root, arguments.operator)
    return 0
