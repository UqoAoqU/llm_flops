"""Import-free registry commands for the benchmark engine."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .registry import FilesystemRegistry, RegistryIssue, RegistrySnapshot


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 2 parser without exposing future execution commands."""

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


def _issue_owner(
    issue: RegistryIssue, repository_root: Path
) -> tuple[str | None, str | None]:
    """Return an issue's lexical ``(operator, candidate)`` path ownership.

    Paths outside the two registry roots, or paths without an operator
    component, are global.  This intentionally inspects path components rather
    than performing a substring match.  It also avoids resolving an issue path,
    because resolving a rejected symlink could erase its registry ownership.
    """

    try:
        relative = issue.path.relative_to(repository_root)
    except ValueError:
        return None, None
    parts = relative.parts
    if len(parts) < 3 or parts[0] != "operators":
        return None, None
    if parts[1] == "references":
        return parts[2], None
    if parts[1] == "candidates":
        candidate_id = parts[3] if len(parts) >= 4 else None
        return parts[2], candidate_id
    return None, None


def _selected_issues(
    snapshot: RegistrySnapshot,
    repository_root: Path,
    operator_ids: Sequence[str],
    candidate_glob: str | None = None,
) -> tuple[RegistryIssue, ...]:
    selected = frozenset(operator_ids)
    applicable: list[RegistryIssue] = []
    for issue in snapshot.issues:
        operator_id, candidate_id = _issue_owner(issue, repository_root)
        if operator_id is None:
            applicable.append(issue)
        elif operator_id not in selected:
            continue
        elif (
            candidate_id is not None
            and candidate_glob is not None
            and not fnmatch.fnmatchcase(candidate_id, candidate_glob)
        ):
            continue
        else:
            applicable.append(issue)
    return tuple(applicable)


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
    issues = _selected_issues(
        snapshot, repository_root, selected, candidate_glob
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
    issues = _selected_issues(snapshot, repository_root, selected)
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
    registry = FilesystemRegistry(repository_root or Path.cwd())
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
