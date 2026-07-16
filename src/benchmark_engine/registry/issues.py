"""Shared registry issue ownership and selector isolation."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from pathlib import Path

from .base import RegistryIssue, RegistrySnapshot


def registry_issue_owner(
    issue: RegistryIssue, repository_root: Path
) -> tuple[str | None, str | None]:
    """Return the lexical ``(operator_id, candidate_id)`` owning an issue.

    An issue outside the reference/candidate registry roots is global and
    returns ``(None, None)``.  Paths are intentionally not resolved: resolving
    a rejected symlink could erase its lexical registry ownership.
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


def selected_registry_issues(
    snapshot: RegistrySnapshot,
    operator_ids: Sequence[str],
    *,
    candidate_patterns: Sequence[str] | None = None,
    repository_root: Path | None = None,
) -> tuple[RegistryIssue, ...]:
    """Return issues applicable to one selected registry scope.

    Values within ``candidate_patterns`` are ORed. ``None`` means all
    candidates, while an empty sequence intentionally selects no candidate
    issues. Global/unowned issues are always returned.
    """

    root = repository_root or snapshot.repository_root
    selected = frozenset(operator_ids)
    applicable: list[RegistryIssue] = []
    for issue in snapshot.issues:
        if root is None:
            applicable.append(issue)
            continue
        operator_id, candidate_id = registry_issue_owner(issue, Path(root))
        if operator_id is None:
            applicable.append(issue)
        elif operator_id not in selected:
            continue
        elif candidate_id is None or candidate_patterns is None:
            applicable.append(issue)
        elif any(
            fnmatch.fnmatchcase(candidate_id, pattern)
            for pattern in candidate_patterns
        ):
            applicable.append(issue)
    return tuple(applicable)


__all__ = ["registry_issue_owner", "selected_registry_issues"]
