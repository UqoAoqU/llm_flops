"""Deterministic suite and CLI selection rules."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from .models import CaseSpec, ImplementationSpec
from .registry import RegistrySnapshot
from .suite import SuiteConfig


class SelectorError(ValueError):
    pass


@dataclass(frozen=True)
class Selectors:
    operators: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    cases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    exclude_operators: tuple[str, ...] = ()


def _matches(value: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def select_operators(
    snapshot: RegistrySnapshot, suite: SuiteConfig, selectors: Selectors
) -> tuple[str, ...]:
    values = tuple(
        operator_id
        for operator_id in snapshot.discovered_operator_ids
        if _matches(operator_id, suite.operator_include)
        and _matches(operator_id, selectors.operators)
        and not any(
            fnmatch.fnmatchcase(operator_id, pattern)
            for pattern in suite.operator_exclude + selectors.exclude_operators
        )
    )
    if not values:
        raise SelectorError("selector matched no operators")
    return tuple(sorted(values))


def select_candidates(
    snapshot: RegistrySnapshot,
    operator_ids: tuple[str, ...],
    selectors: Selectors,
    suite: SuiteConfig | None = None,
) -> dict[str, tuple[ImplementationSpec, ...]]:
    result: dict[str, tuple[ImplementationSpec, ...]] = {}
    # A suite candidate list is a safe default set (for example smoke should
    # not run intentional failure demonstrations).  An explicit CLI selector
    # is an override and must be able to address every valid registry member.
    suite_patterns = (
        ("*",)
        if selectors.candidates or suite is None
        else suite.candidate_include
    )
    for operator_id in operator_ids:
        matching = tuple(
            candidate
            for candidate in snapshot.candidates.get(operator_id, ())
            if _matches(candidate.implementation_id, selectors.candidates)
            and _matches(
                candidate.implementation_id,
                suite_patterns,
            )
        )
        if not matching:
            raise SelectorError(
                f"selector matched no candidates for operator: {operator_id}"
            )
        result[operator_id] = matching
    return result


def select_cases(
    cases: tuple[CaseSpec, ...], suite: SuiteConfig, selectors: Selectors
) -> tuple[CaseSpec, ...]:
    selected = tuple(
        case
        for case in cases
        if _matches(case.case_id, selectors.cases)
        and (not suite.case_tags or bool(case.tags.intersection(suite.case_tags)))
        and (not selectors.tags or bool(case.tags.intersection(selectors.tags)))
    )
    if not selected:
        raise SelectorError("selector matched no cases")
    return tuple(sorted(selected, key=lambda case: case.case_id))


__all__ = [
    "SelectorError",
    "Selectors",
    "select_candidates",
    "select_cases",
    "select_operators",
]
