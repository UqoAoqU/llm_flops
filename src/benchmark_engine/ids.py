"""Identifier validation and safe source/result path construction."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


OPERATOR_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
CANDIDATE_ID_PATTERN = re.compile(
    r"^(?P<task>[a-z0-9][a-z0-9_-]{0,79})__"
    r"(?P<timestamp>[0-9]{8}T[0-9]{6}Z)__"
    r"(?P<hash>[0-9a-f]{8,64})$"
)
EVALUATION_ID_PATTERN = re.compile(
    r"^(?P<timestamp>[0-9]{8}T[0-9]{6}Z)__"
    r"(?P<environment>[0-9a-f]{8,64})__"
    r"(?P<run>[a-z0-9][a-z0-9_-]{2,79})$"
)


class IdentifierError(ValueError):
    """Raised when an identifier cannot safely be used as a path component."""


def _reject_path_syntax(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise IdentifierError(f"{name} must be a string")
    if not value:
        raise IdentifierError(f"{name} must not be empty")
    if Path(value).is_absolute() or value in {".", ".."}:
        raise IdentifierError(f"{name} must be one relative path component")
    if "/" in value or "\\" in value or ".." in value:
        raise IdentifierError(f"{name} must not contain traversal or separators")
    return value


def _validate_utc_timestamp(value: str, name: str) -> None:
    try:
        parsed = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError as error:
        raise IdentifierError(f"{name} contains an invalid UTC timestamp") from error
    if parsed.strftime("%Y%m%dT%H%M%SZ") != value:
        raise IdentifierError(f"{name} contains a non-canonical UTC timestamp")


def validate_operator_id(operator_id: object) -> str:
    value = _reject_path_syntax(operator_id, "operator_id")
    if OPERATOR_ID_PATTERN.fullmatch(value) is None:
        raise IdentifierError(
            "operator_id must match ^[a-z][a-z0-9_]{2,79}$"
        )
    return value


def validate_candidate_id(candidate_id: object) -> str:
    value = _reject_path_syntax(candidate_id, "candidate_id")
    match = CANDIDATE_ID_PATTERN.fullmatch(value)
    if match is None:
        raise IdentifierError(
            "candidate_id must be <task_identifier>__<YYYYMMDDTHHMMSSZ>__"
            "<8-or-more lowercase hex>"
        )
    _validate_utc_timestamp(match.group("timestamp"), "candidate_id")
    return value


def candidate_hash_suffix(candidate_id: str) -> str:
    validate_candidate_id(candidate_id)
    match = CANDIDATE_ID_PATTERN.fullmatch(candidate_id)
    assert match is not None
    return match.group("hash")


def validate_evaluation_id(evaluation_id: object) -> str:
    value = _reject_path_syntax(evaluation_id, "evaluation_id")
    match = EVALUATION_ID_PATTERN.fullmatch(value)
    if match is None:
        raise IdentifierError(
            "evaluation_id must be <YYYYMMDDTHHMMSSZ>__"
            "<environment lowercase hex>__<run short id>"
        )
    _validate_utc_timestamp(match.group("timestamp"), "evaluation_id")
    return value


def _safe_child(root: Path, *components: str) -> Path:
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*components).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise IdentifierError(f"resolved path escapes root {resolved_root}") from error
    return resolved


def candidate_source_path(
    repository_root: Path, operator_id: str, candidate_id: str
) -> Path:
    """Return ``operators/candidates/<operator>/<candidate>`` safely."""

    operator = validate_operator_id(operator_id)
    candidate = validate_candidate_id(candidate_id)
    return _safe_child(
        Path(repository_root), "operators", "candidates", operator, candidate
    )


def candidate_result_path(
    output_root: Path, operator_id: str, candidate_id: str
) -> Path:
    """Mirror a candidate under ``results/<operator>/<candidate>``."""

    operator = validate_operator_id(operator_id)
    candidate = validate_candidate_id(candidate_id)
    return _safe_child(Path(output_root), operator, candidate)


def evaluation_result_path(
    output_root: Path,
    operator_id: str,
    candidate_id: str,
    evaluation_id: str,
) -> Path:
    """Return the safe directory for one candidate evaluation."""

    operator = validate_operator_id(operator_id)
    candidate = validate_candidate_id(candidate_id)
    evaluation = validate_evaluation_id(evaluation_id)
    return _safe_child(Path(output_root), operator, candidate, evaluation)
