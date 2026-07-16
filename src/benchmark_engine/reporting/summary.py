"""Human-readable summaries derived from evaluation manifests."""

from __future__ import annotations

from collections.abc import Mapping


def render_summary(manifest: Mapping[str, object]) -> str:
    """Render the small stable summary available before evaluators exist."""

    identity = manifest["identity"]
    if not isinstance(identity, Mapping):
        raise TypeError("manifest identity must be an object")
    environment = manifest["environment"]
    if not isinstance(environment, Mapping):
        raise TypeError("manifest environment must be an object")
    reason = manifest.get("terminal_reason")
    lines = [
        "# Benchmark evaluation",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Run: `{identity['run_id']}`",
        f"- Operator: `{identity['operator_id']}`",
        f"- Candidate: `{identity['candidate_id']}`",
        f"- Evaluation: `{identity['evaluation_id']}`",
        f"- Suite: `{manifest['suite_id']}`",
        f"- Mode: `{manifest['mode']}`",
        f"- Environment: `{environment['fingerprint']}`",
    ]
    if reason:
        lines.append(f"- Terminal reason: {reason}")
    lines.extend(
        (
            "",
            "This file is generated from `evaluation_manifest.json`; do not edit it.",
            "",
        )
    )
    return "\n".join(lines)


__all__ = ["render_summary"]
