"""Runtime and import-free planning fingerprints."""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any


def environment_fingerprint(observed: dict[str, Any]) -> str:
    return importlib.import_module("benchmark_environment").environment_fingerprint(
        observed
    )


def planning_fingerprint(lock: dict[str, Any]) -> str:
    """Hash normalized lock metadata without collecting or importing packages."""

    encoded = json.dumps(
        lock,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"benchmark-engine-planning-v1\0" + encoded).hexdigest()


__all__ = ["environment_fingerprint", "planning_fingerprint"]
