#!/usr/bin/env python3
"""Run the formal MI300X benchmark-engine smoke suite."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_smoke_command() -> tuple[str, ...]:
    """Return the one authoritative MI300X smoke command."""
    return (str(ROOT / "bench.sh"), "run", "--suite", "mi300x_smoke")


def main() -> int:
    environment = dict(os.environ)
    environment["ROCR_VISIBLE_DEVICES"] = "0"
    environment["HIP_VISIBLE_DEVICES"] = "0"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        build_smoke_command(),
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
