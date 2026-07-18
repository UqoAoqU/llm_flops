#!/usr/bin/env python3
"""Stable entry points for CPU, GPU smoke, B200 regression, and full validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".runtime" / "venv" / "bin" / "python"


def _commands(tier: str, work_root: Path) -> list[list[str]]:
    cpu = [
        [str(PYTHON), "-m", "unittest", "discover", "-s", "tests", "-v"],
        [str(ROOT / "run.sh"), "check"],
        [str(ROOT / "bench.sh"), "validate"],
    ]
    gpu = [
        [str(ROOT / "run.sh"), "smoke"],
        [str(ROOT / "bench.sh"), "run", "--suite", "smoke"],
    ]
    regression = [
        [str(PYTHON), str(ROOT / "tools" / "regress_deepseek_v4.py"),
         "--phase", phase, "--work-dir", str(work_root / phase)]
        for phase in ("prefill", "decode")
    ]
    nightly = [
        [str(ROOT / "bench.sh"), "run", "--suite", "regression"],
        [str(ROOT / "bench.sh"), "run", "--suite", "deepseek_v4_prefill"],
        [str(ROOT / "bench.sh"), "run", "--suite", "deepseek_v4_decode"],
    ]
    if tier == "cpu":
        return cpu
    if tier == "gpu-smoke":
        return gpu
    if tier == "b200-regression":
        return regression
    return cpu + gpu + regression + nightly


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one documented benchmark test tier.")
    parser.add_argument("tier", choices=("cpu", "gpu-smoke", "b200-regression", "full"))
    parser.add_argument(
        "--work-root", type=Path, default=ROOT / ".runtime" / "regression" / "deepseek-v4"
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    commands = _commands(arguments.tier, arguments.work_root.resolve())
    if arguments.dry_run:
        print(json.dumps({"schema_version": 1, "tier": arguments.tier, "commands": commands}, indent=2))
        return 0
    environment = dict(os.environ)
    if arguments.tier != "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
