#!/usr/bin/env python3
"""Print and validate the benchmark runtime."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_environment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--check", *sys.argv[1:]]))
