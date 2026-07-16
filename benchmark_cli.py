"""Dispatch reproducible single-operator and end-to-end benchmarks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


OPERATORS = {
    "dsa_indexer": "dsa_indexer.py",
    "dsa_flashmla": "dsa_flashmla.py",
    "dsa_projection": "dsa_projection.py",
    "mla_flashmla": "mla_flashmla.py",
    "moe_deepgemm": "moe_deepgemm.py",
}

COMMANDS = {
    "prefill": "bench_deepseek_v4_prefill.py",
    "decode": "bench_deepseek_v4_decode.py",
    "compare": "print_deepseek_v4_quant_comparison.py",
    "check": "tools/check_environment.py",
    "smoke": "tools/smoke_test.py",
}

GPU_COMMANDS = {"op", "prefill", "decode", "smoke"}


class CommandError(ValueError):
    pass


def usage() -> str:
    operators = "|".join(OPERATORS)
    return (
        "Usage: run.sh op <operator> [args...]\n"
        "       run.sh {prefill|decode|compare|check|smoke} [args...]\n"
        f"Operators: {operators}"
    )


def resolve_command(argv: list[str], root: Path) -> list[str]:
    if not argv:
        raise CommandError(usage())
    command = argv[0]
    if command == "op":
        if len(argv) < 2 or argv[1] not in OPERATORS:
            requested = "<missing>" if len(argv) < 2 else argv[1]
            raise CommandError(
                f"Unknown operator {requested!r}. Valid operators: "
                + ", ".join(OPERATORS)
            )
        return [str(root / OPERATORS[argv[1]]), *argv[2:]]
    if command not in COMMANDS:
        raise CommandError(f"Unknown command {command!r}.\n{usage()}")
    return [str(root / COMMANDS[command]), *argv[1:]]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parent
    try:
        selected = resolve_command(args, root)
    except CommandError as error:
        print(error, file=sys.stderr)
        return 2

    if args[0] in GPU_COMMANDS:
        status = subprocess.run(
            [sys.executable, "-m", "benchmark_environment", "--check"],
            cwd=root,
            check=False,
        ).returncode
        if status:
            return status
    os.execv(sys.executable, [sys.executable, *selected])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
