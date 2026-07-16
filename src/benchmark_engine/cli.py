"""Command-line interface for the first benchmark-engine phase."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately small Phase 1 parser."""

    parser = argparse.ArgumentParser(
        prog="bench",
        description="Reproducible operator benchmark engine.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    build_parser().parse_args(argv)
    return 0
