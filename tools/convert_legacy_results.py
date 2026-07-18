#!/usr/bin/env python3
"""Convert one legacy DeepSeek V4 CSV into non-rankable engine artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from benchmark_engine.reporting import LegacyImportError, convert_legacy_csv


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import legacy DeepSeek V4 aggregate graph_ms without inventing provenance or samples."
    )
    parser.add_argument("csv", type=Path, help="legacy DeepSeek V4 CSV")
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="path-safe candidate identity used by the mirrored result layout",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fully validate and stage in a temporary directory without publishing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.dry_run:
            with tempfile.TemporaryDirectory(prefix="benchmark-legacy-dry-run-") as temporary:
                report = convert_legacy_csv(
                    arguments.csv,
                    Path(temporary),
                    arguments.candidate_id,
                    repository_root=ROOT,
                )
        else:
            report = convert_legacy_csv(
                arguments.csv,
                arguments.output_root,
                arguments.candidate_id,
                repository_root=ROOT,
            )
    except (LegacyImportError, OSError, ValueError) as error:
        print(f"legacy import error: {error}", file=sys.stderr)
        return 2
    payload = report.to_dict()
    payload["dry_run"] = arguments.dry_run
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
