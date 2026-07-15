"""Print MXFP4 versus FP8/MXFP8 DeepSeek V4 benchmark results."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperatorComparison:
    name: str
    mxfp4_ms: float
    fp8_mxfp8_ms: float

    @property
    def speedup(self):
        return self.mxfp4_ms / self.fp8_mxfp8_ms


@dataclass(frozen=True)
class ProfileComparison:
    case: tuple[str, int, int]
    mxfp4_total_ms: float
    fp8_mxfp8_total_ms: float
    operators: tuple[OperatorComparison, ...]

    @property
    def speedup(self):
        return self.mxfp4_total_ms / self.fp8_mxfp8_total_ms


def _case(rows):
    if not rows:
        raise ValueError("cannot compare empty result rows")
    cases = {
        (row["phase"], int(row["m"]), int(row["context"])) for row in rows
    }
    if len(cases) != 1:
        raise ValueError(f"result file contains multiple cases: {cases}")
    return next(iter(cases))


def _canonical_operator(name):
    replacements = {
        "C4 Indexer FP4 Quant": "C4 Indexer Quant",
        "C4 Indexer FP8 Quant": "C4 Indexer Quant",
        "C4 FP4 Paged MQA Logits": "C4 Paged MQA Logits",
        "C4 FP8 Paged MQA Logits": "C4 Paged MQA Logits",
    }
    return replacements.get(name, name)


def _operator_rows(rows):
    result = {}
    for row in rows:
        name = _canonical_operator(row["operator"])
        if name in result:
            raise ValueError(f"duplicate semantic operator: {name}")
        result[name] = row
    return result


def compare_profile_rows(mxfp4_rows, fp8_mxfp8_rows):
    mx_case = _case(mxfp4_rows)
    fp8_case = _case(fp8_mxfp8_rows)
    if mx_case != fp8_case:
        raise ValueError(f"case mismatch: {mx_case} != {fp8_case}")

    mx = _operator_rows(mxfp4_rows)
    fp8 = _operator_rows(fp8_mxfp8_rows)
    if mx.keys() != fp8.keys():
        raise ValueError(
            f"operator mismatch: {sorted(mx.keys())} != {sorted(fp8.keys())}"
        )

    comparisons = []
    for name, mx_row in mx.items():
        fp8_row = fp8[name]
        if int(mx_row["instances"]) != int(fp8_row["instances"]):
            raise ValueError(f"instance mismatch for {name}")
        comparisons.append(
            OperatorComparison(
                name,
                float(mx_row["model_ms"]),
                float(fp8_row["model_ms"]),
            )
        )

    mx_total = sum(item.mxfp4_ms for item in comparisons)
    fp8_total = sum(item.fp8_mxfp8_ms for item in comparisons)
    return ProfileComparison(mx_case, mx_total, fp8_total, tuple(comparisons))


def _load(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def _result_path(results, profile, phase, m):
    return results / f"deepseek_v4_pro_{profile}_{phase}_kv65536_m{m}.csv"


def main():
    results = Path("results")
    cases = (
        ("prefill", 1024),
        ("prefill", 2048),
        ("prefill", 4096),
        ("decode", 16),
        ("decode", 32),
    )
    current_phase = None
    for phase, m in cases:
        if phase != current_phase:
            print(f"\n===== {phase.upper()} =====")
            current_phase = phase
        comparison = compare_profile_rows(
            _load(_result_path(results, "mxfp4", phase, m)),
            _load(_result_path(results, "fp8_mxfp8", phase, m)),
        )
        print(
            f"M={m:<4} MXFP4={comparison.mxfp4_total_ms:10.6f} ms  "
            f"FP8/MXFP8={comparison.fp8_mxfp8_total_ms:10.6f} ms  "
            f"speedup={comparison.speedup:.4f}x"
        )
        for item in comparison.operators:
            mx_pct = item.mxfp4_ms / comparison.mxfp4_total_ms * 100
            fp8_pct = item.fp8_mxfp8_ms / comparison.fp8_mxfp8_total_ms * 100
            print(
                f"  {item.name:<38} {item.mxfp4_ms:10.6f} -> "
                f"{item.fp8_mxfp8_ms:10.6f} ms  {item.speedup:7.4f}x  "
                f"pct={mx_pct:5.2f}%->{fp8_pct:5.2f}%"
            )


if __name__ == "__main__":
    main()
