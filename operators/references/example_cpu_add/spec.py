from benchmark_engine.correctness import (
    FloatingComparator,
    InputBundle,
    Tolerance,
    clone_input_bundle,
)
from benchmark_engine.models import CaseSpec
import random


class VectorAddSpec:
    operator_id = "example_cpu_add"

    def cases(self):
        return (
            CaseSpec(
                case_id="tiny",
                symbols={"size": 1},
                seed=0,
                tags=frozenset({"smoke"}),
                timeout_s=30,
            ),
            CaseSpec(
                case_id="small",
                symbols={"size": 4},
                seed=0,
                tags=frozenset({"smoke", "representative"}),
                timeout_s=30,
            ),
        )

    def make_inputs(self, case, context):
        size = int(case.symbols["size"])
        generator = random.Random(context.seed)
        return InputBundle(
            args=(
                [generator.uniform(-1.0, 1.0) for _ in range(size)],
                [generator.uniform(-1.0, 1.0) for _ in range(size)],
            )
        )

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        return output

    def comparator(self, case):
        return FloatingComparator(
            operator_default=Tolerance(0.0, 0.0, source="operator"),
            require_explicit=True,
        )

    def cost_model(self, case):
        size = int(case.symbols["size"])
        return {
            "flops": size,
            "estimated_bytes": size * 8 * 3,
            "throughput_units": size,
        }


SPEC = VectorAddSpec()


def cost_model(case):
    return SPEC.cost_model(case)
