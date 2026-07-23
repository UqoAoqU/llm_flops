"""Reference-owned contract for the MI300X BF16 GEMM framework probe."""

import importlib

from benchmark_engine.correctness import (
    FloatingComparator,
    InputBundle,
    Tolerance,
    clone_input_bundle,
)
from benchmark_engine.models import CaseSpec


TOLERANCE = Tolerance(
    rtol=0.001,
    atol=0.01,
    source="mi300x_bf16_gemm",
)


class RocmBf16GemmSpec:
    operator_id = "example_rocm_bf16_gemm"

    def cases(self):
        return (
            CaseSpec(
                case_id="smoke_m256_n256_k256",
                symbols={"m": 256, "n": 256, "k": 256},
                seed=942,
                tags=frozenset({"smoke"}),
                timeout_s=120,
            ),
            CaseSpec(
                case_id="representative_m1024_n4096_k4096",
                symbols={"m": 1024, "n": 4096, "k": 4096},
                seed=942,
                tags=frozenset({"representative"}),
                timeout_s=300,
            ),
        )

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        m, n, k = (int(case.symbols[name]) for name in ("m", "n", "k"))
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("example_rocm_bf16_gemm requires a GPU generator")
        left = torch.randn(
            (m, k),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        right = torch.randn(
            (k, n),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        return InputBundle(args=(left, right))

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        return output

    def comparator(self, case):
        del case
        return FloatingComparator(
            operator_default=TOLERANCE,
            require_explicit=True,
        )

    def cost_model(self, case):
        m, n, k = (int(case.symbols[name]) for name in ("m", "n", "k"))
        return {
            "flops": 2 * m * n * k,
            "estimated_bytes": 2 * (m * k + k * n + m * n),
            "throughput_units": m * n,
        }


SPEC = RocmBf16GemmSpec()


def cost_model(case):
    return SPEC.cost_model(case)
