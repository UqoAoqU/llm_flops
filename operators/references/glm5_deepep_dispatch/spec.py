"""Discoverable description of the legacy multi-GPU DeepEP benchmark."""

from benchmark_engine.correctness import ExactComparator
from benchmark_engine.models import CaseSpec


M_PER_GPU = (512, 1024, 2048, 4096, 8192, 16384)
SCENARIOS = ("balanced", "mild", "medium", "heavy")


class Glm5DeepEpDispatchSpec:
    operator_id = "glm5_deepep_dispatch"

    def cases(self):
        return tuple(
            CaseSpec(
                f"unsupported__{scenario}__m{tokens}",
                {"tokens_per_gpu": tokens, "hidden": 6144, "experts": 256, "topk": 8,
                 "scenario": scenario, "local_gpus": 8, "requires_nccl": True,
                 "requires_deepep": True},
                439,
                frozenset({"legacy_deepep", "multi_gpu", "unsupported", scenario}),
                1800,
            )
            for tokens in M_PER_GPU for scenario in SCENARIOS
        )

    def legacy_mappings(self):
        return {"script": "bench_glm5_deepep.py", "csv": "glm5_deepep_dispatch_perf.csv",
                "tokens_per_gpu": M_PER_GPU, "scenarios": SCENARIOS, "local_gpus": 8,
                "timer": "cuda_event", "warmup": 10, "runs": 30,
                "unsupported_reason": "multi-GPU scheduling and distributed collectives are excluded"}

    def make_inputs(self, case, context):
        del case, context
        raise NotImplementedError(
            "unsupported: GLM-5 DeepEP requires 8 local GPUs and an initialized NCCL/DeepEP group"
        )

    def clone_inputs(self, inputs):
        del inputs
        raise NotImplementedError("unsupported: DeepEP collective state is not cloneable")

    def normalize_output(self, output):
        del output
        raise NotImplementedError("unsupported: no single-worker DeepEP output contract")

    def comparator(self, case):
        del case
        return ExactComparator()

    def cost_model(self, case):
        del case
        return None

    def workspace_bytes(self, case):
        del case
        return None


SPEC = Glm5DeepEpDispatchSpec()


def cost_model(case):
    return SPEC.cost_model(case)
