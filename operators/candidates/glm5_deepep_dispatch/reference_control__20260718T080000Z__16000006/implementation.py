"""Stable unsupported endpoint for a deliberately excluded multi-GPU path."""


def operator(*_args, **_kwargs):
    raise NotImplementedError(
        "unsupported: bench_glm5_deepep.py requires 8 local GPUs, NCCL process groups "
        "and DeepEP collective state; benchmark_engine multi-GPU scheduling is out of scope"
    )
