# MI300X BF16 GEMM framework probe

This operator is an infrastructure control, not a DeepSeek V4 production
reference. It validates seeded GPU inputs, isolated reference/candidate
execution, correctness comparison, HIP graph/event timing, KFD locking, cost
reporting, and artifact generation on `gfx942`.

PyTorch intentionally exposes ROCm devices through its `cuda` Python namespace,
so the manifest uses `device_types: [cuda]` while the environment and result
metadata record `accelerator_backend=rocm`.
