# GLM-5 DeepEP dispatch (unsupported)

This contract makes `bench_glm5_deepep.py` visible without claiming support.
The legacy program always spawns eight workers, creates an NCCL process group,
and owns DeepEP collective buffers. Multi-GPU scheduling and distributed
collectives are excluded from this benchmark-engine milestone, so every case
is tagged `unsupported` and is deliberately absent from `glm5_smoke` and
`glm5_regression`. Running it directly returns a stable unsupported result;
use the unchanged legacy script for an explicitly provisioned eight-GPU host.
The reference endpoint raises the same stable unsupported classification.
