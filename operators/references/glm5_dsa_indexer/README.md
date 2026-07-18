# GLM-5 DSA indexer

This contract migrates the four cuBLAS FP8 GEMMs in `dsa_indexer.py`:
`index_k_proj`, `index_q_upproj`, `index_weights_proj`, and `index_score`.
The original M/context cartesian product remains as `legacy_full` case metadata;
the smoke and regression suites select bounded representatives. Quantization is
input preparation and is excluded from steady-state timing, as in the legacy
script. Results are written under
`results/glm5_dsa_indexer/<candidate_id>/<evaluation_id>/`.
The reference is the optimized cuBLAS/DeepGEMM path, not a PyTorch oracle.
