# GLM-5 dense prefill attention

This is the dense paged FlashMLA contract from `mla_flashmla.py`. Scheduler
metadata is built before measurement and cloned with each isolated input. The
complete 4x4 legacy query/context sweep remains discoverable as `legacy_full`;
the regular suites select bounded smoke/regression cases to avoid accidental
multi-hour runs. The original script remains available for exact CSV replay.
The reference remains the optimized SGL Kernel FlashMLA implementation.
