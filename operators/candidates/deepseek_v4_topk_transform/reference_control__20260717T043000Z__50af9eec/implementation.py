"""Optimized SGLang TopK baseline migrated from the legacy benchmark.

The private cache accessor is deliberately resolved during worker entrypoint
import. Ninja/PTXAS build time is therefore charged to ``import_ms`` and can
never contaminate steady-state samples.
"""

from sglang.jit_kernel.dsv4.topk import (
    _jit_topk_v2_module,
    topk_transform_512_v2,
)


_jit_topk_v2_module()


def operator(scores, seq_lens, page_tables, output, page_size, metadata):
    topk_transform_512_v2(
        scores, seq_lens, page_tables, output, page_size, metadata
    )
    return output
