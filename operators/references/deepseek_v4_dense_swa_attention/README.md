# DeepSeek V4 dense SWA attention

Dense SWA decode is a separate operator because its single fixed 128-token
cache has different inputs and state from sparse dual-cache decode. The
reference keeps the legacy decode `flash_mla_with_kvcache` SWA-only invocation.
It intentionally makes no prefill mapping claim: legacy dense prefill uses the
different `flash_mla_sparse_fwd` call shape. Physical indices are used directly,
so `block_table` is explicitly absent.

The cache is immutable and bounded cache views are observed. Small zero-cache
cases have an independent zero oracle, while representative runs retain 256
oracle samples. Scheduler workspace is clone-local and its byte count remains
unknown rather than being reported as zero.

Executable small cases use batch 2 so the 576-byte padded cache row stride is
preserved. Clone construction recreates the padded backing allocation instead
of using a plain contiguous tensor clone.
