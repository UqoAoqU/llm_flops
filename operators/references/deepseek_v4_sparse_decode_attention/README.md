# DeepSeek V4 sparse decode attention

The reference is the legacy FlashMLA dual-cache call. Every request attends a
128-token SWA cache plus a C4 or C128 compressed cache. Physical indices,
valid lengths, 64-token scheduler padding, tail pages, and the C128 two-token
page layout are part of the contract; `block_table` is intentionally `None`
because this API consumes physical indices directly.

Both caches are immutable inputs and bounded cache views are included in
`observed_state`. Zero-valued small inputs have an independent zero oracle;
large cases retain only 256 deterministic samples. The oracle is never passed
to candidate code. Scheduler/workspace state is created once per clone and its
byte size remains `None` because the backend does not expose it.

Executable small cases use batch 2. A one-page batch lets PyTorch normalize
away the padded leading stride, while FlashMLA requires every cache row stride
to remain a multiple of 576 bytes. Cloning reconstructs padded storage before
copying values.
