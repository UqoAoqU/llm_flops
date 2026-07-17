"""Intentionally invalid TopK output used only by Phase 11 tests."""


def operator(scores, seq_lens, page_tables, output, page_size, metadata):
    del scores, seq_lens, page_tables, page_size, metadata
    output.zero_()
    return output
