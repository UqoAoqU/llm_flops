import unittest

from deepseek_v4_benchmark import (
    ATTN_HEADS,
    E_GLOBAL,
    E_LOCAL,
    INDEX_HEADS,
    INDEX_TOPK,
    MOE_INTERMEDIATE,
    MOE_TOPK,
    BenchmarkRow,
    _moe_fp8_mxfp8_fn,
    adapter_io_shapes,
    case_context,
    compressed_context,
    decode_adapters,
    prefill_adapters,
    summarize_rows,
    local_routed_pairs,
)


class DeepSeekV4BenchmarkTest(unittest.TestCase):
    def test_quant_profiles_use_distinct_indexer_and_moe_backends(self):
        mx = {adapter.name: adapter for adapter in prefill_adapters("mxfp4")}
        fp8 = {
            adapter.name: adapter for adapter in prefill_adapters("fp8_mxfp8")
        }

        self.assertEqual(mx["C4 Indexer FP4 Quant"].kind, "fp4_quant")
        self.assertEqual(fp8["C4 Indexer FP8 Quant"].kind, "fp8_quant")
        self.assertEqual(mx["Routed Expert Fused MoE"].kind, "moe_mxfp4")
        self.assertEqual(
            fp8["Routed Expert Fused MoE"].kind, "moe_fp8_mxfp8"
        )

    def test_unknown_quant_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown quant profile"):
            prefill_adapters("invalid")

    def test_fp8_indexer_shapes_use_c4_context(self):
        adapters = {
            adapter.name: adapter for adapter in prefill_adapters("fp8_mxfp8")
        }
        self.assertEqual(
            adapter_io_shapes(
                adapters["C4 FP8 Paged MQA Logits"], 1024, 65536
            ),
            (
                "q=(1024,1,64,128); raw_kv=65536; c4_kv=16384",
                "logits=(1024,16384)",
            ),
        )

    def test_fp8_mxfp8_moe_is_one_fused_row(self):
        self.assertTrue(callable(_moe_fp8_mxfp8_fn))
        rows = [
            adapter
            for adapter in decode_adapters("fp8_mxfp8")
            if "MoE" in adapter.name
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0].backend,
            "FlashInfer TRTLLM FP8 weight + MXFP8 activation",
        )
        self.assertEqual(rows[0].kind, "moe_fp8_mxfp8")
        self.assertEqual(rows[0].shape, (16, 7168, 3072))

    def test_summary_excludes_unavailable_rows(self):
        rows = [
            BenchmarkRow("attention", "FlashMLA", 2, 1.5, "executed", ""),
            BenchmarkRow("moe", "FlashInfer", 61, None, "unavailable", "not wired"),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary.total_ms, 3.0)
        self.assertEqual(summary.percent_by_name, {"attention": 100.0})

    def test_fixed_raw_kv_derives_compressed_lengths(self):
        self.assertEqual(compressed_context(65536, 4), 16384)
        self.assertEqual(compressed_context(65536, 128), 512)

    def test_prefill_registry_includes_indexer_selection_pipeline(self):
        adapters = {adapter.name: adapter for adapter in prefill_adapters()}
        self.assertEqual(adapters["C4 Indexer FP4 Quant"].instances, 30)
        self.assertEqual(adapters["C4 FP4 Paged MQA Logits"].instances, 30)
        self.assertEqual(adapters["C4 TopK Transform"].instances, 30)

    def test_decode_registry_uses_fp4_indexer_without_compression(self):
        adapters = {adapter.name: adapter for adapter in decode_adapters()}

        self.assertEqual(adapters["C4 Indexer Q Projection"].instances, 30)
        self.assertEqual(adapters["C4 Indexer Head Weight Projection"].instances, 30)
        self.assertEqual(adapters["C4 Indexer FP4 Quant"].instances, 30)
        self.assertEqual(adapters["C4 FP4 Paged MQA Logits"].instances, 30)
        self.assertEqual(adapters["C4 TopK Transform"].instances, 30)
        self.assertEqual(adapters["Sparse Decode Attention C4"].instances, 30)
        self.assertEqual(adapters["Sparse Decode Attention C128"].instances, 30)
        self.assertNotIn("Compressor WKV-Gate GEMM (C4)", adapters)
        self.assertNotIn("Compressor WKV-Gate GEMM (C128)", adapters)

    def test_registry_matches_full_shape_v4_pro_geometry(self):
        self.assertEqual(ATTN_HEADS, 128)
        self.assertEqual(INDEX_HEADS, 64)
        self.assertEqual(INDEX_TOPK, 1024)
        self.assertEqual(E_GLOBAL, 384)
        self.assertEqual(E_LOCAL, 16)
        self.assertEqual(MOE_INTERMEDIATE, 3072)
        self.assertEqual(MOE_TOPK, 6)

        for adapters in (prefill_adapters(), decode_adapters()):
            by_name = {adapter.name: adapter for adapter in adapters}
            self.assertEqual(by_name["Q RMSNorm + WQ_B"].shape, (1536, 65536))
            self.assertEqual(
                by_name["WO_A Grouped Projection"].shape, (16, 4096, 1024)
            )
            self.assertEqual(by_name["WO_B Projection"].shape, (16384, 7168))
            self.assertEqual(by_name["Routed Expert Fused MoE"].instances, 61)
            self.assertEqual(by_name["Routed Expert Fused MoE"].shape, (16, 7168, 3072))
            self.assertEqual(by_name["LM Head"].shape, (7168, 129280))
            self.assertNotEqual(by_name["Dense SWA Attention"].kind, "missing")

    def test_shape_descriptions_cover_full_shape_and_fused_moe(self):
        prefill = {adapter.name: adapter for adapter in prefill_adapters()}
        self.assertEqual(
            adapter_io_shapes(prefill["Q RMSNorm + WQ_B"], 1024, 1024),
            ("x=(1024,1536); weight=(65536,1536)", "y=(1024,65536)"),
        )
        self.assertEqual(
            adapter_io_shapes(prefill["Routed Expert Fused MoE"], 32, 16384),
            (
                "x=(32,7168); topk_ids/weights=(32,6); local_pairs=8; w13=(16,6144,7168); w2=(16,7168,3072)",
                "y=(32,7168)",
            ),
        )

    def test_ep_local_routed_pair_counts_match_requested_cases(self):
        self.assertEqual([local_routed_pairs(m) for m in (1024, 2048, 4096, 16, 32)], [256, 512, 1024, 4, 8])

    def test_prefill_context_defaults_to_sequence_length(self):
        self.assertEqual(case_context("prefill", 2048, None), 2048)
        self.assertEqual(case_context("decode", 16, None), 16384)
        self.assertEqual(case_context("prefill", 2048, 4096), 4096)

    def test_prefill_lm_head_uses_original_full_token_style(self):
        lm_head = {adapter.name: adapter for adapter in prefill_adapters()}["LM Head"]
        self.assertIsNone(lm_head.m_override)


if __name__ == "__main__":
    unittest.main()
