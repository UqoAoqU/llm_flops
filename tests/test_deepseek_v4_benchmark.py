import csv
import tempfile
import unittest
from pathlib import Path

from print_deepseek_v4_quant_comparison import _load, compare_profile_rows

from deepseek_v4_benchmark import (
    ATTN_HEADS,
    C4_LAYERS,
    C128_LAYERS,
    DENSE_LAYERS,
    E_GLOBAL,
    E_LOCAL,
    INDEX_HEADS,
    INDEX_TOPK,
    MOE_INTERMEDIATE,
    MOE_TOPK,
    MODEL_LAYERS,
    BenchmarkRow,
    _moe_fp8_mxfp8_fn,
    adapter_io_shapes,
    case_context,
    compressed_context,
    decode_adapters,
    graph_ms,
    prefill_adapters,
    summarize_rows,
    local_routed_pairs,
    result_exit_code,
    write_result_csv,
)


LEGACY_CSV_HEADER = (
    "phase",
    "quant_profile",
    "environment_fingerprint",
    "m",
    "context",
    "operator",
    "backend",
    "instances",
    "call_ms",
    "model_ms",
    "pct",
    "status",
    "input_shape",
    "output_shape",
    "error",
)

PREFILL_NAMES = {
    "mxfp4": (
        "Fused WQ_A + WKV",
        "Q RMSNorm + WQ_B",
        "Compressor WKV-Gate GEMM (C4)",
        "Compressor WKV-Gate GEMM (C128)",
        "C4 Indexer Q Projection",
        "C4 Indexer Head Weight Projection",
        "C4 Indexer FP4 Quant",
        "C4 FP4 Paged MQA Logits",
        "C4 TopK Transform",
        "Sparse Prefill Attention",
        "Dense SWA Attention",
        "WO_A Grouped Projection",
        "WO_B Projection",
        "Routed Expert Fused MoE",
        "LM Head",
    ),
    "fp8_mxfp8": (
        "Fused WQ_A + WKV",
        "Q RMSNorm + WQ_B",
        "Compressor WKV-Gate GEMM (C4)",
        "Compressor WKV-Gate GEMM (C128)",
        "C4 Indexer Q Projection",
        "C4 Indexer Head Weight Projection",
        "C4 Indexer FP8 Quant",
        "C4 FP8 Paged MQA Logits",
        "C4 TopK Transform",
        "Sparse Prefill Attention",
        "Dense SWA Attention",
        "WO_A Grouped Projection",
        "WO_B Projection",
        "Routed Expert Fused MoE",
        "LM Head",
    ),
}

DECODE_NAMES = {
    "mxfp4": (
        "Fused WQ_A + WKV",
        "Q RMSNorm + WQ_B",
        "C4 Indexer Q Projection",
        "C4 Indexer Head Weight Projection",
        "C4 Indexer FP4 Quant",
        "C4 FP4 Paged MQA Logits",
        "C4 TopK Transform",
        "Sparse Decode Attention C4",
        "Sparse Decode Attention C128",
        "Dense SWA Attention",
        "WO_A Grouped Projection",
        "WO_B Projection",
        "Routed Expert Fused MoE",
        "LM Head",
    ),
    "fp8_mxfp8": (
        "Fused WQ_A + WKV",
        "Q RMSNorm + WQ_B",
        "C4 Indexer Q Projection",
        "C4 Indexer Head Weight Projection",
        "C4 Indexer FP8 Quant",
        "C4 FP8 Paged MQA Logits",
        "C4 TopK Transform",
        "Sparse Decode Attention C4",
        "Sparse Decode Attention C128",
        "Dense SWA Attention",
        "WO_A Grouped Projection",
        "WO_B Projection",
        "Routed Expert Fused MoE",
        "LM Head",
    ),
}

COMMON_ADAPTER_METADATA = {
    "Fused WQ_A + WKV":
        ("DeepGEMM fp8_gemm_nt", 61, "fp8", (7168, 2048), None),
    "Q RMSNorm + WQ_B":
        ("DeepGEMM fp8_gemm_nt", 61, "fp8", (1536, 65536), None),
    "Compressor WKV-Gate GEMM (C4)":
        ("DeepGEMM fp8_gemm_nt", 30, "fp8", (7168, 2048), None),
    "Compressor WKV-Gate GEMM (C128)":
        ("DeepGEMM fp8_gemm_nt", 30, "fp8", (7168, 1024), None),
    "C4 Indexer Q Projection":
        ("DeepGEMM fp8_gemm_nt", 30, "fp8", (1536, 65536), None),
    "C4 Indexer Head Weight Projection":
        ("cuBLAS BF16 GEMM", 30, "bf16", (7168, 64), None),
    "C4 Indexer FP4 Quant":
        ("SGLang fused RoPE/Hadamard FP4", 30, "fp4_quant", None, None),
    "C4 FP4 Paged MQA Logits":
        ("DeepGEMM fp8_fp4_paged_mqa_logits", 30, "fp4_logits", None, None),
    "C4 Indexer FP8 Quant":
        ("SGLang fused RoPE/Hadamard FP8", 30, "fp8_quant", None, None),
    "C4 FP8 Paged MQA Logits":
        ("DeepGEMM fp8_paged_mqa_logits", 30, "fp8_logits", None, None),
    "C4 TopK Transform":
        ("SGLang JIT topk_transform_512_v2", 30, "topk", None, None),
    "Sparse Prefill Attention":
        ("sgl-kernel FlashMLA sparse_fwd", 60, "prefill_attention", None, None),
    "Sparse Decode Attention C4":
        ("sgl-kernel FlashMLA dual-cache C4", 30, "decode_attention_c4", None, None),
    "Sparse Decode Attention C128":
        ("sgl-kernel FlashMLA dual-cache C128", 30, "decode_attention_c128", None, None),
    "WO_A Grouped Projection":
        ("cuBLAS grouped BF16 GEMM", 61, "grouped_bf16", (16, 4096, 1024), None),
    "WO_B Projection":
        ("DeepGEMM fp8_gemm_nt", 61, "fp8", (16384, 7168), None),
    "LM Head":
        ("DeepGEMM FP8 full vocab", 1, "fp8", (7168, 129280), None),
}

PHASE_ADAPTER_METADATA = {
    "prefill": {
        "Dense SWA Attention":
            ("sgl-kernel FlashMLA SWA-only", 1, "dense_prefill_attention", None, None),
    },
    "decode": {
        "Dense SWA Attention":
            ("sgl-kernel FlashMLA SWA-only", 1, "dense_decode_attention", None, None),
    },
}

PROFILE_ADAPTER_METADATA = {
    "mxfp4": {
        "Routed Expert Fused MoE": (
            "FlashInfer TRTLLM MXFP4",
            61,
            "moe_mxfp4",
            (16, 7168, 3072),
            None,
        ),
    },
    "fp8_mxfp8": {
        "Routed Expert Fused MoE": (
            "FlashInfer TRTLLM FP8 weight + MXFP8 activation",
            61,
            "moe_fp8_mxfp8",
            (16, 7168, 3072),
            None,
        ),
    },
}


class DeepSeekV4BenchmarkTest(unittest.TestCase):
    def assert_adapter_contract(self, phase, profile, adapters, expected_names):
        self.assertEqual(tuple(adapter.name for adapter in adapters), expected_names)
        expected_metadata = dict(COMMON_ADAPTER_METADATA)
        expected_metadata.update(PHASE_ADAPTER_METADATA[phase])
        expected_metadata.update(PROFILE_ADAPTER_METADATA[profile])
        for adapter in adapters:
            with self.subTest(phase=phase, profile=profile, adapter=adapter.name):
                self.assertEqual(
                    (
                        adapter.backend,
                        adapter.instances,
                        adapter.kind,
                        adapter.shape,
                        adapter.m_override,
                    ),
                    expected_metadata[adapter.name],
                )

    def test_prefill_adapter_order_and_metadata_are_the_legacy_contract(self):
        for profile, expected_names in PREFILL_NAMES.items():
            with self.subTest(profile=profile):
                self.assert_adapter_contract(
                    "prefill",
                    profile,
                    prefill_adapters(profile),
                    expected_names,
                )

    def test_decode_adapter_order_and_metadata_are_the_legacy_contract(self):
        for profile, expected_names in DECODE_NAMES.items():
            with self.subTest(profile=profile):
                self.assert_adapter_contract(
                    "decode",
                    profile,
                    decode_adapters(profile),
                    expected_names,
                )

    def test_layer_counts_are_the_deepseek_v4_projection_contract(self):
        self.assertEqual(C4_LAYERS, 30)
        self.assertEqual(C128_LAYERS, 30)
        self.assertEqual(DENSE_LAYERS, 1)
        self.assertEqual(MODEL_LAYERS, 61)
        self.assertEqual(
            MODEL_LAYERS,
            C4_LAYERS + C128_LAYERS + DENSE_LAYERS,
        )

    def test_graph_ms_characterizes_cuda_graph_call_order(self):
        events = []

        class FakeGraph:
            def __init__(self):
                events.append("CUDAGraph")

            def replay(self):
                events.append("replay")

        class FakeCapture:
            def __enter__(self):
                events.append("capture_enter")

            def __exit__(self, exc_type, exc_value, traceback):
                events.append("capture_exit")

        class FakeEvent:
            def __init__(self, label, enable_timing):
                self.label = label
                events.append(f"Event:{label}:{enable_timing}")

            def record(self):
                events.append(f"record:{self.label}")

            def elapsed_time(self, other):
                events.append(f"elapsed:{self.label}:{other.label}")
                return 20.0

        class FakeCuda:
            def __init__(self):
                self.event_count = 0

            def synchronize(self):
                events.append("synchronize")

            def CUDAGraph(self):
                return FakeGraph()

            def graph(self, graph):
                self.assert_is_fake_graph(graph)
                events.append("capture_context")
                return FakeCapture()

            def Event(self, enable_timing):
                label = "start" if self.event_count == 0 else "end"
                self.event_count += 1
                return FakeEvent(label, enable_timing)

            @staticmethod
            def assert_is_fake_graph(graph):
                if not isinstance(graph, FakeGraph):
                    raise AssertionError("capture must receive the CUDAGraph")

        class FakeTorch:
            cuda = FakeCuda()

        def fn():
            events.append("fn")

        self.assertEqual(graph_ms(fn, FakeTorch(), warmup=2, runs=4), 5.0)
        self.assertEqual(
            events,
            [
                "fn",
                "fn",
                "synchronize",
                "CUDAGraph",
                "capture_context",
                "capture_enter",
                "fn",
                "fn",
                "fn",
                "fn",
                "capture_exit",
                "synchronize",
                "Event:start:True",
                "Event:end:True",
                "record:start",
                "replay",
                "record:end",
                "synchronize",
                "elapsed:start:end",
            ],
        )

    def test_summary_model_time_percentages_and_zero_total(self):
        rows = [
            BenchmarkRow("attention", "FlashMLA", 2, 1.5, "executed"),
            BenchmarkRow("projection", "DeepGEMM", 1, 1.0, "executed"),
            BenchmarkRow("missing timing", "DeepGEMM", 100, None, "executed"),
            BenchmarkRow("unavailable", "FlashInfer", 61, 9.0, "unavailable"),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary.total_ms, 4.0)
        self.assertEqual(
            summary.percent_by_name,
            {"attention": 75.0, "projection": 25.0},
        )
        self.assertAlmostEqual(sum(summary.percent_by_name.values()), 100.0)

        zero = summarize_rows(
            [BenchmarkRow("zero", "fake", 3, 0.0, "executed")]
        )
        self.assertEqual(zero.total_ms, 0.0)
        self.assertEqual(zero.percent_by_name, {})

    def test_result_exit_code_requires_every_row_to_be_executed(self):
        executed = BenchmarkRow("attention", "FlashMLA", 2, 1.5, "executed")
        no_timing = BenchmarkRow("projection", "DeepGEMM", 1, None, "executed")
        unavailable = BenchmarkRow("moe", "FlashInfer", 61, None, "unavailable")
        failed = BenchmarkRow("indexer", "DeepGEMM", 30, None, "failed")
        self.assertEqual(result_exit_code([executed, no_timing]), 0)
        self.assertEqual(result_exit_code([executed, unavailable]), 1)
        self.assertEqual(result_exit_code([executed, failed]), 1)

    def test_result_csv_locks_header_formatting_and_unavailable_blanks(self):
        rows = [
            BenchmarkRow(
                "attention",
                "FlashMLA",
                2,
                1.23456789,
                "executed",
                input_shape="q=(16,128,512)",
                output_shape="y=(16,128,512)",
            ),
            BenchmarkRow(
                "moe",
                "FlashInfer",
                61,
                None,
                "unavailable",
                "missing backend",
                "x=(16,7168)",
                "y=(16,7168)",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            write_result_csv(
                path,
                "decode",
                "mxfp4",
                16,
                65536,
                rows,
                "4286ebdeb11b",
            )
            with path.open(newline="") as source:
                reader = csv.DictReader(source)
                written = list(reader)
                self.assertEqual(tuple(reader.fieldnames), LEGACY_CSV_HEADER)

        self.assertEqual(
            written[0],
            {
                "phase": "decode",
                "quant_profile": "mxfp4",
                "environment_fingerprint": "4286ebdeb11b",
                "m": "16",
                "context": "65536",
                "operator": "attention",
                "backend": "FlashMLA",
                "instances": "2",
                "call_ms": "1.234568",
                "model_ms": "2.469136",
                "pct": "100.0000",
                "status": "executed",
                "input_shape": "q=(16,128,512)",
                "output_shape": "y=(16,128,512)",
                "error": "",
            },
        )
        self.assertEqual(written[1]["call_ms"], "")
        self.assertEqual(written[1]["model_ms"], "")
        self.assertEqual(written[1]["pct"], "0.0000")
        self.assertEqual(written[1]["environment_fingerprint"], "4286ebdeb11b")
        self.assertEqual(written[1]["input_shape"], "x=(16,7168)")
        self.assertEqual(written[1]["output_shape"], "y=(16,7168)")
        self.assertEqual(written[1]["error"], "missing backend")

    def test_legacy_csv_fixture_is_readable_by_the_comparison_path(self):
        fixture = Path(__file__).parent / "fixtures" / "legacy_deepseek_result.csv"
        with fixture.open(newline="") as source:
            reader = csv.DictReader(source)
            fixture_rows = list(reader)
            self.assertEqual(tuple(reader.fieldnames), LEGACY_CSV_HEADER)

        self.assertEqual(_load(fixture), fixture_rows)
        self.assertEqual(
            {row["status"] for row in fixture_rows},
            {"executed", "unavailable"},
        )
        executed = [row for row in fixture_rows if row["status"] == "executed"]
        comparison = compare_profile_rows(executed, executed)
        self.assertEqual(comparison.case, ("prefill", 1024, 65536))
        self.assertEqual(comparison.mxfp4_total_ms, 6.0)
        self.assertEqual(comparison.fp8_mxfp8_total_ms, 6.0)
        self.assertEqual(comparison.speedup, 1.0)

    def test_result_csv_records_environment_fingerprint(self):
        row = BenchmarkRow("attention", "FlashMLA", 2, 1.5, "executed", "")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            write_result_csv(path, "prefill", "fp8_mxfp8", 16, 512, [row], "abc123")
            with path.open(newline="") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["environment_fingerprint"], "abc123")

    def test_unavailable_row_fails_formal_result(self):
        executed = BenchmarkRow("attention", "FlashMLA", 2, 1.5, "executed", "")
        unavailable = BenchmarkRow("moe", "FlashInfer", 61, None, "unavailable", "missing")
        self.assertEqual(result_exit_code([executed]), 0)
        self.assertEqual(result_exit_code([executed, unavailable]), 1)

    def test_comparison_requires_matching_cases_and_maps_indexer_names(self):
        mx_rows = [
            {
                "phase": "prefill",
                "m": "1024",
                "context": "65536",
                "operator": "C4 FP4 Paged MQA Logits",
                "instances": "30",
                "model_ms": "6.0",
            }
        ]
        fp8_rows = [
            {
                "phase": "prefill",
                "m": "1024",
                "context": "65536",
                "operator": "C4 FP8 Paged MQA Logits",
                "instances": "30",
                "model_ms": "4.0",
            }
        ]

        comparison = compare_profile_rows(mx_rows, fp8_rows)
        self.assertEqual(comparison.case, ("prefill", 1024, 65536))
        self.assertEqual(comparison.speedup, 1.5)
        self.assertEqual(
            comparison.operators[0].name, "C4 Paged MQA Logits"
        )

        fp8_rows[0]["context"] = "16384"
        with self.assertRaisesRegex(ValueError, "case mismatch"):
            compare_profile_rows(mx_rows, fp8_rows)

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
