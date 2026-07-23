from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.engine import _projection_status, build_dry_run_plan
from benchmark_engine.models import CorrectnessStatus, PerformanceStatus
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.projection import DEEPSEEK_V4_PROJECTION
from benchmark_engine.selectors import Selectors
from benchmark_engine.reporting import ArtifactWriter, MODEL_PROJECTION_SCHEMA, RESULTS_SCHEMA, summarize_run
from benchmark_engine.reporting.csv_writer import AtomicCsvTable
from benchmark_engine.reporting.summary import _projection_lines
from tests.reporting_fixtures import manifest, results_row


ROOT = Path(__file__).resolve().parents[1]


def _legacy():
    spec = importlib.util.spec_from_file_location("legacy_deepseek_v4", ROOT / "deepseek_v4_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DeepSeekV4ProjectionTests(unittest.TestCase):
    def _plan(self, suite):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return build_dry_run_plan(
            ROOT, suite, Selectors(), output_root=Path(temporary.name) / "results"
        )

    def test_prefill_and_decode_suites_expand_real_phase_shapes(self):
        high_priority = {
            "deepseek_v4_aiter_block_fp8_gemm": 2,
            "deepseek_v4_fused_qk_norm_rope_store": 2,
            "deepseek_v4_c4_c128_compressor": 4,
            "deepseek_v4_aiter_c4_paged_mqa_logits": 2,
            "deepseek_v4_tilelang_sparse_attention": 5,
            "deepseek_v4_aiter_fp8_fused_moe": 2,
        }
        for suite, phase in (
            ("deepseek_v4_prefill", "prefill"),
            ("deepseek_v4_decode", "decode"),
        ):
            with self.subTest(suite=suite):
                plan = self._plan(suite)
                self.assertEqual(len(plan.jobs), sum(high_priority.values()))
                self.assertEqual({job.case.symbols["phase"] for job in plan.jobs}, {phase})
                actual = {}
                for job in plan.jobs:
                    operator_id = job.identity.operator_id
                    actual[operator_id] = actual.get(operator_id, 0) + 1
                self.assertEqual(actual, high_priority)

    def test_projection_registry_is_a_lossless_legacy_adapter_mapping(self):
        legacy = _legacy()
        for phase, adapters in (("prefill", legacy.prefill_adapters("fp8_mxfp8")),
                                ("decode", legacy.decode_adapters("fp8_mxfp8"))):
            mappings = DEEPSEEK_V4_PROJECTION.mappings(phase, "fp8_mxfp8")
            actual = [(item.display_name, item.backend, item.instances, item.shape, item.kind) for item in mappings]
            expected = [(item.name, item.backend, item.instances, item.shape, item.kind) for item in adapters]
            self.assertEqual(actual, expected)

    def test_projection_is_only_multiplication_and_missing_is_explicit(self):
        mappings = DEEPSEEK_V4_PROJECTION.mappings("prefill", "fp8_mxfp8")
        fused = next(item for item in mappings if item.adapter_id == "fused_wq_a_wkv")
        self.assertEqual(fused.instances * 0.125, 7.625)
        missing = {item.adapter_id for item in mappings if item.operator_id is None}
        self.assertEqual(missing, {"c4_indexer_head_weight", "dense_swa_attention",
                                   "wo_a_grouped_projection"})
        self.assertEqual(DEEPSEEK_V4_PROJECTION.mappings("prefill", "mxfp4"), ())

    def test_run_summary_aggregates_projection_and_never_counts_missing_as_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            created = manifest()
            writer = ArtifactWriter(root)
            writer.initialize(created)
            directory = writer.evaluation_dir(created.identity)
            row = results_row()
            AtomicCsvTable(directory / RESULTS_SCHEMA.filename, RESULTS_SCHEMA).append(row)
            projection = {
                "run_id": created.identity.run_id, "evaluation_id": created.identity.evaluation_id,
                "result_id": row["result_id"], "suite_id": "deepseek_v4_prefill",
                "projection_id": "deepseek_v4_pro", "phase": "prefill",
                "quant_profile": "fp8_mxfp8", "model_input": 1024, "raw_context": 65536,
                "operator_id": "deepseek_v4_fp8_gemm_nt", "candidate_id": created.identity.candidate_id,
                "case_id": "prefill__fused_wq_a_wkv__m1024__ctx65536__fp8_mxfp8",
                "adapter_id": "fused_wq_a_wkv", "display_name": "Fused WQ_A + WKV",
                "backend": "DeepGEMM fp8_gemm_nt", "kind": "fp8", "legacy_shape": "[7168, 2048]", "instances": 61,
                "implementation_role": "candidate", "per_call_ms": 0.125,
                "projected_model_ms": 7.625, "status": "measured",
            }
            AtomicCsvTable(directory / MODEL_PROJECTION_SCHEMA.filename,
                           MODEL_PROJECTION_SCHEMA).append(projection)
            reference = dict(projection, implementation_role="reference", per_call_ms=0.25,
                             projected_model_ms=15.25)
            AtomicCsvTable(directory / MODEL_PROJECTION_SCHEMA.filename,
                           MODEL_PROJECTION_SCHEMA).append(reference)
            summary = summarize_run(root, created.identity.run_id)
            self.assertIn("Candidate measured partial total: 7.625000", summary)
            self.assertIn("Reference measured partial total: 15.250000", summary)
            self.assertIn("Q RMSNorm + WQ_B", summary)
            self.assertIn("missing", summary)

    def test_projection_status_preserves_unsupported_and_numeric_failure(self):
        self.assertEqual(
            _projection_status(CorrectnessStatus.UNSUPPORTED, PerformanceStatus.SKIPPED,
                               "layout unavailable", None),
            ("unsupported", "layout unavailable"),
        )
        self.assertEqual(
            _projection_status(CorrectnessStatus.FAILED, PerformanceStatus.SKIPPED, None, None),
            ("correctness_failed", "failed"),
        )

    def test_projection_summary_combines_unique_per_operator_candidates(self):
        base = {
            "phase": "prefill", "quant_profile": "fp8_mxfp8", "model_input": "1024",
            "raw_context": "65536", "adapter_id": "fused_wq_a_wkv",
            "implementation_role": "candidate", "status": "measured", "reason": "",
            "per_call_ms": "1.0", "projected_model_ms": "61.0", "_seed": "0",
            "evaluation_id": "evaluation_a", "result_id": "result_a",
        }
        rows = [dict(base, candidate_id="candidate_a"),
                dict(base, adapter_id="q_rmsnorm_wq_b", candidate_id="candidate_b",
                     result_id="result_b", per_call_ms="2.0", projected_model_ms="122.0")]
        first = "\n".join(_projection_lines(rows, complete_model=True))
        second = "\n".join(_projection_lines(list(reversed(rows)), complete_model=True))
        self.assertEqual(first, second)
        self.assertEqual(first.count("### prefill"), 1)
        self.assertIn("| Fused WQ_A + WKV | candidate_a |", first)
        self.assertIn("| Q RMSNorm + WQ_B | candidate_b |", first)
        self.assertIn("Candidate measured partial total: 183.000000", first)
        self.assertIn("Ambiguous candidates: none", first)

    def test_projection_summary_excludes_ambiguous_adapter_and_separates_seeds(self):
        base = {
            "phase": "prefill", "quant_profile": "fp8_mxfp8", "model_input": "1024",
            "raw_context": "65536", "adapter_id": "fused_wq_a_wkv",
            "implementation_role": "candidate", "status": "measured", "reason": "",
            "per_call_ms": "1.0", "projected_model_ms": "61.0",
            "evaluation_id": "evaluation_a", "_seed": "0", "result_id": "result_a",
        }
        ambiguous_rows = [dict(base, candidate_id="candidate_a"),
                          dict(base, candidate_id="candidate_b", result_id="result_b",
                               projected_model_ms="122.0"),
                          dict(base, candidate_id="candidate_a", implementation_role="reference",
                               result_id="reference_a", projected_model_ms="30.5"),
                          dict(base, candidate_id="candidate_b", implementation_role="reference",
                               result_id="reference_b", projected_model_ms="31.0")]
        ambiguous = "\n".join(_projection_lines(ambiguous_rows, complete_model=True))
        self.assertIn("candidate_a, candidate_b", ambiguous)
        self.assertIn("| ambiguous |", ambiguous)
        self.assertIn("Candidate measured partial total: 0.000000", ambiguous)
        self.assertIn("Reference measured partial total: 0.000000", ambiguous)
        self.assertIn("Ambiguous candidates: Fused WQ_A + WKV (candidate_a, candidate_b)", ambiguous)

        seeded_rows = [dict(base, candidate_id="candidate_a"),
                       dict(base, candidate_id="candidate_a", result_id="result_seed_1",
                            _seed="1", projected_model_ms="122.0")]
        seeded = "\n".join(_projection_lines(seeded_rows, complete_model=True))
        self.assertEqual(seeded.count("### prefill"), 2)
        self.assertIn("seed=0", seeded)
        self.assertIn("seed=1", seeded)
        self.assertEqual(seeded.count("Ambiguous candidates: none"), 2)
        self.assertIn("Candidate measured partial total: 61.000000", seeded)
        self.assertIn("Candidate measured partial total: 122.000000", seeded)

    def test_stale_incomplete_projection_partition_is_recoverably_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            table = AtomicCsvTable(Path(temporary) / MODEL_PROJECTION_SCHEMA.filename,
                                   MODEL_PROJECTION_SCHEMA)
            base = {
                "run_id": "run_0123456789abcdef", "evaluation_id": "evaluation",
                "result_id": "result", "suite_id": "deepseek_v4_prefill",
                "projection_id": "deepseek_v4_pro", "phase": "prefill",
                "quant_profile": "fp8_mxfp8", "model_input": 1024, "raw_context": 65536,
                "operator_id": "deepseek_v4_fp8_gemm_nt", "candidate_id": "candidate",
                "case_id": "case", "adapter_id": "fused_wq_a_wkv",
                "display_name": "Fused WQ_A + WKV", "backend": "DeepGEMM fp8_gemm_nt",
                "kind": "fp8", "legacy_shape": "[7168, 2048]", "instances": 61, "implementation_role": "candidate",
                "per_call_ms": 1.0, "projected_model_ms": 61.0, "status": "measured",
            }
            table.append(base)
            resumed = dict(base, per_call_ms=0.5, projected_model_ms=30.5)
            table.replace_partitions((resumed,), partition_fields=("result_id",))
            rows = table.read_rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["per_call_ms"], "0.5")


class BoundedProjectionOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        cls.torch = torch

    def _load_module(self, operator):
        path = ROOT / "operators" / "references" / operator / "spec.py"
        spec = importlib.util.spec_from_file_location(f"test_{operator}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def _load_spec(self, operator):
        return self._load_module(operator).SPEC

    def test_gemm_and_topk_large_outputs_are_bounded_and_detect_changes(self):
        torch = self.torch
        for operator, tensor in (
            ("deepseek_v4_fp8_gemm_nt", torch.arange(1024, dtype=torch.float32).reshape(32, 32)),
            ("deepseek_v4_topk_transform", torch.arange(1024, dtype=torch.int32).reshape(4, 256)),
        ):
            with self.subTest(operator=operator):
                spec = self._load_spec(operator)
                reference = spec.normalize_output(tensor)
                control = spec.normalize_output(tensor.clone())
                wrong_tensor = tensor.clone()
                wrong_value = wrong_tensor.reshape(-1)
                if wrong_tensor.is_floating_point():
                    # The DeepGEMM contract intentionally has rtol=1e-2 and
                    # atol=1e-1. A +1 perturbation at ~1023 is valid, so use a
                    # finite sign-changing error that is unambiguously beyond
                    # the production tolerance.
                    wrong_value[-1] = -wrong_value[-1] - 100.0
                else:
                    wrong_value[-1] += 1
                wrong = spec.normalize_output(wrong_tensor)
                self.assertLessEqual(sum(leaf.size for leaf in reference.leaves), 256)
                case = (next(case for case in spec.cases() if "performance_only" in case.tags)
                        if operator == "deepseek_v4_topk_transform" else spec.cases()[0])
                self.assertTrue(spec.comparator(case).compare(reference, control).passed)
                self.assertFalse(spec.comparator(case).compare(reference, wrong).passed)

    def test_paged_performance_payload_uses_bounded_cross_check_and_state_guard(self):
        torch = self.torch
        module = self._load_module("deepseek_v4_fp8_paged_mqa_logits")
        spec = module.SPEC

        def leaf(path, values, shape=None):
            values = tuple(values)
            return OutputLeaf(path, values, "torch.float32", shape or (len(values),),
                              (1,), "strided", "cpu")

        output = leaf("output", (1.0, 2.0), (1024, 16384))
        state = (leaf("state.cache_head", (3.0,)), leaf("state.cache_tail", (4.0,)),
                 leaf("state.block_table_head", (0.0,)), leaf("state.block_table_tail", (1.0,)))
        reference = OutputBundle((output, *state))
        control = OutputBundle((leaf("output", (1.0, 2.0), (1024, 16384)), *state))
        performance_case = next(case for case in spec.cases() if "performance_only" in case.tags)
        comparator = spec.comparator(performance_case)
        self.assertTrue(comparator.compare(reference, control).passed)
        wrong_output = OutputBundle((leaf("output", (1.0, 9.0), (1024, 16384)), *state))
        self.assertFalse(comparator.compare(reference, wrong_output).passed)
        mutated_state = (*state[:-1], leaf("state.block_table_tail", (99.0,)))
        self.assertFalse(comparator.compare(reference, OutputBundle((output, *mutated_state))).passed)

        bounded = module._bounded_observed_state(torch.arange(2048), torch.arange(2048))
        self.assertNotIn("semantic_oracle", bounded)
        self.assertLessEqual(sum(value.numel() for value in bounded.values()), 256)

        small_case = next(case for case in spec.cases() if case.case_id == "smoke_tail_page")
        self.assertFalse(spec.comparator(small_case).compare(reference, control).passed)
        oracle_state = (*state, leaf("state.semantic_oracle", (1.0, 2.0)),
                        leaf("state.oracle_indices", (0.0, 1.0)))
        with_oracle = OutputBundle((output, *oracle_state))
        self.assertTrue(spec.comparator(small_case).compare(with_oracle, with_oracle).passed)

    def test_indexer_derived_payload_is_bounded_per_leaf(self):
        torch = self.torch
        spec = self._load_spec("deepseek_v4_indexer_fp8_quant")
        q = torch.arange(1024, dtype=torch.uint8).reshape(8, 128)
        weights = torch.ones((8, 1), dtype=torch.float32)
        output = spec.normalize_output((q, weights))
        self.assertEqual({leaf.path for leaf in output.leaves},
                         {"output.codes", "output.values", "output.weights", "output.effective"})
        self.assertTrue(all(leaf.size <= 256 for leaf in output.leaves))


if __name__ == "__main__":
    unittest.main()
