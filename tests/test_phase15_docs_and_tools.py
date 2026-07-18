import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.cli import build_parser
from benchmark_engine.reporting import (
    AtomicCsvTable,
    MODEL_PROJECTION_SCHEMA,
    PERFORMANCE_SAMPLES_SCHEMA,
    RESULTS_SCHEMA,
)
import yaml

from tests.reporting_fixtures import results_row


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Phase15DocumentationTests(unittest.TestCase):
    def test_every_reference_operator_has_a_nonempty_readme(self):
        references = ROOT / "operators" / "references"
        directories = sorted(path for path in references.iterdir() if path.is_dir())
        self.assertTrue(directories)
        for directory in directories:
            with self.subTest(operator=directory.name):
                readme = directory / "README.md"
                self.assertTrue(readme.is_file())
                text = readme.read_text(encoding="utf-8")
                self.assertTrue(text.startswith("# "))
                self.assertIn("reference", text.lower())

    def test_documentation_index_links_and_required_phase15_topics(self):
        index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
        for name in (
            "migration-llm-flops.md", "troubleshooting.md", "cli.md",
            "csv-schema.md", "result-layout.md", "testing.md",
        ):
            self.assertIn(name, index)
            self.assertTrue((ROOT / "docs" / name).is_file())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in (
            "Five-minute CPU check", "Reference and candidate", "Exit codes",
            "convert_legacy_results.py", "correctness", "performance", "all",
        ):
            self.assertIn(phrase, readme)

    def test_relative_links_and_yaml_examples_are_valid(self):
        markdown = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
        link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
        for document in markdown:
            text = document.read_text(encoding="utf-8")
            for target in link_pattern.findall(text):
                target = target.strip().strip("<>")
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                relative = target.split("#", 1)[0]
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent / relative).resolve().exists())
            for format_name, block in re.findall(
                r"```(json|yaml|yml)\n(.*?)\n```", text, flags=re.DOTALL
            ):
                with self.subTest(document=document.name, format=format_name):
                    parsed = json.loads(block) if format_name == "json" else yaml.safe_load(block)
                    self.assertIsInstance(parsed, (dict, list))

    def test_documented_engine_commands_parse(self):
        parser = build_parser()
        commands = (
            ("list",),
            ("validate", "--operator", "example_cpu_add"),
            ("env", "--json"),
            ("run", "--suite", "smoke", "--dry-run"),
            ("run", "--mode", "correctness", "--operator", "example_cpu_add"),
            ("run", "--mode", "performance", "--operator", "example_cpu_add"),
            ("run", "--mode", "all", "--operator", "example_cpu_add"),
            ("summarize", "results/example/candidate/evaluation"),
            ("compare", "--result", "current", "--baseline-result", "baseline"),
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNotNone(parser.parse_args(command).command)


class Phase15ToolContractTests(unittest.TestCase):
    def test_test_tiers_are_stable_and_dry_run_has_no_side_effects(self):
        script = ROOT / "tools" / "run_test_tier.py"
        with tempfile.TemporaryDirectory() as directory:
            for tier in ("cpu", "gpu-smoke", "b200-regression", "full"):
                result = subprocess.run(
                    [sys.executable, str(script), tier, "--dry-run", "--work-root", directory],
                    cwd=ROOT, text=True, capture_output=True, check=True,
                )
                payload = json.loads(result.stdout)
                self.assertEqual(payload["tier"], tier)
                self.assertTrue(payload["commands"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_regression_thresholds_are_predeclared_and_empty_compare_fails(self):
        module = load_tool("regress_deepseek_v4.py")
        self.assertEqual(module.RELATIVE_THRESHOLD, 0.10)
        self.assertEqual(module.ABSOLUTE_THRESHOLD_MS, 0.02)
        with self.assertRaisesRegex(module.RegressionError, "non-empty"):
            module._compare({}, {})
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "tools" / "regress_deepseek_v4.py"),
                    "--phase", "prefill", "--work-dir", directory, "--dry-run",
                ],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["timer_pair"]["legacy"], "graph_ms")
            self.assertEqual(payload["timer_pair"]["engine"], "cuda_graph median")
            self.assertEqual(payload["runs"], 20)

    def test_regression_reader_requires_schema_valid_formal_interleaved_samples(self):
        module = load_tool("regress_deepseek_v4.py")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            evaluation = work / "engine" / "op" / "candidate" / "evaluation"
            evaluation.mkdir(parents=True)
            result = results_row()
            result.update(
                mode="all",
                performance_status="passed",
                performance_formal=True,
                ranking_eligible=True,
                performance_gate_status="passed",
                requested_timer="cuda_graph",
                effective_timer="cuda_graph",
                reference_requested_timer="cuda_graph",
                reference_effective_timer="cuda_graph",
                candidate_median_ms=1.0,
                other_compute_processes_detected=False,
                telemetry_error=None,
            )
            AtomicCsvTable(evaluation / "results.csv", RESULTS_SCHEMA).append(result)
            samples = []
            for index in range(5):
                for role, order in (("reference", 2 * index), ("candidate", 2 * index + 1)):
                    samples.append(
                        {
                            "result_id": result["result_id"],
                            "implementation_role": role,
                            "reference_dtype": '{"output":"float32"}',
                            "candidate_dtype": '{"output":"float32"}',
                            "reference_shape": '{"output":[1]}',
                            "candidate_shape": '{"output":[1]}',
                            "sample_index": index,
                            "inner_iterations": 1,
                            "elapsed_ms": 1.0,
                            "per_call_ms": 1.0,
                            "order_index": order,
                            "requested_timer": "cuda_graph",
                            "effective_timer": "cuda_graph",
                        }
                    )
            AtomicCsvTable(
                evaluation / "performance_samples.csv", PERFORMANCE_SAMPLES_SCHEMA
            ).append_many(samples)
            AtomicCsvTable(
                evaluation / "model_projection.csv", MODEL_PROJECTION_SCHEMA
            ).append(
                {
                    "run_id": result["run_id"],
                    "evaluation_id": result["evaluation_id"],
                    "result_id": result["result_id"],
                    "suite_id": result["suite_id"],
                    "projection_id": "deepseek_v4_pro",
                    "phase": "prefill",
                    "quant_profile": "fp8_mxfp8",
                    "model_input": 1024,
                    "raw_context": 65536,
                    "operator_id": result["operator_id"],
                    "candidate_id": result["candidate_id"],
                    "case_id": result["case_id"],
                    "adapter_id": "fused_wq_a_wkv",
                    "display_name": "Fused WQ_A + WKV",
                    "backend": "DeepGEMM fp8_gemm_nt",
                    "kind": "fp8",
                    "legacy_shape": "[7168,2048]",
                    "instances": 61,
                    "implementation_role": "candidate",
                    "per_call_ms": 1.0,
                    "projected_model_ms": 61.0,
                    "status": "measured",
                }
            )
            measured = module._read_engine(work, "prefill", 5)
            self.assertEqual(measured[(1024, "Fused WQ_A + WKV")], 1.0)
            result_path = evaluation / "results.csv"
            result_path.write_text("bad,header\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                module._read_engine(work, "prefill", 5)


if __name__ == "__main__":
    unittest.main()
