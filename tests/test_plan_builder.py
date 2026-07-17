import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from benchmark_engine.config import resolve_evaluation_config
from benchmark_engine.planning import PlanBuilder, PlanningError
from benchmark_engine.registry import FilesystemRegistry, RegistryIssue
from benchmark_engine.selectors import SelectorError, Selectors
from benchmark_engine.suite import SuiteConfig
from tests.registry_fixtures import (
    OPERATOR_ID,
    write_candidate,
    write_reference,
    write_registry,
)


SUITE = SuiteConfig(1, "regression", ("*",), (), ("smoke",), "all", (0, 4), 11, 7)


class PlanBuilderTest(unittest.TestCase):
    def builder(self):
        return PlanBuilder(
            now=lambda: datetime(2026, 7, 16, 9, 15, tzinfo=timezone.utc),
            random_hex=lambda _size: "1" * 32,
        )

    def test_stable_expansion_ids_configs_and_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_registry(root)
            snapshot = FilesystemRegistry(root).discover()
            builder = PlanBuilder(
                now=lambda: datetime(2026, 7, 16, 9, 15, tzinfo=timezone.utc),
                random_hex=lambda _size: "1" * 32,
            )
            plan = builder.build(snapshot, SUITE, Selectors(), environment_fingerprint="a" * 64, output_root=root / "results")
            self.assertEqual(len(plan.jobs), 4)
            order = [(job.identity.candidate_id, job.case.case_id, job.case.seed) for job in plan.jobs]
            self.assertEqual(order, sorted(order))
            self.assertEqual(len({job.result_id for job in plan.jobs}), 4)
            for candidate_id in {job.identity.candidate_id for job in plan.jobs}:
                self.assertEqual(len({job.identity.evaluation_id for job in plan.jobs if job.identity.candidate_id == candidate_id}), 1)
            self.assertTrue(all(job.resolved_config.performance_samples == 11 for job in plan.jobs))
            self.assertEqual(type(plan).from_dict(json.loads(json.dumps(plan.to_dict()))), plan)
            self.assertFalse((root / "results").exists())

    def test_unselected_bad_operator_and_sibling_candidate_do_not_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_reference(root)
            _, good = write_candidate(
                root, "good", "20260716T081500Z", with_manifest=False
            )
            bad_root, bad = write_candidate(
                root, "bad", "20260716T081600Z", with_manifest=False
            )
            (bad_root / "implementation.py").write_text(
                "def operator(left, right):\n    return left - right\n",
                encoding="utf-8",
            )
            broken = write_reference(root, "broken_cpu_add")
            manifest = broken / "operator.yaml"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "schema_version: 1", "schema_version: 2"
                ),
                encoding="utf-8",
            )
            snapshot = FilesystemRegistry(root).discover()
            self.assertTrue(snapshot.issues)

            plan = self.builder().build(
                snapshot,
                SUITE,
                Selectors(operators=(OPERATOR_ID,), candidates=(good,)),
                environment_fingerprint="c" * 64,
                output_root=root / "results",
            )
            self.assertEqual(
                {job.identity.candidate_id for job in plan.jobs}, {good}
            )
            self.assertNotIn(bad, {job.identity.candidate_id for job in plan.jobs})

    def test_global_and_selected_registry_issues_still_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_reference(root)
            _, good = write_candidate(
                root, "good", "20260716T081500Z", with_manifest=False
            )
            bad_root, bad = write_candidate(
                root, "bad", "20260716T081600Z", with_manifest=False
            )
            (bad_root / "implementation.py").write_text(
                "def operator(left, right):\n    return left - right\n",
                encoding="utf-8",
            )
            broken = write_reference(root, "broken_cpu_add")
            manifest = broken / "operator.yaml"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "schema_version: 1", "schema_version: 2"
                ),
                encoding="utf-8",
            )
            snapshot = FilesystemRegistry(root).discover()
            for selectors in (
                Selectors(operators=(OPERATOR_ID,), candidates=(bad,)),
                Selectors(operators=("broken_cpu_add",)),
            ):
                with self.subTest(selectors=selectors), self.assertRaisesRegex(
                    PlanningError, "registry is invalid for selected scope"
                ):
                    self.builder().build(
                        snapshot,
                        SUITE,
                        selectors,
                        environment_fingerprint="d" * 64,
                        output_root=root / "results",
                    )

            global_snapshot = replace(
                snapshot,
                issues=snapshot.issues
                + (
                    RegistryIssue(
                        "registry.global",
                        root / "global.yaml",
                        "$",
                        "global failure",
                    ),
                ),
            )
            with self.assertRaisesRegex(PlanningError, "registry.global"):
                self.builder().build(
                    global_snapshot,
                    SUITE,
                    Selectors(operators=(OPERATOR_ID,), candidates=(good,)),
                    environment_fingerprint="e" * 64,
                    output_root=root / "results",
                )

    def test_multi_operator_candidate_miss_fails_the_whole_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_registry(root)
            write_reference(root, "second_cpu_add")
            snapshot = FilesystemRegistry(root).discover()
            with self.assertRaisesRegex(
                SelectorError, "no candidates for operator: second_cpu_add"
            ):
                self.builder().build(
                    snapshot,
                    SUITE,
                    Selectors(),
                    environment_fingerprint="f" * 64,
                    output_root=root / "results",
                )

    def test_existing_path_requires_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_registry(root)
            snapshot = FilesystemRegistry(root).discover()
            builder = PlanBuilder(now=lambda: datetime(2026, 7, 16, 9, 15, tzinfo=timezone.utc), random_hex=lambda _size: "2" * 32)
            first = builder.build(snapshot, SUITE, Selectors(candidates=("task_one*",)), environment_fingerprint="b" * 64, output_root=root / "results")
            first.jobs[0].output_dir.mkdir(parents=True)
            with self.assertRaisesRegex(PlanningError, "already exists"):
                builder.build(snapshot, SUITE, Selectors(candidates=("task_one*",)), environment_fingerprint="b" * 64, output_root=root / "results")
            resumed = builder.build(snapshot, SUITE, Selectors(candidates=("task_one*",)), environment_fingerprint="b" * 64, output_root=root / "results", resume=True)
            self.assertEqual(len(resumed.jobs), 2)

    def test_config_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_registry(root)
            manifest = FilesystemRegistry(root).discover().operator_manifests["fixture_cpu_add"]
            operator_fallback = SuiteConfig(
                1, "s", ("*",), (), ("smoke",), "performance", (3,), 9, 8
            )
            resolved = resolve_evaluation_config(manifest, operator_fallback)
            self.assertEqual(resolved.performance_timer, "wall_clock")
            self.assertEqual(resolved.performance_warmup, 0)
            self.assertEqual(resolved.performance_samples, 9)
            self.assertEqual(resolved.performance_inner_iterations, 8)

            suite_override = SuiteConfig(
                1,
                "s",
                ("*",),
                (),
                ("smoke",),
                "performance",
                (3,),
                9,
                8,
                performance_warmup=4,
                performance_timer="cuda_event",
            )
            resolved = resolve_evaluation_config(manifest, suite_override)
            self.assertEqual(resolved.performance_timer, "cuda_event")
            self.assertEqual(resolved.performance_warmup, 4)
            self.assertEqual(resolved.performance_samples, 9)
            self.assertEqual(resolved.performance_inner_iterations, 8)

            resolved = resolve_evaluation_config(
                manifest,
                suite_override,
                mode="correctness",
                seeds=(5,),
                performance_timer="cuda_graph",
                performance_warmup=6,
                performance_samples=11,
                performance_inner_iterations=12,
            )
            self.assertEqual(resolved.mode, "correctness")
            self.assertEqual(resolved.correctness_seeds, (5,))
            self.assertEqual(resolved.performance_timer, "cuda_graph")
            self.assertEqual(resolved.performance_warmup, 6)
            self.assertEqual(resolved.performance_samples, 11)
            self.assertEqual(resolved.performance_inner_iterations, 12)
            self.assertEqual(resolved.performance_timeout_s, 10)


if __name__ == "__main__":
    unittest.main()
