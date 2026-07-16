import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from benchmark_engine.models import (
    CaseSpec,
    CorrectnessStatus,
    EvaluationIdentity,
    EvaluationJob,
    EvaluationPlan,
    ImplementationSpec,
    InputBundle,
    OutputBundle,
    OutputLeaf,
    PerformanceStatus,
    ResultStatus,
)


class ModelContractTest(unittest.TestCase):
    def setUp(self):
        self.reference = ImplementationSpec(
            operator_id="demo_op",
            implementation_id="reference",
            role="reference",
            root=Path("operators/references/demo_op"),
            entrypoint="implementation:operator",
            source_hash="a" * 64,
            manifest_version=1,
        )
        self.candidate = ImplementationSpec(
            operator_id="demo_op",
            implementation_id="candidate_1",
            role="candidate",
            root=Path("operators/candidates/demo_op/candidate_1"),
            entrypoint="implementation:operator",
            source_hash="b" * 64,
            manifest_version=1,
        )
        self.case = CaseSpec(
            case_id="small",
            symbols={"m": 8, "transpose": False},
            seed=17,
            tags=frozenset({"smoke", "cpu"}),
            timeout_s=30,
        )
        self.identity = EvaluationIdentity(
            run_id="run-1",
            evaluation_id="eval-1",
            operator_id="demo_op",
            candidate_id="candidate_1",
        )
        self.job = EvaluationJob(
            identity=self.identity,
            reference=self.reference,
            candidate=self.candidate,
            case=self.case,
            mode="all",
            output_dir=Path("results/demo_op/candidate_1/eval-1"),
        )

    def assert_json_round_trip(self, value):
        encoded = value.to_dict()
        wire = json.loads(json.dumps(encoded, allow_nan=False))
        self.assertEqual(type(value).from_dict(wire), value)

    def test_all_data_models_are_frozen(self):
        values = [
            self.reference,
            self.case,
            InputBundle(),
            OutputLeaf("output", "float32", (2, 4), "cpu"),
            OutputBundle(()),
            self.identity,
            self.job,
            EvaluationPlan("run-1", "all", (self.job,), "env-1"),
        ]
        for value in values:
            with self.subTest(model=type(value).__name__):
                with self.assertRaises(FrozenInstanceError):
                    value.extra = "mutation"

    def test_stable_enum_values(self):
        self.assertEqual(CorrectnessStatus.PASSED.value, "passed")
        self.assertEqual(CorrectnessStatus.FAILED.value, "failed")
        self.assertEqual(PerformanceStatus.UNSTABLE.value, "unstable")
        self.assertEqual(PerformanceStatus.SKIPPED.value, "skipped")
        self.assertEqual(ResultStatus.PLANNED.value, "planned")
        self.assertEqual(ResultStatus.RUNNING.value, "running")
        self.assertEqual(ResultStatus.CRASHED.value, "crashed")

    def test_models_round_trip_through_json(self):
        bundle = InputBundle(
            args=(1, ("nested", 2)),
            kwargs={
                "path": Path("fixture/input.json"),
                "status": ResultStatus.PLANNED,
                "list": [1, 2],
            },
            observed_state={"names": frozenset({"cache", "output"})},
        )
        leaf = OutputLeaf(
            path="output.logits",
            dtype="float32",
            shape=(2, 4),
            device="cpu",
            stride=(4, 1),
            layout="strided",
            comparator="floating",
            metadata={"source": Path("candidate")},
        )
        values = [
            self.reference,
            self.case,
            bundle,
            leaf,
            OutputBundle((leaf,), {"kind": "mapping"}),
            self.identity,
            self.job,
            EvaluationPlan("run-1", "all", (self.job,), "env-1"),
        ]
        for value in values:
            with self.subTest(model=type(value).__name__):
                self.assert_json_round_trip(value)

    def test_runtime_objects_are_rejected(self):
        class TensorLike:
            pass

        with self.assertRaisesRegex(TypeError, "non-JSON object"):
            InputBundle(args=(TensorLike(),))
        with self.assertRaisesRegex(TypeError, "non-string mapping key"):
            InputBundle(kwargs={1: "invalid"})
        with self.assertRaisesRegex(TypeError, "finite JSON numbers"):
            InputBundle(kwargs={"loss": float("nan")})

    def test_nested_enum_type_is_preserved(self):
        bundle = InputBundle(kwargs={"status": ResultStatus.PLANNED})
        wire = json.loads(json.dumps(bundle.to_dict()))
        decoded = InputBundle.from_dict(wire)
        self.assertIs(decoded.kwargs["status"], ResultStatus.PLANNED)

    def test_from_dict_rejects_unknown_fields(self):
        encoded = self.reference.to_dict()
        encoded["runtime_object"] = "not part of the schema"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            ImplementationSpec.from_dict(encoded)


if __name__ == "__main__":
    unittest.main()
