import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from benchmark_engine.correctness import OperatorSpec
from benchmark_engine.execution.worker import _validate_operator_spec
from benchmark_engine.execution import WorkerController, WorkerOutcome
from benchmark_engine.models import CaseSpec
from tests.worker_fixtures import make_job


CASE = CaseSpec(
    case_id="small",
    symbols={"size": 2},
    seed=7,
    tags=frozenset({"smoke"}),
)


def request(operator_id="test_operator", case=CASE):
    return SimpleNamespace(
        identity=SimpleNamespace(operator_id=operator_id),
        case=case,
    )


class CompleteSpec:
    operator_id = "test_operator"

    def cases(self): return (CASE,)
    def make_inputs(self, case, context): pass
    def clone_inputs(self, inputs): pass
    def normalize_output(self, output): pass
    def comparator(self, case): pass
    def cost_model(self, case): return None


class OperatorSpecProtocolTests(unittest.TestCase):
    def test_contract_declares_operator_identity_and_runtime_methods(self):
        self.assertIn(OperatorSpec.__annotations__["operator_id"],("str",str))
        for name in ("cases","make_inputs","clone_inputs","normalize_output","comparator","cost_model"):
            self.assertTrue(callable(getattr(OperatorSpec,name,None)))

    def test_worker_preserves_empty_phase5_fixture_compatibility(self):
        self.assertFalse(_validate_operator_spec(object(), request()))

    def test_worker_accepts_only_complete_identity_consistent_formal_specs(self):
        self.assertTrue(_validate_operator_spec(CompleteSpec(), request()))

        partial = SimpleNamespace(
            operator_id="test_operator",
            make_inputs=lambda case, context: None,
        )
        with self.assertRaisesRegex(TypeError, "contract is incomplete; missing: cases"):
            _validate_operator_spec(partial, request())

        mismatched = CompleteSpec()
        mismatched.operator_id = "different_operator"
        with self.assertRaisesRegex(ValueError, "does not match request identity"):
            _validate_operator_spec(mismatched, request())

        wrong_case = CaseSpec(
            case_id="small",
            symbols={"size": 999},
            seed=CASE.seed,
            tags=CASE.tags,
        )
        with self.assertRaisesRegex(ValueError, "does not match its OperatorSpec declaration"):
            _validate_operator_spec(CompleteSpec(), request(case=wrong_case))

    def test_partial_formal_spec_returns_stable_worker_protocol_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            response = WorkerController().run(
                make_job(output), spec_entrypoint="spec_partial:spec"
            )
            self.assertIs(response.outcome, WorkerOutcome.ERROR)
            self.assertEqual(response.stage.value, "correctness")
            self.assertEqual(response.error_type, "TypeError")
            self.assertIn("OperatorSpec contract is incomplete", response.error_message)
            self.assertIsNone(response.result_payload)
            self.assertIsNotNone(response.diagnostic_path)
            self.assertTrue((output / response.diagnostic_path).is_file())


if __name__ == "__main__": unittest.main()
