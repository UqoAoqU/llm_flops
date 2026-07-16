import unittest

from benchmark_engine.correctness import OperatorSpec


class OperatorSpecProtocolTests(unittest.TestCase):
    def test_contract_declares_operator_identity_and_runtime_methods(self):
        self.assertIn(OperatorSpec.__annotations__["operator_id"],("str",str))
        for name in ("cases","make_inputs","clone_inputs","normalize_output","comparator","cost_model"):
            self.assertTrue(callable(getattr(OperatorSpec,name,None)))


if __name__ == "__main__": unittest.main()
