import unittest

from tools.smoke_test import build_smoke_plan


class SmokeTestPlanTest(unittest.TestCase):
    def test_plan_covers_every_single_operator_family(self):
        self.assertEqual(
            [case.name for case in build_smoke_plan()],
            [
                "dsa_indexer",
                "dsa_flashmla",
                "dsa_projection",
                "mla_flashmla",
                "moe_deepgemm",
            ],
        )

    def test_every_case_has_a_callable_runner(self):
        self.assertTrue(all(callable(case.run) for case in build_smoke_plan()))


if __name__ == "__main__":
    unittest.main()
