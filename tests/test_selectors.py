import unittest
from pathlib import Path

from benchmark_engine.models import CaseSpec, ImplementationSpec
from benchmark_engine.registry import RegistrySnapshot
from benchmark_engine.selectors import (
    SelectorError,
    Selectors,
    select_candidates,
    select_cases,
    select_operators,
)
from benchmark_engine.suite import SuiteConfig


def candidate(operator: str, name: str) -> ImplementationSpec:
    return ImplementationSpec(operator, name, "candidate", Path(name), "m:o", "a" * 64, 1)


SUITE = SuiteConfig(1, "test", ("*_op",), (), ("smoke", "representative"), "all", (0,), 3, 1)


class SelectorTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = RegistrySnapshot(
            candidates={
                "alpha_op": (candidate("alpha_op", "first"), candidate("alpha_op", "second")),
                "beta_op": (candidate("beta_op", "third"),),
            },
            discovered_operator_ids=("alpha_op", "beta_op"),
        )

    def test_or_within_and_across_categories_and_exclude_last(self):
        selectors = Selectors(
            operators=("alpha*", "beta*"),
            candidates=("first", "third"),
            cases=("small*", "medium*"),
            tags=("representative",),
            exclude_operators=("beta*",),
        )
        operators = select_operators(self.snapshot, SUITE, selectors)
        self.assertEqual(operators, ("alpha_op",))
        selected_candidates = select_candidates(self.snapshot, operators, selectors)
        self.assertEqual([item.implementation_id for item in selected_candidates["alpha_op"]], ["first"])
        cases = (
            CaseSpec("small", {}, 0, frozenset({"smoke"})),
            CaseSpec("medium", {}, 0, frozenset({"representative"})),
            CaseSpec("large", {}, 0, frozenset({"representative"})),
        )
        self.assertEqual([case.case_id for case in select_cases(cases, SUITE, selectors)], ["medium"])

    def test_no_match_is_an_error(self):
        with self.assertRaisesRegex(SelectorError, "no operators"):
            select_operators(self.snapshot, SUITE, Selectors(operators=("missing",)))

    def test_each_selected_operator_requires_a_matching_candidate(self):
        with self.assertRaisesRegex(
            SelectorError, "no candidates for operator: beta_op"
        ):
            select_candidates(
                self.snapshot,
                ("alpha_op", "beta_op"),
                Selectors(candidates=("first",)),
            )

    def test_explicit_candidate_overrides_suite_default_candidate_set(self):
        suite = SuiteConfig(
            1,
            "smoke",
            ("*_op",),
            (),
            ("smoke",),
            "correctness",
            (0,),
            3,
            1,
            candidate_include=("first",),
        )
        defaults = select_candidates(
            self.snapshot, ("alpha_op",), Selectors(), suite
        )
        self.assertEqual(
            [item.implementation_id for item in defaults["alpha_op"]], ["first"]
        )
        explicit = select_candidates(
            self.snapshot,
            ("alpha_op",),
            Selectors(candidates=("second",)),
            suite,
        )
        self.assertEqual(
            [item.implementation_id for item in explicit["alpha_op"]], ["second"]
        )


if __name__ == "__main__":
    unittest.main()
