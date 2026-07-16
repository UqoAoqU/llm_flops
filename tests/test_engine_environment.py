import sys
import types
import unittest
import warnings
from unittest.mock import patch

from benchmark_engine.environment import (
    collect_environment,
    environment_fingerprint,
    load_lock,
    planning_fingerprint,
    validate_environment,
)


class EngineEnvironmentTest(unittest.TestCase):
    def test_adapter_reuses_legacy_functions(self):
        lock = {"schema_version": 1}
        observed = {"python": "3.12.0"}
        legacy = types.SimpleNamespace(
            load_lock=lambda path="legacy-default": {"path": path},
            collect_environment=lambda value, include_cuda=True: (value, include_cuda),
            validate_environment=lambda expected, actual: ["legacy", expected, actual],
            environment_fingerprint=lambda value: "legacy-fingerprint",
        )
        with patch.dict(sys.modules, {"benchmark_environment": legacy}):
            self.assertEqual(load_lock(), {"path": "legacy-default"})
            self.assertEqual(collect_environment(lock, include_cuda=False), (lock, False))
            self.assertEqual(validate_environment(lock, observed), ["legacy", lock, observed])
            self.assertEqual(environment_fingerprint(observed), "legacy-fingerprint")

    def test_planning_fingerprint_is_stable_and_import_free(self):
        first = planning_fingerprint({"b": [2], "a": 1})
        second = planning_fingerprint({"a": 1, "b": [2]})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_environment_probe_suppresses_only_scoped_third_party_deprecations(self):
        def noisy_probe(lock, include_cuda=True):
            warnings.warn("optional dependency is deprecated", DeprecationWarning)
            return {"lock": lock, "include_cuda": include_cuda}

        legacy = types.SimpleNamespace(collect_environment=noisy_probe)
        with patch.dict(sys.modules, {"benchmark_environment": legacy}):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                observed = collect_environment({"schema_version": 1}, include_cuda=False)
            self.assertEqual(captured, [])
            self.assertFalse(observed["include_cuda"])

        # The adapter must not mutate the process-wide warning policy.
        with warnings.catch_warnings(record=True) as outside:
            warnings.simplefilter("always")
            warnings.warn("outside probe", DeprecationWarning)
        self.assertEqual(len(outside), 1)
        self.assertIs(outside[0].category, DeprecationWarning)


if __name__ == "__main__":
    unittest.main()
