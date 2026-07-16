import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
