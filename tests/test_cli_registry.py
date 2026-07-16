import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.cli import main

from tests.registry_fixtures import OPERATOR_ID, write_reference, write_registry


class RegistryCliTest(unittest.TestCase):
    def call(self, root: Path, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments, repository_root=root)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_list_is_sorted_and_selectors_work_without_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = write_registry(root)
            code, stdout, stderr = self.call(root, "list", "--operator", "fixture_*")
            self.assertEqual(code, 0, stderr)
            positions = [stdout.index(candidate) for candidate in sorted((first, second))]
            self.assertEqual(positions, sorted(positions))
            code, stdout, stderr = self.call(
                root, "list", "--candidate", f"{second.split('__', 1)[0]}*"
            )
            self.assertEqual(code, 0, stderr)
            self.assertIn(second, stdout)
            self.assertNotIn(first, stdout)

    def test_validate_and_no_match_exit_codes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_registry(root)
            code, stdout, stderr = self.call(
                root, "validate", "--operator", OPERATOR_ID
            )
            self.assertEqual(code, 0, stderr)
            self.assertIn("registry valid", stdout)
            code, _, stderr = self.call(root, "validate", "--operator", "missing*")
            self.assertEqual(code, 2)
            self.assertIn("matched no operators", stderr)
            code, _, stderr = self.call(root, "list", "--candidate", "missing*")
            self.assertEqual(code, 2)
            self.assertIn("matched no candidates", stderr)

    def test_registry_error_exits_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_registry(root)
            manifest = root / "operators" / "references" / OPERATOR_ID / "operator.yaml"
            manifest.write_text("schema_version: 99\n")
            code, _, stderr = self.call(root, "list")
            self.assertEqual(code, 2)
            self.assertIn("manifest.", stderr)

    def test_operator_selector_isolates_issues_by_registry_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_registry(root)
            bad_operator = "broken_cpu_add"
            bad_reference = write_reference(root, bad_operator)
            manifest = bad_reference / "operator.yaml"
            manifest.write_text(
                manifest.read_text().replace("schema_version: 1", "schema_version: 2")
            )

            for command in ("list", "validate"):
                with self.subTest(command=command, selector="good"):
                    code, stdout, stderr = self.call(
                        root, command, "--operator", "fixture_*"
                    )
                    self.assertEqual(code, 0, stderr)
                    if command == "list":
                        self.assertIn(OPERATOR_ID, stdout)
                    else:
                        self.assertIn("registry valid", stdout)
                    self.assertEqual(stderr, "")

                for selector in ("broken_*", "*"):
                    with self.subTest(command=command, selector=selector):
                        code, _, stderr = self.call(
                            root, command, "--operator", selector
                        )
                        self.assertEqual(code, 2)
                        self.assertIn("manifest.schema_version", stderr)
                        self.assertIn(bad_operator, stderr)
                        self.assertNotIn(OPERATOR_ID, stderr)


if __name__ == "__main__":
    unittest.main()
