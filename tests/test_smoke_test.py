import unittest
from unittest.mock import patch

from tools.smoke_test import ROOT, build_smoke_command, main


class SmokeTestCommandTest(unittest.TestCase):
    def test_command_runs_the_formal_mi300x_suite(self):
        self.assertEqual(
            build_smoke_command(),
            (str(ROOT / "bench.sh"), "run", "--suite", "mi300x_smoke"),
        )

    @patch("tools.smoke_test.subprocess.run")
    def test_main_pins_every_rocm_visible_device_variable(self, run):
        run.return_value.returncode = 7
        self.assertEqual(main(), 7)
        call = run.call_args
        self.assertEqual(call.args[0], build_smoke_command())
        self.assertEqual(call.kwargs["cwd"], ROOT)
        for name in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            self.assertEqual(call.kwargs["env"][name], "0")
        self.assertFalse(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
