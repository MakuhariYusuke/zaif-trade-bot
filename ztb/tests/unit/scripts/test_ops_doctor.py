import unittest
from unittest.mock import MagicMock, patch

from ztb.ops.monitoring.ops_doctor import run_script


class TestOpsDoctor(unittest.TestCase):
    def test_run_script_success(self):
        """Test successful script run."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            exit_code, stdout, stderr = run_script("test.py", ["--arg"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(stdout, "output")
            self.assertEqual(stderr, "")
            mock_run.assert_called_once()

    def test_run_script_failure(self):
        """Test failed script run."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            exit_code, stdout, stderr = run_script("test.py", [])

            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout, "")
            self.assertEqual(stderr, "error")

    @patch("sys.exit")
    @patch("builtins.print")
    def test_main_success(self, mock_print, mock_exit):
        """Test main with all successful checks."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open"),
        ):
            with patch("sys.argv", ["ops_doctor.py", "--correlation-id", "test123"]):
                from ztb.ops.monitoring.ops_doctor import main

                main()

        # Check summary printed
        calls = [call.args[0] for call in mock_print.call_args_list]
        self.assertIn("Doctor Summary: OK=4, WARN=0, FAIL=0", calls)
        mock_exit.assert_called_with(0)

    @patch("sys.exit")
    @patch("builtins.print")
    def test_main_mixed_results(self, mock_print, mock_exit):
        """Test main with mixed results."""

        def mock_run(cmd, **kwargs):
            mock_result = MagicMock()
            script_name = cmd[1].split("/")[-1]
            if "validate" in script_name:
                mock_result.returncode = 0  # OK
            elif "collect" in script_name:
                mock_result.returncode = 1  # WARN
            elif "progress" in script_name:
                mock_result.returncode = 2  # FAIL
            else:
                mock_result.returncode = 0  # OK
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with (
            patch("subprocess.run", side_effect=mock_run),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open"),
        ):
            with patch("sys.argv", ["ops_doctor.py", "--correlation-id", "test123"]):
                from ztb.ops.monitoring.ops_doctor import main

                main()

        calls = [call.args[0] for call in mock_print.call_args_list]
        self.assertIn("Doctor Summary: OK=2, WARN=1, FAIL=1", calls)
        mock_exit.assert_called_with(1)  # fail_count


if __name__ == "__main__":
    unittest.main()
