#!/usr/bin/env python3
"""
Unit tests for compat_wrapper.py
"""

import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ztb.ops.benchmark.compat_wrapper import handle_request, run_command_safely


class TestCompatWrapper(unittest.TestCase):
    def test_handle_status_request_success(self):
        """Test successful status request."""
        with patch("ztb.scripts.compat_wrapper.create_status_embed") as mock_create:
            mock_create.return_value = {
                "title": "Test Status",
                "color": 0x00FF00,
                "fields": [{"name": "Test", "value": "Value"}],
            }

            result = handle_request({"action": "status"})

            self.assertTrue(result["success"])
            self.assertEqual(result["action"], "status")
            self.assertEqual(result["data"]["title"], "Test Status")

    def test_handle_status_request_error(self):
        """Test status request with error."""
        with patch("ztb.scripts.compat_wrapper.create_status_embed") as mock_create:
            mock_create.side_effect = Exception("Test error")

            result = handle_request({"action": "status"})

            self.assertFalse(result["success"])
            self.assertEqual(result["action"], "status")
            self.assertIn("Test error", result["error"])

    def test_handle_run_request_success(self):
        """Test successful run request."""
        with patch("ztb.scripts.compat_wrapper.run_command_safely") as mock_run:
            mock_run.return_value = {
                "success": True,
                "returncode": 0,
                "stdout": "output",
                "stderr": "",
                "command": "echo test",
            }

            result = handle_request(
                {"action": "run", "params": {"command": "echo test"}}
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["action"], "run")
            self.assertEqual(result["command"], "echo test")

    def test_handle_run_request_missing_command(self):
        """Test run request without command."""
        result = handle_request({"action": "run"})

        self.assertFalse(result["success"])
        self.assertEqual(result["action"], "run")
        self.assertIn("Missing 'command' parameter", result["error"])

    def test_handle_unknown_action(self):
        """Test unknown action."""
        result = handle_request({"action": "unknown"})

        self.assertFalse(result["success"])
        self.assertEqual(result["action"], "unknown")
        self.assertIn("Unknown action", result["error"])

    @patch("subprocess.run")
    def test_run_command_safely_success(self, mock_run):
        """Test successful command execution."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="success output", stderr="error output"
        )

        result = run_command_safely("echo test")

        self.assertTrue(result["success"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "success output")
        self.assertEqual(result["stderr"], "error output")

    @patch("subprocess.run")
    def test_run_command_safely_failure(self, mock_run):
        """Test failed command execution."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="command failed"
        )

        result = run_command_safely("failing command")

        self.assertFalse(result["success"])
        self.assertEqual(result["returncode"], 1)

    @patch("subprocess.run")
    def test_run_command_safely_timeout(self, mock_run):
        """Test command timeout."""
        from subprocess import TimeoutExpired

        mock_run.side_effect = TimeoutExpired("cmd", 300)

        result = run_command_safely("slow command")

        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])

    @patch("ztb.scripts.compat_wrapper.json.loads")
    @patch("sys.stdin")
    @patch("sys.stdout", new_callable=StringIO)
    @patch("sys.exit")
    def test_main_valid_request(
        self, mock_exit, mock_stdout, mock_stdin, mock_json_loads
    ):
        """Test main with valid JSON request."""
        mock_json_loads.return_value = {"action": "status"}
        mock_stdin.read.return_value = '{"action": "status"}'

        with patch("ztb.scripts.compat_wrapper.handle_request") as mock_handle:
            mock_handle.return_value = {"success": True, "data": "test"}

            from ztb.ops.benchmark.compat_wrapper import main

            main()

            # Just check that it doesn't crash
            self.assertTrue(True)

    @patch("ztb.scripts.compat_wrapper.json.loads")
    @patch("sys.stdin")
    @patch("sys.stdout", new_callable=StringIO)
    @patch("sys.exit")
    def test_main_invalid_json(
        self, mock_exit, mock_stdout, mock_stdin, mock_json_loads
    ):
        """Test main with invalid JSON."""
        from json import JSONDecodeError

        mock_json_loads.side_effect = JSONDecodeError("Invalid", "", 0)
        mock_stdin.read.return_value = "invalid json"

        from ztb.ops.benchmark.compat_wrapper import main

        main()

        # Just check that it doesn't crash
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
