#!/usr/bin/env python3
"""
Unit tests for status_check.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from status_check import main


class TestStatusCheck(unittest.TestCase):
    @patch("subprocess.run")
    def test_status_check_basic(self, mock_subprocess):
        """Test status check with correlation ID."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["status_check.py", "--correlation-id", "test123"]):
            main()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        expected_cmd = [
            sys.executable,
            "ztb/ztb/ztb/scripts/status_snapshot.py",
            "--correlation-id",
            "test123",
        ]
        self.assertEqual(args[0], expected_cmd)

    @patch("subprocess.run")
    def test_status_check_with_output(self, mock_subprocess):
        """Test status check with output file."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch(
            "sys.argv",
            ["status_check.py", "--correlation-id", "test123", "--output", "report.md"],
        ):
            main()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        expected_cmd = [
            sys.executable,
            "ztb/ztb/ztb/scripts/status_snapshot.py",
            "--correlation-id",
            "test123",
            "--output",
            "report.md",
        ]
        self.assertEqual(args[0], expected_cmd)

    @patch("subprocess.run")
    def test_status_check_failure(self, mock_subprocess):
        """Test status check failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["status_check.py", "--correlation-id", "test123"]):
            with self.assertRaises(SystemExit) as cm:
                main()

        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
