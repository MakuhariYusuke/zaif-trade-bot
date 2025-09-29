#!/usr/bin/env python3
"""
Unit tests for monitoring_start.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from monitoring_start import main


class TestMonitoringStart(unittest.TestCase):
    @patch("subprocess.run")
    def test_monitoring_start_success(self, mock_subprocess):
        """Test monitoring start success."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["monitoring_start.py", "--correlation-id", "test123"]):
            main()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        expected_cmd = [
            sys.executable,
            "ztb/ztb/ztb/scripts/launch_monitoring.py",
            "--correlation-id",
            "test123",
        ]
        self.assertEqual(args[0], expected_cmd)

    @patch("builtins.print")
    def test_monitoring_start_dry_run(self, mock_print):
        """Test monitoring start dry run."""
        with patch(
            "sys.argv",
            ["monitoring_start.py", "--correlation-id", "test123", "--dry-run"],
        ):
            main()

        mock_print.assert_called()

    @patch("subprocess.run")
    def test_monitoring_start_failure(self, mock_subprocess):
        """Test monitoring start failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["monitoring_start.py", "--correlation-id", "test123"]):
            with self.assertRaises(SystemExit) as cm:
                main()

        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
