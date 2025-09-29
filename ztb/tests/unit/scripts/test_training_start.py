#!/usr/bin/env python3
"""
Unit tests for training_start.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from training_start import main


class TestTrainingStart(unittest.TestCase):
    @patch("subprocess.run")
    @patch("training_start.datetime")
    def test_training_start_with_generated_corr_id(
        self, mock_datetime, mock_subprocess
    ):
        """Test training start with auto-generated correlation ID."""
        # Mock datetime
        mock_dt = MagicMock()
        mock_dt.now.return_value.strftime.return_value = "20250101T120000Z"
        mock_datetime.now.return_value = mock_dt.now.return_value

        # Mock subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        # Mock sys.argv
        with patch("sys.argv", ["training_start.py"]):
            main()

        # Check subprocess was called with correct command
        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        self.assertEqual(args[0], ["make", "1m-start", "CORR=20250101T120000Z"])
        self.assertIn("CORR", kwargs["env"])
        self.assertEqual(kwargs["env"]["CORR"], "20250101T120000Z")

    @patch("subprocess.run")
    def test_training_start_with_provided_corr_id(self, mock_subprocess):
        """Test training start with provided correlation ID."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["training_start.py", "--correlation-id", "test123"]):
            main()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        self.assertEqual(args[0], ["make", "1m-start", "CORR=test123"])

    @patch("builtins.print")
    def test_training_start_dry_run(self, mock_print):
        """Test training start dry run mode."""
        with patch(
            "sys.argv",
            ["training_start.py", "--correlation-id", "test123", "--dry-run"],
        ):
            main()

        # Should print dry run message and not call subprocess
        mock_print.assert_called()

    @patch("subprocess.run")
    def test_training_start_failure(self, mock_subprocess):
        """Test training start when subprocess fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["training_start.py", "--correlation-id", "test123"]):
            with self.assertRaises(SystemExit) as cm:
                main()

        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
