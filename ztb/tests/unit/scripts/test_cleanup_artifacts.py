#!/usr/bin/env python3
"""
Unit tests for cleanup_artifacts.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from cleanup_artifacts import main


class TestCleanupArtifacts(unittest.TestCase):
    @patch("subprocess.run")
    def test_cleanup_artifacts_default(self, mock_subprocess):
        """Test cleanup artifacts with default settings."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["cleanup_artifacts.py"]):
            main()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        expected_cmd = [
            sys.executable,
            "ztb/ztb/ztb/scripts/artifacts_janitor.py",
            "--days",
            "30",
        ]
        self.assertEqual(args[0], expected_cmd)

    @patch("subprocess.run")
    def test_cleanup_artifacts_with_days(self, mock_subprocess):
        """Test cleanup artifacts with custom days."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["cleanup_artifacts.py", "--days", "7"]):
            main()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        expected_cmd = [
            sys.executable,
            "ztb/ztb/ztb/scripts/artifacts_janitor.py",
            "--days",
            "7",
        ]
        self.assertEqual(args[0], expected_cmd)

    @patch("subprocess.run")
    def test_cleanup_artifacts_dry_run(self, mock_subprocess):
        """Test cleanup artifacts dry run."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["cleanup_artifacts.py", "--dry-run"]):
            main()

        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        expected_cmd = [
            sys.executable,
            "ztb/ztb/ztb/scripts/artifacts_janitor.py",
            "--days",
            "30",
            "--dry-run",
        ]
        self.assertEqual(args[0], expected_cmd)

    @patch("subprocess.run")
    def test_cleanup_artifacts_failure(self, mock_subprocess):
        """Test cleanup artifacts failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        with patch("sys.argv", ["cleanup_artifacts.py"]):
            with self.assertRaises(SystemExit) as cm:
                main()

        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
