#!/usr/bin/env python3
"""
Unit tests for cronish catchup functionality
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from cronish import perform_catchup


class TestCronishCatchup(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.kill_file = Path(self.temp_dir) / "ztb.stop"
        self.last_run_file = Path(self.temp_dir) / "cronish_last_run.txt"

    def tearDown(self):
        # Clean up temp files
        if self.last_run_file.exists():
            self.last_run_file.unlink()
        if self.kill_file.exists():
            self.kill_file.unlink()
        Path(self.temp_dir).rmdir()

    @patch("cronish.time.time")
    @patch("cronish.run_command")
    @patch("cronish.time.sleep")
    def test_perform_catchup_no_previous_run(
        self, mock_sleep, mock_run_command, mock_time
    ):
        """Test catchup when no previous run file exists."""
        mock_time.return_value = 1000.0  # Current time

        args = MagicMock()
        args.max_catchup = 5
        args.interval_sec = 60
        args.catchup_cooldown_sec = 1
        args.command = "echo test"
        args.fail_fast = False

        with patch("builtins.open", mock_open()) as mock_file:
            perform_catchup(args, self.kill_file)

        # Should not run any catchup commands
        mock_run_command.assert_not_called()

    @patch("cronish.Path")
    @patch("cronish.time.time")
    @patch("cronish.run_command")
    @patch("cronish.time.sleep")
    def test_perform_catchup_missed_two_nights(
        self, mock_sleep, mock_run_command, mock_time, mock_path
    ):
        """Test catchup with exactly 3 missed nightly runs."""
        # Mock Path to return our temp file
        mock_path.return_value = self.last_run_file

        # Simulate last run 3 days ago (missed 3 intervals)
        last_run_time = 1000.0
        current_time = last_run_time + (3 * 24 * 3600)  # 3 days later

        mock_time.return_value = current_time
        mock_run_command.return_value = 0  # Success

        # Create last run file
        with open(self.last_run_file, "w") as f:
            f.write(str(last_run_time))

        args = MagicMock()
        args.max_catchup = 5  # Allow up to 5 catchups
        args.interval_sec = 24 * 3600  # Daily interval
        args.catchup_cooldown_sec = 1
        args.command = "echo nightly"
        args.fail_fast = False

        perform_catchup(args, self.kill_file)

        # Should run exactly 3 catchup commands (missed 3 intervals)
        self.assertEqual(mock_run_command.call_count, 3)
        # Should sleep twice between catchups (not after last)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("cronish.time.time")
    @patch("cronish.run_command")
    @patch("cronish.time.sleep")
    def test_perform_catchup_with_missed_runs(
        self, mock_sleep, mock_run_command, mock_time
    ):
        """Test catchup with missed runs."""
        mock_time.return_value = 1000.0  # Current time
        mock_run_command.return_value = 0

        # Create last run file with time 100 seconds ago
        with open(self.last_run_file, "w") as f:
            f.write("900.0")

        args = MagicMock()
        args.max_catchup = 5
        args.interval_sec = 60  # 1 minute intervals
        args.catchup_cooldown_sec = 2
        args.command = "echo test"
        args.fail_fast = False

        perform_catchup(args, self.kill_file)

        # Should have missed 1 interval (100 / 60 = 1.66, so 1 missed)
        # But max_catchup=5, so should run 1 catchup
        self.assertEqual(mock_run_command.call_count, 1)
        mock_run_command.assert_called_with("echo test")

        # Should sleep once for cooldown (but since only 1 catchup, no sleep)
        mock_sleep.assert_not_called()

    @patch("cronish.time.time")
    @patch("cronish.run_command")
    @patch("cronish.time.sleep")
    def test_perform_catchup_max_catchup_limit(
        self, mock_sleep, mock_run_command, mock_time
    ):
        """Test catchup respects max_catchup limit."""
        mock_time.return_value = 1000.0
        mock_run_command.return_value = 0

        # Create last run file with time 400 seconds ago
        with open(self.last_run_file, "w") as f:
            f.write("600.0")

        args = MagicMock()
        args.max_catchup = 2  # Limit to 2 catchups
        args.interval_sec = 60
        args.catchup_cooldown_sec = 1
        args.command = "echo test"
        args.fail_fast = False

        perform_catchup(args, self.kill_file)

        # Should have missed 6 intervals (400 / 60 = 6.66), but limited to 2
        self.assertEqual(mock_run_command.call_count, 2)
        # Should sleep once between catchups
        self.assertEqual(mock_sleep.call_count, 1)
        mock_sleep.assert_called_with(1)

    @patch("cronish.time.time")
    @patch("cronish.run_command")
    @patch("cronish.time.sleep")
    def test_perform_catchup_kill_file_detected(
        self, mock_sleep, mock_run_command, mock_time
    ):
        """Test catchup stops when kill file is detected."""
        mock_time.return_value = 1000.0
        mock_run_command.return_value = 0

        # Create last run file
        with open(self.last_run_file, "w") as f:
            f.write("700.0")

        # Create kill file
        self.kill_file.touch()

        args = MagicMock()
        args.max_catchup = 5
        args.interval_sec = 60
        args.catchup_cooldown_sec = 1
        args.command = "echo test"
        args.fail_fast = False

        with self.assertRaises(SystemExit):
            perform_catchup(args, self.kill_file)

        # Should not run any commands
        mock_run_command.assert_not_called()

    @patch("cronish.time.time")
    @patch("cronish.run_command")
    @patch("cronish.time.sleep")
    def test_perform_catchup_fail_fast(self, mock_sleep, mock_run_command, mock_time):
        """Test catchup with fail_fast exits on failure."""
        mock_time.return_value = 1000.0
        mock_run_command.return_value = 1  # Command fails

        # Create last run file
        with open(self.last_run_file, "w") as f:
            f.write("700.0")

        args = MagicMock()
        args.max_catchup = 5
        args.interval_sec = 60
        args.catchup_cooldown_sec = 1
        args.command = "echo test"
        args.fail_fast = True

        with self.assertRaises(SystemExit):
            perform_catchup(args, self.kill_file)

        # Should run one command and exit
        self.assertEqual(mock_run_command.call_count, 1)


if __name__ == "__main__":
    unittest.main()
