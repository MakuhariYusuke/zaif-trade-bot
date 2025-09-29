import argparse
from unittest.mock import MagicMock, patch

import pytest

from ztb.ops.scheduling.cronish import calculate_next_run, format_eta, run_command


def test_run_command() -> None:
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = "error"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        exit_code = run_command("echo test")
        assert exit_code == 0
        mock_run.assert_called_once()


def test_run_command_failure() -> None:
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error"
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        exit_code = run_command("fail cmd")
        assert exit_code == 1


def test_calculate_next_run() -> None:
    with patch("time.time", return_value=1000), patch("random.uniform", return_value=5):
        next_run = calculate_next_run(300, 10)
        assert next_run == 1000 + 300 + 5


def test_format_eta() -> None:
    assert format_eta(30) == "30s"
    assert format_eta(90) == "1.5m"
    assert format_eta(7200) == "2.0h"


def test_main_cycles() -> None:
    # Test multiple cycles with mocked time
    with (
        patch("time.time") as mock_time,
        patch("time.sleep") as mock_sleep,
        patch("random.uniform", return_value=0),
        patch("subprocess.run") as mock_run,
        patch("builtins.print") as mock_print,
    ):
        mock_time.side_effect = [1000, 1001, 1301, 1302]  # Simulate time progression
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Mock args
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(
                interval_sec=300, jitter_sec=0, fail_fast=False, command="echo test"
            ),
        ):
            # Mock to stop after 2 cycles
            with patch(
                "pathlib.Path.exists", side_effect=[False, False, True]
            ):  # Stop on 3rd check
                with pytest.raises(SystemExit):
                    from ztb.ops.scheduling.cronish import main

                    main()

                # Should have run command twice
                assert mock_run.call_count == 2
                assert mock_sleep.call_count == 2
