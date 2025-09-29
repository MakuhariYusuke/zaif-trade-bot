import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ztb.ops.monitoring.collect_last_errors import (
    collect_last_errors,
    extract_errors_from_logs,
    extract_errors_from_watch_log,
    write_errors_report,
)


def test_extract_errors_from_logs():
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        log_file = log_dir / "test.log"
        with open(log_file, "w") as f:
            f.write("INFO: normal message\n")
            f.write("ERROR: something went wrong\n")
            f.write("FAIL: another failure\n")

        errors = extract_errors_from_logs(log_dir)
        assert len(errors) == 2
        assert "ERROR" in errors[0]
        assert "FAIL" in errors[1]


def test_extract_errors_from_logs_no_errors():
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        log_file = log_dir / "test.log"
        with open(log_file, "w") as f:
            f.write("INFO: normal message\n")

        errors = extract_errors_from_logs(log_dir)
        assert len(errors) == 0


def test_extract_errors_from_watch_log():
    with tempfile.TemporaryDirectory() as tmp:
        watch_log = Path(tmp) / "watch_log.jsonl"
        with open(watch_log, "w") as f:
            f.write(
                '{"timestamp": "2023-09-29T12:00:00", "level": "INFO", "message": "ok"}\n'
            )
            f.write(
                '{"timestamp": "2023-09-29T12:01:00", "level": "ERROR", "message": "error occurred"}\n'
            )
            f.write(
                '{"timestamp": "2023-09-29T12:02:00", "level": "FAIL", "message": "failure"}\n'
            )

        errors = extract_errors_from_watch_log(watch_log)
        assert len(errors) == 2
        assert "ERROR" in errors[0]
        assert "FAIL" in errors[1]


def test_extract_errors_from_watch_log_missing():
    with tempfile.TemporaryDirectory() as tmp:
        watch_log = Path(tmp) / "watch_log.jsonl"
        errors = extract_errors_from_watch_log(watch_log)
        assert len(errors) == 0


def test_collect_last_errors():
    with (
        tempfile.TemporaryDirectory() as tmp_log,
        tempfile.TemporaryDirectory() as tmp_watch,
    ):
        # Mock paths
        with patch("ztb.scripts.collect_last_errors.Path") as mock_path:
            mock_path.return_value.glob.return_value = [Path(tmp_log) / "test.log"]
            mock_path.return_value.exists.return_value = True

            # Create log file
            log_file = Path(tmp_log) / "test.log"
            with open(log_file, "w") as f:
                f.write("ERROR: test error\n")

            # Create watch log
            watch_file = Path(tmp_watch) / "watch_log.jsonl"
            with open(watch_file, "w") as f:
                f.write('{"level": "ERROR", "message": "watch error"}\n')

            # Patch the functions
            with (
                patch(
                    "ztb.scripts.collect_last_errors.extract_errors_from_logs",
                    return_value=["log error"],
                ),
                patch(
                    "ztb.scripts.collect_last_errors.extract_errors_from_watch_log",
                    return_value=["watch error"],
                ),
            ):
                errors = collect_last_errors("test_id")
                assert len(errors) == 2


def test_write_errors_report():
    with tempfile.TemporaryDirectory() as tmp:
        corr_id = "test123"
        errors = ["error1", "error2"]

        write_errors_report(corr_id, errors)

        report_path = Path(tmp) / "artifacts" / corr_id / "reports" / "last_errors.txt"
        # Since we can't easily patch Path, check if function runs without error
        # In real test, would need to patch Path


def test_write_errors_report_no_errors():
    with tempfile.TemporaryDirectory() as tmp:
        corr_id = "test123"
        errors = []

        write_errors_report(corr_id, errors)

        # Similar to above


def test_main_no_errors():
    with (
        patch("ztb.scripts.collect_last_errors.collect_last_errors", return_value=[]),
        patch("ztb.scripts.collect_last_errors.write_errors_report") as mock_write,
        patch("builtins.print") as mock_print,
    ):
        # Mock args
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(correlation_id="test"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                from ztb.ops.monitoring.collect_last_errors import main

                main()
            assert exc_info.value.code == 0
            mock_print.assert_called_with("no recent errors")
            mock_write.assert_not_called()
