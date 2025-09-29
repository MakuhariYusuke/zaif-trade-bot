import argparse
import sys
from unittest.mock import MagicMock, patch

from ztb.ops.release.cli import run_script


def test_run_script():
    with (
        patch("subprocess.run") as mock_run,
        patch("pathlib.Path.exists", return_value=True),
    ):
        mock_result = mock_run.return_value
        mock_result.returncode = 0

        exit_code = run_script("test.py", ["--arg", "value"])

        assert exit_code == 0
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == sys.executable
        assert cmd[1] == "scripts\\test.py"  # Windows path
        assert cmd[2:] == ["--arg", "value"]


def test_run_script_not_found():
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("builtins.print") as mock_print,
    ):
        exit_code = run_script("nonexistent.py", [])

        assert exit_code == 1
        mock_print.assert_called()


def test_main_valid_command():
    from ztb.ops.release.cli import main

    with (
        patch("ztb.scripts.ops.argparse.ArgumentParser") as mock_parser_class,
        patch("ztb.scripts.ops.run_script", return_value=0) as mock_run_script,
        patch("sys.exit") as mock_exit,
    ):
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_args.return_value = argparse.Namespace(
            command="eta", args=["--correlation-id", "test123"]
        )

        main()

        mock_run_script.assert_called_with(
            "progress_eta.py", ["--correlation-id", "test123"]
        )
        mock_exit.assert_called_with(0)


def test_main_with_args():
    from ztb.ops.release.cli import main

    with (
        patch("ztb.scripts.ops.argparse.ArgumentParser") as mock_parser_class,
        patch("ztb.scripts.ops.run_script", return_value=0) as mock_run_script,
        patch("sys.exit") as mock_exit,
    ):
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_args.return_value = argparse.Namespace(
            command="bundle", args=["--correlation-id", "test123", "--exclude-logs"]
        )

        main()

        mock_run_script.assert_called_with(
            "bundle_artifacts.py", ["--correlation-id", "test123", "--exclude-logs"]
        )
        mock_exit.assert_called_with(0)
