import argparse
from unittest.mock import MagicMock, patch

from ztb.ops.monitoring.launch_monitoring import MonitoringLauncher


def test_get_commands():
    launcher = MonitoringLauncher("test123", "http://example.com", False)
    commands = launcher.get_commands()
    assert len(commands) == 3
    assert commands[0] == [
        "python",
        "ztb/ztb/ztb/scripts/watch.py",
        "--correlation-id",
        "test123",
    ]
    assert commands[1] == [
        "python",
        "ztb/ztb/ztb/scripts/tb_scrape.py",
        "--correlation-id",
        "test123",
    ]
    assert commands[2] == [
        "python",
        "ztb/ztb/ztb/scripts/alert_notifier.py",
        "--correlation-id",
        "test123",
    ]


def test_get_commands_no_webhook():
    launcher = MonitoringLauncher("test123", None, False)
    commands = launcher.get_commands()
    assert len(commands) == 2
    assert "alert_notifier" not in str(commands)


def test_launch_processes_dry_run():
    launcher = MonitoringLauncher("test123", "http://example.com", True)
    with patch("builtins.print") as mock_print:
        launcher.launch_processes()
        assert mock_print.call_count == 3  # 3 commands


def test_launch_processes_execute():
    launcher = MonitoringLauncher("test123", "http://example.com", False)
    mock_proc = MagicMock()
    mock_proc.pid = 123
    with (
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        patch("builtins.print") as mock_print,
    ):
        launcher.launch_processes()
        assert mock_popen.call_count == 3
        assert len(launcher.processes) == 3
        assert mock_print.call_count == 3


def test_terminate_all():
    launcher = MonitoringLauncher("test123", None, False)
    mock_proc = MagicMock()
    launcher.processes = [mock_proc]

    launcher.terminate_all()
    mock_proc.terminate.assert_called()
    mock_proc.wait.assert_called()


def test_main_dry_run():
    with (
        patch(
            "ztb.scripts.launch_monitoring.MonitoringLauncher"
        ) as mock_launcher_class,
        patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(
                correlation_id="test",
                webhook="http://ex.com",
                dry_run=True,
                execute=False,
            ),
        ),
    ):
        mock_launcher = MagicMock()
        mock_launcher_class.return_value = mock_launcher

        from ztb.ops.monitoring.launch_monitoring import main

        main()

        mock_launcher_class.assert_called_with("test", "http://ex.com", True)
        mock_launcher.launch_processes.assert_called()
        mock_launcher.wait_and_cleanup.assert_called()


def test_main_execute():
    with (
        patch(
            "ztb.scripts.launch_monitoring.MonitoringLauncher"
        ) as mock_launcher_class,
        patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(
                correlation_id="test",
                webhook="http://ex.com",
                dry_run=False,
                execute=True,
            ),
        ),
    ):
        mock_launcher = MagicMock()
        mock_launcher_class.return_value = mock_launcher

        from ztb.ops.monitoring.launch_monitoring import main

        main()

        mock_launcher_class.assert_called_with("test", "http://ex.com", False)
