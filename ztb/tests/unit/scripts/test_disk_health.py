import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ztb.ops.monitoring.disk_health import (
    check_health,
    get_disk_usage,
    get_inode_usage,
    write_alerts,
)


def test_get_disk_usage():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        usage = get_disk_usage(path)
        assert "total" in usage
        assert "free" in usage
        assert usage["free"] > 0


def test_get_inode_usage():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        inodes = get_inode_usage(path)
        if inodes is None:  # Windows
            assert inodes is None
        else:
            assert "total" in inodes
            assert "usage_pct" in inodes


def test_check_health_low_space():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)

        # Mock low space
        mock_stat = MagicMock()
        mock_stat.total = 100 * 1024**3  # 100GB
        mock_stat.used = 95 * 1024**3  # 95GB used
        mock_stat.free = 5 * 1024**3  # 5GB free

        with patch("shutil.disk_usage", return_value=mock_stat):
            alerts = check_health(path, 10.0, 90.0, False)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "low_disk_space"
        assert "5.0GB" in alerts[0]["message"]


def test_check_health_high_inode():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)

        # Mock normal disk
        mock_stat = MagicMock()
        mock_stat.total = 100 * 1024**3
        mock_stat.used = 50 * 1024**3
        mock_stat.free = 50 * 1024**3

        # Mock high inode usage
        mock_statvfs = MagicMock()
        mock_statvfs.f_files = 1000
        mock_statvfs.f_favail = 50  # 95% used

        with (
            patch("shutil.disk_usage", return_value=mock_stat),
            patch(
                "ztb.scripts.disk_health.get_inode_usage",
                return_value={
                    "total": 1000,
                    "used": 950,
                    "free": 50,
                    "usage_pct": 95.0,
                },
            ),
        ):
            alerts = check_health(path, 10.0, 90.0, False)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "high_inode_usage"
        assert "95.0%" in alerts[0]["message"]


def test_check_health_ok():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)

        # Mock good disk
        mock_stat = MagicMock()
        mock_stat.total = 100 * 1024**3
        mock_stat.used = 50 * 1024**3
        mock_stat.free = 50 * 1024**3

        with (
            patch("shutil.disk_usage", return_value=mock_stat),
            patch("ztb.scripts.disk_health.get_inode_usage", return_value=None),
        ):
            alerts = check_health(path, 10.0, 90.0, False)

        assert len(alerts) == 0


def test_write_alerts():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "alerts.jsonl"
        alerts = [{"test": "alert"}]

        write_alerts(alerts, log_path)

        with open(log_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        assert json.loads(lines[0]) == {"test": "alert"}


def test_main_with_alerts():
    with tempfile.TemporaryDirectory() as tmp:
        # Mock low space
        mock_stat = MagicMock()
        mock_stat.total = 100 * 1024**3
        mock_stat.used = 95 * 1024**3
        mock_stat.free = 0.01 * 1024**3  # 0.01GB free

        with (
            patch("shutil.disk_usage", return_value=mock_stat),
            patch("ztb.scripts.disk_health.get_inode_usage", return_value=None),
            patch("sys.exit") as mock_exit,
            patch("builtins.print") as mock_print,
        ):
            with patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(
                    correlation_id="test123",
                    path=Path(tmp),
                    min_free_gb=0.1,
                    max_inode_usage=90.0,
                    check_io=False,
                ),
            ):
                from ztb.ops.monitoring.disk_health import main

                main()

                # Should print JSON and exit 1
                mock_print.assert_called()
                mock_exit.assert_called_with(1)
