from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from scripts.v460.lib.resilience import FillTestHealthMonitor, HealthThresholds


class TestHealthMonitorYamlDefaults:
    def test_default_check_interval_is_60_seconds(self) -> None:
        hm = FillTestHealthMonitor()

        assert hm._thresholds.check_interval_sec == 60.0


class TestHealthMonitorWarnings:
    def test_warning_log_when_rss_exceeds_warn_threshold(self, caplog) -> None:
        hm = FillTestHealthMonitor(HealthThresholds(
            rss_warn_mb=100.0,
            rss_critical_mb=200.0,
            check_interval_sec=0.0,
        ))
        hm._psutil_available = True
        hm._process = MagicMock()
        hm._psutil = MagicMock()
        hm._process.memory_info.return_value = SimpleNamespace(
            rss=int(150 * 1024 * 1024),
        )
        hm._process.cpu_percent.return_value = 12.5
        hm._process.num_threads.return_value = 8
        hm._psutil.disk_usage.return_value = SimpleNamespace(
            free=int(10 * 1024**3),
        )

        status = hm.maybe_check(7)

        assert status is not None
        assert status["level"] == "warning"
        assert "RSS 150MB exceeds warn threshold 100MB" in caplog.text

    def test_warning_rss_triggers_pressure_gc_with_cooldown(self) -> None:
        hm = FillTestHealthMonitor(HealthThresholds(
            rss_warn_mb=100.0,
            rss_critical_mb=200.0,
            check_interval_sec=0.0,
        ))
        hm._psutil_available = True
        hm._process = MagicMock()
        hm._psutil = MagicMock()
        hm._process.memory_info.return_value = SimpleNamespace(
            rss=int(150 * 1024 * 1024),
        )
        hm._process.cpu_percent.return_value = 12.5
        hm._process.num_threads.return_value = 8
        hm._psutil.disk_usage.return_value = SimpleNamespace(
            free=int(10 * 1024**3),
        )

        with patch("scripts.v460.lib.resilience.time.time", return_value=1000.0), patch(
            "scripts.v460.lib.resilience.gc.collect",
            return_value=7,
        ) as mock_gc:
            first_status = hm.maybe_check(1)

        assert first_status is not None
        assert first_status["pressure_gc_collected"] == 7
        mock_gc.assert_called_once()

        with patch("scripts.v460.lib.resilience.time.time", return_value=1010.0), patch(
            "scripts.v460.lib.resilience.gc.collect",
            return_value=5,
        ) as mock_gc:
            second_status = hm.maybe_check(2)

        assert second_status is not None
        assert second_status["pressure_gc_collected"] == 0
        mock_gc.assert_not_called()

        snapshot = hm.snapshot_memory_diagnostics(now_ts=1010.0)
        assert snapshot["last_pressure_gc_collected"] == 7
        assert snapshot["last_pressure_gc_age_sec"] == pytest.approx(10.0)
        assert snapshot["rss_mb"] == pytest.approx(150.0)
        assert snapshot["cpu_percent"] == pytest.approx(12.5)
        assert snapshot["threads"] == 8
