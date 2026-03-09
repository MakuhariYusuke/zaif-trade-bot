from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

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
