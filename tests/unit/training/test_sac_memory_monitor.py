from __future__ import annotations

from unittest.mock import patch

from ztb.training.sac.memory_monitor import build_post_cycle_memory_status


def test_build_post_cycle_memory_status_marks_leak_and_threshold() -> None:
    with patch(
        "ztb.training.sac.memory_monitor.get_memory_usage",
        return_value={"rss": 320.0, "cache_total_entries": 7.0},
    ):
        status = build_post_cycle_memory_status(
            150.0,
            rss_warning_mb=256.0,
        )

    assert status["rss_mb"] == 320.0
    assert status["rss_delta_mb"] == 170.0
    assert status["cache_total_entries"] == 7.0
    assert status["leak_warning"] is True
    assert status["rss_warning"] is True


def test_build_post_cycle_memory_status_skips_leak_on_first_cycle() -> None:
    with patch(
        "ztb.training.sac.memory_monitor.get_memory_usage",
        return_value={"rss": 80.0, "cache_total_entries": 0.0},
    ):
        status = build_post_cycle_memory_status(
            0.0,
            rss_warning_mb=256.0,
        )

    assert status["rss_delta_mb"] == 0.0
    assert status["leak_warning"] is False
    assert status["rss_warning"] is False
