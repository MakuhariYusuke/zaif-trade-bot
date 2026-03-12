from __future__ import annotations

from pathlib import Path


_WATCHDOG_PATH = (
    Path(__file__).resolve().parents[3]
    / "ops"
    / "windows"
    / "fill_test_watchdog.ps1"
)


def test_restart_lock_stale_threshold_extended_to_120_seconds() -> None:
    source = _WATCHDOG_PATH.read_text(encoding="utf-8")

    assert "120秒以上前の restart.lock は stale とみなす (360# OPS-4)" in source
    assert "if ($lockAge -gt 120)" in source


def test_start_process_waits_for_fill_test_lock() -> None:
    source = _WATCHDOG_PATH.read_text(encoding="utf-8")

    assert "$lockWaitMax = 30" in source
    assert "$lockWaitInterval = 2" in source
    assert "fill_test.lock detected after ${lockWaitElapsed}s — startup confirmed" in source
    assert "fill_test.lock not found after ${lockWaitMax}s — startup may have failed" in source
