from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.v460.lib.fill_test_cli import (
    _collect_fill_test_memory_diagnostics,
    _dump_exit_diagnostics,
    _read_lock_heartbeat_age_sec,
)
from tests.unit.v460._fill_test_source import FILL_TEST_CLI, read_source_text


class TestExitDiagnostics:
    def test_read_lock_heartbeat_age_prefers_heartbeat_field(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "fill_test.lock"
        lock_path.write_text("111|100|run-1|175", encoding="utf-8")

        age = _read_lock_heartbeat_age_sec(tmp_path, lock_path=lock_path, now_ts=200.0)

        assert age == pytest.approx(25.0)

    def test_dump_exit_diagnostics_writes_expected_payload(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "fill_test.lock"
        lock_path.write_text("111|100|run-1|175", encoding="utf-8")
        fake_proc = MagicMock()
        fake_proc.pid = 4321
        fake_proc.memory_info.return_value = SimpleNamespace(
            rss=128 * 1024 * 1024,
            vms=512 * 1024 * 1024,
        )
        fixed_now = datetime(1970, 1, 1, 0, 3, 20, tzinfo=timezone.utc)

        with patch("scripts.v460.lib.fill_test_cli.psutil.Process", return_value=fake_proc), patch(
            "scripts.v460.lib.fill_test_cli.datetime",
        ) as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            dump_path = _dump_exit_diagnostics(
                tmp_path,
                "run:test/1",
                stop_reason="completed",
                lock_path=lock_path,
                trigger="unit",
                extra_payload={
                    "gc_counts": [1, 2, 3],
                    "ml_cache_stats": {"total_ml_cache_entries": 4},
                },
            )

        assert dump_path is not None
        assert dump_path.exists()
        assert dump_path.parent == tmp_path / "diagnostics"
        assert "run_test_1" in dump_path.name

        payload = json.loads(dump_path.read_text(encoding="utf-8"))
        assert payload["trigger"] == "unit"
        assert payload["run_id"] == "run:test/1"
        assert payload["stop_reason"] == "completed"
        assert payload["pid"] == 4321
        assert payload["rss_mb"] == pytest.approx(128.0)
        assert payload["vms_mb"] == pytest.approx(512.0)
        assert payload["lock_heartbeat_age_sec"] == pytest.approx(25.0)
        assert payload["gc_counts"] == [1, 2, 3]
        assert payload["ml_cache_stats"]["total_ml_cache_entries"] == 4

    def test_dump_exit_diagnostics_survives_psutil_failure(self, tmp_path: Path) -> None:
        with patch(
            "scripts.v460.lib.fill_test_cli.psutil.Process",
            side_effect=RuntimeError("psutil unavailable"),
        ):
            dump_path = _dump_exit_diagnostics(
                tmp_path,
                "run-2",
                stop_reason=None,
                trigger="unit",
            )

        assert dump_path is not None
        payload = json.loads(dump_path.read_text(encoding="utf-8"))
        assert payload["rss_mb"] is None
        assert payload["vms_mb"] is None
        assert payload["lock_heartbeat_age_sec"] is None

    def test_fill_test_cli_registers_atexit_and_signal_dump(self) -> None:
        source = read_source_text(FILL_TEST_CLI)

        assert "atexit.register(_atexit_hook)" in source
        assert "_emit_exit_diagnostics(\"signal\", signal_reason)" in source

    def test_collect_fill_test_memory_diagnostics_includes_runner_buffers(self) -> None:
        runner = SimpleNamespace(
            _recent_records=[1, 2, 3],
            _batch_persistence=SimpleNamespace(unsaved_batch=["a", "b"]),
            _health_monitor=SimpleNamespace(
                snapshot_memory_diagnostics=lambda: {"last_pressure_gc_collected": 9},
            ),
        )

        with patch(
            "scripts.v460.lib.fill_test_cli.gc.get_count",
            return_value=(1, 2, 3),
        ), patch(
            "scripts.v460.lib.fill_test_cli.gc.get_threshold",
            return_value=(700, 10, 10),
        ), patch(
            "scripts.v460.ml.cache_cleanup.get_ml_data_cache_stats",
            return_value={"total_ml_cache_entries": 5},
        ):
            payload = _collect_fill_test_memory_diagnostics(runner)

        assert payload["gc_counts"] == [1, 2, 3]
        assert payload["gc_thresholds"] == [700, 10, 10]
        assert payload["runner_buffer_stats"] == {
            "recent_records_count": 3,
            "recent_records_maxlen": None,
            "unsaved_batch_count": 2,
        }
        assert payload["ml_cache_stats"] == {"total_ml_cache_entries": 5}
        assert payload["health_monitor"] == {"last_pressure_gc_collected": 9}
