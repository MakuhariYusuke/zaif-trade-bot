"""
148# fill_test イベント系機能のテスト.

Codex §9 #5 対応: event logger / stderr mirror / kill reason 参照の検証.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.v460.lib.event_logger import log_event as _log_event, setup_stderr_mirror as _setup_stderr_mirror, TeeWriter as _TeeWriter


class _RecordingWriter:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.flushed = 0

    def write(self, text: str) -> None:
        self.writes.append(text)

    def flush(self) -> None:
        self.flushed += 1


class _FailingWriter:
    def write(self, _text: str) -> None:
        raise IOError("disk full")

    def flush(self) -> None:
        raise IOError("disk full")


# ======================================================================
# _log_event
# ======================================================================


class TestLogEvent:
    """fill_test_events.jsonl イベントロガーの検証."""

    def test_start_event_creates_jsonl(self, tmp_path: Path) -> None:
        """start イベントが events.jsonl を作成する."""
        _log_event("start", tmp_path, run_id="test-001", git_sha="abc123")
        events_file = tmp_path / "fill_test_events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "start"
        assert record["run_id"] == "test-001"
        assert record["git_sha"] == "abc123"
        assert "timestamp" in record
        assert "pid" in record

    def test_stop_event_with_reason(self, tmp_path: Path) -> None:
        """stop イベントが reason を記録する."""
        _log_event("stop", tmp_path, reason="completed")
        events_file = tmp_path / "fill_test_events.jsonl"
        record = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert record["event"] == "stop"
        assert record["reason"] == "completed"

    def test_crash_event_with_details(self, tmp_path: Path) -> None:
        """crash イベントが details を記録する."""
        _log_event(
            "crash",
            tmp_path,
            reason="crash:ValueError",
            details={"traceback": "Traceback ..."},
        )
        events_file = tmp_path / "fill_test_events.jsonl"
        record = json.loads(events_file.read_text(encoding="utf-8").strip())
        assert record["event"] == "crash"
        assert record["reason"] == "crash:ValueError"
        assert record["details"]["traceback"] == "Traceback ..."

    def test_multiple_events_append(self, tmp_path: Path) -> None:
        """複数イベントが追記される."""
        _log_event("start", tmp_path, run_id="r1")
        _log_event("stop", tmp_path, run_id="r1", reason="completed")
        events_file = tmp_path / "fill_test_events.jsonl"
        lines = events_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "start"
        assert json.loads(lines[1])["event"] == "stop"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """存在しない results_dir が自動作成される."""
        nested = tmp_path / "sub" / "dir"
        _log_event("start", nested)
        assert (nested / "fill_test_events.jsonl").exists()


# ======================================================================
# _TeeWriter
# ======================================================================


class TestTeeWriter:
    """stderr ミラー用 _TeeWriter の検証."""

    def test_writes_to_all_writers(self) -> None:
        """全 writer に書き込まれる."""
        w1 = _RecordingWriter()
        w2 = _RecordingWriter()
        tee = _TeeWriter(w1, w2)
        tee.write("hello")
        assert w1.writes == ["hello"]
        assert w2.writes == ["hello"]

    def test_flush_all_writers(self) -> None:
        """全 writer が flush される."""
        w1 = _RecordingWriter()
        w2 = _RecordingWriter()
        tee = _TeeWriter(w1, w2)
        tee.flush()
        assert w1.flushed == 1
        assert w2.flushed == 1

    def test_write_returns_length(self) -> None:
        """write() が文字数を返す."""
        w = _RecordingWriter()
        tee = _TeeWriter(w)
        assert tee.write("abc") == 3
        assert w.writes == ["abc"]

    def test_writer_exception_suppressed(self) -> None:
        """writer の例外が抑制される."""
        w_bad = _FailingWriter()
        w_good = _RecordingWriter()
        tee = _TeeWriter(w_bad, w_good)
        tee.write("test")  # should not raise
        assert w_good.writes == ["test"]

    def test_flush_exception_suppressed(self) -> None:
        """flush の例外が抑制される."""
        w_bad = _FailingWriter()
        tee = _TeeWriter(w_bad)
        tee.flush()  # should not raise


# ======================================================================
# _setup_stderr_mirror
# ======================================================================


class TestSetupStderrMirror:
    """stderr ミラーリングのセットアップ検証."""

    def test_stderr_becomes_tee(self, tmp_path: Path) -> None:
        """stderr が _TeeWriter に置き換わる."""
        original = sys.stderr
        try:
            _setup_stderr_mirror(tmp_path)
            assert isinstance(sys.stderr, _TeeWriter)
            stderr_log = tmp_path / "logs" / "fill_test_stderr.log"
            assert stderr_log.exists()
        finally:
            sys.stderr = original


# ======================================================================
# KillSwitch.get_reason() 参照テスト (148# §9 #1 回帰テスト)
# ======================================================================


class TestKillSwitchReasonAccess:
    """KillSwitch の reason 参照が get_reason() である検証."""

    def test_get_reason_on_kill(self) -> None:
        """kill() 後に get_reason() が正しい値を返す."""
        from ztb.risk.circuit_breakers import KillSwitch

        ks = KillSwitch("test")
        ks.kill("hard_loss_cap")
        assert ks.get_reason() == "hard_loss_cap"
        assert ks.is_killed()

    def test_get_reason_before_kill(self) -> None:
        """kill 前の get_reason() は空文字."""
        from ztb.risk.circuit_breakers import KillSwitch

        ks = KillSwitch("test")
        assert ks.get_reason() == ""
        assert not ks.is_killed()

    def test_no_reason_attribute(self) -> None:
        """KillSwitch に .reason プロパティが存在しないことを確認."""
        from ztb.risk.circuit_breakers import KillSwitch

        ks = KillSwitch("test")
        assert not hasattr(ks, "reason"), (
            "KillSwitch should not have a .reason attribute; use get_reason()"
        )

    def test_signal_reason(self) -> None:
        """signal 理由の記録と取得."""
        from ztb.risk.circuit_breakers import KillSwitch

        ks = KillSwitch("test")
        ks.kill("signal:SIGINT")
        assert ks.get_reason() == "signal:SIGINT"

    def test_preflight_stop_reason(self) -> None:
        """preflight_stop 理由の記録と取得."""
        from ztb.risk.circuit_breakers import KillSwitch

        ks = KillSwitch("test")
        ks.kill("preflight_stop")
        assert ks.get_reason() == "preflight_stop"
