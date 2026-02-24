"""148# Event Logger — 停止理由の永続化 + stderr ミラーリング.

run_fill_test.py から分離 (158# P2-4: god object 分割).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


def log_event(
    event: str,
    results_dir: str | Path,
    run_id: str = "",
    git_sha: str = "",
    reason: str | None = None,
    details: dict | None = None,
) -> None:
    """fill_test_events.jsonl にイベントを記録.

    148# P0: 停止理由を推定でなく事実として記録するため、
    start/stop/crash/signal イベントを永続化する。

    Args:
        event: イベント種別 (start, stop, crash, signal)
        results_dir: 結果ディレクトリ
        run_id: 実行 ID
        git_sha: Git SHA
        reason: 停止理由 (stop/crash/signal の場合)
        details: 追加詳細情報
    """
    try:
        events_path = Path(results_dir) / "fill_test_events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "run_id": run_id,
            "git_sha": git_sha,
            "pid": os.getpid(),
            "reason": reason,
            "details": details or {},
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        if os.name == "nt":
            # §11 #2: lock/unlock を byte 0 固定で対称化
            # §11 #3: Windows 専用パス (非 Windows は else へ)
            import msvcrt

            with open(events_path, "a", encoding="utf-8") as f:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    f.seek(0, 2)  # EOF for append
                    f.write(line)
                    f.flush()
                finally:
                    f.seek(0)  # back to locked byte 0
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            # 非 Windows: fcntl があれば使用、なければ無ロックで追記
            with open(events_path, "a", encoding="utf-8") as f:
                try:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.write(line)
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    f.write(line)
                    f.flush()
        logger.info(f"[event] {event}: reason={reason}")
    except Exception as e:
        logger.warning(f"[event] Failed to log event: {e}")


class TeeWriter:
    """148# P1: stderr を複数出力先に同時書き込み."""

    def __init__(self, *writers: TextIO) -> None:
        self.writers = writers

    def write(self, s: str) -> int:
        for w in self.writers:
            try:
                w.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        for w in self.writers:
            try:
                w.flush()
            except Exception:
                pass


def setup_stderr_mirror(results_dir: str | Path) -> None:
    """148# P1: stderr をファイルにもミラーリング."""
    try:
        stderr_path = Path(results_dir) / "logs" / "fill_test_stderr.log"
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_file = open(stderr_path, "a", encoding="utf-8")
        sys.stderr = TeeWriter(sys.__stderr__, stderr_file)  # type: ignore[assignment]
        logger.info(f"[148#] stderr mirroring to {stderr_path}")
    except Exception as e:
        logger.warning(f"[148#] Failed to setup stderr mirror: {e}")
