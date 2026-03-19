"""148# Event Logger — 停止理由の永続化 + stderr ミラーリング.

run_fill_test.py から分離 (158# P2-4: god object 分割).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from ztb.data.raw_paths import utc_day_str_from_timestamp

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord

logger = logging.getLogger(__name__)


def _build_event_time_fields(now_ts: float | None = None) -> dict[str, object]:
    """分析しやすい時刻メタを統一して返す."""
    event_ts = time.time() if now_ts is None else now_ts
    event_dt = datetime.fromtimestamp(event_ts, timezone.utc)
    return {
        "timestamp": event_dt.isoformat(),
        "timestamp_epoch": event_ts,
        "utc_day": utc_day_str_from_timestamp(event_ts),
        "utc_hour": event_dt.hour,
    }


def build_cycle_revenue_event_details(record: "FillRecord") -> dict[str, object]:
    """収益分析に直結しやすい cycle 文脈を平たく返す."""
    submitted_at = datetime.fromtimestamp(record.timestamp, timezone.utc)
    return {
        "cycle_id": record.cycle_id,
        "submit_timestamp_epoch": record.timestamp,
        "submit_utc_day": utc_day_str_from_timestamp(record.timestamp),
        "submit_utc_hour": submitted_at.hour,
        "side": record.side,
        "filled": record.filled,
        "cancel_reason": record.cancel_reason,
        "queue_wait_sec": record.queue_wait_sec,
        "order_price": record.order_price,
        "order_quantity": record.order_quantity,
        "order_lot_effective": record.order_lot_effective,
        "spread_at_order": record.spread_at_order,
        "spread_bps": record.spread_bps,
        "effective_offset_used": record.effective_offset_used,
        "skip_gate_reason": record.skip_gate_reason,
        "skip_gate_score": record.skip_gate_score,
        "skip_gate_as_prob": record.skip_gate_as_prob,
        "skip_gate_threshold_used": record.skip_gate_threshold_used,
        "regime": record.regime,
        "regime_confidence": record.regime_confidence,
        "macro_trend": record.macro_trend,
        "macro_aligned": record.macro_aligned,
        "decision_path": record.decision_path,
        "ev_score_pretrade": record.ev_score_pretrade,
        "ev_offset_mult_applied": record.ev_offset_mult_applied,
        "sidecar_offset_bps": record.sidecar_offset_bps,
        "sidecar_confidence": record.sidecar_confidence,
        "sidecar_signal_status": record.sidecar_signal_status,
        "queue_fill_prob_est": record.queue_fill_prob_est,
        "cross_venue_reference_exchange": record.cross_venue_reference_exchange,
        "cross_venue_lead_lag_spread_bps": record.cross_venue_lead_lag_spread_bps,
        "cross_venue_lead_lag_velocity_bps": record.cross_venue_lead_lag_velocity_bps,
        "cross_venue_lead_lag_applied": record.cross_venue_lead_lag_applied,
        "cross_venue_lead_lag_vetoed": record.cross_venue_lead_lag_vetoed,
        "post_fill_30s_pnl": record.post_fill_30s_pnl,
    }


def log_event(
    event: str,
    results_dir: str | Path,
    run_id: str = "",
    git_sha: str = "",
    reason: str | None = None,
    details: dict[str, object] | None = None,
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
            **_build_event_time_fields(),
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
            except Exception as e:
                # 253# bare except → debug ログで可観測性確保
                logger.debug("TeeWriter.write failed for %s: %s", type(w).__name__, e, exc_info=True)
        return len(s)

    def flush(self) -> None:
        for w in self.writers:
            try:
                w.flush()
            except Exception as e:
                # 253# bare except → debug ログで可観測性確保
                logger.debug("TeeWriter.flush failed for %s: %s", type(w).__name__, e, exc_info=True)


def setup_stderr_mirror(results_dir: str | Path) -> None:
    """148# P1: stderr をファイルにもミラーリング."""
    stderr_file = None
    try:
        stderr_path = Path(results_dir) / "logs" / "fill_test_stderr.log"
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_file = open(stderr_path, "a", encoding="utf-8")
        sys.stderr = TeeWriter(sys.__stderr__, stderr_file)  # type: ignore[assignment]
        logger.info(f"[148#] stderr mirroring to {stderr_path}")
    except Exception as e:
        # 327# セットアップ失敗時にファイルハンドルをリークさせない
        if stderr_file is not None:
            stderr_file.close()
        logger.warning(f"[148#] Failed to setup stderr mirror: {e}")
