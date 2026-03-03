"""
BatchPersistence — fill_test のバッチ保存・緊急ダンプ.

119# God Object 分割: run_fill_test.py からバッチ保存ロジックを分離.
ztb/io/common の ensure_parent_dir を活用し、ディレクトリ保証を統一.
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


from ztb.io.common import ensure_parent_dir
from ztb.metrics.fill_quality import FillRecord, format_utc_day, save_fill_records

logger = logging.getLogger(__name__)


class BatchPersistence:
    """fill_test の FillRecord バッチ保存・リカバリ.

    024# R1: 保存失敗耐性・リトライ・緊急ダンプ.
    024# R4: record.timestamp 由来の日付でファイル分割.
    107# R1: 時間ベース定期 flush.
    """

    def __init__(
        self,
        results_dir: Path,
        max_retries: int = 3,
        save_fail_threshold: int = 3,
        retry_backoff_sec: float = 0.5,
        flush_interval_sec: float = 600.0,
    ) -> None:
        self._results_dir = results_dir
        self._max_retries = max_retries
        self._save_fail_threshold = save_fail_threshold
        self._retry_backoff_sec = retry_backoff_sec
        self._flush_interval_sec = flush_interval_sec
        self._save_fail_count: int = 0
        self._last_flush_time: float = time.time()
        self._unsaved_batch: list[FillRecord] = []

    @property
    def unsaved_batch(self) -> list[FillRecord]:
        """前回未保存のレコード."""
        return self._unsaved_batch

    def reset_flush_timer(self) -> None:
        """flush タイマーをリセット (batch_size flush 後などに呼ぶ)."""
        self._last_flush_time = time.time()

    def take_unsaved(self) -> list[FillRecord]:
        """未保存バッチを取得しクリア."""
        batch = list(self._unsaved_batch)
        self._unsaved_batch = []
        return batch

    def try_save_batch(self, batch: list[FillRecord]) -> bool:
        """バッチ保存を試行。失敗時はリトライ + フォールバック.

        024# R1: 保存専用 try/except を分離し、失敗を握り潰さない.
        024# R4: record.timestamp 由来の日付でファイル分割.

        Returns:
            True if save succeeded, False otherwise.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                self._save_batch_by_date(batch)
                self._save_fail_count = 0
                return True
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Batch save attempt {attempt + 1}/{self._max_retries} "
                    f"failed: {e}",
                    exc_info=True,
                )
                time.sleep(self._retry_backoff_sec * (2 ** attempt))

        # 全リトライ失敗
        self._save_fail_count += 1
        logger.error(
            f"Batch save FAILED after {self._max_retries} retries "
            f"(consecutive failures: {self._save_fail_count}): {last_error}"
        )

        # 024# R1: 連続失敗時は緊急ダンプ
        if self._save_fail_count >= self._save_fail_threshold:
            self.emergency_dump(batch, "save_fail")
            self._save_fail_count = 0
            return True  # ダンプ成功ならバッチクリア

        # batch は呼び出し元で保持 → 次回再試行
        self._unsaved_batch = list(batch)
        return False

    def maybe_flush(self, batch: list[FillRecord], context: str) -> list[FillRecord]:
        """107# R1: 時間ベース定期 flush.

        time_filter 抑制中・残高不足待機中など複数箇所で同一の
        "flush_interval 経過でバッチ保存" パターンを統合.

        Returns:
            flush 成功時は空リスト、それ以外は元の batch をそのまま返す.
        """
        if not batch:
            return batch
        now_ts = time.time()
        if now_ts - self._last_flush_time >= self._flush_interval_sec:
            if self.try_save_batch(batch):
                self._last_flush_time = now_ts
                logger.info(f"[batch_flush] Periodic flush during {context}")
                return []
        return batch

    def _save_batch_by_date(self, batch: list[FillRecord]) -> None:
        """024# R4: record.timestamp 由来の日付でファイル分割保存."""
        by_date: dict[str, list[FillRecord]] = {}
        for record in batch:
            day_str = format_utc_day(record.timestamp) or datetime.fromtimestamp(
                record.timestamp, tz=timezone.utc
            ).strftime("%Y%m%d")
            by_date.setdefault(day_str, []).append(record)

        for day_str, day_records in by_date.items():
            path = self._results_dir / f"fill_records_{day_str}.jsonl"
            save_fill_records(day_records, path)

    def emergency_dump(self, batch: list[FillRecord], reason: str) -> None:
        """024# R1: 緊急ダンプ — 通常保存が不可能な場合のフォールバック."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dump_dir = self._results_dir / "emergency"
        ensure_parent_dir(dump_dir / "_placeholder")  # ztb/io/common 活用
        dump_path = dump_dir / f"emergency_{reason}_{ts}.jsonl"

        try:
            save_fill_records(batch, dump_path)
            logger.warning(
                f"Emergency dump: {len(batch)} records saved to {dump_path}"
            )
        except Exception as e:
            import sys
            print(
                f"CRITICAL: Emergency dump also failed: {e}\n"
                f"Unsaved records: {len(batch)}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
