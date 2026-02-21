"""136# P1-01: retrain trigger — データ駆動の再学習トリガー.

retrain_scheduler の固定 interval loop を補完し、以下の事前チェックを行う:
  1. fill_records ファイルの更新有無 (mtime ベース)
  2. trades データの健全性 (trades_health 連携)
  3. 連続スキップ時の適応的バックオフ

Usage:
    trigger = RetainTrigger(results_dir, raw_dir)
    if trigger.should_retrain():
        retrain_model(cfg)
        trigger.record_result("deployed")
    else:
        trigger.record_result("skipped_trigger")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RetainTriggerConfig:
    """トリガー設定."""

    #: fill_records の更新がない場合にスキップする
    check_fill_records_mtime: bool = True
    #: trades 健全性をチェックする (retrain 前ガード)
    check_trades_health: bool = True
    #: trades lookback 日数
    trades_lookback_days: int = 3
    #: trades stale 閾値 (時間)
    trades_stale_threshold_hours: float = 36.0
    #: 連続 skip 時のバックオフ倍率 (1.0=無効, 2.0=倍増)
    backoff_multiplier: float = 2.0
    #: バックオフ最大 interval (秒)
    backoff_max_interval_sec: int = 14400  # 4h
    #: 基本 interval (秒) — バックオフの基底
    base_interval_sec: int = 3600


@dataclass
class RetainTrigger:
    """データ駆動の retrain トリガー.

    retrain_scheduler の while ループ内で should_retrain() を呼び出し、
    不要な retrain 試行 (データ未更新・trades 欠損) を回避する。
    """

    results_dir: Path
    raw_dir: Path | None = None
    config: RetainTriggerConfig = field(default_factory=RetainTriggerConfig)

    # 内部状態
    _last_fill_mtime: float = 0.0
    _consecutive_skips: int = 0
    _last_check_time: float = 0.0

    def _get_fill_records_latest_mtime(self) -> float:
        """fill_records_*.jsonl の最新 mtime を返す."""
        rd = self.results_dir
        if not rd.exists():
            return 0.0
        mtimes: list[float] = []
        for p in rd.glob("fill_records_*.jsonl"):
            try:
                mtimes.append(p.stat().st_mtime)
            except OSError:
                continue
        return max(mtimes) if mtimes else 0.0

    def _check_trades_health(self) -> tuple[bool, str]:
        """trades データの健全性を簡易チェック.

        Returns:
            (healthy, message)
        """
        from ztb.data.trades_health import check_trades_health

        result = check_trades_health(
            raw_dir=self.raw_dir,
            lookback_days=self.config.trades_lookback_days,
            stale_threshold_hours=self.config.trades_stale_threshold_hours,
        )
        return result.healthy, result.message

    def should_retrain(self) -> tuple[bool, str]:
        """retrain を実行すべきか事前チェック.

        Returns:
            (should_run, reason): should_run=False なら reason にスキップ理由。
        """
        self._last_check_time = time.time()

        # Check 1: fill_records 更新チェック
        if self.config.check_fill_records_mtime:
            current_mtime = self._get_fill_records_latest_mtime()
            if current_mtime <= self._last_fill_mtime and self._last_fill_mtime > 0.0:
                reason = (
                    f"fill_records unchanged (mtime={current_mtime:.0f}, "
                    f"last={self._last_fill_mtime:.0f})"
                )
                logger.info(f"[136# P1-01] Retrain skip: {reason}")
                self._consecutive_skips += 1
                return False, reason
            # mtime を先に更新（retrain 結果に依らず次回比較のため）
            self._last_fill_mtime = current_mtime

        # Check 2: trades 健全性チェック
        if self.config.check_trades_health:
            healthy, msg = self._check_trades_health()
            if not healthy:
                reason = f"trades unhealthy: {msg}"
                logger.warning(f"[136# P1-01] Retrain blocked: {reason}")
                self._consecutive_skips += 1
                return False, reason

        return True, "ok"

    def record_result(self, status: str) -> None:
        """retrain 結果を記録し、バックオフ状態を更新."""
        if status in ("deployed", "error"):
            # deploy or error → バックオフリセット
            self._consecutive_skips = 0
        else:
            self._consecutive_skips += 1

    def get_effective_interval(self) -> int:
        """連続スキップに応じた適応的 interval (秒) を返す.

        skips=0 → base, skips=1 → base*mul, skips=2 → base*mul^2, ...
        最大 backoff_max_interval_sec で打ち止め。
        """
        if self._consecutive_skips == 0 or self.config.backoff_multiplier <= 1.0:
            return self.config.base_interval_sec
        factor = self.config.backoff_multiplier ** self._consecutive_skips
        interval = int(self.config.base_interval_sec * factor)
        return min(interval, self.config.backoff_max_interval_sec)

    @property
    def consecutive_skips(self) -> int:
        """連続スキップ回数."""
        return self._consecutive_skips
