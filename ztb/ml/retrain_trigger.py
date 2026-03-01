"""136# P1-01: retrain trigger — データ駆動の再学習トリガー.

retrain_scheduler の固定 interval loop を補完し、以下の事前チェックを行う:
  1. fill_records ファイルの更新有無 (mtime ベース)
  2. trades データの健全性 (trades_health 連携)
  3. 連続スキップ時の適応的バックオフ

Usage:
    trigger = RetrainTrigger(results_dir, raw_dir)
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
class RetrainTriggerConfig:
    """トリガー設定."""

    #: fill_records の更新がない場合にスキップする
    check_fill_records_mtime: bool = True
    #: trades 健全性をチェックする (retrain 前ガード)
    check_trades_health: bool = True
    #: trades lookback 日数
    trades_lookback_days: int = 3
    #: trades stale 閾値 (時間)
    trades_stale_threshold_hours: float = 36.0
    #: 158# trades 欠損日の許容数 (0=厳密, 1=1日欠損許容)
    trades_max_missing_days: int = 1
    #: 連続 skip 時のバックオフ倍率 (1.0=無効, 2.0=倍増)
    backoff_multiplier: float = 2.0
    #: バックオフ最大 interval (秒)
    backoff_max_interval_sec: int = 14400  # 4h
    #: 基本 interval (秒) — バックオフの基底
    base_interval_sec: int = 3600
    #: §9 #2: feature 鮮度チェックを有効化
    check_feature_freshness: bool = False
    #: feature trades stale 閾値 (時間)
    feature_trades_stale_hours: float = 6.0
    #: feature OB stale 閾値 (時間)
    feature_ob_stale_hours: float = 6.0
    #: 145# R-2b: レジーム別 interval 倍率
    #: high_vol → 短い間隔 (市場変動が激しいので頻繁に retrain)
    #: ranging → 長い間隔 (安定レジームでは低頻度で十分)
    regime_interval_multipliers: dict[str, float] = field(default_factory=lambda: {
        "high_vol": 0.5,
        "trending": 0.75,
        "ranging": 1.5,
        "unknown": 1.0,
    })

    def __post_init__(self) -> None:
        """§11-#3: regime_interval_multipliers の値域バリデーション."""
        for regime, mul in self.regime_interval_multipliers.items():
            if mul <= 0:
                raise ValueError(
                    f"regime_interval_multipliers[{regime!r}] must be > 0, got {mul}"
                )
        if self.base_interval_sec < 1:
            raise ValueError(
                f"base_interval_sec must be >= 1, got {self.base_interval_sec}"
            )


@dataclass
class RetrainTrigger:
    """データ駆動の retrain トリガー.

    retrain_scheduler の while ループ内で should_retrain() を呼び出し、
    不要な retrain 試行 (データ未更新・trades 欠損) を回避する。
    """

    results_dir: Path
    raw_dir: Path | None = None
    config: RetrainTriggerConfig = field(default_factory=RetrainTriggerConfig)

    # 内部状態
    _last_fill_mtime: float = 0.0
    _consecutive_skips: int = 0
    _last_check_time: float = 0.0
    _current_regime: str = "unknown"  # 145# R-2b

    def _get_fill_records_latest_mtime(self) -> float:
        """fill_records_*.jsonl の最新 mtime を返す."""
        from ztb.metrics.fill_quality import list_fill_record_files

        rd = self.results_dir
        if not rd.exists():
            return 0.0
        mtimes: list[float] = []
        for p in list_fill_record_files(rd, include_emergency=False):
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
            max_missing_days=self.config.trades_max_missing_days,
        )
        return result.healthy, result.message

    def should_retrain(self) -> tuple[bool, str]:
        """retrain を実行すべきか事前チェック.

        Returns:
            (should_run, reason): should_run=False なら reason にスキップ理由。
        """
        self._last_check_time = time.time()

        # Check 1: fill_records 更新チェック
        # §9 #1 FIX: mtime は全チェック通過後に更新。
        # trades unhealthy で block された場合、mtime を消費しない。
        _pending_mtime: float | None = None
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
            # mtime 更新を保留 (全チェック通過まで確定しない)
            _pending_mtime = current_mtime

        # Check 2: trades 健全性チェック
        if self.config.check_trades_health:
            healthy, msg = self._check_trades_health()
            if not healthy:
                reason = f"trades unhealthy: {msg}"
                logger.warning(f"[136# P1-01] Retrain blocked: {reason}")
                self._consecutive_skips += 1
                return False, reason

        # Check 3: feature 鮮度チェック (§9 #2: freshness → trigger 接続)
        if self.config.check_feature_freshness:
            from ztb.data.trades_health import check_feature_freshness

            freshness = check_feature_freshness(
                raw_dir=self.raw_dir,
                trades_stale_hours=self.config.feature_trades_stale_hours,
                ob_stale_hours=self.config.feature_ob_stale_hours,
            )
            if not freshness.fresh:
                reason = f"feature stale: {freshness.message}"
                logger.warning(f"[136# P1-01] Retrain blocked: {reason}")
                self._consecutive_skips += 1
                return False, reason

        # 全チェック通過: mtime を確定更新
        if _pending_mtime is not None:
            self._last_fill_mtime = _pending_mtime

        return True, "ok"

    def record_result(self, status: str, current_regime: str | None = None) -> None:
        """retrain 結果を記録し、バックオフ状態を更新.

        145# R-2b: current_regime を渡すと interval 計算で使用。
        """
        if status in ("deployed", "error"):
            # deploy or error → バックオフリセット
            self._consecutive_skips = 0
        else:
            self._consecutive_skips += 1
        # 145# R-2b: レジーム情報更新
        if current_regime is not None:
            self._current_regime = current_regime

    def update_regime(self, regime: str) -> None:
        """145# R-2b: 現在レジームを外部から更新."""
        self._current_regime = regime

    def get_effective_interval(self) -> int:
        """連続スキップに応じた適応的 interval (秒) を返す.

        skips=0 → base, skips=1 → base*mul, skips=2 → base*mul^2, ...
        最大 backoff_max_interval_sec で打ち止め。
        145# R-2b: レジーム別 interval 倍率を追加適用。
        §11-#3: 最低 1 秒を保証し busy-loop を防止。
        """
        if self._consecutive_skips == 0 or self.config.backoff_multiplier <= 1.0:
            base = self.config.base_interval_sec
        else:
            factor = self.config.backoff_multiplier ** self._consecutive_skips
            base = int(self.config.base_interval_sec * factor)

        # 145# R-2b: レジーム倍率
        regime_mul = self.config.regime_interval_multipliers.get(
            self._current_regime, 1.0,
        )
        # §11-#3: max(1,...) で 0 秒 interval (busy-loop) を防止
        interval = max(1, int(base * max(regime_mul, 0.0)))
        return min(interval, self.config.backoff_max_interval_sec)

    @property
    def consecutive_skips(self) -> int:
        """連続スキップ回数."""
        return self._consecutive_skips


# §9 #5: 後方互換エイリアス (段階的移行)
RetainTriggerConfig = RetrainTriggerConfig
RetainTrigger = RetrainTrigger
