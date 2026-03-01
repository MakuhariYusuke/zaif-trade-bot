"""188# DefaultCycleStrategy — RegimePolicyConfig ベースの CycleStrategy 実装.

179# で regime_policy.py 内に定義 → 188# で分割抽出。
File size guidance: 200 行以下に維持。

MAX LINES: 200 (超えたら strategy variant を別ファイルに分割)
"""

from __future__ import annotations

import logging
import time

from scripts.v460.lib.regime_policy import CycleStrategy, RegimePolicyConfig  # noqa: F401

logger = logging.getLogger(__name__)


class DefaultCycleStrategy:
    """RegimePolicyConfig と FillTestConfig を参照する標準 CycleStrategy.

    - dynamic_cycle_enabled=False → config.cycle_interval_sec 固定
    - dynamic_wait_enabled=False → config.post_fill_wait_sec 固定
    - chase_enabled=False → Chase 無効
    """

    def __init__(
        self,
        base_interval: float,
        base_wait_buy: float,
        base_wait_sell: float,
        policy: RegimePolicyConfig,
    ) -> None:
        self._base_interval = base_interval
        self._base_wait_buy = base_wait_buy
        self._base_wait_sell = base_wait_sell
        self._policy = policy
        # 停止条件によるフォールバック状態
        self._fallback_active: bool = False
        self._fallback_until: float = 0.0
        # 182# Trend Mode 厳格化: サイクル冒頭で更新
        self._current_confidence: float = 0.0
        # 186# ヒステリシス状態
        self._in_trend_mode: bool = False
        self._trend_dwell: int = 0

    @property
    def policy(self) -> RegimePolicyConfig:
        return self._policy

    def activate_fallback(self, duration_sec: float = 3600.0) -> None:
        """停止条件トリガー: 一定時間 ranging モードにフォールバック."""
        self._fallback_active = True
        self._fallback_until = time.time() + duration_sec
        logger.warning(
            f"[179# CycleStrategy] Fallback activated for {duration_sec:.0f}s "
            f"— all cycle intervals revert to base"
        )

    def _check_fallback(self) -> bool:
        """フォールバック期間中かどうかチェック."""
        if self._fallback_active:
            if time.time() >= self._fallback_until:
                self._fallback_active = False
                logger.info("[179# CycleStrategy] Fallback expired — resuming dynamic mode")
                return False
            return True
        return False

    def update_confidence(self, confidence: float) -> None:
        """182# サイクル冒頭で呼び出し、最新 confidence をキャッシュ."""
        self._current_confidence = confidence

    def gated_regime(self, regime: str | None, confidence: float | None = None) -> str | None:
        """182#/186# Trend Mode ヒステリシス付き confidence gating.

        Enter: confidence >= trend_min_confidence
        Exit:  confidence < trend_exit_confidence AND dwell >= trend_min_dwell
        """
        if regime is None:
            return regime
        c = confidence if confidence is not None else self._current_confidence
        is_trending_input = regime.startswith("trending")

        if self._in_trend_mode:
            # Exit: regime が non-trending、または confidence 低下 + dwell 経過
            if not is_trending_input or (
                c < self._policy.trend_exit_confidence
                and self._trend_dwell >= self._policy.trend_min_dwell
            ):
                self._in_trend_mode = False
                self._trend_dwell = 0
                logger.debug(
                    "[186# gated_regime] EXIT trend mode: regime=%s conf=%.3f dwell=%d",
                    regime, c, self._trend_dwell,
                )
                return "ranging" if is_trending_input else regime
            self._trend_dwell += 1
            return regime  # trend 維持
        else:
            # Enter: confidence >= 閾値
            if is_trending_input and c >= self._policy.trend_min_confidence:
                self._in_trend_mode = True
                self._trend_dwell = 1
                logger.debug(
                    "[186# gated_regime] ENTER trend mode: regime=%s conf=%.3f",
                    regime, c,
                )
                return regime
            return "ranging" if is_trending_input else regime

    def effective_interval(self, regime: str | None) -> float:
        """C: regime 別サイクル間隔 (182# confidence gating 内包)."""
        if not self._policy.dynamic_cycle_enabled or self._check_fallback():
            return self._base_interval
        regime = self.gated_regime(regime)
        if regime is None:
            return self._base_interval
        return self._policy.cycle_intervals.get(regime, self._base_interval)

    def effective_post_fill_wait(
        self, side: str, regime: str | None, *, vol_ratio: float | None = None,
    ) -> float:
        """D: regime × side 別 post-fill wait (182# confidence gating 内包).

        200# G: vol_ratio による動的スケーリング:
        - 低 vol (< 1.0): wait 延長 (利益確定に時間が必要)
        - 高 vol (> 1.0): wait 短縮 (素早い判断が必要)
        - vol_ratio=None: 従来通り固定値
        """
        if not self._policy.dynamic_wait_enabled or self._check_fallback():
            base = self._base_wait_sell if side == "sell" else self._base_wait_buy
        else:
            regime = self.gated_regime(regime)
            if regime is None:
                base = self._base_wait_sell if side == "sell" else self._base_wait_buy
            else:
                regime_waits = self._policy.post_fill_wait.get(regime)
                if regime_waits is None:
                    base = self._base_wait_sell if side == "sell" else self._base_wait_buy
                else:
                    base = regime_waits.get(
                        side,
                        self._base_wait_sell if side == "sell" else self._base_wait_buy,
                    )
        # 200# G: volatility-scaled wait (opt-in: vol_ratio が渡された場合のみ)
        if vol_ratio is not None and vol_ratio > 0:
            # vol_ratio=0.5 → wait ×1.3, vol_ratio=1.5 → wait ×0.85
            # 上下限: 0.7x ~ 1.5x で暴走防止
            _vol_scale = max(0.7, min(1.5, 1.0 / vol_ratio ** 0.3))
            return base * _vol_scale
        return base

    def is_chase_enabled(self, regime: str | None, side: str | None = None) -> bool:
        """Chase: trending 系 regime 限定で有効 (187# 方向制限追加).

        187# B-1: trending_up → buy のみ chase 許可 (sell は cancel-only)
                  trending_down → sell のみ chase 許可 (buy は cancel-only)
                  trending (方向不明) → 両方許可 (後方互換)
        """
        if not self._policy.chase_enabled or self._check_fallback():
            return False
        regime = self.gated_regime(regime)
        if regime is None:
            return False
        if regime not in self._policy.chase_regimes:
            return False
        # 187# B-1: 方向フィルタリング
        if side is not None:
            if regime == "trending_up" and side != "buy":
                return False
            if regime == "trending_down" and side != "sell":
                return False
        return True

    def chase_drift_bps(self) -> float:
        return self._policy.chase_drift_bps

    def chase_max_reprice(self) -> int:
        return self._policy.chase_max_reprice
