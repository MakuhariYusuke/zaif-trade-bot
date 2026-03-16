"""188# Macro Regime Detector — 5m/15m slope ベースの中長期マーケット状態判定.

185#-§5.1 Phase D: fill_test サイクルの mid_price を時間バケット集約し、
5 分・15 分スロープを算出。micro regime と組み合わせて
マクロ的なトレンド/レンジ/ボラ判定を行う基盤モジュール。

市場理論的根拠:
  **Regime-Switching Model** — Hamilton (1989) "A New Approach to the Economic
  Analysis of Nonstationary Time Series and the Business Cycle".
  マクロレベルの状態遷移を slope とボラティリティで検知。
  micro regime (120s) が短期ノイズを含むのに対し、
  macro regime (5m/15m) は構造的トレンドを捕捉する。

  **Micro-Macro 矛盾検出**: micro = ranging なのに macro = trending の
  場合、micro がノイズに惑わされている可能性がある。
  compose_regimes() で矛盾を検出し、ログまたは regime 補正を行う。

設計原則:
  - 入力: FillTestRegimeDetector と同じ (timestamp, mid_price) ストリーム
  - 5m/15m スロープは OLS 線形回帰で算出 (ノイズ耐性)
  - micro regime より遅延するが、トレンド方向の確度が高い
  - compose_regimes() で micro + macro を統合し、矛盾検出 + regime 増強

MAX LINES: 250
"""""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum


import numpy as np

logger = logging.getLogger(__name__)


class MacroTrend(str, Enum):
    """マクロ方向分類."""

    STRONG_UP = "macro_strong_up"       # 5m/15m 双方上昇
    WEAK_UP = "macro_weak_up"           # 片方のみ上昇
    NEUTRAL = "macro_neutral"           # 横ばい
    WEAK_DOWN = "macro_weak_down"       # 片方のみ下降
    STRONG_DOWN = "macro_strong_down"   # 5m/15m 双方下降
    INSUFFICIENT = "macro_insufficient" # データ不足


@dataclass
class MacroRegimeConfig:
    """MacroRegimeDetector 設定."""

    # バケットサイズ (秒) — mid_price をこの間隔で集約
    bucket_sec: float = 30.0

    # スロープ算出ウィンドウ (バケット数)
    slope_window_5m: int = 10    # 30s × 10 = 5 分
    slope_window_15m: int = 30   # 30s × 30 = 15 分

    # スロープ閾値 (bps/min) — これ以上で trending 判定
    slope_threshold_bps_per_min: float = 1.0

    # 強いトレンドの閾値 (bps/min)
    strong_slope_threshold_bps_per_min: float = 3.0

    # バッファ上限 (バケット数)
    max_buckets: int = 60  # 30min 分

    # micro regime との矛盾時に macro を優先する confidence 閾値
    macro_override_confidence: float = 0.7

    # 458# ヒステリシス: トレンド判定のフラッピング防止
    # トレンド遷移に必要な連続一致回数 (update 呼び出しベース)
    hysteresis_count: int = 3
    # 確定後の最小保持回数 (この回数に達するまで同方向維持)
    hold_count: int = 2


@dataclass
class MacroRegimeResult:
    """MacroRegimeDetector の判定結果."""

    trend: MacroTrend
    slope_5m_bps_per_min: float = 0.0    # 5 分スロープ (bps/min)
    slope_15m_bps_per_min: float = 0.0   # 15 分スロープ (bps/min)
    confidence: float = 0.0              # 0.0–1.0
    buckets_available: int = 0           # 蓄積バケット数
    micro_macro_aligned: bool = True     # micro/macro 一致フラグ

    def to_dict(self) -> dict[str, object]:
        """JSON serializable dict."""
        return {
            "trend": self.trend.value,
            "slope_5m": round(self.slope_5m_bps_per_min, 4),
            "slope_15m": round(self.slope_15m_bps_per_min, 4),
            "confidence": round(self.confidence, 4),
            "buckets": self.buckets_available,
            "aligned": self.micro_macro_aligned,
        }


class MacroRegimeDetector:
    """5m/15m スロープによるマクロ regime 判定.

    使い方:
        macro = MacroRegimeDetector(config)
        # fill_test サイクルごとに更新
        result = macro.update(timestamp, mid_price)
        # micro regime と組み合わせ
        composed = compose_regimes(micro_result, result)
    """

    def __init__(self, config: MacroRegimeConfig | None = None) -> None:
        self.config = config or MacroRegimeConfig()
        # バケット: (bucket_start_ts, [prices_in_bucket])
        self._buckets: list[tuple[float, float]] = []  # (ts_center, avg_price)
        self._current_bucket_prices: list[float] = []
        self._current_bucket_start: float = 0.0
        # 458# ヒステリシス状態
        self._confirmed_trend: MacroTrend = MacroTrend.NEUTRAL
        self._pending_trend: MacroTrend = MacroTrend.NEUTRAL
        self._pending_count: int = 0   # pending 方向の連続回数
        self._hold_remaining: int = 0  # 確定後の残り保持回数

    @property
    def buckets_available(self) -> int:
        return len(self._buckets)

    def update(self, timestamp: float, mid_price: float) -> MacroRegimeResult:
        """mid_price を投入し、マクロ regime を更新.

        Args:
            timestamp: エポック秒.
            mid_price: 板の mid price.

        Returns:
            MacroRegimeResult
        """
        if not math.isfinite(mid_price) or mid_price <= 0:
            return self._insufficient_result()

        # バケット集約
        if self._current_bucket_start == 0.0:
            self._current_bucket_start = timestamp

        bucket_age = timestamp - self._current_bucket_start
        self._current_bucket_prices.append(mid_price)

        if bucket_age >= self.config.bucket_sec and self._current_bucket_prices:
            # バケット確定 → 平均価格を記録
            avg = sum(self._current_bucket_prices) / len(self._current_bucket_prices)
            ts_center = self._current_bucket_start + bucket_age / 2
            self._buckets.append((ts_center, avg))
            self._current_bucket_prices = [mid_price]
            self._current_bucket_start = timestamp

            # バッファ上限
            if len(self._buckets) > self.config.max_buckets:
                self._buckets = self._buckets[-self.config.max_buckets:]

        # データ不足判定
        if len(self._buckets) < self.config.slope_window_5m:
            return self._insufficient_result()

        # スロープ算出
        slope_5m = self._compute_slope(self.config.slope_window_5m)
        slope_15m = (
            self._compute_slope(self.config.slope_window_15m)
            if len(self._buckets) >= self.config.slope_window_15m
            else 0.0
        )

        # トレンド分類
        raw_trend, confidence = self._classify(slope_5m, slope_15m)

        # 458# ヒステリシス: フラッピング防止
        trend = self._apply_hysteresis(raw_trend)

        return MacroRegimeResult(
            trend=trend,
            slope_5m_bps_per_min=slope_5m,
            slope_15m_bps_per_min=slope_15m,
            confidence=confidence,
            buckets_available=len(self._buckets),
        )

    def _apply_hysteresis(self, raw_trend: MacroTrend) -> MacroTrend:
        """458# ヒステリシス: 連続 N 回一致でトレンド確定、確定後は最低 M 回保持.

        フラッピング防止用。raw の classify 結果をフィルタし、安定したトレンドのみ返す。
        """
        hyst_n = self.config.hysteresis_count
        hold_m = self.config.hold_count

        # 保持期間中は確定済みトレンドを維持
        if self._hold_remaining > 0:
            self._hold_remaining -= 1
            if raw_trend == self._confirmed_trend:
                # 同方向なら保持カウンタをリセット（延命）
                self._hold_remaining = hold_m
            return self._confirmed_trend

        # 新しい raw が pending と同じなら連続カウント加算
        if raw_trend == self._pending_trend:
            self._pending_count += 1
        else:
            self._pending_trend = raw_trend
            self._pending_count = 1

        # 連続 N 回到達で確定
        if self._pending_count >= hyst_n:
            if self._confirmed_trend != raw_trend:
                logger.debug(
                    "[458# macro_hyst] trend confirmed: %s → %s (after %d consecutive)",
                    self._confirmed_trend.value, raw_trend.value, hyst_n,
                )
            self._confirmed_trend = raw_trend
            self._hold_remaining = hold_m
            self._pending_count = 0

        return self._confirmed_trend

    def _compute_slope(self, window: int) -> float:
        """直近 window バケットの OLS 線形回帰スロープを bps/min で返す.

        Returns:
            スロープ (bps/min). 正=上昇, 負=下降.
        """
        if len(self._buckets) < window:
            return 0.0

        recent = self._buckets[-window:]
        ts = np.array([b[0] for b in recent], dtype=np.float64)
        px = np.array([b[1] for b in recent], dtype=np.float64)

        if px[0] <= 0 or not np.all(np.isfinite(px)):
            return 0.0

        # 時間を分単位に正規化
        t_min = (ts - ts[0]) / 60.0

        # OLS: slope = Σ(t-t̄)(p-p̄) / Σ(t-t̄)²
        t_mean = t_min.mean()
        p_mean = px.mean()
        dt = t_min - t_mean
        dp = px - p_mean
        denom = (dt * dt).sum()
        if denom < 1e-12:
            return 0.0

        slope_jpy_per_min = (dt * dp).sum() / denom
        # JPY/min → bps/min (基準価格 = 区間先頭)
        slope_bps = slope_jpy_per_min / px[0] * 10_000
        return float(slope_bps)

    def _classify(
        self, slope_5m: float, slope_15m: float,
    ) -> tuple[MacroTrend, float]:
        """スロープからマクロトレンドと confidence を分類."""
        thr = self.config.slope_threshold_bps_per_min
        strong_thr = self.config.strong_slope_threshold_bps_per_min

        up_5 = slope_5m > thr
        down_5 = slope_5m < -thr
        up_15 = slope_15m > thr
        down_15 = slope_15m < -thr

        has_15m = len(self._buckets) >= self.config.slope_window_15m

        if has_15m:
            if up_5 and up_15:
                excess = min(abs(slope_5m), abs(slope_15m)) / strong_thr
                conf = min(1.0, 0.6 + excess * 0.3)
                return MacroTrend.STRONG_UP, conf
            if down_5 and down_15:
                excess = min(abs(slope_5m), abs(slope_15m)) / strong_thr
                conf = min(1.0, 0.6 + excess * 0.3)
                return MacroTrend.STRONG_DOWN, conf
            if up_5 or up_15:
                return MacroTrend.WEAK_UP, 0.4
            if down_5 or down_15:
                return MacroTrend.WEAK_DOWN, 0.4
        else:
            # 15m データ不足 — 5m 単独判定
            if up_5:
                conf = min(0.7, 0.4 + abs(slope_5m) / strong_thr * 0.3)
                return MacroTrend.WEAK_UP, conf
            if down_5:
                conf = min(0.7, 0.4 + abs(slope_5m) / strong_thr * 0.3)
                return MacroTrend.WEAK_DOWN, conf

        return MacroTrend.NEUTRAL, 0.5

    def _insufficient_result(self) -> MacroRegimeResult:
        return MacroRegimeResult(
            trend=MacroTrend.INSUFFICIENT,
            buckets_available=len(self._buckets),
        )


def compose_regimes(
    micro_regime: str | None,
    micro_confidence: float,
    macro_result: MacroRegimeResult,
) -> tuple[str | None, bool]:
    """micro regime と macro trend を照合し、矛盾フラグを返す.

    Returns:
        (effective_regime, is_aligned)
        - is_aligned=False の場合、micro/macro が矛盾 (例: micro=trending_up, macro=strong_down)
        - 呼び出し元は矛盾時に regime を ranging に降格する等の制御に使用
    """
    if micro_regime is None or macro_result.trend == MacroTrend.INSUFFICIENT:
        return micro_regime, True

    aligned = True

    # 矛盾検出: micro trending_up ↔ macro strong_down (またはその逆)
    if micro_regime == "trending_up" and macro_result.trend in (
        MacroTrend.STRONG_DOWN, MacroTrend.WEAK_DOWN,
    ):
        aligned = False
    elif micro_regime == "trending_down" and macro_result.trend in (
        MacroTrend.STRONG_UP, MacroTrend.WEAK_UP,
    ):
        aligned = False

    return micro_regime, aligned
