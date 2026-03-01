"""
軽量レジーム検知 — fill_test 実測サイクルの mid_price 系列からマーケット状態を分類.

035# §4 準拠.

設計原則:
  - 4 状態: trending / ranging / high_vol / unknown (035# §4.2 #1)
  - ヒステリシス: 連続 N サイクル一致で状態確定 (035# §4.2 #2)
  - 信頼度ゲート: confidence 低時は unknown で適応停止 (035# §4.2 #3)
  - レジーム別評価を必須化 (035# §4.2 #4)

既存資産再利用:
  - ztb/metrics/metrics.py::classify_market_regime の分類ロジックを軽量化
  - fill_test サイクル ≈120 秒で得られる mid_price のみを入力とする
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FillTestRegime(str, Enum):
    """fill_test 用の軽量レジーム分類.

    035# §4.2: 4 状態から開始.
    156# D-4: trending を方向別に分解 (trending_up / trending_down).
    後方互換: .is_trending プロパティで TRENDING/TRENDING_UP/TRENDING_DOWN を統一判定.
    """

    TRENDING = "trending"            # 方向不明 (レガシー互換)
    TRENDING_UP = "trending_up"      # 156# D-4: 上昇トレンド
    TRENDING_DOWN = "trending_down"  # 156# D-4: 下降トレンド
    RANGING = "ranging"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"

    @property
    def is_trending(self) -> bool:
        """trending 系 (方向問わず) かどうか."""
        return self in (FillTestRegime.TRENDING, FillTestRegime.TRENDING_UP, FillTestRegime.TRENDING_DOWN)


@dataclass
class RegimeConfig:
    """レジーム検知の設定."""

    # 検知ウィンドウ (直近 N 観測)
    window: int = 20

    # トレンド閾値: window 区間の価格変化率 (%) 以上でトレンド判定
    trend_threshold_pct: float = 0.5

    # 高ボラ判定: 現在のボラティリティが baseline の X 倍以上
    high_vol_multiplier: float = 2.0

    # ヒステリシス: 状態確定までに必要な連続一致数 (035# §4.2 #2)
    hysteresis_count: int = 3

    # 信頼度ゲート: この値未満は unknown 扱い (035# §4.2 #3)
    min_confidence: float = 0.4

    # 200# H: velocity modulation (opt-in)
    # trending regime の confidence を velocity で補正
    # velocity が trend 方向と一致 → confidence 強化, 不一致 → 弱化
    velocity_modulation: bool = False
    velocity_window_ratio: float = 0.25  # window の何割を velocity 計算に使うか


@dataclass
class RegimeResult:
    """レジーム検知の結果."""

    regime: FillTestRegime
    confidence: float  # 0.0–1.0
    stability: int  # 連続同一レジーム数
    trend_pct: float  # window 区間の価格変化率 (%)
    volatility_ratio: float  # 現在 vol / baseline vol

    def to_dict(self) -> dict[str, object]:
        """JSON serializable dict."""
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "stability": self.stability,
            "trend_pct": round(self.trend_pct, 4),
            "volatility_ratio": round(self.volatility_ratio, 4),
        }


class FillTestRegimeDetector:
    """fill_test サイクルの mid_price からレジームを軽量判定.

    使い方:
        detector = FillTestRegimeDetector(config)
        result = detector.update(timestamp, mid_price)
        if result.regime == FillTestRegime.UNKNOWN:
            # 適応停止
    """

    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        self.config = config or RegimeConfig()
        self._prices: list[tuple[float, float]] = []  # (timestamp, mid_price)
        self._raw_history: list[FillTestRegime] = []  # ヒステリシス用
        self._confirmed_regime: FillTestRegime = FillTestRegime.UNKNOWN
        self._stability_count: int = 0

    @property
    def current_regime(self) -> FillTestRegime:
        """現在確定中のレジーム."""
        return self._confirmed_regime

    @property
    def last_volatility_ratio(self) -> float:
        """168# 直近の volatility_ratio (maker_price 低 vol boost 用)."""
        if hasattr(self, "_last_result") and self._last_result is not None:
            return self._last_result.volatility_ratio
        return 1.0  # デフォルト: ブースト不発動

    @property
    def current_confidence(self) -> float:
        """182# 直近の confidence (Trend Mode 厳格化用)."""
        if hasattr(self, "_last_result") and self._last_result is not None:
            return self._last_result.confidence
        return 0.0

    @property
    def observation_count(self) -> int:
        """蓄積済み観測数."""
        return len(self._prices)

    def update(self, timestamp: float, mid_price: float) -> RegimeResult:
        """新しい mid_price を投入し、レジーム判定を更新.

        Args:
            timestamp: エポック秒.
            mid_price: 板の mid price.

        Returns:
            RegimeResult with current regime assessment.
        """
        self._prices.append((timestamp, mid_price))

        # バッファ上限: window の 3 倍まで保持 (baseline 算出用)
        max_buffer = self.config.window * 3
        if len(self._prices) > max_buffer:
            self._prices = self._prices[-max_buffer:]

        # データ不足 → unknown (confidence=0)
        if len(self._prices) < self.config.window:
            return RegimeResult(
                regime=FillTestRegime.UNKNOWN,
                confidence=0.0,
                stability=0,
                trend_pct=0.0,
                volatility_ratio=0.0,
            )

        # 指標算出
        trend_pct, vol_ratio = self._compute_indicators()

        # 分類
        raw_regime, confidence = self._classify(trend_pct, vol_ratio)

        # 信頼度ゲート (035# §4.2 #3)
        if confidence < self.config.min_confidence:
            raw_regime = FillTestRegime.UNKNOWN

        # ヒステリシス適用 (035# §4.2 #2)
        confirmed = self._apply_hysteresis(raw_regime)

        result = RegimeResult(
            regime=confirmed,
            confidence=confidence,
            stability=self._stability_count,
            trend_pct=trend_pct,
            volatility_ratio=vol_ratio,
        )
        # 168# §9.10: maker_price 低 vol boost 用にキャッシュ
        self._last_result = result
        return result

    def _compute_indicators(self) -> tuple[float, float]:
        """直近 window の trend% と volatility ratio を算出.

        Returns:
            (trend_pct, volatility_ratio)
        """
        recent = self._prices[-self.config.window :]
        prices = np.array([p[1] for p in recent], dtype=float)

        # trend: window 区間の価格変化率 (%)
        if (
            prices.size >= 2
            and np.isfinite(prices[0])
            and np.isfinite(prices[-1])
            and abs(prices[0]) > 1e-12
        ):
            trend_pct = (prices[-1] - prices[0]) / prices[0] * 100
        else:
            trend_pct = 0.0

        # returns (隣接比) — zero/invalid denominator は除外して NaN/inf 伝播を防止
        returns = self._safe_returns(prices)
        current_vol = float(np.std(returns)) if len(returns) > 1 else 0.0

        # baseline: 全バッファの returns の std
        all_prices = np.array([p[1] for p in self._prices], dtype=float)
        all_returns = self._safe_returns(all_prices)
        baseline_vol = float(np.std(all_returns)) if len(all_returns) > 1 else current_vol

        vol_ratio = current_vol / baseline_vol if baseline_vol > 1e-12 else 1.0

        # 200# H: 短期 velocity 計算 (velocity_modulation 有効時に _classify で使用)
        self._last_velocity_pct = 0.0
        if self.config.velocity_modulation:
            vel_n = max(2, int(self.config.window * self.config.velocity_window_ratio))
            vel_prices = prices[-vel_n:]
            if (
                vel_prices.size >= 2
                and np.isfinite(vel_prices[0])
                and abs(vel_prices[0]) > 1e-12
            ):
                self._last_velocity_pct = (vel_prices[-1] - vel_prices[0]) / vel_prices[0] * 100

        return trend_pct, vol_ratio

    @staticmethod
    def _safe_returns(prices: np.ndarray) -> np.ndarray:
        """価格列から有限な return のみを抽出する."""
        if prices.size < 2:
            return np.array([], dtype=float)
        prev = prices[:-1]
        diff = np.diff(prices)
        valid = np.isfinite(prev) & np.isfinite(diff) & (np.abs(prev) > 1e-12)
        if not np.any(valid):
            return np.array([], dtype=float)
        returns = diff[valid] / prev[valid]
        return returns[np.isfinite(returns)]

    def _classify(
        self, trend_pct: float, vol_ratio: float
    ) -> tuple[FillTestRegime, float]:
        """指標からレジームと信頼度を算出.

        Returns:
            (regime, confidence)
        """
        abs_trend = abs(trend_pct)
        threshold = self.config.trend_threshold_pct

        # 高ボラ判定が最優先
        if vol_ratio >= self.config.high_vol_multiplier:
            # 信頼度: multiplier をどれだけ超えたか (最大 1.0)
            excess = (vol_ratio - self.config.high_vol_multiplier) / self.config.high_vol_multiplier
            confidence = min(1.0, 0.6 + excess * 0.4)
            return FillTestRegime.HIGH_VOL, confidence

        # トレンド判定 — 156# D-4: 方向別に分解
        if abs_trend >= threshold:
            # 信頼度: threshold をどれだけ超えたか
            excess = (abs_trend - threshold) / threshold
            confidence = min(1.0, 0.5 + excess * 0.3)
            regime = (
                FillTestRegime.TRENDING_UP
                if trend_pct > 0
                else FillTestRegime.TRENDING_DOWN
            )
            # 200# H: velocity modulation — 短期 velocity が方向と一致するか
            if self.config.velocity_modulation:
                _vel = getattr(self, "_last_velocity_pct", 0.0)
                _trend_sign = 1.0 if trend_pct > 0 else -1.0
                _vel_sign = 1.0 if _vel > 0 else (-1.0 if _vel < 0 else 0.0)
                if _vel_sign == _trend_sign:
                    # 一致: confidence を最大 +0.15 強化
                    confidence = min(1.0, confidence + 0.15 * min(1.0, abs(_vel) / max(threshold, 0.01)))
                elif _vel_sign == -_trend_sign:
                    # 不一致: confidence を最大 -0.20 弱化 (反転兆候)
                    confidence = max(0.0, confidence - 0.20 * min(1.0, abs(_vel) / max(threshold, 0.01)))
            return regime, confidence

        # レンジ: トレンドも高ボラもない
        # 信頼度: threshold からの距離 (0 に近いほど確信が高い)
        proximity = 1.0 - (abs_trend / threshold) if threshold > 0 else 1.0
        confidence = min(1.0, 0.4 + proximity * 0.4)
        return FillTestRegime.RANGING, confidence

    def _apply_hysteresis(self, raw_regime: FillTestRegime) -> FillTestRegime:
        """ヒステリシス: raw 判定が N 回連続で一致して初めて状態遷移.

        035# §4.2 #2: 連続 N サイクル一致で状態確定.
        152# A: UNKNOWN → first regime は (N-1) 連続で確定 (初回遷移の加速).
        152# B: UNKNOWN が長期化した場合、直近 raw の最頻分類で仮確定.
        """
        self._raw_history.append(raw_regime)
        # raw_history もバウンド
        if len(self._raw_history) > self.config.hysteresis_count * 3:
            self._raw_history = self._raw_history[-self.config.hysteresis_count * 3 :]

        # 直近 N 回の連続一致をカウント
        consecutive = 0
        for r in reversed(self._raw_history):
            if r == raw_regime:
                consecutive += 1
            else:
                break

        if raw_regime == self._confirmed_regime:
            # 既確定レジームが継続
            self._stability_count = consecutive
            return self._confirmed_regime

        # 152# A: 初回遷移 (UNKNOWN →) は閾値を 1 下げて高速確定
        if self._confirmed_regime == FillTestRegime.UNKNOWN:
            threshold = max(2, self.config.hysteresis_count - 1)
        else:
            threshold = self.config.hysteresis_count

        if consecutive >= threshold:
            # 新レジームが十分な連続一致 → 遷移
            old = self._confirmed_regime
            self._confirmed_regime = raw_regime
            self._stability_count = consecutive
            logger.info(
                f"[Regime] transition: {old.value} → {raw_regime.value} "
                f"(consecutive={consecutive}, threshold={threshold})"
            )
            return raw_regime

        # 152# B: UNKNOWN が長期化 → 最頻分類フォールバック
        if (
            self._confirmed_regime == FillTestRegime.UNKNOWN
            and len(self._raw_history) >= self.config.hysteresis_count * 2
        ):
            from collections import Counter

            recent_raw = self._raw_history[-self.config.hysteresis_count * 2 :]
            non_unknown = [r for r in recent_raw if r != FillTestRegime.UNKNOWN]
            if non_unknown:
                majority, majority_count = Counter(non_unknown).most_common(1)[0]
                # 過半数以上の一致を要求
                if majority_count > len(recent_raw) // 2:
                    self._confirmed_regime = majority
                    self._stability_count = majority_count
                    logger.info(
                        f"[Regime] majority fallback: unknown → {majority.value} "
                        f"(count={majority_count}/{len(recent_raw)})"
                    )
                    return majority

        # 遷移未確定 → 旧レジーム維持
        self._stability_count += 1  # 旧が暫定的に続く
        return self._confirmed_regime

    def reset(self) -> None:
        """内部状態をリセット."""
        self._prices.clear()
        self._raw_history.clear()
        self._confirmed_regime = FillTestRegime.UNKNOWN
        self._stability_count = 0

    # --- 121# A4: state persistence support ---

    def get_state(self) -> dict:
        """永続化用の状態辞書を返す.

        FillTestStatePersistence に保存して再起動時の warm-up を省略.
        """
        return {
            "confirmed": self._confirmed_regime.value,
            "stability": self._stability_count,
            "prices": list(self._prices),  # [(ts, price), ...]
            "raw_history": [r.value for r in self._raw_history],
        }

    def restore_state(self, state: dict) -> bool:
        """永続化された状態から復元. 成功時 True.

        Args:
            state: get_state() で保存した辞書.

        Returns:
            復元に成功した場合 True.
        """
        try:
            confirmed_val = state.get("confirmed", "unknown")
            self._confirmed_regime = FillTestRegime(confirmed_val)
            self._stability_count = int(state.get("stability", 0))

            prices = state.get("prices", [])
            self._prices = [(float(p[0]), float(p[1])) for p in prices]

            raw_history = state.get("raw_history", [])
            self._raw_history = [FillTestRegime(v) for v in raw_history]

            logger.info(
                f"[Regime] state restored: regime={self._confirmed_regime.value}, "
                f"stability={self._stability_count}, prices={len(self._prices)}"
            )
            return True
        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning(f"[Regime] state restore failed: {e}")
            return False
