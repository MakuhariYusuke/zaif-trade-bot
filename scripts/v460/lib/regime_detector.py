"""
軽量レジーム検知 — fill_test 実測サイクルの mid_price 系列からマーケット状態を分類.

035# §4 準拠.

設計原則:
  - 4 状態: trending / ranging / high_vol / unknown (035# §4.2 #1)
  - ヒステリシス: 連続 N サイクル一致で状態確定 (035# §4.2 #2)
  - 信頼度ゲート: confidence 低時は unknown で適応停止 (035# §4.2 #3)
  - レジーム別評価を必須化 (035# §4.2 #4)

市場理論的根拠:
  **Markov-Switching Model** — Hamilton (1989) "A New Approach to the Economic
  Analysis of Nonstationary Time Series and the Business Cycle".
  市場状態を隠れマルコフ過程としてモデル化。本モジュールは
  二次モーメントと線形回帰スロープを状態変数として使用し、
  隠れ状態を trending / ranging / high_vol / unknown に分類する。

  **Adaptive Market Hypothesis (AMH)** — Lo (2004) "The Adaptive Markets
  Hypothesis: Market Efficiency from an Evolutionary Perspective".
  市場効率性は時間変動し、レジームに依存する。レジーム検知は
  AMH が予測する「市場状態依存の最適戦略」を実現する基盤となる。

  **ヒステリシスの意義**: 状態確定に連続 N サイクルを要求するのは、
  Bayes 更新 (posterior が十分な evidence で確定するまで待つ) の離散近似。

既存資産再利用:
  - ztb/metrics/metrics.py::classify_market_regime の分類ロジックを軽量化
  - fill_test サイクル ≈120 秒で得られる mid_price のみを入力とする
"""""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from scripts.v460.lib.bayesian_regime_filter import (
        BayesianRegimeFilter,
        BayesianRegimeResult,
    )

logger = logging.getLogger(__name__)


# =====================================================================
# 366# T5: Welford Online Variance — O(1) per update
# =====================================================================

class WelfordOnlineVar:
    """Welford (1962) online variance — sliding window 対応.

    従来の ``np.std(returns)`` は毎回 O(n) で全 window を再計算していたが、
    新旧サンプルの add/remove で O(1) 更新を実現する。

    数値安定性: Welford 原論文の再帰公式を使用し、
    catastrophic cancellation を回避。

    References:
        B. P. Welford (1962) "Note on a Method for Calculating Corrected
        Sums of Squares and Products". Technometrics 4(3):419-420.
    """

    __slots__ = ("_count", "_mean", "_m2")

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    def add(self, x: float) -> None:
        """サンプル追加 — O(1)."""
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def remove(self, x: float) -> None:
        """サンプル除去 (sliding window 用) — O(1).

        count=0 へのアンダーフロー時はリセット。
        """
        if self._count <= 1:
            self._count = 0
            self._mean = 0.0
            self._m2 = 0.0
            return
        delta = x - self._mean
        self._count -= 1
        self._mean -= delta / self._count
        delta2 = x - self._mean
        self._m2 -= delta * delta2
        # 数値誤差で M2 が微小負になる場合のガード
        if self._m2 < 0:
            self._m2 = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def variance(self) -> float:
        """母分散 (np.std と同等)."""
        return self._m2 / self._count if self._count > 0 else 0.0

    @property
    def std(self) -> float:
        """母標準偏差."""
        return math.sqrt(self.variance)

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0


@runtime_checkable
class RegimeDetectorLike(Protocol):
    """257# regime_detector の型安全 Protocol.

    maker_price / order_monitor / adaptation_engine 共用。
    ``object | None`` → ``RegimeDetectorLike | None`` に置換し、
    getattr / hasattr を排除する。
    """

    @property
    def current_regime(self) -> "FillTestRegime": ...

    @property
    def last_volatility_ratio(self) -> float: ...


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

    @property
    def is_high_vol(self) -> bool:
        """225# 高ボラティリティレジームかどうか."""
        return self == FillTestRegime.HIGH_VOL


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

    # 324# RSI 補強 (opt-in) — ztb 既存実装の活用
    # RSI がトレンド方向を確認 → confidence +0.10
    # RSI がトレンド方向と不一致 → confidence -0.15 (反転兆候)
    rsi_modulation: bool = True   # ztb/analysis/regime の RSI 計算を再利用
    rsi_period: int = 14          # Wilder RSI 標準期間


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

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
        self._prices: list[tuple[float, float]] = []  # (timestamp, mid_price)
        self._raw_history: list[FillTestRegime] = []  # ヒステリシス用
        self._confirmed_regime: FillTestRegime = FillTestRegime.UNKNOWN
        self._stability_count: int = 0
        # 230# H-4: 明示的初期化 (hasattr 排除)
        self._last_result: RegimeResult | None = None
        self._last_velocity_pct: float = 0.0
        # 366# T5: Welford online variance — O(1) per update
        self._window_welford = WelfordOnlineVar()   # 直近 window の returns 用
        self._all_welford = WelfordOnlineVar()       # 全 buffer の returns 用 (baseline)
        # C2 fix: 実際に add した return 値を deque に保持し、正確な remove を保証
        self._window_returns: deque[float] = deque()
        self._all_returns: deque[float] = deque()
        # 366# M2: Bayesian Regime Filter (オプショナル)
        self._bayesian_filter: BayesianRegimeFilter | None = None
        self._last_bayesian_result: BayesianRegimeResult | None = None

    @property
    def current_regime(self) -> FillTestRegime:
        """現在確定中のレジーム."""
        return self._confirmed_regime

    @property
    def last_volatility_ratio(self) -> float:
        """168# 直近の volatility_ratio (maker_price 低 vol boost 用)."""
        if self._last_result is not None:
            return self._last_result.volatility_ratio
        return 1.0  # デフォルト: ブースト不発動

    @property
    def current_confidence(self) -> float:
        """182# 直近の confidence (Trend Mode 厳格化用)."""
        if self._last_result is not None:
            return self._last_result.confidence
        return 0.0

    @property
    def observation_count(self) -> int:
        """蓄積済み観測数."""
        return len(self._prices)

    @property
    def bayesian_offset_multiplier(self) -> float:
        """366# M2: ベイズ事後確率加重の offset 乗数 (未配線時は 1.0)."""
        if self._last_bayesian_result is not None:
            return self._last_bayesian_result.offset_multiplier
        return 1.0

    def set_bayesian_filter(self, bf: BayesianRegimeFilter) -> None:
        """366# M2: Bayesian Regime Filter を注入."""
        self._bayesian_filter = bf
        logger.info("[Regime] Bayesian filter injected")

    def update(self, timestamp: float, mid_price: float) -> RegimeResult:
        """新しい mid_price を投入し、レジーム判定を更新.

        Args:
            timestamp: エポック秒.
            mid_price: 板の mid price.

        Returns:
            RegimeResult with current regime assessment.
        """
        # 366# T5: Welford incremental update
        prev_price = self._prices[-1][1] if self._prices else None
        self._prices.append((timestamp, mid_price))

        # 新しい return を Welford に追加
        if prev_price is not None and abs(prev_price) > 1e-12:
            new_ret = (mid_price - prev_price) / prev_price
            if math.isfinite(new_ret):
                self._all_welford.add(new_ret)
                self._all_returns.append(new_ret)
                self._window_welford.add(new_ret)
                self._window_returns.append(new_ret)

        # バッファ上限: window の 3 倍まで保持 (baseline 算出用)
        max_buffer = self.config.window * 3
        if len(self._prices) > max_buffer:
            self._prices = self._prices[-max_buffer:]
            # C2 fix: all_returns も同期してトリム
            while len(self._all_returns) > max_buffer - 1:
                old_ret = self._all_returns.popleft()
                self._all_welford.remove(old_ret)

        # C2 fix: window_returns から実際に add した値を pop して正確に remove
        window = self.config.window
        while len(self._window_returns) > max(0, window - 1):
            old_ret = self._window_returns.popleft()
            self._window_welford.remove(old_ret)

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

        # 366# M2: Bayesian filter で confidence を補完
        if self._bayesian_filter is not None and prev_price is not None and prev_price > 0:
            ret = (mid_price - prev_price) / prev_price
            bay_result = self._bayesian_filter.update(ret)
            self._last_bayesian_result = bay_result
            # ベイズ事後確率の MAP 確率で confidence を補正 (加重平均)
            bayes_conf = bay_result.map_probability
            result = RegimeResult(
                regime=confirmed,
                confidence=0.6 * confidence + 0.4 * bayes_conf,
                stability=self._stability_count,
                trend_pct=trend_pct,
                volatility_ratio=vol_ratio,
            )

        # 168# §9.10: maker_price 低 vol boost 用にキャッシュ
        self._last_result = result
        return result

    def _compute_indicators(self) -> tuple[float, float]:
        """直近 window の trend% と volatility ratio を算出.

        366# T5: Welford online variance で O(1) 化。
        _window_welford (window 内 returns の std) と
        _all_welford (全 buffer returns の std) を使用。
        np.std() フォールバックは Welford count 不足時のみ。

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

        # 366# T5: Welford O(1) variance (primary path)
        if self._window_welford.count >= 2:
            current_vol = self._window_welford.std
        else:
            # フォールバック: 従来の O(n) 計算
            returns = self._safe_returns(prices)
            current_vol = float(np.std(returns)) if len(returns) > 1 else 0.0

        if self._all_welford.count >= 2:
            baseline_vol = self._all_welford.std
        else:
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

    def _apply_rsi_modulation(
        self, confidence: float, trend_pct: float,
    ) -> float:
        """324# RSI による confidence 補強 — ztb 既存実装の活用.

        ztb/analysis/regime/advanced_regime_detector.py の
        TechnicalIndicators.calculate_rsi() と同一アルゴリズム (Wilder RSI)。
        DRY: 3 行の核心計算で inline 化し import 依存を回避。

        市場理論:
          J. Welles Wilder Jr. (1978) "New Concepts in Technical Trading Systems"
          RSI は相対的な上昇・下降の勢いを 0-100 で定量化。
          RSI > 50: 上昇モメンタム優勢 → trending_up の確信材料
          RSI < 50: 下降モメンタム優勢 → trending_down の確信材料
          RSI がトレンド方向と不一致 → 反転兆候 (divergence)

        Args:
            confidence: 現在の confidence (0.0-1.0)
            trend_pct: 価格変化率 (%) — 正=up, 負=down

        Returns:
            調整後の confidence
        """
        prices = np.array([p[1] for p in self._prices], dtype=float)
        period = self.config.rsi_period
        if prices.size < period + 1:
            return confidence

        # Wilder RSI core (ztb advanced_regime_detector 互換)
        deltas = np.diff(prices[-period - 1:])
        avg_gain = float(np.mean(np.maximum(deltas, 0.0)))
        avg_loss = float(np.mean(np.maximum(-deltas, 0.0)))
        if avg_loss < 1e-12:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # RSI とトレンド方向の一致/不一致で confidence を調整
        _trending_up = trend_pct > 0
        if _trending_up and rsi >= 55.0:
            # RSI がトレンド方向を確認 → confidence +0.10
            confidence = min(1.0, confidence + 0.10)
        elif _trending_up and rsi < 45.0:
            # RSI がトレンドと不一致 (bearish divergence) → confidence -0.15
            confidence = max(0.0, confidence - 0.15)
        elif not _trending_up and rsi <= 45.0:
            # RSI がトレンド方向を確認 → confidence +0.10
            confidence = min(1.0, confidence + 0.10)
        elif not _trending_up and rsi > 55.0:
            # RSI がトレンドと不一致 (bullish divergence) → confidence -0.15
            confidence = max(0.0, confidence - 0.15)
        # RSI 45-55 (中立) → 変更なし

        return confidence

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
                _vel = self._last_velocity_pct
                _trend_sign = 1.0 if trend_pct > 0 else -1.0
                _vel_sign = 1.0 if _vel > 0 else (-1.0 if _vel < 0 else 0.0)
                if _vel_sign == _trend_sign:
                    # 一致: confidence を最大 +0.15 強化
                    confidence = min(1.0, confidence + 0.15 * min(1.0, abs(_vel) / max(threshold, 0.01)))
                elif _vel_sign == -_trend_sign:
                    # 不一致: confidence を最大 -0.20 弱化 (反転兆候)
                    confidence = max(0.0, confidence - 0.20 * min(1.0, abs(_vel) / max(threshold, 0.01)))
            # 324# RSI modulation — ztb 既存実装の活用
            if self.config.rsi_modulation:
                confidence = self._apply_rsi_modulation(confidence, trend_pct)
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
        # C2 fix: Welford + return deque もリセット
        self._window_welford.reset()
        self._all_welford.reset()
        self._window_returns.clear()
        self._all_returns.clear()

    # --- 121# A4: state persistence support ---

    def get_state(self) -> dict:
        """永続化用の状態辞書を返す.

        FillTestStatePersistence に保存して再起動時の warm-up を省略.
        366# M2: Bayesian filter 状態も含める。
        """
        state: dict = {
            "confirmed": self._confirmed_regime.value,
            "stability": self._stability_count,
            "prices": list(self._prices),  # [(ts, price), ...]
            "raw_history": [r.value for r in self._raw_history],
        }
        # 366# M2: Bayesian filter state persistence
        if self._bayesian_filter is not None:
            state["bayesian_filter"] = self._bayesian_filter.get_state()
        return state

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

            # 366# M2: Bayesian filter state restoration
            bayes_state = state.get("bayesian_filter")
            if bayes_state is not None and self._bayesian_filter is not None:
                ok = self._bayesian_filter.restore_state(bayes_state)
                if ok:
                    logger.info("[Regime] Bayesian filter state restored")
                else:
                    logger.warning("[Regime] Bayesian filter state restore failed — using fresh prior")

            logger.info(
                f"[Regime] state restored: regime={self._confirmed_regime.value}, "
                f"stability={self._stability_count}, prices={len(self._prices)}"
            )
            return True
        except (ValueError, KeyError, IndexError, TypeError) as e:
            logger.warning(f"[Regime] state restore failed: {e}")
            return False
