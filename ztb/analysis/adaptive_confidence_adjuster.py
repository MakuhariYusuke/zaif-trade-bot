"""
Phase 3-2: パラメータ最適化 - 動的信頼度閾値調整システム

市場条件とパフォーマンスに基づいてエントリー信頼度閾値を動的に調整します。
トレンド、ボラティリティ、レジームに応じた適応型閾値管理を実装します。
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

from ztb.analysis.market_regime_types import MarketRegime
from ztb.utils.performance_profiler import PerformanceProfiler


@dataclass
class ConfidenceThresholds:
    """信頼度閾値設定 - 12レジーム拡張版"""

    base_threshold: float = 0.7  # 基本閾値

    # トレンド系閾値
    strong_bull_threshold: float = 0.7   # 強い強気トレンド時
    moderate_bull_threshold: float = 0.65 # 中程度の強気トレンド時
    moderate_bear_threshold: float = 0.65 # 中程度の弱気トレンド時
    strong_bear_threshold: float = 0.6   # 強い弱気トレンド時

    # レンジ系閾値
    tight_range_threshold: float = 0.75    # 狭いレンジ時
    wide_range_threshold: float = 0.7      # 広いレンジ時
    volatile_range_threshold: float = 0.8  # ボラティリティの高いレンジ時
    quiet_range_threshold: float = 0.6     # ボラティリティの低いレンジ時

    # 特殊パターン閾値
    breakout_up_threshold: float = 0.7     # 上方向ブレイクアウト時
    breakout_down_threshold: float = 0.7   # 下方向ブレイクアウト時
    reversal_up_threshold: float = 0.75    # 上方向転換時
    reversal_down_threshold: float = 0.75  # 下方向転換時

    # 適応型調整パラメータ
    performance_adjustment_factor: float = 0.1  # パフォーマンス調整係数
    volatility_adjustment_factor: float = 0.05  # ボラティリティ調整係数
    min_threshold: float = 0.5  # 最小閾値
    max_threshold: float = 0.9  # 最大閾値

    def get_threshold_for_regime(self, regime: MarketRegime) -> float:
        """レジームに応じた閾値を取得"""
        regime_thresholds = {
            # トレンド系
            MarketRegime.STRONG_BULL: self.strong_bull_threshold,
            MarketRegime.MODERATE_BULL: self.moderate_bull_threshold,
            MarketRegime.MODERATE_BEAR: self.moderate_bear_threshold,
            MarketRegime.STRONG_BEAR: self.strong_bear_threshold,

            # レンジ系
            MarketRegime.TIGHT_RANGE: self.tight_range_threshold,
            MarketRegime.WIDE_RANGE: self.wide_range_threshold,
            MarketRegime.VOLATILE_RANGE: self.volatile_range_threshold,
            MarketRegime.QUIET_RANGE: self.quiet_range_threshold,

            # 特殊パターン
            MarketRegime.BREAKOUT_UP: self.breakout_up_threshold,
            MarketRegime.BREAKOUT_DOWN: self.breakout_down_threshold,
            MarketRegime.REVERSAL_UP: self.reversal_up_threshold,
            MarketRegime.REVERSAL_DOWN: self.reversal_down_threshold
        }
        return regime_thresholds.get(regime, self.base_threshold)

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'base_threshold': self.base_threshold,
            # トレンド系
            'strong_bull_threshold': self.strong_bull_threshold,
            'moderate_bull_threshold': self.moderate_bull_threshold,
            'moderate_bear_threshold': self.moderate_bear_threshold,
            'strong_bear_threshold': self.strong_bear_threshold,
            # レンジ系
            'tight_range_threshold': self.tight_range_threshold,
            'wide_range_threshold': self.wide_range_threshold,
            'volatile_range_threshold': self.volatile_range_threshold,
            'quiet_range_threshold': self.quiet_range_threshold,
            # 特殊パターン
            'breakout_up_threshold': self.breakout_up_threshold,
            'breakout_down_threshold': self.breakout_down_threshold,
            'reversal_up_threshold': self.reversal_up_threshold,
            'reversal_down_threshold': self.reversal_down_threshold,
            # 調整パラメータ
            'performance_adjustment_factor': self.performance_adjustment_factor,
            'volatility_adjustment_factor': self.volatility_adjustment_factor,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold
        }


@dataclass
class AdaptiveThresholdDecision:
    """適応型閾値決定結果"""

    current_threshold: float
    market_regime: MarketRegime
    base_threshold: float
    performance_adjustment: float
    volatility_adjustment: float
    final_threshold: float
    confidence_score: float
    reasoning: str
    timestamp: pd.Timestamp

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'current_threshold': self.current_threshold,
            'market_regime': self.market_regime.value,
            'base_threshold': self.base_threshold,
            'performance_adjustment': self.performance_adjustment,
            'volatility_adjustment': self.volatility_adjustment,
            'final_threshold': self.final_threshold,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }


class MarketRegimeDetector:
    """市場レジーム検出器"""

    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)

    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        市場レジームを検出 (12分類拡張版)

        Args:
            data: OHLCデータ

        Returns:
            検出された市場レジーム
        """
        if len(data) < self.lookback_periods:
            return MarketRegime.QUIET_RANGE

        recent_data = data.tail(self.lookback_periods)

        # 基本指標計算
        trend_strength = self._calculate_trend_strength(recent_data)
        volatility = self._calculate_volatility(recent_data)
        is_breakout = self._detect_breakout(recent_data)
        is_reversal = self._detect_reversal(recent_data)

        # トレンド強度による分類
        abs_trend = abs(trend_strength)

        if abs_trend > 0.7:  # 強いトレンド
            if trend_strength > 0:
                return MarketRegime.STRONG_BULL
            else:
                return MarketRegime.STRONG_BEAR
        elif abs_trend > 0.3:  # 中程度のトレンド
            if trend_strength > 0:
                return MarketRegime.MODERATE_BULL
            else:
                return MarketRegime.MODERATE_BEAR

        # レンジ/特殊パターン分類
        if is_reversal:
            # 転換パターン
            recent_trend = self._calculate_trend_strength(recent_data.head(10))
            if trend_strength > 0.2 and recent_trend < -0.2:
                return MarketRegime.REVERSAL_UP
            elif trend_strength < -0.2 and recent_trend > 0.2:
                return MarketRegime.REVERSAL_DOWN

        if is_breakout:
            # ブレイクアウトパターン
            breakout_direction = self._detect_breakout_direction(recent_data)
            if breakout_direction > 0:
                return MarketRegime.BREAKOUT_UP
            else:
                return MarketRegime.BREAKOUT_DOWN

        # ボラティリティベースのレンジ分類
        if volatility > 0.6:
            return MarketRegime.VOLATILE_RANGE
        elif volatility > 0.3:
            return MarketRegime.WIDE_RANGE
        elif volatility < 0.15:
            return MarketRegime.QUIET_RANGE
        else:
            return MarketRegime.TIGHT_RANGE

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """トレンド強度を計算"""
        closes = data['close']

        # 全体的な価格変化
        total_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]

        # ボラティリティで正規化
        returns = closes.pct_change().dropna()
        if len(returns) > 0:
            volatility = returns.std()
            vol_value = volatility.item() if hasattr(volatility, 'item') else float(volatility)
            if vol_value > 0:
                trend_strength = total_change / vol_value
                return np.clip(trend_strength, -5, 5)
            else:
                return total_change * 10  # ボラティリティがゼロの場合は大きな値
        else:
            return 0.0

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """ボラティリティを計算"""
        returns = data['close'].pct_change().dropna()

        if len(returns) < 5:
            return 0.5

        # 標準偏差をパーセンタイルに変換
        vol_std = returns.std()
        q10 = returns.quantile(0.1)
        q90 = returns.quantile(0.9)

        # ゼロ除算を防ぐ
        denominator = q90 - q10
        if denominator == 0 or np.isnan(denominator):
            # 全ての値が同じ場合（低ボラティリティ）
            return 0.0 if vol_std == 0 else 0.5

        vol_percentile = (vol_std - q10) / denominator
        vol_percentile = np.clip(vol_percentile, 0, 1)

        return vol_percentile

    def _detect_breakout(self, data: pd.DataFrame) -> bool:
        """ブレイクアウトを検出"""
        if len(data) < 20:
            return False

        recent_high = data['high'].tail(10).max()
        recent_low = data['low'].tail(10).min()
        prev_high = data['high'].iloc[-20:-10].max()
        prev_low = data['low'].iloc[-20:-10].min()

        # 最近の高値が過去の高値を大幅に上回る、または安値が過去の安値を大幅に下回る
        breakout_up = recent_high > prev_high * 1.05  # 5%以上のブレイク
        breakout_down = recent_low < prev_low * 0.95  # 5%以上のブレイク

        return bool(breakout_up or breakout_down)

    def _detect_breakout_direction(self, data: pd.DataFrame) -> int:
        """ブレイクアウトの方向を検出 (1: 上方向, -1: 下方向, 0: なし)"""
        if len(data) < 20:
            return 0

        recent_high = data['high'].tail(10).max()
        recent_low = data['low'].tail(10).min()
        prev_high = data['high'].iloc[-20:-10].max()
        prev_low = data['low'].iloc[-20:-10].min()

        breakout_up = recent_high > prev_high * 1.05
        breakout_down = recent_low < prev_low * 0.95

        if breakout_up and not breakout_down:
            return 1
        elif breakout_down and not breakout_up:
            return -1
        else:
            return 0

    def _detect_reversal(self, data: pd.DataFrame) -> bool:
        """トレンド転換を検出"""
        if len(data) < 30:
            return False

        # 短期トレンドと中期トレンドの比較
        short_trend = self._calculate_trend_strength(data.tail(10))
        medium_trend = self._calculate_trend_strength(data.tail(20))

        # トレンドの方向が逆転している場合
        trend_reversal = (short_trend * medium_trend < 0) and (abs(short_trend) > 0.3)

        # RSIのダイバージェンスも考慮（簡易版）
        if len(data) >= 14:
            closes = data['close']
            rsi_short = self._calculate_rsi(closes.tail(7))
            rsi_medium = self._calculate_rsi(closes.tail(14))

            rsi_divergence = abs(rsi_short - rsi_medium) > 20

            return bool(trend_reversal or rsi_divergence)

        return bool(trend_reversal)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSIを計算"""
        if len(prices) < period + 1:
            return 50.0

        deltas = prices.diff()
        gain = (deltas.where(deltas > 0, 0)).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50.0


class AdaptiveConfidenceAdjuster:
    """適応型信頼度調整器"""

    def __init__(self, thresholds: Optional[ConfidenceThresholds] = None):
        self.thresholds = thresholds or ConfidenceThresholds()
        self.regime_detector = MarketRegimeDetector()
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)

        # パフォーマンス履歴
        self.performance_history: List[Dict[str, Any]] = []
        self.threshold_history: List[float] = []

    def calculate_adaptive_threshold(
        self,
        data: pd.DataFrame,
        recent_performance: Optional[List[Dict[str, Any]]] = None,
        current_volatility: Optional[float] = None
    ) -> AdaptiveThresholdDecision:
        """
        適応型信頼度閾値を計算

        Args:
            data: 市場データ
            recent_performance: 最近のパフォーマンスデータ
            current_volatility: 現在のボラティリティ（Noneの場合は自動計算）

        Returns:
            適応型閾値決定結果
        """
        # 市場レジーム検出
        market_regime = self.regime_detector.detect_regime(data)

        # 基本閾値取得
        base_threshold = self.thresholds.get_threshold_for_regime(market_regime)

        # パフォーマンス調整
        performance_adjustment = self._calculate_performance_adjustment(recent_performance)

        # ボラティリティ調整
        volatility_adjustment = self._calculate_volatility_adjustment(data, current_volatility)

        # 最終閾値計算
        final_threshold = base_threshold + performance_adjustment + volatility_adjustment
        final_threshold = np.clip(final_threshold, self.thresholds.min_threshold, self.thresholds.max_threshold)

        # 信頼度スコア計算
        confidence_score = self._calculate_confidence_score(
            market_regime, performance_adjustment, volatility_adjustment
        )

        # 履歴更新
        self.threshold_history.append(final_threshold)
        if len(self.threshold_history) > 100:
            self.threshold_history.pop(0)

        reasoning = self._generate_reasoning(
            market_regime, base_threshold, performance_adjustment, volatility_adjustment, final_threshold
        )

        return AdaptiveThresholdDecision(
            current_threshold=final_threshold,
            market_regime=market_regime,
            base_threshold=base_threshold,
            performance_adjustment=performance_adjustment,
            volatility_adjustment=volatility_adjustment,
            final_threshold=final_threshold,
            confidence_score=confidence_score,
            reasoning=reasoning,
            timestamp=pd.Timestamp.now()
        )

    def _calculate_performance_adjustment(self, recent_performance: Optional[List[Dict[str, Any]]]) -> float:
        """パフォーマンスに基づく調整を計算"""
        if not recent_performance or len(recent_performance) < 5:
            return 0.0

        # 最近の勝率を計算
        recent_trades = recent_performance[-20:]  # 最新20トレード
        winning_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        win_rate = winning_trades / len(recent_trades)

        # 目標勝率（例: 60%）
        target_win_rate = 0.6
        win_rate_deviation = win_rate - target_win_rate

        # 勝率が低い場合は閾値を上げる、高い場合は下げる
        adjustment = -win_rate_deviation * self.thresholds.performance_adjustment_factor

        return adjustment

    def _calculate_volatility_adjustment(self, data: pd.DataFrame, current_volatility: Optional[float]) -> float:
        """ボラティリティに基づく調整を計算"""
        if current_volatility is None:
            # ATRベースのボラティリティ計算
            if len(data) >= 14:
                returns = data['close'].pct_change().dropna()
                current_volatility = float(returns.tail(14).std())
            else:
                current_volatility = 0.02  # デフォルト2%

        # ボラティリティの正規化（過去の分布に基づく）
        if len(self.threshold_history) > 10:
            historical_vol = np.array(self.threshold_history)
            vol_percentile = (current_volatility - np.percentile(historical_vol, 10)) / \
                           (np.percentile(historical_vol, 90) - np.percentile(historical_vol, 10))
            vol_percentile = np.clip(vol_percentile, 0, 1)
        else:
            vol_percentile = 0.5

        # 高ボラティリティ時は閾値を上げる
        if vol_percentile > 0.7:
            adjustment = self.thresholds.volatility_adjustment_factor
        elif vol_percentile < 0.3:
            adjustment = -self.thresholds.volatility_adjustment_factor
        else:
            adjustment = 0.0

        return adjustment

    def _calculate_confidence_score(self, regime: MarketRegime, perf_adj: float, vol_adj: float) -> float:
        """信頼度スコアを計算"""
        # レジームの安定性スコア
        regime_stability = {
            # トレンド系 (強いトレンドは不安定)
            MarketRegime.STRONG_BULL: 0.6,
            MarketRegime.MODERATE_BULL: 0.7,
            MarketRegime.MODERATE_BEAR: 0.7,
            MarketRegime.STRONG_BEAR: 0.6,
            
            # レンジ系 (安定した順)
            MarketRegime.QUIET_RANGE: 0.9,
            MarketRegime.TIGHT_RANGE: 0.8,
            MarketRegime.WIDE_RANGE: 0.7,
            MarketRegime.VOLATILE_RANGE: 0.4,
            
            # 特殊パターン
            MarketRegime.BREAKOUT_UP: 0.5,
            MarketRegime.BREAKOUT_DOWN: 0.5,
            MarketRegime.REVERSAL_UP: 0.45,
            MarketRegime.REVERSAL_DOWN: 0.45
        }.get(regime, 0.5)

        # 調整の安定性スコア
        adjustment_stability = 1.0 - abs(perf_adj + vol_adj) * 2  # 調整が大きいほどスコア低下

        return (regime_stability + adjustment_stability) / 2.0

    def _generate_reasoning(
        self,
        regime: MarketRegime,
        base_threshold: float,
        perf_adj: float,
        vol_adj: float,
        final_threshold: float
    ) -> str:
        """決定理由を生成"""
        parts = [
            f"市場レジーム: {regime.value}",
            f"基本閾値: {base_threshold:.2f}"
        ]

        if abs(perf_adj) > 0.01:
            direction = "引き上げ" if perf_adj > 0 else "引き下げ"
            parts.append(f"パフォーマンス調整: {direction} ({perf_adj:+.3f})")

        if abs(vol_adj) > 0.01:
            direction = "引き上げ" if vol_adj > 0 else "引き下げ"
            parts.append(f"ボラティリティ調整: {direction} ({vol_adj:+.3f})")

        parts.append(f"最終閾値: {final_threshold:.2f}")

        return " | ".join(parts)

    @PerformanceProfiler.profile
    def optimize_thresholds(
        self,
        historical_data: pd.DataFrame,
        performance_data: List[Dict[str, Any]],
        threshold_ranges: Optional[Dict[str, List[float]]] = None
    ) -> ConfidenceThresholds:
        """
        閾値パラメータを最適化

        Args:
            historical_data: 過去の市場データ
            performance_data: パフォーマンスデータ
            threshold_ranges: 最適化範囲

        Returns:
            最適化された閾値設定
        """
        if threshold_ranges is None:
            threshold_ranges = {
                'base_threshold': [0.6, 0.65, 0.7, 0.75, 0.8],
                'performance_adjustment_factor': [0.05, 0.1, 0.15, 0.2],
                'volatility_adjustment_factor': [0.02, 0.05, 0.08, 0.1]
            }

        best_thresholds = None
        best_score = float('-inf')

        # グリッドサーチ
        for base_thresh in threshold_ranges['base_threshold']:
            for perf_factor in threshold_ranges['performance_adjustment_factor']:
                for vol_factor in threshold_ranges['volatility_adjustment_factor']:

                    test_thresholds = ConfidenceThresholds(
                        base_threshold=base_thresh,
                        performance_adjustment_factor=perf_factor,
                        volatility_adjustment_factor=vol_factor
                    )

                    # 一時的に設定を変更して評価
                    original_thresholds = self.thresholds
                    self.thresholds = test_thresholds

                    try:
                        score = self._evaluate_thresholds(historical_data, performance_data)
                        if score > best_score:
                            best_score = score
                            best_thresholds = test_thresholds
                    finally:
                        self.thresholds = original_thresholds

        self.logger.info(f"閾値最適化完了: 最高スコア = {best_score:.3f}")
        return best_thresholds or self.thresholds

    def _evaluate_thresholds(self, data: pd.DataFrame, performance: List[Dict[str, Any]]) -> float:
        """閾値設定を評価"""
        total_score = 0
        evaluation_periods = min(50, len(data) // 10)  # 最大50期間

        for i in range(evaluation_periods):
            start_idx = i * 10
            end_idx = min((i + 1) * 10, len(data))

            period_data = data.iloc[start_idx:end_idx]
            period_performance = performance[start_idx:end_idx] if start_idx < len(performance) else []

            # 適応型閾値計算
            decision = self.calculate_adaptive_threshold(period_data, period_performance)

            # 勝率をスコアとして使用
            win_rate = sum(1 for p in period_performance if p.get('pnl', 0) > 0) / len(period_performance) \
                      if period_performance else 0.5

            # 閾値が適切かどうかを評価（勝率が60%付近が理想）
            threshold_score = 1.0 - abs(win_rate - 0.6) * 2

            total_score += threshold_score

        return total_score / evaluation_periods if evaluation_periods > 0 else 0.0


# ===== 使用例 =====

if __name__ == "__main__":
    """使用例"""

    # 適応型信頼度調整器の初期化
    adjuster = AdaptiveConfidenceAdjuster()

    # サンプルOHLCデータ（強気トレンド）
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    np.random.seed(42)

    # 強気トレンドのシミュレーション
    base_price = 100
    trend = np.linspace(0, 20, 50)  # 上昇トレンド
    noise = np.random.randn(50) * 2

    sample_data = pd.DataFrame({
        'open': base_price + trend + noise,
        'high': base_price + trend + noise + 1,
        'low': base_price + trend + noise - 1,
        'close': base_price + trend + noise + 0.5
    }, index=dates)

    # サンプルパフォーマンスデータ
    sample_performance = [
        {'pnl': 100, 'confidence': 0.8},
        {'pnl': -50, 'confidence': 0.6},
        {'pnl': 150, 'confidence': 0.9},
        {'pnl': -30, 'confidence': 0.7},
        {'pnl': 200, 'confidence': 0.85},
    ]

    # 適応型閾値計算
    decision = adjuster.calculate_adaptive_threshold(sample_data, sample_performance)

    print(f"適応型閾値決定:")
    print(f"  市場レジーム: {decision.market_regime.value}")
    print(f"  基本閾値: {decision.base_threshold:.2f}")
    print(f"  パフォーマンス調整: {decision.performance_adjustment:+.3f}")
    print(f"  ボラティリティ調整: {decision.volatility_adjustment:+.3f}")
    print(f"  最終閾値: {decision.final_threshold:.2f}")
    print(f"  信頼度スコア: {decision.confidence_score:.2f}")
    print(f"  理由: {decision.reasoning}")

    # 市場レジーム検出器のテスト
    detector = MarketRegimeDetector()
    regime = detector.detect_regime(sample_data)
    print(f"\n検出された市場レジーム: {regime.value}")