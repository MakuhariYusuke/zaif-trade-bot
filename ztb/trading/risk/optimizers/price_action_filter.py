"""
Phase 3-1: シグナル品質向上 - 価格アクションフィルタ

価格アクションのパターンを分析してシグナルの信頼性を向上させます。
ピンバー、インサイドバー、キーレベルブレイクなどのパターンを検知します。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.utils.performance_profiler import PerformanceProfiler


class PriceActionPattern(Enum):
    """価格アクションパターン"""

    PIN_BAR = "pin_bar"
    INSIDE_BAR = "inside_bar"
    OUTSIDE_BAR = "outside_bar"
    MARUBOZU = "marubozu"
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULL = "engulfing_bull"
    ENGULFING_BEAR = "engulfing_bear"
    KEY_LEVEL_BREAK = "key_level_break"
    SUPPORT_TEST = "support_test"
    RESISTANCE_TEST = "resistance_test"
    NORMAL = "normal"


@dataclass
class PriceActionAnalysisResult:
    """価格アクション分析結果"""

    pattern: PriceActionPattern
    strength: float  # 0-1のパターン強度
    direction: str  # 'bullish', 'bearish', 'neutral'
    key_level_proximity: float  # キーレベルへの近接度（0-1）
    momentum_alignment: float  # モメンタムとの整合性（0-1）
    risk_level: str  # 'low', 'medium', 'high'

    @property
    def is_bullish(self) -> bool:
        """強気パターンか"""
        return self.direction == "bullish" and self.strength > 0.6

    @property
    def is_bearish(self) -> bool:
        """弱気パターンか"""
        return self.direction == "bearish" and self.strength > 0.6

    @property
    def is_high_probability(self) -> bool:
        """高確率パターンか"""
        return self.strength > 0.7 and self.key_level_proximity > 0.8


@dataclass
class PriceActionFilterCriteria:
    """価格アクションフィルタ基準"""

    require_pattern_confirmation: bool = True
    min_pattern_strength: float = 0.6
    require_key_level_alignment: bool = True
    min_key_level_proximity: float = 0.7
    require_momentum_alignment: bool = True
    min_momentum_alignment: float = 0.6

    # パターン特化設定
    bullish_patterns_only: bool = False
    bearish_patterns_only: bool = False
    high_probability_only: bool = False

    # 動的調整
    adaptive_filtering: bool = True
    market_regime_adjustment: bool = True


class PriceActionFilter:
    """価格アクションフィルタ"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.filter_criteria = PriceActionFilterCriteria()
        self.pattern_history: List[PriceActionAnalysisResult] = []
        self.max_history_size = 1000  # メモリ管理のため履歴サイズを制限

        # キーレベル追跡
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []

    def analyze_price_action(
        self,
        market_data: pd.DataFrame,
        signal_timestamp: datetime,
        lookback_periods: int = 20,
    ) -> PriceActionAnalysisResult:
        """
        指定時刻の価格アクションを分析

        Args:
            market_data: 市場データ（OHLCカラムを含む）
            signal_timestamp: シグナル発生時刻
            lookback_periods: ルックバック期間

        Returns:
            PriceActionAnalysisResult: 価格アクション分析結果
        """
        required_columns = ["open", "high", "low", "close"]
        if (
            not all(col in market_data.columns for col in required_columns)
            or market_data.empty
        ):
            return PriceActionAnalysisResult(
                pattern=PriceActionPattern.NORMAL,
                strength=0.5,
                direction="neutral",
                key_level_proximity=0.5,
                momentum_alignment=0.5,
                risk_level="medium",
            )

        # 指定時刻の価格データを取得
        current_candle = self._get_candle_at_timestamp(market_data, signal_timestamp)

        if current_candle is None:
            return PriceActionAnalysisResult(
                pattern=PriceActionPattern.NORMAL,
                strength=0.5,
                direction="neutral",
                key_level_proximity=0.5,
                momentum_alignment=0.5,
                risk_level="medium",
            )

        # 過去データの取得
        historical_data = self._get_historical_price_data(
            market_data, signal_timestamp, lookback_periods
        )

        # パターン認識
        pattern, strength, direction = self._recognize_pattern(
            current_candle, historical_data
        )

        # キーレベル近接度計算
        key_level_proximity = self._calculate_key_level_proximity(
            current_candle, historical_data
        )

        # モメンタム整合性評価
        momentum_alignment = self._calculate_momentum_alignment(
            current_candle, historical_data
        )

        # リスクレベル評価
        risk_level = self._assess_risk_level(
            pattern, strength, key_level_proximity, momentum_alignment
        )

        result = PriceActionAnalysisResult(
            pattern=pattern,
            strength=strength,
            direction=direction,
            key_level_proximity=key_level_proximity,
            momentum_alignment=momentum_alignment,
            risk_level=risk_level,
        )

        # 履歴に保存（メモリ管理）
        self.pattern_history.append(result)
        if len(self.pattern_history) > self.max_history_size:
            # 古いデータを削除してメモリを節約
            self.pattern_history = self.pattern_history[-self.max_history_size // 2 :]

        # キーレベル更新
        self._update_key_levels(current_candle)

        return result

    def _get_candle_at_timestamp(
        self, market_data: pd.DataFrame, timestamp: datetime
    ) -> Optional[Dict[str, float]]:
        """指定時刻のローソク足データを取得"""
        if market_data.index.empty:
            return None

        # 最も近いタイムスタンプを見つける
        time_diffs = market_data.index - timestamp
        closest_idx = np.abs(time_diffs).argmin()

        # 差が大きすぎる場合はNone
        time_diff = abs(time_diffs[closest_idx])
        if time_diff > timedelta(hours=1):  # 1時間以内のデータのみ使用
            return None

        row = market_data.iloc[closest_idx]
        return {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
        }

    def _get_historical_price_data(
        self,
        market_data: pd.DataFrame,
        signal_timestamp: datetime,
        lookback_periods: int,
    ) -> pd.DataFrame:
        """過去の価格データを取得"""
        if market_data.index.empty:
            return pd.DataFrame()

        # 指定時刻より前のデータを取得
        mask = market_data.index <= signal_timestamp
        historical_data = market_data[mask]

        if len(historical_data) == 0:
            return pd.DataFrame()

        # 指定期間分のデータを返す
        return historical_data.tail(lookback_periods)

    def _recognize_pattern(
        self, current_candle: Dict[str, float], historical_data: pd.DataFrame
    ) -> Tuple[PriceActionPattern, float, str]:
        """価格アクションパターンを認識"""
        o, h, l, c = (
            current_candle["open"],
            current_candle["high"],
            current_candle["low"],
            current_candle["close"],
        )

        # 基本的な価格変動を計算
        body_size = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        if total_range == 0:
            return PriceActionPattern.DOJI, 0.8, "neutral"

        body_ratio = body_size / total_range
        upper_ratio = upper_shadow / total_range
        lower_ratio = lower_shadow / total_range

        # ピンバー検知（影が長く、実体が小さい）
        if body_ratio < 0.3:
            if upper_ratio > 0.6:
                return (
                    PriceActionPattern.PIN_BAR,
                    min(upper_ratio, 0.9),
                    "bearish",
                )  # 上髭ピンバー
            elif lower_ratio > 0.6:
                return (
                    PriceActionPattern.HAMMER,
                    min(lower_ratio, 0.9),
                    "bullish",
                )  # ハンマー

        # 丸坊主（実体がほとんど全体）
        if body_ratio > 0.8:
            direction = "bullish" if c > o else "bearish"
            return PriceActionPattern.MARUBOZU, body_ratio, direction

        # 十字線（実体が非常に小さい）
        if body_ratio < 0.1:
            return PriceActionPattern.DOJI, 0.7, "neutral"

        # インサイドバー検知（前回のレンジ内に収まる）
        if not historical_data.empty and len(historical_data) >= 1:
            prev_candle = historical_data.iloc[-1]
            prev_high, prev_low = prev_candle["high"], prev_candle["low"]

            if h <= prev_high and l >= prev_low:
                return PriceActionPattern.INSIDE_BAR, 0.8, "neutral"

        # アウトサイドバー検知（前回のレンジを突破）
        if not historical_data.empty and len(historical_data) >= 1:
            prev_candle = historical_data.iloc[-1]
            prev_high, prev_low = prev_candle["high"], prev_candle["low"]

            if h > prev_high and l < prev_low:
                direction = "bullish" if c > o else "bearish"
                return PriceActionPattern.OUTSIDE_BAR, 0.7, direction

        # エンゴルフィングパターン検知
        if not historical_data.empty and len(historical_data) >= 1:
            prev_candle = historical_data.iloc[-1]
            prev_o, prev_c = prev_candle["open"], prev_candle["close"]

            # 強気エンゴルフィング
            if c > o and prev_c < prev_o and c > prev_o and o < prev_c:
                return PriceActionPattern.ENGULFING_BULL, 0.8, "bullish"

            # 弱気エンゴルフィング
            if c < o and prev_c > prev_o and c < prev_o and o > prev_c:
                return PriceActionPattern.ENGULFING_BEAR, 0.8, "bearish"

        # 流星線（上髭が長く、下髭が短い）
        if upper_ratio > 0.6 and lower_ratio < 0.2 and c < o:
            return PriceActionPattern.SHOOTING_STAR, upper_ratio, "bearish"

        return PriceActionPattern.NORMAL, 0.5, "neutral"

    def _calculate_key_level_proximity(
        self, current_candle: Dict[str, float], historical_data: pd.DataFrame
    ) -> float:
        """キーレベルへの近接度を計算"""
        if not self.support_levels and not self.resistance_levels:
            return 0.5

        h, l, c = current_candle["high"], current_candle["low"], current_candle["close"]

        max_proximity = 0.0

        # サポートレベルとの近接度
        for support in self.support_levels:
            proximity = 1.0 - min(abs(c - support) / support, 1.0)
            max_proximity = max(max_proximity, proximity)

        # レジスタンスレベルとの近接度
        for resistance in self.resistance_levels:
            proximity = 1.0 - min(abs(c - resistance) / resistance, 1.0)
            max_proximity = max(max_proximity, proximity)

        return max_proximity

    def _calculate_momentum_alignment(
        self, current_candle: Dict[str, float], historical_data: pd.DataFrame
    ) -> float:
        """モメンタムとの整合性を計算"""
        if historical_data.empty or len(historical_data) < 5:
            return 0.5

        # 最近の価格トレンドを計算
        recent_prices = historical_data["close"].tail(5).values
        if len(recent_prices) < 2:
            return 0.5

        # 線形回帰でトレンドを計算
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]

        # 現在のキャンドルの方向
        current_direction = (
            1 if current_candle["close"] > current_candle["open"] else -1
        )

        # トレンドと方向の整合性
        trend_direction = 1 if slope > 0 else -1

        if current_direction == trend_direction:
            alignment = 0.8
        else:
            alignment = 0.3

        # トレンドの強さで調整
        trend_strength = min(abs(slope) / np.mean(recent_prices) * 100, 1.0)
        alignment = alignment * (0.5 + 0.5 * trend_strength)

        return alignment

    def _assess_risk_level(
        self,
        pattern: PriceActionPattern,
        strength: float,
        key_level_proximity: float,
        momentum_alignment: float,
    ) -> str:
        """リスクレベルを評価"""
        risk_score = 0.0

        # パターンによるリスク評価
        pattern_risks = {
            PriceActionPattern.PIN_BAR: 0.3,
            PriceActionPattern.INSIDE_BAR: 0.4,
            PriceActionPattern.OUTSIDE_BAR: 0.5,
            PriceActionPattern.MARUBOZU: 0.6,
            PriceActionPattern.DOJI: 0.7,
            PriceActionPattern.HAMMER: 0.3,
            PriceActionPattern.SHOOTING_STAR: 0.3,
            PriceActionPattern.ENGULFING_BULL: 0.4,
            PriceActionPattern.ENGULFING_BEAR: 0.4,
            PriceActionPattern.KEY_LEVEL_BREAK: 0.2,
            PriceActionPattern.SUPPORT_TEST: 0.3,
            PriceActionPattern.RESISTANCE_TEST: 0.3,
            PriceActionPattern.NORMAL: 0.5,
        }

        risk_score = pattern_risks.get(pattern, 0.5)

        # 強度による調整（弱いパターンはリスク高い）
        if strength < 0.5:
            risk_score += 0.2

        # キーレベル近接度による調整
        if key_level_proximity > 0.8:
            risk_score -= 0.2  # キーレベル付近は信頼性高い
        elif key_level_proximity < 0.3:
            risk_score += 0.1  # キーレベルから離れている

        # モメンタム整合性による調整
        if momentum_alignment > 0.7:
            risk_score -= 0.1
        elif momentum_alignment < 0.4:
            risk_score += 0.2

        risk_score = max(0.0, min(1.0, risk_score))

        # リスクレベル判定
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    def _update_key_levels(self, current_candle: Dict[str, float]):
        """キーレベルを更新"""
        h, l = current_candle["high"], current_candle["low"]

        # サポートレベル更新（安値の蓄積）
        if not self.support_levels or min(self.support_levels) > l:
            self.support_levels.append(l)
            self.support_levels = sorted(self.support_levels)[-10:]  # 最新10個保持

        # レジスタンスレベル更新（高値の蓄積）
        if not self.resistance_levels or max(self.resistance_levels) < h:
            self.resistance_levels.append(h)
            self.resistance_levels = sorted(self.resistance_levels)[
                -10:
            ]  # 最新10個保持

    def should_filter_signal(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        custom_criteria: Optional[PriceActionFilterCriteria] = None,
    ) -> Tuple[bool, str, PriceActionAnalysisResult]:
        """
        シグナルをフィルタリングすべきかを判定

        Returns:
            Tuple[bool, str, PriceActionAnalysisResult]: (フィルタリング可否, 理由, 分析結果)
        """
        criteria = custom_criteria or self.filter_criteria

        # 価格アクション分析
        try:
            signal_timestamp = pd.to_datetime(signal.get("timestamp", datetime.now()))
        except (ValueError, TypeError):
            signal_timestamp = datetime.now()

        analysis_result = self.analyze_price_action(market_data, signal_timestamp)

        # パターン強度チェック
        if (
            criteria.require_pattern_confirmation
            and analysis_result.strength < criteria.min_pattern_strength
        ):
            return (
                True,
                f"パターン強度が不十分です: {analysis_result.strength:.2f}",
                analysis_result,
            )

        # キーレベル整合性チェック
        if (
            criteria.require_key_level_alignment
            and analysis_result.key_level_proximity < criteria.min_key_level_proximity
        ):
            return (
                True,
                f"キーレベル近接度が不十分です: {analysis_result.key_level_proximity:.2f}",
                analysis_result,
            )

        # モメンタム整合性チェック
        if (
            criteria.require_momentum_alignment
            and analysis_result.momentum_alignment < criteria.min_momentum_alignment
        ):
            return (
                True,
                f"モメンタム整合性が不十分です: {analysis_result.momentum_alignment:.2f}",
                analysis_result,
            )

        # パターン方向チェック
        signal_type = signal.get("action", "hold")
        if criteria.bullish_patterns_only and not analysis_result.is_bullish:
            return True, "強気パターンのみ許可されています", analysis_result

        if criteria.bearish_patterns_only and not analysis_result.is_bearish:
            return True, "弱気パターンのみ許可されています", analysis_result

        # 高確率パターンのみチェック
        if criteria.high_probability_only and not analysis_result.is_high_probability:
            return True, "高確率パターンのみ許可されています", analysis_result

        return False, "シグナルは基準を満たしています", analysis_result

    def update_criteria_based_on_market(self, market_data: pd.DataFrame):
        """市場状況に応じてフィルタ基準を更新"""
        if not self.filter_criteria.adaptive_filtering:
            return

        # 最近の市場ボラティリティを計算
        if len(market_data) >= 20:
            recent_returns = market_data["close"].pct_change().tail(20)
            volatility = recent_returns.std()

            # 高ボラティリティ時は基準を厳しく
            if volatility > 0.03:
                self.filter_criteria.min_pattern_strength = min(
                    0.7, self.filter_criteria.min_pattern_strength + 0.1
                )
                self.filter_criteria.min_key_level_proximity = min(
                    0.8, self.filter_criteria.min_key_level_proximity + 0.1
                )
            # 低ボラティリティ時は基準を緩く
            elif volatility < 0.01:
                self.filter_criteria.min_pattern_strength = max(
                    0.5, self.filter_criteria.min_pattern_strength - 0.1
                )
                self.filter_criteria.min_key_level_proximity = max(
                    0.6, self.filter_criteria.min_key_level_proximity - 0.1
                )

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """パターン統計を取得"""
        if not self.pattern_history:
            return {}

        patterns = [result.pattern for result in self.pattern_history]
        strengths = [result.strength for result in self.pattern_history]
        proximities = [result.key_level_proximity for result in self.pattern_history]
        alignments = [result.momentum_alignment for result in self.pattern_history]

        return {
            "total_analyses": len(self.pattern_history),
            "pattern_distribution": {
                pattern.value: patterns.count(pattern) for pattern in set(patterns)
            },
            "average_strength": np.mean(strengths),
            "average_key_level_proximity": np.mean(proximities),
            "average_momentum_alignment": np.mean(alignments),
            "bullish_patterns": sum(
                1 for result in self.pattern_history if result.is_bullish
            ),
            "bearish_patterns": sum(
                1 for result in self.pattern_history if result.is_bearish
            ),
            "high_probability_patterns": sum(
                1 for result in self.pattern_history if result.is_high_probability
            ),
            "high_risk_patterns": sum(
                1 for result in self.pattern_history if result.risk_level == "high"
            ),
        }
