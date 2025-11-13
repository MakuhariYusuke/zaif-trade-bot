"""
Enhanced Risk Manager with Multi-Timeframe Analysis
Phase 3統合: リスク調整済みシグナルスコアリング
実装完了: 2025年11月12日
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RiskAdjustedSignal:
    """リスク調整済みシグナル"""
    action: str
    score: float
    risk_multiplier: float
    position_size: float
    confidence: float


class EnhancedRiskManager:
    """マルチタイムフレームリスク管理"""

    def __init__(self):
        self.timeframes = ['1m', '5m', '15m', '1h']
        self.risk_limits = {
            'max_drawdown': 0.02,  # 2%
            'max_position': 0.10,  # 10%
            'min_confidence': 0.7   # 70%
        }

    def calculate_risk_adjusted_score(self,
                                     phase2_score: float,
                                     market_data: Dict[str, np.ndarray],
                                     volatility: float) -> RiskAdjustedSignal:
        """
        Phase 2スコアをリスク調整
        """
        # マルチタイムフレーム収束分析
        convergence_score = self._analyze_convergence(market_data)

        # 統計的バリデーション
        statistical_confidence = self._validate_statistically(market_data)

        # リスク乗数計算
        risk_multiplier = self._calculate_risk_multiplier(
            volatility, convergence_score, statistical_confidence
        )

        # リスク調整スコア
        adjusted_score = phase2_score * risk_multiplier

        # 動的ポジションサイズ
        position_size = self._calculate_dynamic_position(
            adjusted_score, volatility, convergence_score
        )

        # アクション判定
        action = self._determine_action(adjusted_score, position_size)

        return RiskAdjustedSignal(
            action=action,
            score=adjusted_score,
            risk_multiplier=risk_multiplier,
            position_size=position_size,
            confidence=statistical_confidence
        )

    def _analyze_convergence(self, market_data: Dict[str, np.ndarray]) -> float:
        """マルチタイムフレーム収束スコア"""
        convergence_scores = []

        for tf in self.timeframes:
            if tf in market_data:
                # トレンド方向の一致度を計算
                trend_alignment = self._calculate_trend_alignment(market_data[tf])
                convergence_scores.append(trend_alignment)

        return np.mean(convergence_scores) if convergence_scores else 0.5

    def _calculate_trend_alignment(self, prices: np.ndarray) -> float:
        """トレンド方向の一致度計算"""
        if len(prices) < 10:
            return 0.5

        # 短期トレンド（直近5期間）
        short_trend = np.polyfit(range(5), prices[-5:], 1)[0]

        # 中期トレンド（直近10期間）
        medium_trend = np.polyfit(range(10), prices[-10:], 1)[0]

        # 一致度：同じ方向なら1.0、逆方向なら0.0
        if (short_trend > 0 and medium_trend > 0) or (short_trend < 0 and medium_trend < 0):
            return 1.0
        elif (short_trend > 0 and medium_trend < 0) or (short_trend < 0 and medium_trend > 0):
            return 0.0
        else:
            return 0.5

    def _validate_statistically(self, market_data: Dict[str, np.ndarray]) -> float:
        """統計的バリデーション"""
        # 簡易的な統計的信頼度計算
        # 実際の実装ではより詳細な統計テストを行う
        confidence_scores = []

        for tf in self.timeframes:
            if tf in market_data:
                prices = market_data[tf]
                if len(prices) >= 20:
                    # トレンドの統計的有意性を計算
                    returns = np.diff(np.log(prices))
                    if len(returns) > 0:
                        # t検定でトレンドの有意性を確認
                        from scipy import stats
                        t_stat, p_value = stats.ttest_1samp(returns, 0)
                        confidence = max(0, min(1, 1 - p_value))  # p値から信頼度に変換
                        confidence_scores.append(confidence)

        return np.mean(confidence_scores) if confidence_scores else 0.5

    def _calculate_risk_multiplier(self,
                                 volatility: float,
                                 convergence: float,
                                 confidence: float) -> float:
        """リスク乗数計算（0.1-2.0倍）"""
        # ボラティリティによる調整
        vol_multiplier = 1.0 / (1.0 + volatility * 2)

        # 収束度による調整
        conv_multiplier = 0.5 + convergence * 0.5

        # 信頼性による調整
        conf_multiplier = 0.8 + confidence * 0.4

        # 総合リスク乗数
        risk_multiplier = vol_multiplier * conv_multiplier * conf_multiplier

        # 範囲制限
        return np.clip(risk_multiplier, 0.1, 2.0)

    def _calculate_dynamic_position(self,
                                  adjusted_score: float,
                                  volatility: float,
                                  convergence: float) -> float:
        """動的ポジションサイズ計算"""
        # ベースサイズ：スコアに基づく
        base_size = min(adjusted_score / 100.0, 1.0) * 0.05  # 最大5%

        # ボラティリティ調整：高ボラティリティ時は小さく
        vol_adjustment = 1.0 / (1.0 + volatility * 5)

        # 収束度調整：高収束時は大きく
        conv_adjustment = 0.5 + convergence * 0.5

        position_size = base_size * vol_adjustment * conv_adjustment

        # リスク制限
        return min(position_size, self.risk_limits['max_position'])

    def _determine_action(self, adjusted_score: float, position_size: float) -> str:
        """アクション判定"""
        if adjusted_score >= 70 and position_size > 0.001:
            return "BUY"
        elif adjusted_score <= 30:
            return "SELL"
        else:
            return "HOLD"