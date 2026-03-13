#!/usr/bin/env python3
"""
Enhanced Risk Manager for SAC v445
Phase 3: Risk Management & Statistical Validation

既存のRiskManagerを拡張し、Phase 2のマルチタイムフレーム分析を統合した
高度なリスク管理システム。
"""

from typing import Any
import pandas as pd

from ztb.risk.risk_manager import RiskManager
from ztb.trading.signal.multi_timeframe_analyzer import MultiTimeframeAnalyzer, Timeframe, ConvergenceAnalysis
from ztb.trading.signal.trend_convergence_calculator import TrendConvergenceCalculator
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class EnhancedRiskManager(RiskManager):
    """
    Phase 2統合拡張リスクマネージャー

    既存のRiskManager機能を継承し、マルチタイムフレーム分析を統合。
    収束スコアと時間軸別リスク評価による高度なリスク管理を実現。
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: 拡張リスク管理設定
        """
        # 親クラスの初期化
        super().__init__(config)

        # Phase 2統合設定
        self.multi_timeframe_enabled = config.get("multi_timeframe_enabled", True)
        self.convergence_risk_weight = config.get("convergence_risk_weight", 0.3)
        self.timeframe_risk_weights = config.get("timeframe_risk_weights", {
            Timeframe.M1: 0.2,
            Timeframe.M5: 0.3,
            Timeframe.M15: 0.5
        })

        # Phase 2コンポーネント初期化
        if self.multi_timeframe_enabled:
            self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
            self.convergence_calculator = TrendConvergenceCalculator()
            logger.info("EnhancedRiskManager: Multi-timeframe analysis enabled")
        else:
            self.multi_timeframe_analyzer = None
            self.convergence_calculator = None
            logger.info("EnhancedRiskManager: Multi-timeframe analysis disabled")

    def calculate_enhanced_risk_adjusted_position(
        self,
        base_position: float,
        current_price: float,
        portfolio_value: float,
        atr: float,
        df: pd.DataFrame | None = None,
        step: int = 0,
    ) -> dict[str, Any]:
        """
        Phase 2統合拡張リスク調整済みポジション計算

        Args:
            base_position: 基本ポジションサイズ
            current_price: 現在の価格
            portfolio_value: ポートフォリオ価値
            atr: ATR値
            df: 市場データ（マルチタイムフレーム分析用）
            step: 現在のステップ

        Returns:
            拡張リスク調整情報辞書
        """
        # 基本リスク調整（既存RiskManagerを使用）
        basic_risk_info = self.calculate_risk_adjusted_position(
            base_position, current_price, portfolio_value, atr, df, step
        )

        # 緊急停止の場合はそのまま返す
        if basic_risk_info.get("control_active") and basic_risk_info.get("risk_level", 0.0) >= 1.0:
            return basic_risk_info

        # Phase 2統合拡張
        if self.multi_timeframe_enabled and df is not None:
            enhanced_info = self._apply_multi_timeframe_risk_adjustment(
                basic_risk_info, df, step
            )
            return enhanced_info
        else:
            return basic_risk_info

    def _apply_multi_timeframe_risk_adjustment(
        self,
        basic_risk_info: dict[str, Any],
        df: pd.DataFrame,
        step: int
    ) -> dict[str, Any]:
        """
        マルチタイムフレームベースのリスク調整を適用

        Args:
            basic_risk_info: 基本リスク調整情報
            df: 市場データ
            step: 現在のステップ

        Returns:
            拡張リスク調整情報
        """
        try:
            # マルチタイムフレーム分析実行
            convergence_analysis = self.multi_timeframe_analyzer.analyze_convergence()

            # 時間軸別トレンド分析取得
            timeframe_analyses = {}
            for timeframe in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
                analysis = self.multi_timeframe_analyzer.analyze_timeframe_trend(timeframe)
                if analysis:
                    timeframe_analyses[timeframe] = analysis

            # 収束スコアベースのリスク調整
            convergence_risk_multiplier = self._calculate_convergence_risk_multiplier(
                convergence_analysis, timeframe_analyses
            )

            # 時間軸別リスク評価
            timeframe_risk_multiplier = self._calculate_timeframe_risk_multiplier(
                timeframe_analyses
            )

            # 統合リスク乗数
            integrated_risk_multiplier = (
                convergence_risk_multiplier * self.convergence_risk_weight +
                timeframe_risk_multiplier * (1 - self.convergence_risk_weight)
            )

            # ポジションサイズ調整
            original_position = basic_risk_info["adjusted_position"]
            enhanced_position = original_position * integrated_risk_multiplier

            # リスクレベル更新
            original_risk_level = basic_risk_info.get("risk_level", 0.0)
            enhanced_risk_level = min(1.0, original_risk_level + (1 - integrated_risk_multiplier) * 0.3)

            # 拡張情報追加
            enhanced_info = basic_risk_info.copy()
            enhanced_info.update({
                "adjusted_position": enhanced_position,
                "risk_level": enhanced_risk_level,
                "multi_timeframe_adjusted": True,
                "convergence_score": convergence_analysis.convergence_score,
                "dominant_trend": convergence_analysis.dominant_trend.value,
                "timeframe_agreement": convergence_analysis.timeframe_agreement,
                "convergence_risk_multiplier": convergence_risk_multiplier,
                "timeframe_risk_multiplier": timeframe_risk_multiplier,
                "integrated_risk_multiplier": integrated_risk_multiplier,
                "reasons": basic_risk_info.get("reasons", []) + [
                    f"Multi-timeframe risk adjustment: {integrated_risk_multiplier:.3f}"
                ]
            })

            logger.debug(
                f"Enhanced risk adjustment: original={original_position:.4f}, "
                f"enhanced={enhanced_position:.4f}, multiplier={integrated_risk_multiplier:.3f}"
            )

            return enhanced_info

        except Exception as e:
            logger.warning(f"Multi-timeframe risk adjustment failed: {e}")
            return basic_risk_info

    def _calculate_convergence_risk_multiplier(
        self,
        convergence_analysis: ConvergenceAnalysis,
        timeframe_analyses: dict[Timeframe, Any]
    ) -> float:
        """
        収束スコアベースのリスク乗数計算

        Args:
            convergence_analysis: 収束分析結果
            timeframe_analyses: 時間軸別分析結果

        Returns:
            リスク乗数（0.5-1.5の範囲）
        """
        convergence_score = convergence_analysis.convergence_score

        # 収束スコアに基づくリスク評価
        # 高収束（80以上）：リスク低減（乗数上昇）
        # 低収束（50以下）：リスク増大（乗数低下）
        if convergence_score >= 80:
            base_multiplier = 1.2  # 高信頼度でポジション増加
        elif convergence_score >= 60:
            base_multiplier = 1.0  # 中程度の信頼度
        elif convergence_score >= 40:
            base_multiplier = 0.8  # 低信頼度でポジション減少
        else:
            base_multiplier = 0.6  # 非常に低い信頼度

        # 時間軸一致度による調整
        agreement_bonus = convergence_analysis.timeframe_agreement * 0.2

        return min(1.5, max(0.5, base_multiplier + agreement_bonus))

    def _calculate_timeframe_risk_multiplier(
        self,
        timeframe_analyses: dict[Timeframe, Any]
    ) -> float:
        """
        時間軸別リスク乗数計算

        Args:
            timeframe_analyses: 時間軸別分析結果

        Returns:
            リスク乗数（0.5-1.5の範囲）
        """
        if not timeframe_analyses:
            return 1.0

        risk_multipliers = []

        for timeframe, analysis in timeframe_analyses.items():
            weight = self.timeframe_risk_weights.get(timeframe, 0.33)

            # トレンド強度に基づくリスク評価
            strength = analysis.strength
            momentum = analysis.momentum

            # 強気トレンド：リスク低減
            # 弱気トレンド：リスク増大
            if strength > 70 and momentum > 20:
                timeframe_multiplier = 1.2
            elif strength > 50 and momentum > 0:
                timeframe_multiplier = 1.0
            elif strength < 30 or momentum < -20:
                timeframe_multiplier = 0.7
            else:
                timeframe_multiplier = 0.9

            risk_multipliers.append(timeframe_multiplier * weight)

        return min(1.5, max(0.5, sum(risk_multipliers)))

    def get_risk_dashboard(self) -> dict[str, Any]:
        """
        リスク管理ダッシュボード情報取得

        Returns:
            リスク状態の総合情報
        """
        dashboard = {
            "basic_risk_status": {
                "position_sizer_active": True,
                "drawdown_control_active": True,
                "market_adaptor_active": True
            },
            "multi_timeframe_status": {
                "enabled": self.multi_timeframe_enabled,
                "analyzer_available": self.multi_timeframe_analyzer is not None,
                "convergence_calculator_available": self.convergence_calculator is not None
            },
            "risk_weights": {
                "convergence_weight": self.convergence_risk_weight,
                "timeframe_weights": {tf.value: w for tf, w in self.timeframe_risk_weights.items()}
            }
        }

        # マルチタイムフレーム情報追加
        if self.multi_timeframe_enabled and self.multi_timeframe_analyzer:
            try:
                convergence = self.multi_timeframe_analyzer.analyze_convergence()
                dashboard["current_convergence"] = {
                    "score": convergence.convergence_score,
                    "dominant_trend": convergence.dominant_trend.value,
                    "agreement": convergence.timeframe_agreement
                }
            except Exception as e:
                dashboard["current_convergence"] = {"error": str(e)}

        return dashboard
