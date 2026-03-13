"""
Phase 3-1: シグナル品質向上 - 統合シグナルフィルタ

複数のフィルタを統合して総合的なシグナル品質評価を行います。
ボリュームフィルタ、価格アクションフィルタ、コンフィデンススコアリングを組み合わせます。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ztb.trading.risk.optimizers.price_action_filter import PriceActionFilter
from ztb.trading.risk.optimizers.volume_filter import VolumeFilter
from ztb.trading.strategies.risk_management.confidence_scoring_engine import (
    ConfidenceScoringEngine,
)
from ztb.utils.performance_profiler import PerformanceProfiler

class SignalQuality(Enum):
    """シグナル品質レベル"""

    EXCELLENT = "excellent"  # 最高品質
    HIGH = "high"  # 高品質
    MEDIUM = "medium"  # 中品質
    LOW = "low"  # 低品質
    POOR = "poor"  # 品質不良

@dataclass
class IntegratedFilterResult:
    """統合フィルタ結果"""

    overall_quality: SignalQuality
    quality_score: float  # 0-1の総合品質スコア
    confidence_score: float
    volume_analysis: Any  # VolumeAnalysisResult
    price_action_analysis: Any  # PriceActionAnalysisResult
    filter_reasons: list[str]
    recommended_action: str  # 'accept', 'reject', 'review'
    risk_assessment: str

    @property
    def should_accept(self) -> bool:
        """シグナルを受け入れるべきか"""
        return self.recommended_action == "accept"

    @property
    def needs_review(self) -> bool:
        """レビューが必要か"""
        return self.recommended_action == "review"

@dataclass
class IntegratedFilterCriteria:
    """統合フィルタ基準"""

    # 品質閾値
    min_quality_score: float = 0.6
    min_confidence_score: float = 0.7

    # フィルタ重み付け
    volume_weight: float = 0.3
    price_action_weight: float = 0.4
    confidence_weight: float = 0.3

    # 厳格さ設定
    strict_mode: bool = False  # 厳格モード（全ての基準を満たす必要）
    permissive_mode: bool = False  # 寛容モード（一部の基準で可）

    # 動的調整
    adaptive_filtering: bool = True
    market_regime_adjustment: bool = True

class IntegratedSignalFilter:
    """統合シグナルフィルタ"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.filter_criteria = IntegratedFilterCriteria()

        # 個別フィルタの初期化
        self.confidence_engine = ConfidenceScoringEngine()
        self.volume_filter = VolumeFilter()
        self.price_action_filter = PriceActionFilter()

        self.filter_history: list[IntegratedFilterResult] = []
        self.max_history_size = 1000  # メモリ管理のため履歴サイズを制限

    def evaluate_signal_quality(
        self,
        signal: dict[str, Any],
        market_data: pd.DataFrame,
        additional_context: dict[str, Any] | None = None,
    ) -> IntegratedFilterResult:
        """
        シグナルの総合品質を評価

        Args:
            signal: シグナル情報
            market_data: 市場データ
            additional_context: 追加コンテキスト

        Returns:
            IntegratedFilterResult: 統合評価結果
        """
        filter_reasons = []

        # 1. コンフィデンススコアリング
        confidence_result = self.confidence_engine.calculate_confidence_score(
            signal, market_data
        )
        confidence_score = confidence_result.total_score

        # 2. ボリューム分析
        (
            volume_should_filter,
            volume_reason,
            volume_analysis,
        ) = self.volume_filter.should_filter_signal(signal, market_data)
        if volume_should_filter:
            filter_reasons.append(f"ボリューム: {volume_reason}")

        # 3. 価格アクション分析
        (
            price_action_should_filter,
            price_action_reason,
            price_action_analysis,
        ) = self.price_action_filter.should_filter_signal(signal, market_data)
        if price_action_should_filter:
            filter_reasons.append(f"価格アクション: {price_action_reason}")

        # 個別スコアの計算
        volume_score = 1.0 - (
            0.5 if volume_should_filter else 0.0
        )  # フィルタ通過で1.0、失敗で0.5
        price_action_score = 1.0 - (0.5 if price_action_should_filter else 0.0)

        # 重み付けによる総合品質スコア計算
        quality_score = (
            self.filter_criteria.volume_weight * volume_score
            + self.filter_criteria.price_action_weight * price_action_score
            + self.filter_criteria.confidence_weight * confidence_score
        )

        # 品質レベルの判定
        overall_quality = self._determine_quality_level(quality_score, confidence_score)

        # 推奨アクションの決定
        recommended_action, risk_assessment = self._determine_recommended_action(
            quality_score, confidence_score, filter_reasons, signal
        )

        result = IntegratedFilterResult(
            overall_quality=overall_quality,
            quality_score=quality_score,
            confidence_score=confidence_score,
            volume_analysis=volume_analysis,
            price_action_analysis=price_action_analysis,
            filter_reasons=filter_reasons,
            recommended_action=recommended_action,
            risk_assessment=risk_assessment,
        )

        # 履歴に保存（メモリ管理）
        self.filter_history.append(result)
        if len(self.filter_history) > self.max_history_size:
            # 古いデータを削除してメモリを節約
            self.filter_history = self.filter_history[-self.max_history_size // 2 :]

        # 市場状況に応じた基準更新
        self._update_criteria_based_on_performance()

        return result

    def _determine_quality_level(
        self, quality_score: float, confidence_score: float
    ) -> SignalQuality:
        """品質レベルを判定"""
        combined_score = (quality_score + confidence_score) / 2

        if combined_score >= 0.85:
            return SignalQuality.EXCELLENT
        elif combined_score >= 0.75:
            return SignalQuality.HIGH
        elif combined_score >= 0.60:
            return SignalQuality.MEDIUM
        elif combined_score >= 0.45:
            return SignalQuality.LOW
        else:
            return SignalQuality.POOR

    def _determine_recommended_action(
        self,
        quality_score: float,
        confidence_score: float,
        filter_reasons: list[str],
        signal: dict[str, Any],
    ) -> tuple[str, str]:
        """推奨アクションを決定"""
        # 厳格モード
        if self.filter_criteria.strict_mode:
            if (
                quality_score >= self.filter_criteria.min_quality_score
                and confidence_score >= self.filter_criteria.min_confidence_score
                and not filter_reasons
            ):
                return "accept", "low"
            else:
                return "reject", "high"

        # 寛容モード
        if self.filter_criteria.permissive_mode:
            if quality_score >= 0.4 or confidence_score >= 0.5:
                risk_level = "medium" if quality_score < 0.6 else "low"
                return "accept", risk_level
            else:
                return "reject", "high"

        # 通常モード
        min_quality = self.filter_criteria.min_quality_score
        min_confidence = self.filter_criteria.min_confidence_score

        if quality_score >= min_quality and confidence_score >= min_confidence:
            if not filter_reasons:
                return "accept", "low"
            else:
                return "review", "medium"
        elif (
            quality_score >= min_quality * 0.8
            or confidence_score >= min_confidence * 0.8
        ):
            return "review", "medium"
        else:
            return "reject", "high"

    def _update_criteria_based_on_performance(self):
        """パフォーマンスに基づいて基準を更新"""
        if not self.filter_criteria.adaptive_filtering or len(self.filter_history) < 10:
            return

        # 最近の結果を分析
        recent_results = self.filter_history[-10:]
        accept_rate = sum(
            1 for r in recent_results if r.recommended_action == "accept"
        ) / len(recent_results)

        # 受け入れ率が低すぎる場合、基準を緩く
        if accept_rate < 0.3:
            self.filter_criteria.min_quality_score = max(
                0.4, self.filter_criteria.min_quality_score - 0.05
            )
            self.filter_criteria.min_confidence_score = max(
                0.5, self.filter_criteria.min_confidence_score - 0.05
            )

        # 受け入れ率が高すぎる場合、基準を厳しく
        elif accept_rate > 0.7:
            self.filter_criteria.min_quality_score = min(
                0.8, self.filter_criteria.min_quality_score + 0.05
            )
            self.filter_criteria.min_confidence_score = min(
                0.9, self.filter_criteria.min_confidence_score + 0.05
            )

    def batch_evaluate_signals(
        self, signals: list[dict[str, Any]], market_data: pd.DataFrame
    ) -> list[IntegratedFilterResult]:
        """
        複数のシグナルを一括評価

        Args:
            signals: シグナルリスト
            market_data: 市場データ

        Returns:
            list[IntegratedFilterResult]: 評価結果リスト
        """
        results = []

        for signal in signals:
            result = self.evaluate_signal_quality(signal, market_data)
            results.append(result)

        return results

    def get_filter_statistics(self) -> dict[str, Any]:
        """フィルタ統計を取得"""
        if not self.filter_history:
            return {}

        qualities = [result.overall_quality for result in self.filter_history]
        quality_scores = [result.quality_score for result in self.filter_history]
        confidence_scores = [result.confidence_score for result in self.filter_history]
        actions = [result.recommended_action for result in self.filter_history]

        return {
            "total_evaluations": len(self.filter_history),
            "quality_distribution": {
                quality.value: qualities.count(quality) for quality in set(qualities)
            },
            "action_distribution": {
                action: actions.count(action) for action in set(actions)
            },
            "average_quality_score": np.mean(quality_scores),
            "average_confidence_score": np.mean(confidence_scores),
            "median_quality_score": np.median(quality_scores),
            "acceptance_rate": actions.count("accept") / len(actions),
            "rejection_rate": actions.count("reject") / len(actions),
            "review_rate": actions.count("review") / len(actions),
        }

    def update_market_regime(self, market_regime: str):
        """市場レジームに応じてフィルタを調整"""
        if not self.filter_criteria.market_regime_adjustment:
            return

        if market_regime == "bull":
            # 強気市場では基準をやや緩く
            self.filter_criteria.min_quality_score = max(
                0.5, self.filter_criteria.min_quality_score - 0.1
            )
            self.filter_criteria.min_confidence_score = max(
                0.6, self.filter_criteria.min_confidence_score - 0.1
            )

        elif market_regime == "bear":
            # 弱気市場では基準を厳しく
            self.filter_criteria.min_quality_score = min(
                0.7, self.filter_criteria.min_quality_score + 0.1
            )
            self.filter_criteria.min_confidence_score = min(
                0.8, self.filter_criteria.min_confidence_score + 0.1
            )

        elif market_regime == "sideways":
            # 横ばい市場ではバランスよく
            self.filter_criteria.min_quality_score = 0.6
            self.filter_criteria.min_confidence_score = 0.7

        elif market_regime == "volatile":
            # 高ボラティリティ市場では厳しく
            self.filter_criteria.min_quality_score = min(
                0.8, self.filter_criteria.min_quality_score + 0.1
            )
            self.filter_criteria.min_confidence_score = min(
                0.9, self.filter_criteria.min_confidence_score + 0.1
            )

    def reset_adaptive_criteria(self):
        """適応基準をリセット"""
        self.filter_criteria.min_quality_score = 0.6
        self.filter_criteria.min_confidence_score = 0.7

    def export_filter_configuration(self) -> dict[str, Any]:
        """フィルタ設定をエクスポート"""
        return {
            "criteria": self.filter_criteria.__dict__,
            "statistics": self.get_filter_statistics(),
            "individual_filters": {
                "volume": self.volume_filter.get_volume_statistics(),
                "price_action": self.price_action_filter.get_pattern_statistics(),
                "confidence": {
                    "total_evaluations": len(self.confidence_engine.historical_scores),
                    "average_score": np.mean(
                        [
                            s.total_score
                            for s in self.confidence_engine.historical_scores
                        ]
                    )
                    if self.confidence_engine.historical_scores
                    else 0.0,
                },
            },
        }
