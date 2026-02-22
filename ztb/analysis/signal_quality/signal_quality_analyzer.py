"""
Phase 3-1: シグナル品質向上 - シグナル品質評価メトリクス

シグナルの品質を評価するためのメトリクスと分析ツールを提供します。
偽陽性削減と真陽性増加のための基盤となる評価システムです。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ztb.metrics.metrics import max_drawdown
from ztb.utils.performance_profiler import PerformanceProfiler


@dataclass
class SignalQualityMetrics:
    """シグナル品質メトリクス"""

    # 基本メトリクス
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0

    # 品質メトリクス
    true_positives: int = 0  # 正しく予測した買いシグナル
    false_positives: int = 0  # 誤った買いシグナル
    true_negatives: int = 0  # 正しく予測した売りシグナル
    false_negatives: int = 0  # 見逃した売りシグナル

    # 性能メトリクス
    precision: float = 0.0  # 適合率: TP / (TP + FP)
    recall: float = 0.0  # 再現率: TP / (TP + FN)
    f1_score: float = 0.0  # F1スコア: 2 * (precision * recall) / (precision + recall)

    # 市場適応メトリクス
    market_alignment: float = 0.0  # 市場トレンドとの整合性
    volume_confirmation: float = 0.0  # 出来高確認の割合
    timeframe_consistency: float = 0.0  # 複数時間軸整合性

    # リスクメトリクス
    drawdown_impact: float = 0.0  # ドローダウンへの影響度
    volatility_alignment: float = 0.0  # ボラティリティ適応度


@dataclass
class SignalEvaluationResult:
    """シグナル評価結果"""

    signal_id: str
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    quality_score: float
    market_context: Dict[str, Any]
    evaluation_metrics: SignalQualityMetrics

    # 評価結果
    is_high_quality: bool = False
    recommended_action: str = "hold"
    risk_adjusted_score: float = 0.0


class SignalQualityAnalyzer:
    """シグナル品質分析器"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.evaluation_history: List[SignalEvaluationResult] = []
        self.max_history_size = 1000  # メモリ管理のため履歴サイズを制限

    def evaluate_signal_quality(
        self,
        signals: List[Dict[str, Any]],
        market_data: pd.DataFrame,
        evaluation_window: int = 24,  # 評価期間（時間）
    ) -> SignalQualityMetrics:
        """
        シグナルの品質を評価

        Args:
            signals: 評価対象のシグナルリスト
            market_data: 市場データ
            evaluation_window: 評価期間（時間）

        Returns:
            SignalQualityMetrics: 品質メトリクス
        """
        metrics = SignalQualityMetrics()
        metrics.total_signals = len(signals)

        for signal in signals:
            self._categorize_signal(signal, metrics)

        # 品質メトリクス計算
        self._calculate_quality_metrics(
            metrics, signals, market_data, evaluation_window
        )

        return metrics

    def _categorize_signal(self, signal: Dict[str, Any], metrics: SignalQualityMetrics):
        """シグナルを分類"""
        signal_type = signal.get("action", signal.get("signal_type", "hold"))

        if signal_type in ["buy", "long"]:
            metrics.buy_signals += 1
        elif signal_type in ["sell", "short"]:
            metrics.sell_signals += 1
        else:
            metrics.hold_signals += 1

    def _calculate_quality_metrics(
        self,
        metrics: SignalQualityMetrics,
        signals: List[Dict[str, Any]],
        market_data: pd.DataFrame,
        evaluation_window: int,
    ):
        """品質メトリクスを計算"""
        # 市場適応性の評価
        metrics.market_alignment = self._evaluate_market_alignment(signals, market_data)
        metrics.volume_confirmation = self._evaluate_volume_confirmation(
            signals, market_data
        )
        metrics.timeframe_consistency = self._evaluate_timeframe_consistency(
            signals, market_data
        )

        # リスク適応性の評価
        metrics.drawdown_impact = self._evaluate_drawdown_impact(signals, market_data)
        metrics.volatility_alignment = self._evaluate_volatility_alignment(
            signals, market_data
        )

        # 予測性能の評価（バックテストデータがある場合）
        if len(market_data) > evaluation_window:
            self._evaluate_prediction_performance(
                metrics, signals, market_data, evaluation_window
            )

    def _evaluate_market_alignment(
        self, signals: List[Dict[str, Any]], market_data: pd.DataFrame
    ) -> float:
        """市場トレンドとの整合性を評価"""
        if not signals or market_data.empty:
            return 0.0

        alignment_score = 0.0
        total_signals = 0

        for signal in signals:
            timestamp = pd.to_datetime(signal.get("timestamp"))
            if timestamp in market_data.index:
                # シグナル発生時の市場トレンドを取得
                current_idx = market_data.index.get_loc(timestamp)
                if current_idx >= 20:  # 十分な過去データがある場合
                    past_data = market_data.iloc[current_idx - 20 : current_idx + 1]
                    trend = self._calculate_trend(past_data)

                    signal_type = signal.get("action", "hold")
                    if signal_type == "buy" and trend > 0.1:
                        alignment_score += 1.0
                    elif signal_type == "sell" and trend < -0.1:
                        alignment_score += 1.0
                    elif signal_type == "hold":
                        alignment_score += 0.5  # ホールドは中立的

                    total_signals += 1

        return alignment_score / max(total_signals, 1)

    def _evaluate_volume_confirmation(
        self, signals: List[Dict[str, Any]], market_data: pd.DataFrame
    ) -> float:
        """出来高確認の割合を評価"""
        if not signals or "volume" not in market_data.columns:
            return 0.0

        confirmation_count = 0

        for signal in signals:
            timestamp = pd.to_datetime(signal.get("timestamp"))
            if timestamp in market_data.index:
                current_volume = market_data.loc[timestamp, "volume"]
                avg_volume = market_data["volume"].rolling(20).mean().loc[timestamp]

                # 出来高が平均の1.5倍以上なら確認されたとみなす
                if current_volume > avg_volume * 1.5:
                    confirmation_count += 1

        return confirmation_count / max(len(signals), 1)

    def _evaluate_timeframe_consistency(
        self, signals: List[Dict[str, Any]], market_data: pd.DataFrame
    ) -> float:
        """複数時間軸での整合性を評価"""
        # 簡易実装: 同じ方向のシグナルが複数時間軸で発生しているかを評価
        if not signals:
            return 0.0

        consistency_count = 0

        for signal in signals:
            # 複数時間軸のデータがある場合の整合性チェック
            # （実際の実装では4時間足、1時間足などの比較を行う）
            signal_type = signal.get("action", "hold")
            confidence = signal.get("confidence", 0.5)

            # 高コンフィデンスのシグナルは整合性が高いと仮定
            if confidence > 0.7:
                consistency_count += 1

        return consistency_count / max(len(signals), 1)

    def _evaluate_drawdown_impact(
        self, signals: List[Dict[str, Any]], market_data: pd.DataFrame
    ) -> float:
        """ドローダウンへの影響度を評価"""
        if not signals or market_data.empty:
            return 0.0

        # シグナル発生後の価格変動を評価
        impact_scores = []

        for signal in signals:
            timestamp = pd.to_datetime(signal.get("timestamp"))
            if timestamp in market_data.index:
                current_idx = market_data.index.get_loc(timestamp)
                if current_idx < len(market_data) - 24:  # 24時間分のデータがある場合
                    future_prices = market_data.iloc[current_idx : current_idx + 24][
                        "close"
                    ]
                    max_drawdown = max_drawdown(future_prices)
                    impact_scores.append(max_drawdown)

        return np.mean(impact_scores) if impact_scores else 0.0

    def _evaluate_volatility_alignment(
        self, signals: List[Dict[str, Any]], market_data: pd.DataFrame
    ) -> float:
        """ボラティリティ適応度を評価"""
        if not signals or market_data.empty:
            return 0.0

        alignment_scores = []

        for signal in signals:
            timestamp = pd.to_datetime(signal.get("timestamp"))
            if timestamp in market_data.index:
                current_idx = market_data.index.get_loc(timestamp)
                if current_idx >= 20:
                    past_data = market_data.iloc[current_idx - 20 : current_idx + 1]
                    volatility = past_data["close"].pct_change().std()

                    signal_type = signal.get("action", "hold")
                    confidence = signal.get("confidence", 0.5)

                    # 高ボラティリティ時は低コンフィデンス、高ボラティリティ時は高コンフィデンスを好ましい
                    if (
                        volatility > 0.02 and confidence < 0.6
                    ):  # 高ボラティリティ時は慎重
                        alignment_scores.append(1.0)
                    elif (
                        volatility <= 0.02 and confidence > 0.7
                    ):  # 低ボラティリティ時は積極的
                        alignment_scores.append(1.0)
                    else:
                        alignment_scores.append(0.5)

        return np.mean(alignment_scores) if alignment_scores else 0.0

    def _evaluate_prediction_performance(
        self,
        metrics: SignalQualityMetrics,
        signals: List[Dict[str, Any]],
        market_data: pd.DataFrame,
        evaluation_window: int,
    ):
        """予測性能を評価"""
        # 実際の価格変動とシグナルの整合性を評価
        for signal in signals:
            timestamp = pd.to_datetime(signal.get("timestamp"))
            if timestamp in market_data.index:
                current_idx = market_data.index.get_loc(timestamp)
                if current_idx < len(market_data) - evaluation_window:
                    future_prices = market_data.iloc[
                        current_idx : current_idx + evaluation_window
                    ]["close"]
                    actual_return = (
                        future_prices.iloc[-1] - future_prices.iloc[0]
                    ) / future_prices.iloc[0]

                    signal_type = signal.get("action", "hold")

                    if signal_type == "buy" and actual_return > 0.01:  # 1%以上の上昇
                        metrics.true_positives += 1
                    elif signal_type == "buy" and actual_return <= 0.01:
                        metrics.false_positives += 1
                    elif (
                        signal_type == "sell" and actual_return < -0.01
                    ):  # 1%以上の下落
                        metrics.true_negatives += 1
                    elif signal_type == "sell" and actual_return >= -0.01:
                        metrics.false_negatives += 1

        # 性能メトリクス計算
        if metrics.true_positives + metrics.false_positives > 0:
            metrics.precision = metrics.true_positives / (
                metrics.true_positives + metrics.false_positives
            )

        if metrics.true_positives + metrics.false_negatives > 0:
            metrics.recall = metrics.true_positives / (
                metrics.true_positives + metrics.false_negatives
            )

        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = (
                2
                * (metrics.precision * metrics.recall)
                / (metrics.precision + metrics.recall)
            )

    def _calculate_trend(self, data: pd.DataFrame) -> float:
        """トレンドを計算（簡易版）"""
        if len(data) < 2:
            return 0.0

        # 線形回帰によるトレンド計算
        x = np.arange(len(data))
        y = data["close"].values
        slope = np.polyfit(x, y, 1)[0]

        # 正規化
        avg_price = np.mean(y)
        return slope / avg_price if avg_price != 0 else 0.0

    def generate_quality_report(self, metrics: SignalQualityMetrics) -> Dict[str, Any]:
        """品質レポートを生成"""
        return {
            "summary": {
                "total_signals": metrics.total_signals,
                "signal_distribution": {
                    "buy": metrics.buy_signals,
                    "sell": metrics.sell_signals,
                    "hold": metrics.hold_signals,
                },
            },
            "performance": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
            },
            "market_adaptation": {
                "market_alignment": metrics.market_alignment,
                "volume_confirmation": metrics.volume_confirmation,
                "timeframe_consistency": metrics.timeframe_consistency,
            },
            "risk_metrics": {
                "drawdown_impact": metrics.drawdown_impact,
                "volatility_alignment": metrics.volatility_alignment,
            },
            "recommendations": self._generate_recommendations(metrics),
        }

    def _generate_recommendations(self, metrics: SignalQualityMetrics) -> List[str]:
        """改善 recommendations を生成"""
        recommendations = []

        if metrics.precision < 0.4:
            recommendations.append(
                "適合率が低いため、偽陽性を減らすフィルタリングを強化してください"
            )

        if metrics.recall < 0.4:
            recommendations.append(
                "再現率が低いため、真陽性を増やすためのシグナル検出を改善してください"
            )

        if metrics.market_alignment < 0.5:
            recommendations.append(
                "市場トレンドとの整合性が低いため、トレンドフィルタを追加してください"
            )

        if metrics.volume_confirmation < 0.3:
            recommendations.append(
                "出来高確認が不十分なため、ボリュームフィルタを強化してください"
            )

        if metrics.volatility_alignment < 0.5:
            recommendations.append(
                "ボラティリティ適応が不十分なため、市場変動に応じた閾値調整を実装してください"
            )

        if not recommendations:
            recommendations.append(
                "全体的に良好なシグナル品質です。継続的な監視を推奨します"
            )

        return recommendations
