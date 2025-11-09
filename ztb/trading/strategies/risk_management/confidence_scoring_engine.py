"""
Phase 3-1: シグナル品質向上 - コンフィデンススコアリング改善

シグナルの信頼度を動的に評価し、品質ベースのフィルタリングを提供します。
既存のActionSignalGuideAdapterと連携して品質向上を実現します。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.analysis.signal_quality.signal_quality_analyzer import SignalQualityAnalyzer
from ztb.utils.performance_profiler import PerformanceProfiler


@dataclass
class ConfidenceScore:
    """コンフィデンススコア"""

    base_score: float  # 基本スコア
    market_alignment: float  # 市場整合性
    volume_confirmation: float  # 出来高確認
    timeframe_consistency: float  # 時間軸整合性
    volatility_adaptation: float  # ボラティリティ適応

    @property
    def total_score(self) -> float:
        """総合スコアを計算"""
        weights = {
            "base_score": 0.4,
            "market_alignment": 0.2,
            "volume_confirmation": 0.15,
            "timeframe_consistency": 0.15,
            "volatility_adaptation": 0.1,
        }

        total = (
            self.base_score * weights["base_score"]
            + self.market_alignment * weights["market_alignment"]
            + self.volume_confirmation * weights["volume_confirmation"]
            + self.timeframe_consistency * weights["timeframe_consistency"]
            + self.volatility_adaptation * weights["volatility_adaptation"]
        )

        return min(max(total, 0.0), 1.0)  # 0-1の範囲に制限


@dataclass
class SignalFilterCriteria:
    """シグナルフィルタ基準"""

    min_confidence_threshold: float = 0.6
    min_market_alignment: float = 0.5
    min_volume_confirmation: float = 0.3
    require_timeframe_consistency: bool = True
    max_volatility_risk: float = 0.8

    # 動的調整パラメータ
    adaptive_thresholds: bool = True
    market_condition_adjustment: bool = True


class ConfidenceScoringEngine:
    """コンフィデンススコアリングエンジン"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.quality_analyzer = SignalQualityAnalyzer()
        self.historical_scores: List[ConfidenceScore] = []
        self.max_history_size = 1000  # メモリ管理のため履歴サイズを制限

        # デフォルトのフィルタ基準
        self.filter_criteria = SignalFilterCriteria()

    def calculate_confidence_score(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> ConfidenceScore:
        """
        シグナルのコンフィデンススコアを計算

        Args:
            signal: 評価対象のシグナル
            market_data: 市場データ
            context_data: 追加の文脈データ

        Returns:
            ConfidenceScore: 計算されたコンフィデンススコア
        """
        # 基本スコアの取得
        base_score = self._calculate_base_score(signal)

        # 市場整合性の評価
        market_alignment = self._evaluate_market_alignment(signal, market_data)

        # 出来高確認の評価
        volume_confirmation = self._evaluate_volume_confirmation(signal, market_data)

        # 時間軸整合性の評価
        timeframe_consistency = self._evaluate_timeframe_consistency(
            signal, market_data
        )

        # ボラティリティ適応の評価
        volatility_adaptation = self._evaluate_volatility_adaptation(
            signal, market_data
        )

        confidence_score = ConfidenceScore(
            base_score=base_score,
            market_alignment=market_alignment,
            volume_confirmation=volume_confirmation,
            timeframe_consistency=timeframe_consistency,
            volatility_adaptation=volatility_adaptation,
        )

        # 履歴に保存（メモリ管理）
        self.historical_scores.append(confidence_score)
        if len(self.historical_scores) > self.max_history_size:
            # 古いデータを削除してメモリを節約
            self.historical_scores = self.historical_scores[
                -self.max_history_size // 2 :
            ]

        return confidence_score

    def _calculate_base_score(self, signal: Dict[str, Any]) -> float:
        """基本スコアを計算"""
        # シグナルの強度に基づく基本スコア
        signal_strength = signal.get("strength", 0.5)
        confidence = signal.get("confidence", 0.5)

        # 数値変換（無効な入力に対応）
        try:
            signal_strength = (
                float(signal_strength) if signal_strength is not None else 0.5
            )
            confidence = float(confidence) if confidence is not None else 0.5
        except (ValueError, TypeError):
            signal_strength = 0.5
            confidence = 0.5

        # シグナルタイプによる調整
        signal_type = signal.get("action", "hold")
        type_multiplier = {"buy": 1.0, "sell": 1.0, "hold": 0.7}.get(
            signal_type, 0.5
        )  # ホールドは基本的に低いスコア

        base_score = (signal_strength + confidence) / 2.0 * type_multiplier
        return min(max(base_score, 0.0), 1.0)

    def _evaluate_market_alignment(
        self, signal: Dict[str, Any], market_data: pd.DataFrame
    ) -> float:
        """市場トレンドとの整合性を評価"""
        if market_data.empty:
            return 0.5

        timestamp = pd.to_datetime(signal.get("timestamp", datetime.now()))
        if timestamp not in market_data.index:
            return 0.5

        current_idx = market_data.index.get_loc(timestamp)
        if current_idx < 20:
            return 0.5

        # 過去20期間のトレンドを計算
        past_data = market_data.iloc[current_idx - 20 : current_idx + 1]
        trend = self._calculate_trend_strength(past_data)

        signal_type = signal.get("action", "hold")

        # トレンドとの整合性を評価
        if signal_type == "buy" and trend > 0.1:
            return 0.9
        elif signal_type == "sell" and trend < -0.1:
            return 0.9
        elif signal_type == "buy" and trend < -0.1:
            return 0.2  # 逆トレンド
        elif signal_type == "sell" and trend > 0.1:
            return 0.2  # 逆トレンド
        else:
            return 0.6  # 中立的

    def _evaluate_volume_confirmation(
        self, signal: Dict[str, Any], market_data: pd.DataFrame
    ) -> float:
        """出来高確認を評価"""
        if "volume" not in market_data.columns:
            return 0.5

        timestamp = pd.to_datetime(signal.get("timestamp", datetime.now()))
        if timestamp not in market_data.index:
            return 0.5

        current_volume = market_data.loc[timestamp, "volume"]

        # 移動平均出来高を計算
        current_idx = market_data.index.get_loc(timestamp)
        if current_idx >= 20:
            avg_volume = (
                market_data["volume"].iloc[current_idx - 20 : current_idx].mean()
            )
        else:
            avg_volume = market_data["volume"].iloc[: current_idx + 1].mean()

        if avg_volume == 0:
            return 0.5

        volume_ratio = current_volume / avg_volume

        # 出来高レベルの評価
        if volume_ratio > 2.0:
            return 0.9  # 非常に高い出来高
        elif volume_ratio > 1.5:
            return 0.8  # 高い出来高
        elif volume_ratio > 1.0:
            return 0.6  # 平均的出来高
        else:
            return 0.3  # 低い出来高

    def _evaluate_timeframe_consistency(
        self, signal: Dict[str, Any], market_data: pd.DataFrame
    ) -> float:
        """複数時間軸での整合性を評価"""
        # 簡易実装: 同じ方向のシグナルが異なる時間軸で確認できるかを評価
        # （実際の実装では4時間足、1時間足などのデータを比較）

        signal_type = signal.get("action", "hold")
        confidence = signal.get("confidence", 0.5)

        # 高コンフィデンスのシグナルは時間軸整合性が高いと仮定
        if confidence > 0.8:
            return 0.9
        elif confidence > 0.6:
            return 0.7
        else:
            return 0.4

    def _evaluate_volatility_adaptation(
        self, signal: Dict[str, Any], market_data: pd.DataFrame
    ) -> float:
        """ボラティリティ適応を評価"""
        if market_data.empty:
            return 0.5

        timestamp = pd.to_datetime(signal.get("timestamp", datetime.now()))
        if timestamp not in market_data.index:
            return 0.5

        current_idx = market_data.index.get_loc(timestamp)
        if current_idx < 20:
            return 0.5

        # 過去20期間のボラティリティを計算
        past_returns = (
            market_data["close"].iloc[current_idx - 20 : current_idx + 1].pct_change()
        )
        volatility = past_returns.std()

        confidence = signal.get("confidence", 0.5)

        # ボラティリティに応じたコンフィデンス評価
        if volatility > 0.03:  # 高ボラティリティ
            if confidence < 0.6:  # 慎重なアプローチ
                return 0.8
            else:  # 過度に自信過剰
                return 0.4
        else:  # 低ボラティリティ
            if confidence > 0.7:  # 積極的なアプローチ
                return 0.8
            else:  # 過度に慎重
                return 0.4

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """トレンド強度を計算"""
        if len(data) < 5:
            return 0.0

        # 線形回帰によるトレンド計算
        x = np.arange(len(data))
        y = data["close"].values
        slope = np.polyfit(x, y, 1)[0]

        # トレンド強度を正規化
        avg_price = np.mean(y)
        trend_strength = slope / avg_price if avg_price != 0 else 0.0

        return trend_strength

    def should_accept_signal(
        self,
        signal: Dict[str, Any],
        market_data: pd.DataFrame,
        custom_criteria: Optional[SignalFilterCriteria] = None,
    ) -> Tuple[bool, str, ConfidenceScore]:
        """
        シグナルを受け入れるべきかを判定

        Returns:
            Tuple[bool, str, ConfidenceScore]: (受け入れ可否, 理由, スコア)
        """
        criteria = custom_criteria or self.filter_criteria

        # コンフィデンススコア計算
        confidence_score = self.calculate_confidence_score(signal, market_data)

        # 基準チェック
        if confidence_score.total_score < criteria.min_confidence_threshold:
            return (
                False,
                f"コンフィデンススコアが低すぎます: {confidence_score.total_score:.2f}",
                confidence_score,
            )

        if confidence_score.market_alignment < criteria.min_market_alignment:
            return (
                False,
                f"市場整合性が不十分です: {confidence_score.market_alignment:.2f}",
                confidence_score,
            )

        if confidence_score.volume_confirmation < criteria.min_volume_confirmation:
            return (
                False,
                f"出来高確認が不十分です: {confidence_score.volume_confirmation:.2f}",
                confidence_score,
            )

        if (
            criteria.require_timeframe_consistency
            and confidence_score.timeframe_consistency < 0.5
        ):
            return (
                False,
                f"時間軸整合性が不十分です: {confidence_score.timeframe_consistency:.2f}",
                confidence_score,
            )

        if confidence_score.volatility_adaptation < criteria.max_volatility_risk:
            return (
                False,
                f"ボラティリティリスクが高すぎます: {confidence_score.volatility_adaptation:.2f}",
                confidence_score,
            )

        return True, "シグナルは基準を満たしています", confidence_score

    def update_filter_criteria(self, market_conditions: Dict[str, Any]):
        """市場状況に応じてフィルタ基準を動的調整"""
        if not self.filter_criteria.adaptive_thresholds:
            return

        # 市場ボラティリティに基づく調整
        volatility = market_conditions.get("volatility", 0.02)

        if volatility > 0.03:  # 高ボラティリティ時
            self.filter_criteria.min_confidence_threshold = min(
                0.7, self.filter_criteria.min_confidence_threshold + 0.1
            )
            self.filter_criteria.min_market_alignment = min(
                0.6, self.filter_criteria.min_market_alignment + 0.1
            )
        elif volatility < 0.01:  # 低ボラティリティ時
            self.filter_criteria.min_confidence_threshold = max(
                0.5, self.filter_criteria.min_confidence_threshold - 0.1
            )
            self.filter_criteria.min_market_alignment = max(
                0.4, self.filter_criteria.min_market_alignment - 0.1
            )

    def get_quality_statistics(self) -> Dict[str, Any]:
        """品質統計を取得"""
        if not self.historical_scores:
            return {}

        scores = [s.total_score for s in self.historical_scores]

        return {
            "average_confidence": np.mean(scores),
            "median_confidence": np.median(scores),
            "confidence_std": np.std(scores),
            "high_quality_signals": sum(1 for s in scores if s > 0.8),
            "total_signals": len(scores),
            "quality_distribution": {
                "excellent": sum(1 for s in scores if s > 0.9),
                "good": sum(1 for s in scores if 0.7 < s <= 0.9),
                "fair": sum(1 for s in scores if 0.5 < s <= 0.7),
                "poor": sum(1 for s in scores if s <= 0.5),
            },
        }
