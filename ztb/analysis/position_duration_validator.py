"""
Position Duration Validator for SAC v428
ポジション継続時間の検証とトレーニング中の監視を行うクラス
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class PositionDurationValidator:
    """ポジション継続時間検証クラス"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.target_sell_buy_duration = 8.0
        self.target_buy_sell_duration = 8.0
        self.min_hold_ratio = 0.20

    def validate_position_durations(self, actions: list[int]) -> dict[str, Any]:
        """ポジション継続時間を検証"""
        durations = self._calculate_position_durations(actions)

        validation_results = {
            "sell_buy_duration_ok": durations["sell_to_buy"]["mean"]
            >= self.target_sell_buy_duration,
            "buy_sell_duration_ok": durations["buy_to_sell"]["mean"]
            >= self.target_buy_sell_duration,
            "hold_ratio_ok": durations["hold"]["ratio"] >= self.min_hold_ratio,
            "overall_score": self._calculate_overall_score(durations),
        }

        return {
            "durations": durations,
            "validation": validation_results,
            "recommendations": self._generate_recommendations(validation_results),
        }

    def _calculate_position_durations(self, actions: list[int]) -> dict[str, Any]:
        """ポジション継続時間を計算"""
        sell_to_buy_durations = []
        buy_to_sell_durations = []
        hold_durations = []

        current_position = 0  # 0: HOLD, 1: BUY, 2: SELL
        position_start = 0

        for i, action in enumerate(actions):
            if action != current_position:
                # ポジション変更
                duration = i - position_start
                if current_position == 2:  # SELL -> BUY/SELL
                    sell_to_buy_durations.append(duration)
                elif current_position == 1:  # BUY -> SELL/HOLD
                    buy_to_sell_durations.append(duration)
                elif current_position == 0:  # HOLD -> BUY/SELL
                    hold_durations.append(duration)

                current_position = action
                position_start = i

        # 最後のポジション
        if position_start < len(actions):
            duration = len(actions) - position_start
            if current_position == 2:
                sell_to_buy_durations.append(duration)
            elif current_position == 1:
                buy_to_sell_durations.append(duration)
            elif current_position == 0:
                hold_durations.append(duration)

        return {
            "sell_to_buy": {
                "durations": sell_to_buy_durations,
                "mean": np.mean(sell_to_buy_durations) if sell_to_buy_durations else 0,
                "count": len(sell_to_buy_durations),
            },
            "buy_to_sell": {
                "durations": buy_to_sell_durations,
                "mean": np.mean(buy_to_sell_durations) if buy_to_sell_durations else 0,
                "count": len(buy_to_sell_durations),
            },
            "hold": {
                "durations": hold_durations,
                "mean": np.mean(hold_durations) if hold_durations else 0,
                "count": len(hold_durations),
                "ratio": len(hold_durations) / len(actions) if actions else 0,
            },
        }

    def _calculate_overall_score(self, durations: dict[str, Any]) -> float:
        """全体スコアを計算"""
        sell_buy_score = min(
            durations["sell_to_buy"]["mean"] / self.target_sell_buy_duration, 1.0
        )
        buy_sell_score = min(
            durations["buy_to_sell"]["mean"] / self.target_buy_sell_duration, 1.0
        )
        hold_score = min(durations["hold"]["ratio"] / self.min_hold_ratio, 1.0)

        return (sell_buy_score + buy_sell_score + hold_score) / 3.0

    def _generate_recommendations(self, validation: dict[str, Any]) -> list[str]:
        """改善推奨を生成"""
        recommendations = []

        if not validation["sell_buy_duration_ok"]:
            recommendations.append(
                "SELL→BUY継続時間を延ばすため、ポジション安定性ボーナスを強化"
            )

        if not validation["buy_sell_duration_ok"]:
            recommendations.append("BUY→SELL継続時間を延ばすため、最小保持時間を延長")

        if not validation["hold_ratio_ok"]:
            recommendations.append("HOLD比率を上げるため、HOLDボーナスを増加")

        if validation["overall_score"] < 0.5:
            recommendations.append("全体的な改善のため、アンサンブル合意要件を強化")

        return recommendations

    def analyze_training_durations(
        self, training_results: dict[str, Any]
    ) -> dict[str, Any]:
        """トレーニング中のポジション継続時間を分析"""
        if "actions" not in training_results:
            return {"error": "No actions data in training results"}

        actions = training_results["actions"]
        analysis = self.validate_position_durations(actions)

        # トレーニング進捗との相関
        analysis["training_correlation"] = self._analyze_training_correlation(
            training_results, analysis["durations"]
        )

        return analysis

    def _analyze_training_correlation(
        self, training_results: dict[str, Any], durations: dict[str, Any]
    ) -> dict[str, Any]:
        """トレーニング進捗とポジション継続時間の相関を分析"""
        # 簡易的な相関分析（実際の実装ではより詳細に）
        return {
            "duration_improvement_trend": "analyzing",
            "correlation_with_reward": 0.0,  # 仮の値
            "stability_vs_performance_tradeoff": "monitoring",
        }
