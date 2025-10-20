"""
Model Selection and Rollback Logic for A/B Testing
リスク管理と自動化を重視した実装
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .config import ABTestConfig
from .types import (
    ABTestConfiguration,
    ABTestResult,
    ABTestResultSummary,
    ABTestState,
    ABTestVariant,
)

logger = logging.getLogger(__name__)


class ModelSelector:
    """モデル選択・ロールバックエンジン"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.deployment_history: List[Dict[str, Any]] = []
        self.rollback_triggers: Dict[str, Callable] = {}

    def select_model(
        self,
        test_config: ABTestConfiguration,
        test_state: ABTestState,
        result_summary: ABTestResultSummary,
    ) -> Dict[str, Any]:
        """
        テスト結果に基づいてモデルを選択
        リスク評価と自動化を考慮
        """
        decision = {
            "selected_variant": None,
            "action": "hold",  # "deploy", "rollback", "hold"
            "confidence_level": 0.0,
            "risk_assessment": {},
            "reasoning": [],
            "recommended_traffic_percentage": 0.0,
        }

        # 勝者の決定
        if result_summary.result == ABTestResult.WINNER_A:
            decision["selected_variant"] = test_config.variant_a
            decision["action"] = "deploy"
        elif result_summary.result == ABTestResult.WINNER_B:
            decision["selected_variant"] = test_config.variant_b
            decision["action"] = "deploy"
        else:
            decision["reasoning"].append(
                "No clear winner - continuing with current model"
            )
            return decision

        # リスク評価
        risk_assessment = self._assess_deployment_risks(
            test_config, test_state, result_summary
        )
        decision["risk_assessment"] = risk_assessment

        # 信頼性チェック
        confidence_level = self._calculate_confidence_level(result_summary)
        decision["confidence_level"] = confidence_level

        # アクション決定
        action, reasoning = self._determine_action(
            result_summary, risk_assessment, confidence_level
        )
        decision["action"] = action
        decision["reasoning"].extend(reasoning)

        # トラフィック割合の決定
        decision["recommended_traffic_percentage"] = self._calculate_traffic_percentage(
            confidence_level, risk_assessment
        )

        # ロールバックトリガーの設定
        if action == "deploy":
            self._setup_rollback_triggers(
                test_config.test_id, decision["selected_variant"], risk_assessment
            )

        logger.info(
            f"Model selection decision for {test_config.test_id}: {decision['action']}"
        )
        return decision

    def _assess_deployment_risks(
        self,
        test_config: ABTestConfiguration,
        test_state: ABTestState,
        result_summary: ABTestResultSummary,
    ) -> Dict[str, Any]:
        """デプロイメントリスクを評価"""
        risks = {
            "regression_risk": "low",
            "sample_size_risk": "low",
            "statistical_risk": "low",
            "performance_risk": "low",
            "overall_risk": "low",
        }

        # サンプルサイズリスク
        min_samples = test_config.minimum_sample_size
        if (
            test_state.metrics_a.sample_count < min_samples
            or test_state.metrics_b.sample_count < min_samples
        ):
            risks["sample_size_risk"] = "high"
            risks["overall_risk"] = "high"

        # 統計的リスク
        if result_summary.statistical_result.p_value > 0.1:
            risks["statistical_risk"] = "high"
            risks["overall_risk"] = max(risks["overall_risk"], "medium")

        # 回帰リスク
        if test_state.regression_detected:
            risks["regression_risk"] = "high"
            risks["overall_risk"] = "high"

        # パフォーマンスリスク
        effect_size = result_summary.statistical_result.effect_size
        if effect_size < test_config.minimum_effect_size:
            risks["performance_risk"] = "medium"
            risks["overall_risk"] = max(risks["overall_risk"], "medium")

        return risks

    def _calculate_confidence_level(self, result_summary: ABTestResultSummary) -> float:
        """信頼性を計算"""
        stat_result = result_summary.statistical_result

        # p値に基づく信頼性
        if stat_result.p_value < 0.001:
            p_confidence = 1.0
        elif stat_result.p_value < 0.01:
            p_confidence = 0.9
        elif stat_result.p_value < 0.05:
            p_confidence = 0.8
        else:
            p_confidence = 0.5

        # 効果量に基づく信頼性
        if stat_result.effect_size > 0.8:
            effect_confidence = 1.0
        elif stat_result.effect_size > 0.5:
            effect_confidence = 0.9
        elif stat_result.effect_size > 0.2:
            effect_confidence = 0.7
        else:
            effect_confidence = 0.5

        # サンプルサイズに基づく信頼性
        total_samples = stat_result.sample_size_a + stat_result.sample_size_b
        if total_samples > 10000:
            sample_confidence = 1.0
        elif total_samples > 5000:
            sample_confidence = 0.9
        elif total_samples > 1000:
            sample_confidence = 0.7
        else:
            sample_confidence = 0.5

        # 総合信頼性
        return p_confidence * 0.4 + effect_confidence * 0.4 + sample_confidence * 0.2

    def _determine_action(
        self,
        result_summary: ABTestResultSummary,
        risk_assessment: Dict[str, Any],
        confidence_level: float,
    ) -> tuple[str, List[str]]:
        """アクションを決定"""
        reasoning = []

        # 高リスクの場合は保留
        if risk_assessment["overall_risk"] == "high":
            reasoning.append("High risk detected - holding deployment")
            return "hold", reasoning

        # 信頼性が不十分な場合は保留
        if confidence_level < 0.7:
            reasoning.append(
                f"Low confidence ({confidence_level:.2f}) - continuing testing"
            )
            return "hold", reasoning

        # 勝者がいてリスクが許容範囲内ならデプロイ
        if result_summary.result in [ABTestResult.WINNER_A, ABTestResult.WINNER_B]:
            winner = "A" if result_summary.result == ABTestResult.WINNER_A else "B"
            reasoning.append(
                f"Variant {winner} shows clear improvement with acceptable risk"
            )
            reasoning.append(f"Confidence level: {confidence_level:.2f}")
            return "deploy", reasoning

        # それ以外の場合は保留
        reasoning.append("No clear winner or insufficient confidence")
        return "hold", reasoning

    def _calculate_traffic_percentage(
        self, confidence_level: float, risk_assessment: Dict[str, Any]
    ) -> float:
        """推奨トラフィック割合を計算"""
        base_percentage = confidence_level * 50  # 最大50%

        # リスクに応じて調整
        if risk_assessment["overall_risk"] == "high":
            base_percentage *= 0.3
        elif risk_assessment["overall_risk"] == "medium":
            base_percentage *= 0.6

        # 最小・最大値の制限
        return max(5.0, min(base_percentage, 100.0))

    def _setup_rollback_triggers(
        self,
        test_id: str,
        selected_variant: ABTestVariant,
        risk_assessment: Dict[str, Any],
    ):
        """ロールバックトリガーを設定"""
        trigger_conditions = []

        # 高リスクの場合は厳格なトリガーを設定
        if risk_assessment["overall_risk"] == "high":
            trigger_conditions.extend(
                [
                    {"metric": "error_rate", "threshold": 0.1, "duration_minutes": 5},
                    {
                        "metric": "performance_degradation",
                        "threshold": 0.2,
                        "duration_minutes": 10,
                    },
                ]
            )

        # 中リスクの場合は中程度のトリガー
        elif risk_assessment["overall_risk"] == "medium":
            trigger_conditions.extend(
                [
                    {"metric": "error_rate", "threshold": 0.15, "duration_minutes": 15},
                    {
                        "metric": "performance_degradation",
                        "threshold": 0.3,
                        "duration_minutes": 30,
                    },
                ]
            )

        # 低リスクの場合は緩いトリガー
        else:
            trigger_conditions.extend(
                [
                    {"metric": "error_rate", "threshold": 0.2, "duration_minutes": 30},
                    {
                        "metric": "performance_degradation",
                        "threshold": 0.5,
                        "duration_minutes": 60,
                    },
                ]
            )

        self.rollback_triggers[test_id] = self._create_rollback_trigger(
            test_id, selected_variant, trigger_conditions
        )

        logger.info(f"Rollback triggers set up for test {test_id}")

    def _create_rollback_trigger(
        self,
        test_id: str,
        selected_variant: ABTestVariant,
        conditions: List[Dict[str, Any]],
    ) -> Callable:
        """ロールバックトリガーを作成"""

        def rollback_trigger(metric_name: str, value: float, timestamp: datetime):
            """ロールバックを実行"""
            for condition in conditions:
                if (
                    metric_name == condition["metric"]
                    and value > condition["threshold"]
                ):
                    logger.warning(
                        f"Rollback triggered for {test_id}: {metric_name}={value}"
                    )
                    self._execute_rollback(
                        test_id, selected_variant, f"{metric_name} exceeded threshold"
                    )
                    return True
            return False

        return rollback_trigger

    def _execute_rollback(
        self, test_id: str, deployed_variant: ABTestVariant, reason: str
    ):
        """ロールバックを実行"""
        rollback_record = {
            "test_id": test_id,
            "deployed_variant": deployed_variant.variant_id,
            "rollback_time": datetime.now(),
            "reason": reason,
            "action": "rolled_back_to_previous",
        }

        self.deployment_history.append(rollback_record)

        # ロールバックトリガーを削除
        if test_id in self.rollback_triggers:
            del self.rollback_triggers[test_id]

        logger.info(f"Executed rollback for test {test_id}: {reason}")

    def check_rollback_conditions(
        self, test_id: str, metrics: Dict[str, float], timestamp: datetime
    ) -> bool:
        """ロールバック条件をチェック"""
        if test_id not in self.rollback_triggers:
            return False

        trigger = self.rollback_triggers[test_id]

        # 各メトリクスをチェック
        for metric_name, value in metrics.items():
            if trigger(metric_name, value, timestamp):
                return True

        return False

    def get_deployment_history(
        self, test_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """デプロイメント履歴を取得"""
        if test_id:
            return [
                record
                for record in self.deployment_history
                if record["test_id"] == test_id
            ]
        return self.deployment_history.copy()

    def get_active_rollback_triggers(self) -> List[str]:
        """アクティブなロールバックトリガーを取得"""
        return list(self.rollback_triggers.keys())

    def force_rollback(self, test_id: str, reason: str = "Manual rollback"):
        """強制ロールバックを実行"""
        if test_id in self.rollback_triggers:
            # デプロイされたバリアントを取得（簡易実装）
            deployed_variant = ABTestVariant(
                variant_id="unknown",
                model_path="",
                model_version="",
                description="Force rolled back",
            )
            self._execute_rollback(test_id, deployed_variant, reason)
            return True

        return False


class TrafficManager:
    """トラフィック管理エンジン"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.traffic_allocations: Dict[str, Dict[str, float]] = {}

    def allocate_traffic(
        self,
        test_id: str,
        variant_a: ABTestVariant,
        variant_b: ABTestVariant,
        percentage: float,
    ) -> Dict[str, float]:
        """トラフィックを割り当て"""
        allocation = {
            variant_a.variant_id: (100 - percentage) / 100,
            variant_b.variant_id: percentage / 100,
        }

        self.traffic_allocations[test_id] = allocation

        logger.info(f"Traffic allocated for {test_id}: {allocation}")
        return allocation

    def ramp_up_traffic(
        self,
        test_id: str,
        target_percentage: float,
        steps: int = 5,
        interval_minutes: int = 30,
    ) -> List[Dict[str, Any]]:
        """トラフィックを段階的に増加"""
        if test_id not in self.traffic_allocations:
            raise ValueError(f"Test {test_id} not found")

        current_allocation = self.traffic_allocations[test_id]
        current_percentage = (
            current_allocation[list(current_allocation.keys())[1]] * 100
        )

        ramp_schedule = []
        step_size = (target_percentage - current_percentage) / steps

        for step in range(1, steps + 1):
            new_percentage = current_percentage + (step_size * step)
            new_percentage = max(0, min(new_percentage, 100))

            schedule_item = {
                "step": step,
                "percentage": new_percentage,
                "scheduled_time": datetime.now()
                + timedelta(minutes=interval_minutes * step),
            }
            ramp_schedule.append(schedule_item)

        return ramp_schedule

    def get_traffic_allocation(self, test_id: str) -> Optional[Dict[str, float]]:
        """トラフィック割り当てを取得"""
        return self.traffic_allocations.get(test_id)

    def update_traffic_allocation(self, test_id: str, new_percentage: float) -> bool:
        """トラフィック割り当てを更新"""
        if test_id not in self.traffic_allocations:
            return False

        allocation = self.traffic_allocations[test_id]
        variant_ids = list(allocation.keys())

        allocation[variant_ids[0]] = (100 - new_percentage) / 100
        allocation[variant_ids[1]] = new_percentage / 100

        logger.info(f"Updated traffic allocation for {test_id}: {allocation}")
        return True
