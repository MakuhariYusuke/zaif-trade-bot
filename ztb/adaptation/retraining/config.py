"""
Configuration management for Automatic Retraining Triggers
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .types import TriggerCondition, TriggerPriority, TriggerType


@dataclass
class RetrainingConfig:
    """再訓練トリガー設定"""

    # 基本設定
    enabled: bool = True
    max_concurrent_retraining: int = 1
    retraining_timeout_hours: int = 24

    # トリガー条件
    trigger_conditions: List[TriggerCondition] = field(default_factory=list)

    # パフォーマンス監視設定
    performance_check_interval_minutes: int = 15
    performance_history_window_hours: int = 24
    performance_baseline_period_days: int = 7

    # データ分布監視設定
    distribution_check_interval_minutes: int = 60
    distribution_history_window_hours: int = 168  # 1週間
    distribution_drift_threshold: float = 0.1

    # 時間ベース設定
    time_based_schedules: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"interval_hours": 24, "description": "Daily retraining"},
            {"interval_hours": 168, "description": "Weekly retraining"},
        ]
    )

    # 出来高ベース設定
    volume_based_thresholds: Dict[str, int] = field(
        default_factory=lambda: {
            "min_new_samples": 1000,
            "max_samples_without_retraining": 10000,
        }
    )

    # リソース管理設定
    resource_limits: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_cpu_percent": 80.0,
            "max_memory_mb": 4096,
            "max_gpu_memory_mb": 8192,
            "max_concurrent_jobs": 1,
        }
    )

    # メモリ管理設定（メモリリーク防止）
    max_history_size: int = 1000  # 履歴データの最大サイズ
    cleanup_interval_hours: int = 24  # クリーンアップ間隔
    compression_enabled: bool = True  # 古いデータの圧縮

    # 優先度設定
    priority_weights: Dict[TriggerPriority, float] = field(
        default_factory=lambda: {
            TriggerPriority.LOW: 1.0,
            TriggerPriority.MEDIUM: 2.0,
            TriggerPriority.HIGH: 3.0,
            TriggerPriority.CRITICAL: 5.0,
        }
    )

    # 通知設定
    notifications_enabled: bool = True
    alert_thresholds: Dict[str, Any] = field(
        default_factory=lambda: {
            "failed_retraining_rate": 0.1,
            "average_retraining_time_hours": 12,
        }
    )

    def __post_init__(self) -> None:
        """設定の検証と初期化"""
        if self.max_concurrent_retraining < 1:
            raise ValueError("max_concurrent_retraining must be at least 1")

        if self.retraining_timeout_hours < 1:
            raise ValueError("retraining_timeout_hours must be at least 1")

        if self.max_history_size < 100:
            raise ValueError("max_history_size must be at least 100")

        # デフォルトのトリガー条件を設定
        if not self.trigger_conditions:
            self.trigger_conditions = [
                TriggerCondition(
                    trigger_type=TriggerType.PERFORMANCE,
                    metric_name="win_rate",
                    operator="lt",
                    threshold=0.45,
                    duration_minutes=60,
                    cooldown_minutes=240,
                    priority=TriggerPriority.HIGH,
                ),
                TriggerCondition(
                    trigger_type=TriggerType.PERFORMANCE,
                    metric_name="sharpe_ratio",
                    operator="lt",
                    threshold=1.0,
                    duration_minutes=120,
                    cooldown_minutes=480,
                    priority=TriggerPriority.MEDIUM,
                ),
                TriggerCondition(
                    trigger_type=TriggerType.DATA_DISTRIBUTION,
                    metric_name="distribution_drift",
                    operator="gt",
                    threshold=0.15,
                    duration_minutes=30,
                    cooldown_minutes=180,
                    priority=TriggerPriority.MEDIUM,
                ),
            ]


@dataclass
class RetrainingPolicy:
    """再訓練ポリシー"""

    policy_name: str
    trigger_conditions: List[TriggerCondition]
    retraining_strategy: str  # "full", "incremental", "fine_tuning"
    resource_requirements: Dict[str, Any]
    max_execution_time_hours: int
    success_criteria: Dict[str, float]
    enabled: bool = True

    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """ポリシーに基づいてトリガーを判定"""
        for condition in self.trigger_conditions:
            value = metrics.get(condition.metric_name)
            if value is None:
                continue

            if self._check_condition(value, condition):
                return True
        return False

    def _check_condition(self, value: float, condition: TriggerCondition) -> bool:
        """条件チェック"""
        if condition.operator == "gt":
            return value > condition.threshold
        elif condition.operator == "lt":
            return value < condition.threshold
        elif condition.operator == "gte":
            return value >= condition.threshold
        elif condition.operator == "lte":
            return value <= condition.threshold
        elif condition.operator == "eq":
            return abs(value - condition.threshold) < 1e-6
        elif condition.operator == "ne":
            return abs(value - condition.threshold) >= 1e-6
        return False
