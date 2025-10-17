"""
Configuration management for A/B Testing Framework
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .types import TrafficSplitType, StatisticalTest, RollbackCondition


@dataclass
class ABTestingConfig:
    """A/Bテスト設定"""

    # 基本設定
    test_duration_hours: int = 24
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    statistical_tests: List[StatisticalTest] = field(default_factory=lambda: [
        StatisticalTest.T_TEST,
        StatisticalTest.CONFIDENCE_INTERVAL
    ])

    # トラフィック分割設定
    traffic_split_type: TrafficSplitType = TrafficSplitType.PROBABILISTIC
    traffic_split_config: Dict[str, Any] = field(default_factory=dict)

    # ロールバック設定
    enable_auto_rollback: bool = True
    rollback_conditions: List[RollbackCondition] = field(default_factory=list)

    # マルチアームドバンディット設定
    enable_bandit_optimization: bool = False
    bandit_epsilon: float = 0.1  # 探索率
    bandit_update_frequency: int = 100  # 更新頻度（取引数）

    # モニタリング設定
    metrics_update_interval: int = 300  # 秒
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "win_rate_drop": 0.05,
        "pnl_drop": 0.1,
        "drawdown_increase": 0.02
    })

    # 安全設定
    max_traffic_percentage: float = 0.5  # 新バリアントの最大割合
    gradual_rollout_steps: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    gradual_rollout_interval_hours: int = 6

    # レポート設定
    enable_detailed_reporting: bool = True
    report_generation_interval: int = 3600  # 秒
    report_storage_path: str = "reports/ab_testing"

    def __post_init__(self):
        """設定の検証と初期化"""
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if self.min_sample_size < 100:
            raise ValueError("min_sample_size must be at least 100")

        if self.max_traffic_percentage > 1.0:
            raise ValueError("max_traffic_percentage cannot exceed 1.0")

        # デフォルトのロールバック条件を設定
        if not self.rollback_conditions:
            self.rollback_conditions = [
                RollbackCondition(
                    metric_name="win_rate",
                    threshold=0.05,
                    comparison="less_than",
                    consecutive_periods=3
                ),
                RollbackCondition(
                    metric_name="total_pnl",
                    threshold=0.1,
                    comparison="absolute_change",
                    consecutive_periods=2
                )
            ]


@dataclass
class TestConfiguration:
    """個別テスト設定"""

    test_id: str
    variants: List[str]
    traffic_distribution: Dict[str, float]
    test_duration_hours: int
    success_metrics: List[str]
    guardrail_metrics: List[str]

    def validate(self) -> bool:
        """設定の妥当性検証"""
        # トラフィック分布の合計が1.0であることを確認
        total_traffic = sum(self.traffic_distribution.values())
        if abs(total_traffic - 1.0) > 0.001:
            return False

        # すべてのバリアントがトラフィック分布に含まれていることを確認
        if set(self.variants) != set(self.traffic_distribution.keys()):
            return False

        return True