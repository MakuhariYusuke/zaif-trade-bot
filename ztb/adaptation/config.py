"""
SAC Adaptation Configuration
SAC適応設定
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .monitoring.config import MonitoringConfig
from .safety.config import SafetyConfig
from .online_learning.config import OnlineLearningConfig
from .operations.config import OperationsConfig


@dataclass
class SACConfig:
    """SAC適応システム全体設定"""

    # 基本設定
    adaptation_enabled: bool = True
    adaptation_mode: str = "online"  # online, offline, hybrid

    # 各コンポーネント設定
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    online_learning: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)
    operations: OperationsConfig = field(default_factory=OperationsConfig)

    # 統合設定
    integration_enabled: bool = True
    cross_component_coordination: bool = True
    global_cooldown_seconds: int = 60

    # パフォーマンス設定
    max_concurrent_adaptations: int = 3
    adaptation_timeout_seconds: int = 300
    resource_limits: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 80.0,
        "memory_percent": 85.0,
        "gpu_percent": 90.0
    })

    # ログと監視
    log_level: str = "INFO"
    metrics_collection_enabled: bool = True
    metrics_retention_days: int = 30

    def __post_init__(self):
        """設定の検証"""
        if self.max_concurrent_adaptations <= 0:
            raise ValueError("max_concurrent_adaptations must be positive")

        if self.adaptation_timeout_seconds <= 0:
            raise ValueError("adaptation_timeout_seconds must be positive")

        valid_modes = ["online", "offline", "hybrid"]
        if self.adaptation_mode not in valid_modes:
            raise ValueError(f"adaptation_mode must be one of {valid_modes}")