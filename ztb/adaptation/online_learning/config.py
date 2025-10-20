"""
Configuration management for Online Learning Pipeline
"""

from dataclasses import dataclass, field
from typing import Optional

from .types import LearningMode, MemoryStrategy, StreamingConfig, UpdateStrategy


@dataclass
class OnlineLearningConfig:
    """オンライン学習設定"""

    # 学習設定
    learning_mode: LearningMode = LearningMode.INCREMENTAL
    update_strategy: UpdateStrategy = UpdateStrategy.ADAM
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs_per_update: int = 1

    # メモリ管理
    memory_strategy: MemoryStrategy = MemoryStrategy.SLIDING_WINDOW
    max_memory_samples: int = 10000
    memory_decay_factor: float = 0.95
    importance_threshold: float = 0.1

    # ストリーミング設定
    streaming_config: StreamingConfig = field(
        default_factory=lambda: StreamingConfig(
            batch_size=32,
            buffer_size=1000,
            max_delay_ms=1000,
            checkpoint_interval=1000,
            data_source="redis",
        )
    )

    # 適応設定
    enable_drift_adaptation: bool = True
    adaptation_trigger_threshold: float = 0.1
    adaptation_cooldown_hours: int = 1

    # パフォーマンス設定
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 5
    validation_interval: int = 100

    # リソース管理
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_mb: float = 4096
    gpu_memory_limit_mb: Optional[int] = None

    # チェックポイント設定
    checkpoint_interval_updates: int = 1000
    max_checkpoints_to_keep: int = 10
    checkpoint_storage_path: str = "checkpoints/online_learning"

    # モニタリング設定
    metrics_update_interval: int = 60  # 秒
    performance_alert_threshold: float = 0.05

    def __post_init__(self):
        """設定の検証と初期化"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if self.max_memory_samples < 1000:
            raise ValueError("max_memory_samples must be at least 1000")

        if (
            self.adaptation_trigger_threshold <= 0
            or self.adaptation_trigger_threshold >= 1
        ):
            raise ValueError("adaptation_trigger_threshold must be between 0 and 1")


@dataclass
class LearningSchedule:
    """学習スケジュール設定"""

    initial_learning_rate: float
    decay_type: str  # "exponential", "linear", "step"
    decay_rate: float
    decay_steps: int
    min_learning_rate: float

    def get_learning_rate(self, step: int) -> float:
        """指定ステップでの学習率を取得"""
        if self.decay_type == "exponential":
            lr = self.initial_learning_rate * (
                self.decay_rate ** (step // self.decay_steps)
            )
        elif self.decay_type == "linear":
            decay = (self.initial_learning_rate - self.min_learning_rate) * (
                step / self.decay_steps
            )
            lr = max(self.initial_learning_rate - decay, self.min_learning_rate)
        elif self.decay_type == "step":
            lr = self.initial_learning_rate * (
                self.decay_rate ** (step // self.decay_steps)
            )
        else:
            lr = self.initial_learning_rate

        return max(lr, self.min_learning_rate)
