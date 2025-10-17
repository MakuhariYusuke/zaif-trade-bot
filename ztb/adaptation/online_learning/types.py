"""
Type definitions for Online Learning Pipeline
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np


class LearningMode(Enum):
    """学習モード"""
    INCREMENTAL = "incremental"      # インクリメンタル学習
    ONLINE = "online"               # オンライン学習
    MINI_BATCH = "mini_batch"       # ミニバッチ学習
    STREAMING = "streaming"         # ストリーミング学習


class UpdateStrategy(Enum):
    """更新戦略"""
    SGD = "sgd"                     # 確率的勾配降下法
    ADAM = "adam"                   # Adam最適化
    ADAGRAD = "adagrad"             # AdaGrad最適化
    RMS_PROP = "rms_prop"           # RMSProp最適化


class MemoryStrategy(Enum):
    """メモリ管理戦略"""
    RESERVOIR = "reservoir"         # リザーバーサンプリング
    SLIDING_WINDOW = "sliding_window"  # スライディングウィンドウ
    TIME_DECAY = "time_decay"       # 時間減衰
    IMPORTANCE_SAMPLING = "importance_sampling"  # 重要度サンプリング


@dataclass
class LearningState:
    """学習状態"""
    model_version: str
    total_samples_processed: int
    current_learning_rate: float
    gradient_norm: float
    loss_history: List[float]
    last_update_time: datetime
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float]


@dataclass
class DataBatch:
    """データバッチ"""
    features: np.ndarray
    targets: np.ndarray
    weights: Optional[np.ndarray]
    timestamps: List[datetime]
    batch_id: str
    priority: float = 1.0


@dataclass
class UpdateResult:
    """学習更新結果"""
    success: bool
    loss_change: float
    gradient_norm: float
    parameter_updates: int
    processing_time_ms: float
    memory_delta_mb: float
    error_message: Optional[str]


@dataclass
class StreamingConfig:
    """ストリーミング設定"""
    batch_size: int
    buffer_size: int
    max_delay_ms: int
    checkpoint_interval: int
    data_source: str  # "kafka", "redis", "file"


@dataclass
class ModelCheckpoint:
    """モデルチェックポイント"""
    version: str
    timestamp: datetime
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    metrics: Dict[str, float]
    data_signature: str  # データ分布のハッシュ


@dataclass
class DriftAdaptation:
    """ドリフト適応情報"""
    drift_detected: bool
    drift_type: str
    adaptation_applied: bool
    adaptation_params: Dict[str, Any]
    performance_impact: float


@dataclass
class ResourceMetrics:
    """リソース使用量メトリクス"""
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float]
    disk_io_mb_per_sec: float
    network_io_mb_per_sec: float
    timestamp: datetime