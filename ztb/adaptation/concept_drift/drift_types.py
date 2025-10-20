"""
Type definitions for Concept Drift Detection
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class DriftType(Enum):
    """コンセプトドリフトの種類"""

    NONE = "none"  # ドリフトなし
    SUDDEN = "sudden"  # 突然の変化
    GRADUAL = "gradual"  # 徐々の変化
    RECURRENT = "recurrent"  # 周期的な変化
    INCREMENTAL = "incremental"  # 段階的な変化
    CONCEPT_DRIFT = "concept_drift"  # コンセプトドリフト（一般）


class DriftSeverity(Enum):
    """ドリフトの深刻度"""

    NONE = "none"  # ドリフトなし
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetectionResult:
    """ドリフト検知結果"""

    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    confidence: float
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class DriftStatistics:
    """ドリフト統計情報"""

    total_drift_events: int
    drift_frequency: float  # 1日あたりのドリフト発生数
    average_severity: float
    last_drift_time: Optional[datetime]
    drift_types_distribution: Dict[str, int]


@dataclass
class DriftThresholds:
    """ドリフト検知の閾値設定"""

    ks_test_p_value: float = 0.05
    cusum_threshold: float = 5.0
    adwin_delta: float = 0.002
    ddm_warning_level: float = 2.0
    ddm_drift_level: float = 3.0
    eddm_alpha: float = 0.95
