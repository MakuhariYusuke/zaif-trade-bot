"""
Common type definitions for evaluation and monitoring
評価と監視の共通型定義
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .common import AlertLevel


class AlertType(Enum):
    """アラートタイプ"""

    PERFORMANCE = "performance"
    SAFETY = "safety"
    DRIFT = "drift"
    SYSTEM = "system"


@dataclass
class EvaluationMetrics:
    """統合された評価メトリクス"""

    # ML/Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Financial metrics
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None

    # Additional metrics
    consistency_score: Optional[float] = None
    recovery_factor: Optional[float] = None


@dataclass
class EvaluationResult:
    """評価結果"""

    timestamp: datetime
    performance_metrics: Optional[EvaluationMetrics] = None
    safety_metrics: Optional[Dict[str, Any]] = None
    drift_detected: bool = False