"""
Common type definitions for evaluation and monitoring
評価と監視の共通型定義
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


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
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Additional metrics
    consistency_score: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class EvaluationResult:
    """評価結果"""

    timestamp: datetime
    performance_metrics: Optional[EvaluationMetrics] = None
    safety_metrics: Optional[Dict[str, Any]] = None
    drift_detected: bool = False
