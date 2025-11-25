"""
Common type definitions for alerts and monitoring
アラートと監視の共通型定義
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AlertLevel(Enum):
    """アラートレベル"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """アラートステータス"""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class AlertCondition:
    """アラート条件"""

    metric_name: str
    operator: str  # "gt", "lt", "eq", "ne", "gte", "lte"
    threshold: float
    duration_seconds: int
    cooldown_seconds: int
    alert_level: AlertLevel
    description: str
    auto_resolve: bool = True


@dataclass
class Alert:
    """アラート"""

    id: str
    condition: AlertCondition
    current_value: float
    threshold: float
    level: AlertLevel
    status: AlertStatus
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    description: str = ""
    context: Optional[Dict[str, Any]] = None
