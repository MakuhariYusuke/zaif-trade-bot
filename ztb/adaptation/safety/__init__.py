"""
Safety Mechanisms and Fallback Systems
安全メカニズムとフォールバックシステム

This module provides comprehensive safety mechanisms for SAC v421 including:
- Anomaly Detection: Statistical and ML-based anomaly detection
- Fallback Manager: Multiple fallback strategies (conservative, circuit breaker, gradual degradation)
- Recovery Manager: Automated system recovery with multiple strategies
- Integrated Safety Manager: Orchestrates all safety components

SAC v421の包括的な安全メカニズムを提供:
- 異常検知: 統計的およびMLベースの異常検知
- フォールバックマネージャー: 複数のフォールバック戦略（保守的、遮断器、段階的劣化）
- リカバリーマネージャー: 複数の戦略による自動システムリカバリー
- 統合安全マネージャー: すべての安全コンポーネントをオーケストレーション

Legacy Features:
- Staged rollouts: 段階的デプロイメント
- Circuit breakers: 自動遮断システム
- Fallback strategies: バックアップ戦略
- Anomaly detection: 異常検知と対応
"""

from ..monitoring.safety import SafetyManager
from .config import SafetyConfig
from .types import SafetyLevel, FallbackStrategy, CircuitBreakerState, FallbackStatus

# New SAC v421 Safety Components
from .fallback_manager import FallbackManager, FallbackConfig
from .anomaly_manager import AnomalyDetectionManager, AnomalyConfig, AnomalyResult
from .recovery_manager import RecoveryManager, RecoveryConfig, RecoveryAttempt
from .integrated_safety_manager import IntegratedSafetyManager, IntegratedSafetyConfig, SafetyEventRecord
from .types import (
    SafetyEvent,
    SafetyAction,
    AnomalyType,
    AnomalyDetection,
    FallbackMode,
    RecoveryStatus,
    RecoveryStrategy
)

__all__ = [
    # Legacy Components
    "SafetyManager",
    "SafetyConfig",
    "SafetyLevel",
    "FallbackStrategy",
    "CircuitBreakerState",
    "FallbackStatus",

    # New SAC v421 Components
    "FallbackManager",
    "AnomalyDetectionManager",
    "RecoveryManager",
    "IntegratedSafetyManager",

    # New Configurations
    "FallbackConfig",
    "AnomalyConfig",
    "RecoveryConfig",
    "IntegratedSafetyConfig",

    # New Types
    "SafetyEvent",
    "SafetyAction",
    "AnomalyType",
    "AnomalyDetection",
    "FallbackMode",
    "RecoveryStatus",
    "RecoveryStrategy",

    # New Data Classes
    "AnomalyResult",
    "RecoveryAttempt",
    "SafetyEventRecord"
]

__version__ = "4.2.1"
__author__ = "SAC v421 Safety Team"
__description__ = "Comprehensive safety mechanisms and fallback systems for SAC v421"