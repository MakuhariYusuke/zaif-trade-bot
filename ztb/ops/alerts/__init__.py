"""
Alerting system for Zaif Trade Bot health monitoring.

This package provides notification capabilities for health check failures
and critical system issues.
"""

from .alert_system import (
    AlertConfig,
    AlertManager,
    AlertPriority,
    HealthAlert,
    create_alert_manager,
)

__all__ = [
    "AlertConfig",
    "AlertManager",
    "AlertPriority",
    "HealthAlert",
    "create_alert_manager",
]
