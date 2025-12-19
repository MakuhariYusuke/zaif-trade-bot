"""Risk management utilities for training and evaluation."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def setup_risk_management_config(risk_config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup risk management configuration.

    Args:
        risk_config: Risk configuration dict

    Returns:
        Risk manager configuration
    """
    return {
        "position_sizer": {
            "enabled": risk_config.get("dynamic_position_sizing", True),
            "volatility_adjustment": risk_config.get("volatility_adjustment", True),
            "min_position_size": 0.001,
            "max_position_size": 0.2,
            "base_position_size": 0.1,
        },
        "drawdown_controller": {
            "enabled": risk_config.get("drawdown_control", True),
            "max_drawdown_limit": risk_config.get("max_drawdown_limit", 0.1),
            "emergency_stop_threshold": 0.15,
            "recovery_threshold": 0.05,
        },
        "market_adaptor": {
            "enabled": True,
            "adaptation_window": 50,
            "volatility_threshold": 0.02,
            "trend_strength_threshold": 0.01,
            "regime_change_threshold": 0.7,
        },
    }