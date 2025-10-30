#!/usr/bin/env python3
"""
Environment configuration for trading environments.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

DEFAULT_INITIAL_BALANCE = 10000.0
DEFAULT_MAX_STEPS = 1000
DEFAULT_COMMISSION = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%


@dataclass
class EnvironmentConfig:
    """
    Configuration for trading environment.
    """

    initial_balance: float = DEFAULT_INITIAL_BALANCE
    max_steps: int = DEFAULT_MAX_STEPS
    commission: float = DEFAULT_COMMISSION
    slippage: float = DEFAULT_SLIPPAGE
    max_position_size: float = 1.0
    min_trade_size: float = 1e-5  # Allow very small trades for aggressive scalping
    min_position_change: float = 1e-5  # Minimum delta in position required to execute
    reward_scaling: float = 1.0
    observation_window: int = 60
    feature_names: Optional[list] = None
    curriculum_stage: str = "pnl_focused"  # Default for v439 scalping
    continuous_to_discrete_threshold: float = 0.02  # Lowered for frequent actions
    continuous_to_discrete_threshold_neg: Optional[float] = None
    signal_guidance_enabled: bool = True
    signal_guidance: Dict[str, Any] = field(default_factory=dict)
    scalping_optimization: Dict[str, Any] = field(default_factory=dict)
    use_continuous_actions: bool = False
    behavior_optimization: Optional[Dict[str, Any]] = None
    market_regime: Optional[Dict[str, Any]] = None
    dynamic_reward_shaping: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "initial_balance": self.initial_balance,
            "max_steps": self.max_steps,
            "commission": self.commission,
            "slippage": self.slippage,
            "max_position_size": self.max_position_size,
            "min_trade_size": self.min_trade_size,
            "min_position_change": self.min_position_change,
            "reward_scaling": self.reward_scaling,
            "observation_window": self.observation_window,
            "feature_names": self.feature_names,
            "curriculum_stage": self.curriculum_stage,
            "continuous_to_discrete_threshold": self.continuous_to_discrete_threshold,
            "continuous_to_discrete_threshold_neg": self.continuous_to_discrete_threshold_neg,
            "signal_guidance_enabled": self.signal_guidance_enabled,
            "signal_guidance": self.signal_guidance,
            "scalping_optimization": self.scalping_optimization,
            "use_continuous_actions": self.use_continuous_actions,
            "behavior_optimization": self.behavior_optimization,
            "market_regime": self.market_regime,
            "dynamic_reward_shaping": self.dynamic_reward_shaping,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnvironmentConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
