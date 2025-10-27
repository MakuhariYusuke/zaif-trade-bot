"""
Trading Strategies Package - v436 Action Signal Guide Implementation

This package provides trading strategies and signal guidance systems
for reinforcement learning agents.
"""

from .action_signal_guide import ActionSignalGuide, GuidanceMode
from .signal_definitions import SignalDefinitions, SignalStrength, SignalType
from .signal_evaluator import BacktestResult, SignalEvaluator, SignalPerformance
from .signal_integration import SignalIntegration
from .signal_reward_integrator import SignalRewardIntegrator

__all__ = [
    # Core classes
    "ActionSignalGuide",
    "SignalDefinitions",
    "SignalEvaluator",
    "SignalRewardIntegrator",
    "SignalIntegration",
    # Enums and types
    "GuidanceMode",
    "SignalType",
    "SignalStrength",
    # Data classes
    "SignalPerformance",
    "BacktestResult",
]
