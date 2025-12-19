"""
Trading strategies package.

Keep imports lazy: many strategy modules pull in heavy pattern-recognition code
that is not required for core backtesting/training flows when signal guidance is
disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .action_signal_guide import GuidanceLevel, GuidanceMode  # noqa: F401
    from .action_signal_guide.action_signal_guide import ActionSignalGuide  # noqa: F401
    from .signal_definitions import SignalDefinitions, SignalStrength, SignalType  # noqa: F401
    from .signal_evaluator import BacktestResult, SignalEvaluator, SignalPerformance  # noqa: F401
    from .signal_integration import SignalIntegration  # noqa: F401
    from .signal_reward_integrator import SignalRewardIntegrator  # noqa: F401

__all__ = [
    "ActionSignalGuide",
    "SignalDefinitions",
    "SignalEvaluator",
    "SignalRewardIntegrator",
    "SignalIntegration",
    "GuidanceLevel",
    "GuidanceMode",
    "SignalType",
    "SignalStrength",
    "SignalPerformance",
    "BacktestResult",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ActionSignalGuide": (
        "ztb.trading.strategies.action_signal_guide.action_signal_guide",
        "ActionSignalGuide",
    ),
    "GuidanceLevel": ("ztb.trading.strategies.action_signal_guide", "GuidanceLevel"),
    "GuidanceMode": ("ztb.trading.strategies.action_signal_guide", "GuidanceMode"),
    "SignalDefinitions": ("ztb.trading.strategies.signal_definitions", "SignalDefinitions"),
    "SignalStrength": ("ztb.trading.strategies.signal_definitions", "SignalStrength"),
    "SignalType": ("ztb.trading.strategies.signal_definitions", "SignalType"),
    "BacktestResult": ("ztb.trading.strategies.signal_evaluator", "BacktestResult"),
    "SignalEvaluator": ("ztb.trading.strategies.signal_evaluator", "SignalEvaluator"),
    "SignalPerformance": ("ztb.trading.strategies.signal_evaluator", "SignalPerformance"),
    "SignalIntegration": ("ztb.trading.strategies.signal_integration", "SignalIntegration"),
    "SignalRewardIntegrator": (
        "ztb.trading.strategies.signal_reward_integrator",
        "SignalRewardIntegrator",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
