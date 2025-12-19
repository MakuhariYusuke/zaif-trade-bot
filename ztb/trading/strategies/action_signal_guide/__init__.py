"""
Action Signal Guide Package

This package provides classical technical analysis signals for the SAC reinforcement
learning system, integrating traditional Japanese candlestick patterns and Western
technical indicators to enhance trading decision-making.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .action_signal_guide import (  # noqa: F401
        ActionSignal,
        ActionSignalGuide,
        ActionSignalGuideConfig,
        RecognizerConfig,
    )


class GuidanceMode(Enum):
    """Modes of signal guidance."""

    FULL_GUIDANCE = "full"  # Strong guidance for early training
    PARTIAL_GUIDANCE = "partial"  # Moderate guidance
    MINIMAL_GUIDANCE = "minimal"  # Light guidance for advanced training
    FADE_OUT = "fade_out"  # Guidance that fades out over time
    NO_GUIDANCE = "none"  # No guidance (pure RL)


# Export GuidanceMode as GuidanceLevel for backward compatibility
GuidanceLevel = GuidanceMode

__all__ = [
    "ActionSignalGuide",
    "GuidanceLevel",
    "GuidanceMode",
    "ActionSignalGuideConfig",
    "ActionSignal",
    "RecognizerConfig",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ActionSignal": ("ztb.trading.strategies.action_signal_guide.action_signal_guide", "ActionSignal"),
    "ActionSignalGuide": (
        "ztb.trading.strategies.action_signal_guide.action_signal_guide",
        "ActionSignalGuide",
    ),
    "ActionSignalGuideConfig": (
        "ztb.trading.strategies.action_signal_guide.action_signal_guide",
        "ActionSignalGuideConfig",
    ),
    "RecognizerConfig": (
        "ztb.trading.strategies.action_signal_guide.action_signal_guide",
        "RecognizerConfig",
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
