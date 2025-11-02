"""
Action Signal Guide Package

This package provides classical technical analysis signals for the SAC reinforcement
learning system, integrating traditional Japanese candlestick patterns and Western
technical indicators to enhance trading decision-making.
"""

from enum import Enum

from .action_signal_guide import (
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
