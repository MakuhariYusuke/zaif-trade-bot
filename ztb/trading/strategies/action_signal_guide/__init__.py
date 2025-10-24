"""
Action Signal Guide Package

This package provides classical technical analysis signals for the SAC reinforcement
learning system, integrating traditional Japanese candlestick patterns and Western
technical indicators to enhance trading decision-making.
"""

from .action_signal_guide import ActionSignalGuide, GuidanceLevel

# Export GuidanceLevel as GuidanceMode for backward compatibility
GuidanceMode = GuidanceLevel

__all__ = ['ActionSignalGuide', 'GuidanceLevel', 'GuidanceMode']