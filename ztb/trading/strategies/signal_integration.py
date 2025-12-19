"""
Signal Integration - Legacy Compatibility Module

This module provides a backward-compatible alias `SignalIntegration`
for the newer `SignalRewardIntegrator` implementation. Historically many
components imported the `SignalIntegration` class from here; to preserve
backwards compatibility, we re-export the newer class under the old name.

The heavier implementation lives in `signal_reward_integrator.py`.
"""

from .signal_reward_integrator import SignalRewardIntegrator as SignalIntegration

__all__ = ["SignalIntegration"]
