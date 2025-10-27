"""
Signal Integration - Legacy Compatibility Module

This module provides backward compatibility for the old SignalIntegration class.
All functionality has been moved to signal_reward_integrator.py
"""

# Re-export for backward compatibility
from .signal_reward_integrator import SignalRewardIntegrator as SignalIntegration
