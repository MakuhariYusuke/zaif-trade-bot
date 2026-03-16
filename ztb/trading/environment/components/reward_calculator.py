"""Backward compatibility shim for reward_calculator.

Some tests and external scripts import `ztb.trading.environment.components.reward_calculator`.
The actual implementation lives under `components.calculators.reward_calculator`.
This shim re-exports the main classes to maintain backward compatibility.
"""
from ztb.trading.environment.components.calculators.reward_calculator import (
    RewardCalculator,
)
from ztb.trading.environment.components.calculators.v457_reward_calculator import (
    V457RewardCalculator,
)

__all__ = ["RewardCalculator", "V457RewardCalculator"]
