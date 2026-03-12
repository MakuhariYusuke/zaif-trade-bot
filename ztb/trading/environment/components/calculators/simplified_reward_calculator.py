"""Minimal simplified reward calculator used by script-level tests."""
from typing import Sequence

def calculate_downside_risk_reward(*args: object, **kwargs: object) -> float:
    # Delegates to metrics module in normal code; provide a trivial implementation
    returns = args[0] if args else []
    try:
        return float(sum(r for r in returns if r < 0) / (len([r for r in returns if r < 0]) or 1))
    except Exception:
        return 0.0

def calculate_trading_reward(returns: Sequence[float], penalty: float = 1.0) -> float:
    avg = sum(returns) / (len(returns) or 1)
    downside = calculate_downside_risk_reward(returns)
    return float(avg) - penalty * float(downside)

class SimplifiedRewardCalculator:
    """Backwards-compatible wrapper exposing a simple class-based API used in some
    script-level tests. Delegates to the functional helpers above.
    """

    def __init__(self, config: object = None, reward_settings: object = None, initial_balance: float = 0.0) -> None:
        self.config = config
        self.reward_settings = reward_settings
        self.initial_balance = initial_balance

    def compute(self, returns: Sequence[float]) -> float:
        penalty = 1.0
        if isinstance(self.reward_settings, dict):
            penalty = self.reward_settings.get("penalty", penalty)
        return calculate_trading_reward(returns, penalty=penalty)

__all__ = [
    "calculate_downside_risk_reward",
    "calculate_trading_reward",
    "SimplifiedRewardCalculator",
]
