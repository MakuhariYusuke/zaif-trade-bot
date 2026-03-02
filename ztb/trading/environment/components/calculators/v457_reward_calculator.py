from typing import Any
import numpy as np

class V457RewardCalculator:
    """
    v457 'Reset' Reward Calculator.
    Implements the 'Legacy Asset' logic from v451:
    - Pure Mark-to-Market PnL driven
    - Asymmetric Loss Aversion (Loss Multiplier 1.2)
    - No complex penalties (Hold, Regime, Confidence etc defined in v456)
    """

    def __init__(self, config: Any, reward_settings: Any, initial_portfolio_value: float):
        self.config = config
        self.initial_portfolio_value = initial_portfolio_value
        
        # Hardcoded 'Asset' values from v451 (or passed via config if present)
        # But we default to the 'Golden Era' values.
        self.reward_scale = 1.0
        
        # PnL Centered logic
        rs = reward_settings if isinstance(reward_settings, dict) else {}
        self.loss_multiplier = rs.get("loss_multiplier", 1.2)
        self.profit_multiplier = rs.get("profit_multiplier", 1.0)
        
        self.last_reward_components = {}
        
        # Trend Detector stub to satisfy HeavyEnv interface checks
        self.trend_detector = None 

    def reset(self) -> None:
        """Reset internal state."""
        self.last_reward_components = {}

    def get_last_reward_components(self) -> dict:
        """Return the components of the last calculated reward."""
        return self.last_reward_components

    def calculate_reward(
        self,
        action: int,
        current_price: float,
        position: float,
        portfolio_value: float,
        atr: float,
        transaction_cost: float,
        reward_scaling: float,
        pnl: float,
        old_position: float,
        step: int,
        observation: np.ndarray | None,
        reward_history: list[float],
        portfolio_value_history: list[float],
        continuous_action_value: float | None = None,
        trade_pnl: float = 0.0,
    ) -> float:
        """
        Calculates simple PnL-based reward.
        Reward = (Step PnL) * Scale * (Loss Multiplier if negative)
        """
        
        step_pnl = float(pnl)
        
        # Apply asymmetric loss aversion (v451 Logic)
        weighted_pnl = step_pnl
        if step_pnl > 0:
            weighted_pnl *= self.profit_multiplier
        elif step_pnl < 0:
            weighted_pnl *= self.loss_multiplier
            
        # Apply global scaling
        reward = weighted_pnl * reward_scaling
        
        self.last_reward_components = {
            "pnl": step_pnl,
            "weighted_pnl": weighted_pnl,
            "reward": reward,
            "trade_pnl": trade_pnl
        }
        
        return reward
