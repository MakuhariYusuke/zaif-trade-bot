"""
Fast Intraday / HFT Reward Function
"""

import numpy as np


def compute_hft_reward(
    price_prev: float,
    price_now: float,
    position_prev: float,
    position_now: float,
    atr: float,
    fee_paid: float,
    slippage_paid: float,
    holding_steps: int,
    max_position: float,
    alpha: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    min_edge_mult: float = 1.0,
    edge_penalty_rate: float = 0.0,
    vol_floor: float = 0.0,
    vol_floor_penalty: float = 0.0,
    hold_grace: int = 0,
    hold_ramp: float = 0.0,
    eps: float = 1e-8,
) -> tuple[float, dict[str, float]]:
    """
    Compute reward for Fast Intraday strategy.
    
    SIMPLIFIED VERSION: Focus on PnL - Costs
    
    Formula:
    r_t = pnl - fee_paid - slippage_paid
    
    Args:
        price_prev: Price at previous step.
        price_now: Price at current step.
        position_prev: Position size at previous step (absolute units, e.g. BTC).
        position_now: Position size at current step.
        atr: Current ATR (volatility).
        fee_paid: Fee paid in this step (JPY).
        slippage_paid: Slippage cost paid in this step (JPY).
        holding_steps: Number of steps the current position has been held.
        max_position: Maximum allowed position size (for normalization).
        alpha: Coefficient for position change penalty (churn) - DISABLED.
        beta: Coefficient for holding time penalty - DISABLED.
        gamma: Coefficient for inventory risk - DISABLED.
        min_edge_mult: Required multiple of trade cost vs ATR-based move.
        edge_penalty_rate: Penalty rate for trades with insufficient edge.
        vol_floor: Minimum ATR/price ratio to trade; below this adds penalty.
        vol_floor_penalty: Penalty rate for holding in low volatility.
        hold_grace: Steps held before extra time-decay penalty applies.
        hold_ramp: Extra per-step penalty after hold_grace.
        eps: Epsilon for numerical stability.
        
    Returns:
        Calculated reward.
    """
    # Mark-to-market PnL (JPY)
    pnl = position_prev * (price_now - price_prev)

    # Total cost
    total_cost = fee_paid + slippage_paid

    # Simple reward: PnL - Costs
    # Normalize by max_position for scale consistency
    reward = (pnl - total_cost) / max(max_position, eps)
    
    reward_info = {
        "pnl": pnl,
        "fee_paid": fee_paid,
        "slippage_paid": slippage_paid,
        "total_cost": total_cost,
        "reward_raw": reward,
    }
    
    return reward, reward_info
