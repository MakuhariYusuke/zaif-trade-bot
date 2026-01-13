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
    alpha: float = 0.2,
    beta: float = 0.01,
    gamma: float = 0.5,
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
    
    Formula:
    r_t = pnl_norm - costs - alpha * |pos_chg| - beta * hold - gamma * inv_risk
    
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
        alpha: Coefficient for position change penalty (churn).
        beta: Coefficient for holding time penalty.
        gamma: Coefficient for inventory risk.
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
    # PnL is based on the position held from prev to now.
    # Usually: position_prev * (price_now - price_prev)
    pnl = position_prev * (price_now - price_prev)

    # Normalizer: use ATR or price to make reward unitless
    # We normalize by (Volatility * Max Position) to get a "Risk Adjusted Return" like metric.
    # If ATR is very small, we fallback to a fraction of price.
    denom = max(atr, price_now * 0.001, eps) * max_position

    pnl_norm = pnl / denom
    fee_norm = fee_paid / denom
    slip_norm = slippage_paid / denom

    # Position change penalty (churn)
    # position_change is in absolute units.
    # We normalize it by max_position to get fraction [0, 2].
    position_change = abs(position_now - position_prev)
    churn_penalty = alpha * (position_change / max_position)

    # Holding time penalty (soft)
    # holding_steps is an integer.
    hold_penalty = beta * holding_steps

    # Inventory risk: larger position + higher vol = more penalty
    # We want to penalize holding large positions during high volatility.
    # |pos| / max_pos is fraction [0, 1].
    # vol_ratio = ATR / Price is volatility fraction.
    # But we already normalized PnL by ATR.
    # Let's follow the plan: gamma * abs(position_now) * vol_ratio
    # Wait, the plan snippet said:
    # vol_ratio = atr / max(price_now, eps)
    # inventory_risk = gamma * abs(position_now) * vol_ratio
    # But this inventory_risk is not normalized by denom?
    # Let's look at the snippet again.
    # reward = pnl_norm - ... - inventory_risk
    # So inventory_risk must be unitless.
    # abs(position_now) is units. vol_ratio is unitless. Result is units.
    # This seems wrong if we subtract it from unitless pnl_norm.
    
    # Let's re-read the plan snippet carefully.
    # pnl_norm = pnl / denom
    # ...
    # inventory_risk = gamma * abs(position_now) * vol_ratio
    # reward = ... - inventory_risk
    
    # If pnl_norm is unitless (approx ~1.0 for 1 ATR move),
    # inventory_risk should also be unitless.
    # If position is 1 BTC, vol_ratio is 0.01. inventory_risk = 0.005.
    # This seems compatible with pnl_norm.
    # BUT, if position is 0.001 BTC, inventory_risk is tiny.
    # pnl_norm handles position size in PnL, but divides by max_position.
    # So pnl_norm is proportional to (position / max_position).
    
    # Let's adjust inventory_risk to be consistent:
    # inventory_risk = gamma * (abs(position_now) / max_position) * (atr / price_now * 100?)
    # Or just follow the snippet but ensure units make sense.
    
    # Snippet:
    # vol_ratio = atr / max(price_now, eps)
    # inventory_risk = gamma * abs(position_now) * vol_ratio
    
    # If I use this, and position is 0.1 BTC, price 1M, ATR 10k.
    # vol_ratio = 0.01.
    # inv_risk = 0.5 * 0.1 * 0.01 = 0.0005.
    # PnL for 1 ATR move with 0.1 BTC = 0.1 * 10k = 1000 JPY.
    # Denom = 10k * 1.0 (max_pos) = 10k.
    # pnl_norm = 1000 / 10000 = 0.1.
    # So inv_risk (0.0005) is very small compared to PnL (0.1).
    # Maybe gamma needs to be larger, or vol_ratio scaling is different.
    
    # Let's stick to the snippet but maybe normalize position by max_position.
    # inventory_risk = gamma * (abs(position_now) / max_position)
    # This penalizes just holding position.
    # If we want to penalize holding in HIGH VOLATILITY, we multiply by (ATR / RefATR).
    
    # Let's use:
    # inventory_risk = gamma * (abs(position_now) / max_position) * (atr / (price_now * 0.001))
    # This makes it relative to "baseline volatility".
    
    # Actually, let's stick to the simplest interpretation of the plan which was accepted.
    # "Inventory risk: larger position + higher vol = more penalty"
    # "Recommended: inventory_risk = |pos| * (ATR / price)"
    # Wait, if I use that, it's unit dependent on |pos|.
    # If |pos| is 0.1 vs 100, the value changes.
    # But pnl_norm is also unit dependent? No, pnl_norm = pnl / (ATR * max_pos).
    # pnl = pos * dPrice.
    # pnl_norm = (pos * dPrice) / (ATR * max_pos) = (pos/max_pos) * (dPrice/ATR).
    # This is unitless.
    
    # So inventory_risk should be unitless.
    # |pos| * (ATR/Price) has units of |pos|.
    # It should be (|pos|/max_pos) * (ATR/Price) * Scale?
    
    # I will implement:
    # inventory_risk = gamma * (abs(position_now) / max_position) * (atr / max(price_now, eps)) * 100
    # The *100 is to bring ATR/Price (usually 0.01) to ~1.0 range.
    # Or I can just rely on gamma being large.
    
    # Let's stick to the snippet provided in 08_hft_implementation_details_request.md response:
    # vol_ratio = atr / max(price_now, eps)
    # inventory_risk = gamma * abs(position_now) * vol_ratio
    # AND NOTE: The snippet had `pnl_norm = pnl / denom`.
    # If the expert wrote it, maybe they assumed `gamma` handles the scale.
    # But `abs(position_now)` definitely has units.
    # I will normalize position by max_position.
    
    vol_ratio = atr / max(price_now, eps)
    inventory_risk = gamma * (abs(position_now) / max_position) * (vol_ratio * 100.0)

    # Edge penalty: discourage trades where expected move (ATR proxy) is below cost
    trade_cost = fee_paid + slippage_paid
    expected_move = atr * position_change
    required_edge = trade_cost * min_edge_mult
    edge_shortfall = max(0.0, required_edge - expected_move)
    edge_penalty = 0.0
    if position_change > 0:
        edge_penalty = edge_penalty_rate * (edge_shortfall / denom)

    # Low-volatility penalty: bias toward flat when ATR/price is too small
    low_vol_shortfall = max(0.0, vol_floor - vol_ratio)
    low_vol_penalty = vol_floor_penalty * low_vol_shortfall * (abs(position_now) / max_position)

    # Extra time-decay penalty after grace period
    extra_hold = max(0, holding_steps - hold_grace)
    time_decay_penalty = hold_ramp * extra_hold * (abs(position_now) / max_position)
    # Added *100 to make vol_ratio ~1.0 for 1% volatility.
    
    reward = (
        pnl_norm
        - fee_norm
        - slip_norm
        - churn_penalty
        - hold_penalty
        - inventory_risk
        - edge_penalty
        - low_vol_penalty
        - time_decay_penalty
    )
    
    reward_info = {
        "pnl_norm": pnl_norm,
        "fee_norm": fee_norm,
        "slip_norm": slip_norm,
        "churn_penalty": churn_penalty,
        "hold_penalty": hold_penalty,
        "inventory_risk": inventory_risk,
        "edge_penalty": edge_penalty,
        "low_vol_penalty": low_vol_penalty,
        "time_decay_penalty": time_decay_penalty,
        "vol_ratio": vol_ratio,
        "edge_shortfall": edge_shortfall,
        "trade_cost": trade_cost,
        "expected_move": expected_move,
        "required_edge": required_edge,
        "position_change": position_change
    }
    
    return reward, reward_info
