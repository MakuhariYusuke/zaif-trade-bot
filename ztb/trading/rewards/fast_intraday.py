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
    fee_penalty_weight: float = 0.0,
    eps: float = 1e-8,
    **kwargs,
) -> tuple[float, dict[str, float]]:
    """
    Compute reward for Fast Intraday strategy.
    
    SIMPLIFIED VERSION: Focus on PnL - Costs
    
    Formula:
    r_t = pnl - fee_paid - slippage_paid - (fee_paid + slippage_paid) * fee_penalty_weight
    
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
        fee_penalty_weight: Additional penalty multiplier for transaction costs.
        eps: Epsilon for numerical stability.
        
    Returns:
        Calculated reward.
    """
    # Mark-to-market PnL (JPY)
    pnl = position_prev * (price_now - price_prev)

    # Total cost
    base_cost = fee_paid + slippage_paid
    
    # Extra Fee Penalty (v457.2 Strategy)
    # Only penalize if there was a cost (i.e. a trade occurred)
    extra_fee_penalty = base_cost * fee_penalty_weight
    
    total_cost = base_cost + extra_fee_penalty

    # --- v455/v457 Anti-Freeze Logic Restoration ---
    
    # 1. Cost of Business / Edge Penalty
    # We want to penalize entering trades that don't have enough "Edge" (ATR) to cover costs.
    # delta = abs(position_now - position_prev)
    # But since we don't have delta easily here without re-computing, we rely on the caller or just skip delta scaling for now?
    # Actually, let's just use the current state.
    # Note: This logic assumes we want to punish "bad entries". 
    # For simplicity, we apply penalties based on holding state and environment conditions.

    penalty_term = 0.0
    
    # A) Edge Penalty (Simplified)
    # If we are holding a position, and ATR is too low compared to costs, it's a bad state.
    # required_edge = total_cost * min_edge_mult  (Note: total_cost is per-step realized cost, so this only fires on trade steps)
    
    if total_cost > 0: # This step had a trade
        # Expected move roughly per step is ATR? No, ATR is per candle. 
        # Let's say we need ATR > cost * min_edge_mult to justify trading.
        if atr < (total_cost * min_edge_mult):
             # Shortfall
             shortfall = (total_cost * min_edge_mult) - atr
             # Weigh by how much we traded? For now just flat penalty scaled by rate.
             # But wait, edge_penalty_rate in docs was: edge_penalty_rate * (shortfall / denom)
             # Let's assume denom ~ price or 1.0. Let's keep it simple.
             p_edge = edge_penalty_rate * shortfall
             penalty_term += p_edge
    
    # B) Low Volatility Penalty (Inventory Risk in dead market)
    # If we hold a position when Volatility is super low (ATR/Price < vol_floor), we bleed.
    if abs(position_now) > eps:
        # vol_ratio = atr / price_now
        # Avoid div by zero
        if price_now > eps:
            vol_ratio = atr / price_now
            if vol_ratio < vol_floor:
                shortfall = vol_floor - vol_ratio
                # Scale by position size ratio
                pos_ratio = abs(position_now) / max(max_position, eps)
                p_vol = vol_floor_penalty * shortfall * pos_ratio
                penalty_term += p_vol

    # C) Time-Decay Penalty (The Anti-Freeze Nuke)
    # v457 Update: Fixed Amount Penalty (No Position Scaling)
    # Prevents "Small Position Escape" where agent holds 0.01 BTC to minimize penalty.
    # Now, holding ANY position > eps incurs the full ramp penalty.
    if holding_steps > hold_grace:
        extra_hold = holding_steps - hold_grace
        
        # Determine fixed penalty scaling
        # If position is non-zero, apply full penalty regardless of size
        if abs(position_now) > eps:
             # Legacy: p_time = hold_ramp * extra_hold * (abs(position_now) / max_position)
             # New: p_time = hold_ramp * extra_hold * 1.0
             p_time = hold_ramp * extra_hold
             penalty_term += p_time

    # Simple reward: PnL - Costs - Penalties
    # Normalize by max_position for scale consistency?
    # NO. If we are using Real-Money-ish reward (JPY), we should output raw JPY.
    # However, existing wrapper might expect normalized.
    # v457: We decided to use raw JPY reward / 100000 in Env.
    # So here we should return "Raw JPY Reward".
    # But wait, (pnl-cost) is raw JPY. 
    # v456 logic was: reward = (pnl - total_cost) / max_position
    # This normalizes "Per 1 Unit" reward.
    # If we want PnL-based (Absolute), we should NOT divide by max_position.
    # BUT changing this now breaks continuity with previous steps in standardizers.
    # Let's keep the Division but remove it for the Fixed Penalty Term.
    
    # Actually, to make "Fixed Penalty" work in a normalized system:
    # If reward is per-unit, then penalty should be per-unit.
    # If we add a huge fixed penalty, we need to divide it by max_position to keep units consistent?
    # NO. The simplest way is to treat `penalty_term` as "Value to subtract from Total Reward numerator".
    
    # Let's revert to PnL (Raw) - Penalty (Raw) / MaxPos logic?
    # If max_pos=1.0, it's identity.
    # If max_pos=0.1 (Backtest), then valid.
    
    # To fix "Small Pos Escape":
    # The penalty `p_time` is now absolute JPY (e.g. 200 * steps).
    # We must ensuring it hits hard.
    
    # Calculation:
    # reward_numerator = pnl - total_cost - penalty_term
    # reward = reward_numerator / max(max_position, eps)
    
    # If position is 0.01, PnL is small. Cost is small.
    # If Penalty is 200 (Large), then numerator is -200.
    # Reward = -200 / 1.0 = -200. 
    # This is huge negative. GOOD.
    # This will FORCE exit.
    
    reward = (pnl - total_cost - penalty_term) / max(max_position, eps)
    
    reward_info = {
        "pnl": pnl,
        "fee_paid": fee_paid,
        "slippage_paid": slippage_paid,
        "total_cost": total_cost,
        "total_penalty": penalty_term,
        "reward_raw": reward,
    }
    
    return reward, reward_info
