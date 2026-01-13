"""
Fast Intraday Trading Environment
Specialized lightweight environment for HFT/Scalping strategies.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ztb.processing.online_scaler import OnlineScaler
from ztb.trading.rewards.fast_intraday import compute_hft_reward
from ztb.utils.fee_model import ExchangeFeeModel

logger = logging.getLogger(__name__)


class FastIntradayEnv(gym.Env):
    """
    Fast Intraday Trading Environment.
    
    Action Space: Box(low=[-1, 0], high=[1, 1])
        - target_position: [-1, 1] (Fraction of max_position)
        - ttl_fraction: [0, 1] (Fraction of max_ttl)
        
    Observation Space: Box
        - Market Features (Scaled)
        - Account State (Normalized)
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        initial_balance: float = 1_000_000.0,
        max_position: float = 1.0,  # e.g. 1.0 BTC
        max_steps: Optional[int] = None,
        commission_rate: float = 0.001,  # 0.1%
        max_ttl_steps: int = 60, # 60 minutes
        cooldown_steps: int = 5,
        max_delta_per_step: float = 0.2, # Max 20% position change per step
        min_delta: float = 0.01, # Deadband
        drawdown_limit: float = 0.1, # 10% drawdown kills episode
        prewarm_steps: int = 100,
        reward_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.max_steps = max_steps
        self.max_ttl_steps = max_ttl_steps
        self.cooldown_steps = cooldown_steps
        self.max_delta_per_step = max_delta_per_step
        self.min_delta = min_delta
        self.drawdown_limit = drawdown_limit
        self.prewarm_steps = prewarm_steps
        
        self.reward_params = reward_params or {}
        
        # Fee Model - Using ExchangeFeeModel as required
        # We configure a generic exchange with the provided commission rate
        self.fee_model = ExchangeFeeModel(exchange_fees={
            "generic": {"buy": commission_rate, "sell": commission_rate}
        })
        self.fee_model.set_exchange("generic")
        
        # Pre-convert data to numpy for performance
        # Features
        self.features_data = self.df[feature_columns].values.astype(np.float32)
        # Price data for PnL
        self.close_prices = self.df["close"].values.astype(np.float32)
        # ATR for normalization/slippage (assume 'atr' column exists)
        if "atr" in self.df.columns:
            self.atr_data = self.df["atr"].values.astype(np.float32)
        else:
            # Fallback if ATR not present (should be added by feature eng)
            logger.warning("ATR column not found, using 1% of close price")
            self.atr_data = self.close_prices * 0.01
            
        # Impact proxy for slippage
        if "impact_proxy" in self.df.columns:
            self.impact_data = self.df["impact_proxy"].values.astype(np.float32)
        else:
            # Default to 0.0 if not present to avoid excessive slippage
            # Reviewer noted 1.0 is too high.
            self.impact_data = np.zeros_like(self.close_prices)

        # Store length and clear dataframe to save memory
        self.data_len = len(self.df)
        del self.df
        self.df = None # Explicitly release reference

        # Action Space
        # [target_position (-1 to 1), ttl_fraction (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation Space
        # Market Features + Account State
        # Account State: [current_position_norm, remaining_ttl_norm, last_cost_norm]
        self.account_state_dim = 3
        obs_dim = len(feature_columns) + self.account_state_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Scaler
        self.scaler = OnlineScaler(shape=(len(feature_columns),))
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Random start
        data_len = self.data_len
        # Ensure we have enough data for prewarm and at least some steps
        min_start = self.prewarm_steps
        max_start = data_len - (self.max_steps if self.max_steps else 1000)
        
        if max_start <= min_start:
            self.current_step = min_start
        else:
            self.current_step = self.np_random.integers(min_start, max_start)
            
        # Reset State
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_ttl = 0
        self.steps_held = 0
        self.cooldown_counter = 0
        self.total_pnl = 0.0
        self.max_balance = self.initial_balance
        self.last_step_cost = 0.0
        self.steps_in_episode = 0
        
        # Reset Scaler and Pre-warm
        self.scaler = OnlineScaler(shape=(len(self.feature_columns),))
        # Feed prewarm data
        prewarm_data = self.features_data[self.current_step - self.prewarm_steps : self.current_step]
        self.scaler.batch_update(prewarm_data)
            
        return self._get_observation(), {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 0. Prepare Data
        price_now = self.close_prices[self.current_step]
        price_prev = self.close_prices[self.current_step - 1] if self.current_step > 0 else price_now
        atr = self.atr_data[self.current_step]
        
        # 1. Parse Action
        target_pos_fraction = float(np.clip(action[0], -1.0, 1.0))
        ttl_fraction = float(np.clip(action[1], 0.0, 1.0))
        
        raw_target_position = target_pos_fraction * self.max_position
        
        # 2. Apply Constraints (Cooldown, TTL Expiration, TTL=0)
        
        # Check if TTL expired in previous step (position_ttl <= 0 and we have a position)
        # If expired, we force exit (target=0) and set cooldown
        # Use -1 to mark "Expired and Unwinding" to prevent resetting cooldown every step
        if self.position_ttl <= 0 and abs(self.position) > 1e-6:
            raw_target_position = 0.0
            # Only set cooldown once when transitioning from >0 (or 0) to Expired state
            # We use position_ttl == 0 to detect the "Just Expired" moment
            if self.position_ttl == 0:
                self.cooldown_counter = self.cooldown_steps
                self.position_ttl = -1 # Mark as handled
            
        # If cooling down, force flat
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            raw_target_position = 0.0
            
        # If requested TTL is effectively 0, treat as "No Entry" (force target=0)
        # Only applies if we are trying to enter or reverse
        # But we haven't calculated delta yet.
        # If we are flat, and ttl_fraction is 0, we stay flat.
        # If we are holding, and ttl_fraction is 0, do we exit?
        # Usually TTL is set on Entry.
        # If we are holding, we ignore ttl_fraction unless we reverse.
        # So we check this later or assume raw_target is valid for now.
        
        # 3. Target Transition (Deadband & Clipping)
        delta = raw_target_position - self.position
        
        # Deadband
        if abs(delta) < self.min_delta * self.max_position:
            delta = 0.0
            
        # Clipping (Liquidity constraint)
        max_delta = self.max_delta_per_step * self.max_position
        delta = np.clip(delta, -max_delta, max_delta)
        
        new_position = self.position + delta
        
        # 4. TTL Management (Update & Decrement)
        
        # Check for Entry or Reversal based on ACTUAL new position
        is_reversal = np.sign(new_position) != np.sign(self.position) and abs(new_position) > 1e-6 and abs(self.position) > 1e-6
        is_entry = abs(self.position) < 1e-6 and abs(new_position) > 1e-6
        
        if is_reversal or is_entry:
            # We are entering a new trade.
            # Check if TTL fraction is valid (>= 0.001).
            if ttl_fraction < 1e-3:
                # Invalid TTL for entry.
                # We should have prevented this.
                # Since we already moved, we must revert or force exit?
                # Reverting is cleaner: "Entry Rejected".
                new_position = self.position # Revert
                delta = 0.0
                # No TTL update
            else:
                # Valid Entry
                self.position_ttl = max(1, int(ttl_fraction * self.max_ttl_steps))
                self.steps_held = 0
        
        # Decrement TTL
        # Always decrement if we have a position (even on entry step, to count it as 1 step used)
        if abs(new_position) < 1e-6:
            self.position_ttl = 0
            self.steps_held = 0
        else:
            # Holding (or just entered)
            if self.position_ttl > 0:
                self.position_ttl -= 1
            
            self.steps_held += 1
            
        # 5. Execute Trade & Cost
        trade_value = abs(delta) * price_now
        fee = 0.0
        slippage = 0.0
        
        if abs(delta) > 1e-6:
            # Fee
            trade_type = "buy" if delta > 0 else "sell"
            fee = self.fee_model.calculate_fee(trade_value, trade_type=trade_type)
            
            # Slippage
            # impact_proxy is (High-Low) [JPY/BTC] (Spread Proxy)
            # Cost = Impact * Delta [JPY]
            # Linear Slippage Model
            impact = self.impact_data[self.current_step]
            
            # Cap impact to avoid data outliers killing the episode (Max 1% slippage)
            max_impact = price_now * 0.01 
            impact = min(impact, max_impact)
            
            slippage = impact * abs(delta)
            
        self.last_step_cost = fee + slippage
        
        # Update Balance (Cash Flow)
        # Buying (delta > 0): balance decreases by cost + trade_value
        # Selling (delta < 0): balance increases by trade_value - cost
        # balance -= (delta * price) + cost
        self.balance -= (delta * price_now + self.last_step_cost)
        
        # 6. Reward
        # PnL from holding position from prev to now
        # Note: We use position BEFORE trade for PnL of this step
        old_position = self.position
        pnl = old_position * (price_now - price_prev)
        
        reward, reward_info = compute_hft_reward(
            price_prev=price_prev,
            price_now=price_now,
            position_prev=old_position,
            position_now=new_position,
            atr=atr,
            fee_paid=fee,
            slippage_paid=slippage,
            holding_steps=self.steps_held,
            max_position=self.max_position,
            **self.reward_params
        )
        
        # Update State
        self.position = new_position
        self.total_pnl += pnl - self.last_step_cost
        
        # Update Scaler with CURRENT features (t) BEFORE moving to next step
        # This prevents leakage (using t+1 stats to normalize t+1 obs)
        self.scaler.update(self.features_data[self.current_step])
        
        self.current_step += 1
        self.steps_in_episode += 1
        
        # Check Done
        terminated = False
        truncated = False
        
        if self.current_step >= self.data_len - 1:
            terminated = True
        
        if self.max_steps and self.steps_in_episode >= self.max_steps:
             truncated = True
             
        # Drawdown check
        current_equity = self.balance + self.position * price_now
        if current_equity > self.max_balance:
            self.max_balance = current_equity
        
        drawdown = (self.max_balance - current_equity) / self.max_balance
        if drawdown > self.drawdown_limit:
            terminated = True
            
        # Get Next Observation
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        info = {
            "balance": self.balance,
            "position": self.position,
            "ttl": self.position_ttl,
            "drawdown": drawdown,
            "step_cost": self.last_step_cost,
            **reward_info
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self) -> np.ndarray:
        # Market Features (Scaled)
        raw_feat = self.features_data[self.current_step]
        # OnlineScaler stores mean/var. We need to standardize.
        # (x - mean) / sqrt(var + eps)
        scaled_feat = (raw_feat - self.scaler.mean) / np.sqrt(self.scaler.var + self.scaler.epsilon)
        scaled_feat = np.clip(scaled_feat, -self.scaler.clip, self.scaler.clip)
        
        # Account State
        # [pos/max, ttl/max, cost/max_cost?]
        pos_norm = self.position / self.max_position
        # Clamp TTL to 0 for observation (handle -1 state)
        ttl_norm = max(0, self.position_ttl) / self.max_ttl_steps
        cost_norm = np.tanh(self.last_step_cost / 100.0) # Soft clip cost
        
        account_state = np.array([pos_norm, ttl_norm, cost_norm], dtype=np.float32)
        
        return np.concatenate([scaled_feat, account_state])

