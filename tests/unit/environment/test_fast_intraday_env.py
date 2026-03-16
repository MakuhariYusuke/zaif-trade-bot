"""
Unit tests for FastIntradayEnv (Pytest version)
"""

import pytest
import numpy as np
import pandas as pd
from ztb.trading.environment.fast_intraday_env import FastIntradayEnv

@pytest.fixture
def env_setup():
    # Create dummy data
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
    df = pd.DataFrame({
        "open": np.linspace(100, 110, 1000),
        "high": np.linspace(101, 111, 1000),
        "low": np.linspace(99, 109, 1000),
        "close": np.linspace(100, 110, 1000),
        "volume": np.random.rand(1000) * 100,
        "atr": np.ones(1000) * 1.0,
        "impact_proxy": np.ones(1000) * 0.1,
        "clv": np.zeros(1000),
        "vol_pressure": np.zeros(1000),
        "vol_regime": np.ones(1000) * 0.01,
        "trend_persistence": np.zeros(1000)
    }, index=dates)
    
    feature_columns = ["clv", "vol_pressure", "impact_proxy", "vol_regime", "trend_persistence"]
    
    env = FastIntradayEnv(
        df=df,
        feature_columns=feature_columns,
        max_steps=100,
        prewarm_steps=10,
        max_ttl_steps=10,
        cooldown_steps=2
    )
    return env, feature_columns

def test_initialization(env_setup):
    env, feature_columns = env_setup
    obs, info = env.reset(seed=42)
    assert obs.shape[0] == len(feature_columns) + 3
    assert env.position == 0.0
    assert env.balance == 1_000_000.0

def test_action_mapping_and_ttl(env_setup):
    env, _ = env_setup
    env.reset(seed=42)
    
    # Action: Target 0.5, TTL 0.5 (5 steps)
    action = np.array([0.5, 0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check position (should be clipped by max_delta_per_step=0.2)
    # 0 -> 0.5. Delta 0.5. Max delta 0.2. New pos 0.2.
    assert env.position == pytest.approx(0.2)
    # Check TTL (should be set to 5)
    # Logic: set to 5. Decrement -> 4.
    assert env.position_ttl == 4
    
    # Step 2: Hold target 0.5
    obs, reward, terminated, truncated, info = env.step(action)
    # 0.2 -> 0.5. Delta 0.3. Max delta 0.2. New pos 0.4.
    assert env.position == pytest.approx(0.4)
    # Decrement 4 -> 3
    assert env.position_ttl == 3

def test_ttl_expiration(env_setup):
    env, _ = env_setup
    env.reset(seed=42)
    # Set TTL to 2 steps
    action = np.array([0.5, 0.2], dtype=np.float32) # TTL 0.2 * 10 = 2
    
    env.step(action) # Step 1: TTL set to 2. Decr -> 1. Pos 0.2.
    env.step(action) # Step 2: TTL 1 -> 0. Pos 0.4.
    
    # Step 3: TTL 0 -> Expired. Target forced to 0.
    env.step(action)
    
    # Logic: if position_ttl <= 0: target=0.
    # Step 3: decr -> 0. Expired. target=0. cooldown=2.
    # Delta = 0 - 0.4 = -0.4. Max delta 0.2.
    # New Pos = 0.4 - 0.2 = 0.2.
    
    assert env.position == pytest.approx(0.2)
    # TTL is -1 to indicate "Expired and Unwinding"
    assert env.position_ttl == -1
    
    # Step 4: Continue unwinding
    env.step(action)
    assert env.position == 0.0
    # Fully unwound -> TTL reset to 0
    assert env.position_ttl == 0

def test_deadband_no_ttl_reset(env_fix_setup):
    env = env_fix_setup
    env.reset(seed=42)
    
    # 1. Enter Position (Target 1.0, TTL 5)
    action = np.array([1.0, 0.5], dtype=np.float32) # 0.5 * 60 = 30
    env.step(action)
    assert env.position == 1.0
    # TTL set to 30. Decremented -> 29.
    assert env.position_ttl == 29
    
    # 2. Small change (Target 0.995) -> Delta -0.005.
    # Deadband is 0.01 * 1.0 = 0.01.
    # Delta < Deadband -> Delta = 0.
    # New Position = 1.0.
    # Should NOT be treated as Entry/Reversal.
    # TTL should decrement (29 -> 28).
    
    action = np.array([0.995, 0.5], dtype=np.float32)
    env.step(action)
    
    assert env.position == 1.0
    assert env.position_ttl == 28 # Decremented, not reset to 30
    
    # 3. Large change (Target -1.0) -> Reversal
    action = np.array([-1.0, 0.5], dtype=np.float32)
    env.step(action)
    
    assert env.position == -1.0
    # Reset to 30. Decremented -> 29.
    assert env.position_ttl == 29

def test_reproducibility(env_setup):
    env, _ = env_setup
    obs1, _ = env.reset(seed=123)
    actions = [env.action_space.sample() for _ in range(10)]
    rewards1 = []
    for act in actions:
        _, r, _, _, _ = env.step(act)
        rewards1.append(r)
        
    obs2, _ = env.reset(seed=123)
    rewards2 = []
    for act in actions:
        _, r, _, _, _ = env.step(act)
        rewards2.append(r)
        
    np.testing.assert_array_almost_equal(obs1, obs2)
    np.testing.assert_array_almost_equal(rewards1, rewards2)

# --- New Tests for Bug Fixes ---

@pytest.fixture
def env_fix_setup():
    # Create dummy data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
    # Constant price 100 to simplify PnL checks initially
    df = pd.DataFrame({
        "open": np.ones(100) * 100.0,
        "high": np.ones(100) * 100.0,
        "low": np.ones(100) * 100.0,
        "close": np.ones(100) * 100.0,
        "volume": np.ones(100) * 1000,
        "atr": np.ones(100) * 1.0,
        "impact_proxy": np.ones(100) * 1.0, # Impact 1.0
        "clv": np.zeros(100),
        "vol_pressure": np.zeros(100),
        "vol_regime": np.zeros(100),
        "trend_persistence": np.zeros(100)
    }, index=dates)
    
    feature_columns = ["clv"]
    
    env = FastIntradayEnv(
        df=df,
        feature_columns=feature_columns,
        initial_balance=10000.0,
        max_position=1.0,
        commission_rate=0.0, # Zero fee to isolate slippage/balance logic first
        max_delta_per_step=2.0, # Allow full reversal (1 to -1) in one step
        prewarm_steps=10
    )
    return env

def test_balance_update_logic(env_fix_setup):
    env = env_fix_setup
    env.reset(seed=42)
    
    # Force price to be 100 at current step
    current_step = env.current_step
    env.close_prices[current_step] = 100.0
    env.close_prices[current_step+1] = 100.0 # Next price also 100
    
    # Action: Buy 1.0 (Full position)
    # Cost: Fee=0. Slippage = impact(1.0) * abs(Delta(1.0)) = 1.0
    # Trade Value = 100.
    # Balance should decrease by 100 + 1.0 = 101.0
    
    action = np.array([1.0, 1.0], dtype=np.float32)
    env.step(action)
    
    expected_balance = 10000.0 - 100.0 - 1.0
    assert env.balance == pytest.approx(expected_balance, abs=1e-4)
    assert env.position == 1.0
    
    # Step 2: Sell 1.0 (Close position)
    # Price still 100.
    # Trade Value = 100.
    # Slippage = 1.0 * abs(1.0) = 1.0.
    # Balance should increase by 100 - 1.0 = 99.0.
    # Total Balance = 10000 - 101.0 + 99.0 = 10000 - 2.0.
    
    action = np.array([0.0, 1.0], dtype=np.float32) # Target 0
    env.step(action)
    
    expected_final_balance = 10000.0 - 2.0
    assert env.balance == pytest.approx(expected_final_balance, abs=1e-4)
    assert env.position == 0.0

def test_holding_steps_counter(env_fix_setup):
    env = env_fix_setup
    env.reset(seed=42)
    
    assert env.steps_held == 0
    
    # Enter
    action = np.array([1.0, 1.0], dtype=np.float32)
    env.step(action)
    assert env.steps_held == 1 # Held for 1 step (the entry step)
    
    # Hold
    env.step(action)
    assert env.steps_held == 2
    
    # Reverse
    action = np.array([-1.0, 1.0], dtype=np.float32)
    env.step(action)
    # Should reset to 0 (or 1 if we count the new entry step as held)
    # Logic: is_reversal -> steps_held = 0. Then update holding steps -> if abs(pos)>0 -> +=1.
    # So it should be 1.
    assert env.steps_held == 1
    assert env.position == -1.0
    
    # Exit
    action = np.array([0.0, 1.0], dtype=np.float32)
    env.step(action)
    # Logic: target 0. new_pos 0. abs(pos) not > 1e-6. steps_held = 0.
    assert env.steps_held == 0

def test_slippage_calculation(env_fix_setup):
    env = env_fix_setup
    env.reset(seed=42)
    
    # Set specific values
    current_step = env.current_step
    price = 200.0
    atr = 2.0
    impact = 0.5
    
    env.close_prices[current_step] = price
    env.atr_data[current_step] = atr
    env.impact_data[current_step] = impact
    # Also set for next step since we step twice
    env.impact_data[current_step + 1] = impact
    
    # Buy 1.0
    # Slippage = impact(0.5) * abs(Delta(1.0)) = 0.5
    
    action = np.array([1.0, 1.0], dtype=np.float32)
    env.step(action)
    
    assert env.last_step_cost == pytest.approx(0.5, abs=1e-4)
    
    # Step 2: Partial Sell (Delta = -0.5)
    # Current Pos = 1.0. Target = 0.5. Delta = -0.5.
    # Slippage = impact(0.5) * abs(-0.5) = 0.25
    # (If quadratic, it would be 0.5 * 0.25 = 0.125)
    
    action = np.array([0.5, 1.0], dtype=np.float32)
    env.step(action)
    
    assert env.last_step_cost == pytest.approx(0.25, abs=1e-4)

def test_ttl_zero_logic(env_fix_setup):
    env = env_fix_setup
    env.reset(seed=42)
    
    # Action: Target 1.0, TTL 0.0
    # Should result in NO entry (target forced to 0)
    action = np.array([1.0, 0.0], dtype=np.float32)
    env.step(action)
    
    assert env.position == 0.0
    assert env.position_ttl == 0
    
    # Action: Target 1.0, TTL very small (e.g. 0.02 -> 1.2 steps -> round to 1)
    # max_ttl = 60 (default). 0.02 * 60 = 1.2 -> 1.
    action = np.array([1.0, 0.02], dtype=np.float32) 
    env.step(action)
    
    # Should enter and hold for this step (TTL=1, decremented to 0 immediately)
    assert env.position == 1.0
    assert env.position_ttl == 0
    
    # Next step: TTL decrements to 0 -> Expire -> Target 0 -> Exit
    # We pass same action, but TTL logic should override target because expired
    # Wait, if we pass same action, is_entry/is_reversal is False (pos=1, target=1).
    # So we go to Decrement TTL.
    # TTL 1 -> 0. Expired. Target -> 0.
    env.step(action)
    
    assert env.position == 0.0
    assert env.position_ttl == 0
    # Cooldown set to 5, then decremented to 4 at end of step
    assert env.cooldown_counter == env.cooldown_steps - 1
    
    # Next step: Cooldown should decrement to 3 (NOT reset to 4)
    env.step(action)
    assert env.cooldown_counter == env.cooldown_steps - 2

def test_fee_trade_type(env_fix_setup):
    env = env_fix_setup
    env.reset(seed=42)
    
    # Mock fee model to check calls
    from unittest.mock import MagicMock
    env.fee_model = MagicMock()
    env.fee_model.calculate_fee.return_value = 0.0
    
    # Buy
    action = np.array([1.0, 1.0], dtype=np.float32)
    env.step(action)
    env.fee_model.calculate_fee.assert_called_with(pytest.approx(100.0), trade_type="buy")
    
    # Sell
    action = np.array([0.0, 1.0], dtype=np.float32)
    env.step(action)
    env.fee_model.calculate_fee.assert_called_with(pytest.approx(100.0), trade_type="sell")

