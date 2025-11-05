#!/usr/bin/env python3
"""
Deep investigation: Is environment ACTUALLY using SAC continuous actions?
"""

import json
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import environment components
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.constants import ACTION_BUY, ACTION_SELL, ACTION_HOLD
import pandas as pd

def create_sample_data(periods=100):
    """Create minimal test data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=periods, freq="1h")
    base_price = 5000000
    price_changes = np.random.normal(0, 0.005, periods).cumsum()
    close = pd.Series(base_price * (1 + price_changes), index=dates)
    
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * (1 + np.abs(np.random.normal(0, 0.002, periods))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.002, periods))),
        "close": close,
        "volume": pd.Series(np.random.uniform(1000, 10000, periods), index=dates),
        "timestamp": dates,
    })
    
    # Add indicators
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["RSI"] = 50
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["BB_Upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["BB_Lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
    
    return df.ffill().bfill()

def test_environment_initialization():
    """Test 1: Environment initialization."""
    print("="*80)
    print("TEST 1: Environment Initialization & Action Space")
    print("="*80)
    
    config_path = Path("config/sac_v444_3_balanced_penalty_scale_200.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Prepare environment config
    env_config = config_dict['environment'].copy()
    env_config.update(env_config['behavior_optimization'])
    env_config.update(env_config['action_bonuses'])
    
    # Add curriculum_stage
    curriculum_config = config_dict['training']['curriculum_learning']
    env_config['curriculum_stage'] = curriculum_config['curriculum_stage']
    
    print(f"\n✓ use_continuous_actions: {env_config.get('use_continuous_actions')}")
    print(f"✓ use_standardized_observations: {env_config.get('use_standardized_observations')}")
    print(f"✓ curriculum_stage: {env_config.get('curriculum_stage')}")
    
    # Create environment
    df = create_sample_data()
    env = HeavyTradingEnv(df, env_config)
    
    print(f"\n✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Action space type: {type(env.action_space)}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Config use_continuous_actions: {env.config.use_continuous_actions}")
    
    # Test some steps
    print(f"\n✓ Stepping through environment:")
    obs, info = env.reset()
    
    for step in range(5):
        # Sample continuous action from SAC
        action = env.action_space.sample()
        print(f"\n  Step {step+1}:")
        print(f"    Continuous action: {action} (type: {type(action).__name__})")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"    Reward: {reward}")
        print(f"    Position: {info.get('position', 'N/A')}")
        
        # Check action tracking in RewardCalculator
        if hasattr(env, '_reward_calculator') and hasattr(env._reward_calculator, '_recent_actions'):
            recent = list(env._reward_calculator._recent_actions)
            if recent:
                last_action = recent[-1]
                print(f"    Recent actions tail: {recent[-3:]}")
                print(f"    Last discrete action: {last_action} ({['HOLD', 'BUY', None, None, None, None, None, None, 'SELL'].get(last_action, 'UNKNOWN')})")
        
        if terminated or truncated:
            print(f"    Episode ended")
            break

def test_action_conversion():
    """Test 2: Continuous-to-discrete action conversion."""
    print("\n" + "="*80)
    print("TEST 2: Continuous-to-Discrete Action Conversion")
    print("="*80)
    
    config_path = Path("config/sac_v444_3_balanced_penalty_scale_200.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    
    env_config = config_dict['environment'].copy()
    env_config.update(env_config['behavior_optimization'])
    env_config.update(env_config['action_bonuses'])
    curriculum_config = config_dict['training']['curriculum_learning']
    env_config['curriculum_stage'] = curriculum_config['curriculum_stage']
    
    df = create_sample_data()
    env = HeavyTradingEnv(df, env_config)
    
    # Get threshold from config
    threshold = env.config.continuous_to_discrete_threshold
    threshold_neg = env.config.continuous_to_discrete_threshold_neg
    
    print(f"\nConversion thresholds:")
    print(f"  Positive threshold: {threshold} (BUY if action > {threshold})")
    print(f"  Negative threshold: {threshold_neg} (SELL if action < {threshold_neg})")
    print(f"  HOLD: {threshold_neg} <= action <= {threshold}")
    
    # Test various continuous actions
    test_actions = [
        -1.0,      # Extreme SELL
        -0.5,      # Medium SELL
        -0.08,     # At negative threshold
        -0.07,     # Should be HOLD
        0.0,       # Center
        0.07,      # Should be HOLD
        0.08,      # At positive threshold
        0.5,       # Medium BUY
        1.0,       # Extreme BUY
    ]
    
    print(f"\nTesting continuous action conversion:")
    obs, _ = env.reset()
    
    for cont_action in test_actions:
        # Manually check what the environment does
        # The environment should convert continuous to discrete internally
        print(f"\n  Continuous: {cont_action:6.3f}", end=" → ")
        
        # Manually calculate expected discrete action based on thresholds
        if cont_action > threshold:
            expected = 1  # BUY
            expected_name = "BUY"
        elif cont_action < threshold_neg:
            expected = -1  # SELL
            expected_name = "SELL"
        else:
            expected = 0  # HOLD
            expected_name = "HOLD"
        
        print(f"Expected: {expected_name}", end="")
        
        # Take step and check what was actually used
        obs, reward, _, _, info = env.step(cont_action)
        
        if hasattr(env, '_reward_calculator') and hasattr(env._reward_calculator, '_recent_actions'):
            recent = list(env._reward_calculator._recent_actions)
            if recent:
                actual_discrete = recent[-1]
                actual_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}
                actual_name = actual_names.get(actual_discrete, "UNKNOWN")
                
                match = "✓" if actual_discrete == expected else "✗ MISMATCH"
                print(f", Actual: {actual_name} {match}")

def main():
    """Run all tests."""
    print("\n" + "█"*80)
    print("█ DEEP INVESTIGATION: Environment Action Processing")
    print("█"*80 + "\n")
    
    try:
        test_environment_initialization()
        test_action_conversion()
        
        print("\n" + "="*80)
        print("Investigation complete.")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
