#!/usr/bin/env python3
"""
Debug script to test if SELL actions are executable in the environment.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from ztb.trading.environment.environment import HeavyTradingEnv

def test_sell_action():
    """Test if SELL action can be executed properly."""

    # Load data
    data_path = Path(__file__).parent / "ml-dataset-enhanced.csv"
    df = pd.read_csv(data_path)

    # Create environment with simple reward
    env_config = {
        "reward_scaling": 6.0,
        "transaction_cost": 0.001,
        "max_position_size": 1.0,  # 安全なポジションサイズ
        "risk_free_rate": 0.02,
        "feature_set": "full",
        "initial_portfolio_value": 1000000.0,
        "curriculum_stage": "simple_portfolio",  # Use simple reward
        "exchange": "bitflyer",  # 手数料がかかる取引所でテスト
        "stop_loss_threshold": 0.05,  # 5%ストップロス
        "max_consecutive_trades": 5,  # 最大連続取引回数
        "min_holding_period": 3,  # 最小ホールド期間
        "volatility_trade_threshold": 0.02,  # ボラティリティ取引閾値
        "reward_settings": {
            "enable_forced_diversity": False,  # Disable forced diversity for this test
            "profit_bonus_multipliers": [1.0, 1.0, 1.0],
        },
    }

    env = HeavyTradingEnv(
        df=df,
        config=env_config,
        streaming_pipeline=None,
        stream_batch_size=1000,
        max_features=68
    )

    print("=== Testing SELL Action Execution ===")

    # Reset environment
    env.reset()
    print(f"Initial position: {env.position}")
    print(f"Initial portfolio value: {env.portfolio_value}")

    # Test HOLD action
    print("\n--- Testing HOLD action ---")
    obs, reward, terminated, truncated, info = env.step(0)  # HOLD
    print(f"After HOLD - Position: {env.position}, Reward: {reward}")

    # Test BUY action
    print("\n--- Testing BUY action ---")
    obs, reward, terminated, truncated, info = env.step(1)  # BUY
    print(f"After BUY - Position: {env.position}, Reward: {reward}")

    # Test SELL action (should work if we have a position)
    print("\n--- Testing SELL action ---")
    obs, reward, terminated, truncated, info = env.step(2)  # SELL
    print(f"After SELL - Position: {env.position}, Reward: {reward}")

    # Test SELL again (should work if flat)
    print("\n--- Testing SELL action again ---")
    obs, reward, terminated, truncated, info = env.step(2)  # SELL
    print(f"After SELL again - Position: {env.position}, Reward: {reward}")

    print("\n=== Reward Function Test ===")
    print("Expected rewards for simple_portfolio stage:")
    print("HOLD (0): -1.0")
    print("BUY (1): -0.5")
    print("SELL (2): 2.0")

    # Test reward calculation by simulating the step method
    print("\n--- Testing reward calculation ---")
    for action in [0, 1, 2]:
        # Simulate step to get reward
        env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action {action} reward: {reward}")

    print("\n=== Legal Actions Test ===")
    print("Testing get_legal_actions() for different positions:")

    # Test with flat position
    env.reset()
    legal = env.get_legal_actions()
    print(f"Flat position - Legal actions: HOLD={legal[0]}, BUY={legal[1]}, SELL={legal[2]}")

    # Test after BUY
    env.step(1)  # BUY
    legal = env.get_legal_actions()
    print(f"After BUY (position={env.position}) - Legal actions: HOLD={legal[0]}, BUY={legal[1]}, SELL={legal[2]}")

    # Test after SELL
    env.step(2)  # SELL
    legal = env.get_legal_actions()
    print(f"After SELL (position={env.position}) - Legal actions: HOLD={legal[0]}, BUY={legal[1]}, SELL={legal[2]}")

    # Test insufficient balance scenario
    print("\n--- Testing insufficient balance ---")
    env.reset()
    # Simulate very low balance
    env.total_pnl = -999999  # Almost no money left (portfolio_value ≈ 1)
    legal = env.get_legal_actions()
    print(f"Very low balance (portfolio_value≈{env.initial_portfolio_value + env.total_pnl}) - Legal actions: HOLD={legal[0]}, BUY={legal[1]}, SELL={legal[2]}")

    # Test with no balance at all
    env.total_pnl = -1000000  # No money left
    legal = env.get_legal_actions()
    print(f"No balance (portfolio_value≈{env.initial_portfolio_value + env.total_pnl}) - Legal actions: HOLD={legal[0]}, BUY={legal[1]}, SELL={legal[2]}")

    env.close()

if __name__ == "__main__":
    test_sell_action()