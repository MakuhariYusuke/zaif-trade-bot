"""
SAC v397e Backtest - Reward Redesign Validation

Evaluates the reward redesign approach:
- Threshold: 0.15 (BUY sensitivity increase)
- Reward scale: 100.0 (PnL reduction)
- Trade bonus: 0.5 (10x increase)
- Inactivity penalty: 0.02 (10x increase)
- Per-trade cost: 0.1 (new)

Expected improvements over v397d:
- BUY/SELL balance: 20-30% each
- Zero rewards: <50%
- Trade count: 100-500
- Positive rewards: >10%
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

os.environ["MPLBACKEND"] = "Agg"

from stable_baselines3 import SAC
from ztb.trading.environment.heavy_env import HeavyTradingEnv


def main():
    print("=" * 80)
    print("SAC v397e Backtest - Reward Redesign Validation")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load model
    model_path = project_root / "checkpoints" / "sac_session" / "sac_model_final.zip"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📦 Loading model: {model_path}")
    model = SAC.load(str(model_path))
    
    # Load data
    data_path = project_root / "btc_jpy_real_dataset.csv"
    df = pd.read_csv(data_path)
    max_steps = 5000
    if len(df) > max_steps:
        df = df.head(max_steps)
    
    # Create environment with v397e config
    config = {
        "initial_portfolio_value": 200000,
        "transaction_cost": 0.001,
        "max_position_size": 0.05,
        "enable_action_masking": False,
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.15,  # v397e: 0.15 (was 0.20)
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 100.0,  # v397e: 100.0 (was 1000.0)
            "reward_clip_min": -10.0,
            "reward_clip_max": 10.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.02,  # v397e: 0.02 (was 0.002)
            "inactivity_penalty_window": 3,
            "inactivity_hold_threshold": 0.01,
            "enable_opportunity_cost": False,
            "enable_trade_execution_bonus": True,
            "trade_execution_bonus_rate": 0.5,  # v397e: 0.5 (was 0.05)
            "trade_execution_position_threshold": 0.005,
            "trade_execution_action_multiplier": 1.5
        }
    }
    
    env = HeavyTradingEnv(df=df, config=config, random_start=False)
    
    print(f"📊 Environment created:")
    print(f"  Data: {len(df)} rows")
    print(f"  Threshold: {config['continuous_to_discrete_threshold']}")
    print(f"  Reward Scale: {config['reward_settings']['reward_scale']}")
    print(f"  Trade Bonus: {config['reward_settings']['trade_execution_bonus_rate']}")
    print(f"  Inactivity Penalty: {config['reward_settings']['inactivity_penalty_rate']}")
    print()
    
    # Run backtest
    print(f"🚀 Starting backtest ({max_steps} steps)...")
    
    obs, _ = env.reset()
    done = False
    step = 0
    
    # Tracking
    actions = []
    rewards = []
    portfolio_values = []
    positions = []
    pnls = []
    position_changes = []
    
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        actions.append(action[0])
        rewards.append(reward)
        portfolio_values.append(info.get("portfolio_value", 0))
        positions.append(info.get("position", 0))
        pnls.append(info.get("pnl", 0))
        
        step += 1
        
        if step % 1000 == 0:
            print(f"  Step {step}/{max_steps} - Portfolio: ¥{portfolio_values[-1]:,.0f}, "
                  f"Position: {positions[-1]:.4f}, Reward: {reward:.4f}")
    
    print(f"✅ Backtest completed: {step} steps")
    print()
    
    # Analysis
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    # Action distribution
    threshold = config["continuous_to_discrete_threshold"]
    buy_count = np.sum(actions > threshold)
    sell_count = np.sum(actions < -threshold)
    hold_count = len(actions) - buy_count - sell_count
    
    buy_pct = 100 * buy_count / len(actions)
    sell_pct = 100 * sell_count / len(actions)
    hold_pct = 100 * hold_count / len(actions)
    
    # Reward stats
    positive_rewards = np.sum(rewards > 0)
    negative_rewards = np.sum(rewards < 0)
    zero_rewards = np.sum(rewards == 0)
    
    pos_pct = 100 * positive_rewards / len(rewards)
    neg_pct = 100 * negative_rewards / len(rewards)
    zero_pct = 100 * zero_rewards / len(rewards)
    
    # Performance
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = 100 * (final_value - initial_value) / initial_value
    
    # Results
    results = {
        "model": "v397e_reward_redesign",
        "timestamp": datetime.now().isoformat(),
        "steps": step,
        "action_distribution": {
            "buy_count": int(buy_count),
            "buy_percentage": float(buy_pct),
            "sell_count": int(sell_count),
            "sell_percentage": float(sell_pct),
            "hold_count": int(hold_count),
            "hold_percentage": float(hold_pct)
        },
        "reward_stats": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "positive_count": int(positive_rewards),
            "positive_percentage": float(pos_pct),
            "negative_count": int(negative_rewards),
            "negative_percentage": float(neg_pct),
            "zero_count": int(zero_rewards),
            "zero_percentage": float(zero_pct)
        },
        "performance": {
            "initial_portfolio_value": float(initial_value),
            "final_portfolio_value": float(final_value),
            "total_return_percentage": float(total_return),
            "realized_pnl": float(sum(pnls))
        },
        "position_stats": {
            "mean": float(np.mean(positions)),
            "std": float(np.std(positions)),
            "min": float(np.min(positions)),
            "max": float(np.max(positions))
        }
    }
    
    # Save results
    output_path = project_root / "docs" / "evaluation" / "backtest_sac_v397e_reward_redesign_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Display results
    print("=" * 80)
    print("📊 Backtest Results Summary")
    print("=" * 80)
    print()
    print("🎯 Action Distribution:")
    print(f"  BUY:  {buy_pct:5.1f}% ({buy_count:,} actions)")
    print(f"  HOLD: {hold_pct:5.1f}% ({hold_count:,} actions)")
    print(f"  SELL: {sell_pct:5.1f}% ({sell_count:,} actions)")
    print()
    print("💰 Performance:")
    print(f"  Initial Portfolio: ¥{initial_value:,.0f}")
    print(f"  Final Portfolio:   ¥{final_value:,.0f}")
    print(f"  Total Return:      {total_return:+.2f}%")
    print(f"  Realized PnL:      ¥{sum(pnls):,.0f}")
    print()
    print("🎁 Reward Statistics:")
    print(f"  Mean:     {np.mean(rewards):.4f}")
    print(f"  Std:      {np.std(rewards):.4f}")
    print(f"  Range:    [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
    print(f"  Positive: {pos_pct:5.2f}% ({positive_rewards:,} steps)")
    print(f"  Negative: {neg_pct:5.2f}% ({negative_rewards:,} steps)")
    print(f"  Zero:     {zero_pct:5.2f}% ({zero_rewards:,} steps)")
    print()
    print("📈 Position Statistics:")
    print(f"  Mean:  {np.mean(positions):.4f} BTC")
    print(f"  Range: [{np.min(positions):.4f}, {np.max(positions):.4f}] BTC")
    print()
    print("=" * 80)
    print("🎯 Target Achievement Check:")
    print("=" * 80)
    
    # Check targets
    targets_met = 0
    total_targets = 4
    
    print(f"1. BUY/SELL Balance (20-30% each):")
    if 20 <= buy_pct <= 30 and 20 <= sell_pct <= 30:
        print(f"   ✅ PASS - BUY {buy_pct:.1f}%, SELL {sell_pct:.1f}%")
        targets_met += 1
    else:
        print(f"   ❌ FAIL - BUY {buy_pct:.1f}%, SELL {sell_pct:.1f}%")
    
    print(f"2. Zero Rewards (<50%):")
    if zero_pct < 50:
        print(f"   ✅ PASS - {zero_pct:.1f}%")
        targets_met += 1
    else:
        print(f"   ❌ FAIL - {zero_pct:.1f}%")
    
    print(f"3. Trade Count (100-500):")
    trade_count = buy_count + sell_count
    if 100 <= trade_count <= 500:
        print(f"   ✅ PASS - {trade_count:,} trades")
        targets_met += 1
    else:
        print(f"   ❌ FAIL - {trade_count:,} trades")
    
    print(f"4. Positive Rewards (>10%):")
    if pos_pct > 10:
        print(f"   ✅ PASS - {pos_pct:.2f}%")
        targets_met += 1
    else:
        print(f"   ❌ FAIL - {pos_pct:.2f}%")
    
    print()
    print(f"Overall: {targets_met}/{total_targets} targets met")
    print("=" * 80)
    print()
    print(f"📝 Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
