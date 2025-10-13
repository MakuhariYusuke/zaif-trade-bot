"""
SAC v397g Backtest - Realistic Buy Size

設計:
- max_position_size: 0.01 (BTC価格500万円なら約5万円分)
- initial_balance: 200,000円 (0.04 BTC分)
- continuous_to_discrete_threshold: 0.10 (BUY/SELL判定敏感化)
- reward_scale: 1000.0 (PnL報酬強調)
- trade_execution_bonus_rate: 0.2
- inactivity_penalty_rate: 0.005
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.environ["MPLBACKEND"] = "Agg"

from stable_baselines3 import SAC
from ztb.trading.environment.heavy_env import HeavyTradingEnv

def main():
    print("=" * 80)
    print("SAC v397g Backtest - Realistic Buy Size")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    model_path = project_root / "checkpoints" / "sac_session" / "sac_model_final.zip"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    print(f"📦 Loading model: {model_path}")
    model = SAC.load(str(model_path))
    data_path = project_root / "btc_jpy_real_dataset.csv"
    df = pd.read_csv(data_path)
    max_steps = 5000
    if len(df) > max_steps:
        df = df.head(max_steps)
    config = {
        "initial_portfolio_value": 200000,
        "transaction_cost": 0.001,
        "max_position_size": 0.01,
        "enable_action_masking": False,
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.10,
        "reward_settings": {
            "use_simple_reward": True,
            "reward_scale": 1000.0,
            "reward_clip_min": -10.0,
            "reward_clip_max": 10.0,
            "enable_inactivity_penalty": True,
            "inactivity_penalty_rate": 0.005,
            "inactivity_penalty_window": 3,
            "inactivity_hold_threshold": 0.01,
            "enable_opportunity_cost": False,
            "enable_trade_execution_bonus": True,
            "trade_execution_bonus_rate": 0.2,
            "trade_execution_position_threshold": 0.005,
            "trade_execution_action_multiplier": 1.5
        }
    }
    env = HeavyTradingEnv(df=df, config=config, random_start=False)
    print(f"📊 Environment created:")
    print(f"  Data: {len(df)} rows")
    print(f"  Threshold: {config['continuous_to_discrete_threshold']}")
    print(f"  max_position_size: {config['max_position_size']}")
    print(f"  Reward Scale: {config['reward_settings']['reward_scale']}")
    print(f"  Trade Bonus: {config['reward_settings']['trade_execution_bonus_rate']}")
    print(f"  Inactivity Penalty: {config['reward_settings']['inactivity_penalty_rate']}")
    print()
    print(f"🚀 Starting backtest ({max_steps} steps)...")
    obs, _ = env.reset()
    done = False
    step = 0
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
        if info:
            pnls.append(info.get("pnl", 0.0))
            portfolio_values.append(info.get("portfolio_value", env.portfolio_value))
        if hasattr(env, 'position'):
            positions.append(env.position)
            if len(positions) > 1:
                position_changes.append(abs(positions[-1] - positions[-2]))
        step += 1
        if step % 1000 == 0:
            pv = portfolio_values[-1] if portfolio_values else 0
            pos = positions[-1] if positions else 0
            print(f"  Step {step}/{max_steps} - Portfolio: ¥{pv:,.0f}, Position: {pos:.4f}, Reward: {reward:.4f}")
    print(f"✅ Backtest completed: {step} steps")
    print()
    actions = np.array(actions)
    rewards = np.array(rewards)
    threshold = config["continuous_to_discrete_threshold"]
    buy_count = np.sum(actions > threshold)
    sell_count = np.sum(actions < -threshold)
    hold_count = len(actions) - buy_count - sell_count
    buy_pct = 100 * buy_count / len(actions)
    sell_pct = 100 * sell_count / len(actions)
    hold_pct = 100 * hold_count / len(actions)
    trade_count = buy_count + sell_count
    significant_changes = sum(1 for c in position_changes if c > 0.005)
    positive_rewards = np.sum(rewards > 0)
    negative_rewards = np.sum(rewards < 0)
    zero_rewards = np.sum(rewards == 0)
    pos_pct = 100 * positive_rewards / len(rewards)
    neg_pct = 100 * negative_rewards / len(rewards)
    zero_pct = 100 * zero_rewards / len(rewards)
    initial_value = portfolio_values[0] if portfolio_values else 200000
    final_value = portfolio_values[-1] if portfolio_values else 200000
    total_return = 100 * (final_value - initial_value) / initial_value
    results = {
        "model": "v397g_realistic_buy_size",
        "timestamp": datetime.now().isoformat(),
        "steps": step,
        "action_distribution": {
            "buy_count": int(buy_count),
            "buy_percentage": float(buy_pct),
            "sell_count": int(sell_count),
            "sell_percentage": float(sell_pct),
            "hold_count": int(hold_count),
            "hold_percentage": float(hold_pct),
            "total_trades": int(trade_count),
            "significant_changes": int(significant_changes)
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
            "realized_pnl": float(sum(pnls)) if pnls else 0.0
        },
        "position_stats": {
            "mean": float(np.mean(positions)) if positions else 0.0,
            "std": float(np.std(positions)) if positions else 0.0,
            "min": float(np.min(positions)) if positions else 0.0,
            "max": float(np.max(positions)) if positions else 0.0
        }
    }
    output_path = project_root / "docs" / "evaluation" / "backtest_sac_v397g_realistic_buy_size_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 80)
    print("📊 Backtest Results Summary")
    print("=" * 80)
    print()
    print("🎯 Action Distribution:")
    print(f"  BUY:  {buy_pct:5.1f}% ({buy_count:,} actions)")
    print(f"  HOLD: {hold_pct:5.1f}% ({hold_count:,} actions)")
    print(f"  SELL: {sell_pct:5.1f}% ({sell_count:,} actions)")
    print(f"  Total Trades: {trade_count:,}")
    print(f"  Significant Changes (>0.5%): {significant_changes:,}")
    print()
    print("💰 Performance:")
    print(f"  Initial Portfolio: ¥{initial_value:,.0f}")
    print(f"  Final Portfolio:   ¥{final_value:,.0f}")
    print(f"  Total Return:      {total_return:+.2f}%")
    print(f"  Realized PnL:      ¥{sum(pnls) if pnls else 0:,.0f}")
    print()
    print("🎁 Reward Statistics:")
    print(f"  Mean:     {np.mean(rewards):.4f}")
    print(f"  Std:      {np.std(rewards):.4f}")
    print(f"  Range:    [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
    print(f"  Positive: {pos_pct:5.2f}% ({positive_rewards:,} steps)")
    print(f"  Negative: {neg_pct:5.2f}% ({negative_rewards:,} steps)")
    print(f"  Zero:     {zero_pct:5.2f}% ({zero_rewards:,} steps)")
    print()
    if positions:
        print("📈 Position Statistics:")
        print(f"  Mean:  {np.mean(positions):.4f} BTC")
        print(f"  Range: [{np.min(positions):.4f}, {np.max(positions):.4f}] BTC")
        print()
    print("=" * 80)
    print(f"📝 Results saved to: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
