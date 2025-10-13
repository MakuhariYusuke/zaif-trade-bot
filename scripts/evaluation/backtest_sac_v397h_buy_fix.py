"""
SAC v397h Backtest - BUY Action Learning Fix

設計:
- max_position_size: 0.05 (BTC価格500万円なら約25万円分)
- initial_balance: 200,000円 (0.04 BTC分)
- continuous_to_discrete_threshold: 0.05 (BUY/SELL判定緩和)
- reward_scale: 1000.0 (PnL報酬強調)
- trade_execution_bonus_rate: 0.2
- buy_immediate_bonus_rate: 0.5 (BUY即時ボーナス)
- target_entropy: -2.0 (探索増加)
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
    print("SAC v397h Backtest - BUY Action Learning Fix")
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
        "max_position_size": 0.05,
        "enable_action_masking": False,
        "use_continuous_actions": True,
        "use_standardized_observations": True,
        "continuous_to_discrete_threshold": 0.05,
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
            "immediate_bonus_rate": 0.5,
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
    print(f"  Immediate Bonus: {config['reward_settings']['immediate_bonus_rate']}")
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
            print(f"  Step {step}/{max_steps} completed")
    print(f"✅ Backtest completed: {step} steps")
    print()
    # 分析
    actions = np.array(actions)
    rewards = np.array(rewards)
    pnls = np.array(pnls)
    portfolio_values = np.array(portfolio_values) if portfolio_values else np.array([200000] * len(actions))
    positions = np.array(positions) if positions else np.array([0.0] * len(actions))
    position_changes = np.array(position_changes) if position_changes else np.array([])
    # 行動分布分析
    threshold = config['continuous_to_discrete_threshold']
    buy_actions = np.sum(actions > threshold)
    sell_actions = np.sum(actions < -threshold)
    hold_actions = np.sum((actions >= -threshold) & (actions <= threshold))
    total_actions = len(actions)
    print("🎯 Action Distribution:")
    print(f"  BUY (> {threshold}): {buy_actions} actions ({buy_actions/total_actions*100:.1f}%)")
    print(f"  HOLD (-{threshold} to {threshold}): {hold_actions} actions ({hold_actions/total_actions*100:.1f}%)")
    print(f"  SELL (< -{threshold}): {sell_actions} actions ({sell_actions/total_actions*100:.1f}%)")
    print()
    # 収益分析
    initial_value = 200000
    final_value = portfolio_values[-1] if len(portfolio_values) > 0 else initial_value
    total_return = (final_value - initial_value) / initial_value * 100
    print("💰 Performance:")
    print(f"  Initial Portfolio: ¥{initial_value:,.0f}")
    print(f"  Final Portfolio: ¥{final_value:,.0f}")
    print(f"  Total Return: {total_return:.2f}%")
    if len(pnls) > 0:
        print(f"  Average PnL: ¥{np.mean(pnls):,.2f}")
        print(f"  PnL Std: ¥{np.std(pnls):,.2f}")
    print()
    # ポジション分析
    if len(positions) > 0:
        print("📊 Position Analysis:")
        print(f"  Final Position: {positions[-1]:.4f} BTC")
        print(f"  Max Position: {np.max(positions):.4f} BTC")
        print(f"  Min Position: {np.min(positions):.4f} BTC")
        print(f"  Position Changes: {len(position_changes)}")
        if len(position_changes) > 0:
            print(f"  Avg Position Change: {np.mean(position_changes):.4f} BTC")
    print()
    # 結果保存
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": str(model_path),
        "config": config,
        "backtest_steps": step,
        "action_distribution": {
            "buy_count": int(buy_actions),
            "buy_percent": float(buy_actions/total_actions*100),
            "hold_count": int(hold_actions),
            "hold_percent": float(hold_actions/total_actions*100),
            "sell_count": int(sell_actions),
            "sell_percent": float(sell_actions/total_actions*100)
        },
        "performance": {
            "initial_portfolio": float(initial_value),
            "final_portfolio": float(final_value),
            "total_return_percent": float(total_return),
            "avg_pnl": float(np.mean(pnls)) if len(pnls) > 0 else 0.0,
            "pnl_std": float(np.std(pnls)) if len(pnls) > 0 else 0.0
        },
        "position_stats": {
            "final_position": float(positions[-1]) if len(positions) > 0 else 0.0,
            "max_position": float(np.max(positions)) if len(positions) > 0 else 0.0,
            "min_position": float(np.min(positions)) if len(positions) > 0 else 0.0,
            "position_changes": int(len(position_changes)),
            "avg_position_change": float(np.mean(position_changes)) if len(position_changes) > 0 else 0.0
        },
        "raw_data": {
            "actions": actions.tolist(),
            "rewards": rewards.tolist(),
            "portfolio_values": portfolio_values.tolist() if len(portfolio_values) > 0 else [],
            "positions": positions.tolist() if len(positions) > 0 else [],
            "pnls": pnls.tolist() if len(pnls) > 0 else []
        }
    }
    output_path = project_root / "docs" / "evaluation" / "backtest_sac_v397h_buy_fix_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📝 Results saved to: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()