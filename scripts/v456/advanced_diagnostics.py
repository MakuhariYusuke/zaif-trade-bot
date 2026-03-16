#!/usr/bin/env python3
"""
Week 4: Advanced Diagnostics
環境の詳細な信号解析
- Reward distribution
- Balance trajectory
- Position management
- Fee impact
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.utils.data_manager import DataManager


def run_detailed_analysis(episodes: int = 5):
    """詳細な環境診断"""
    dm = DataManager()
    df = dm.load_cached_or_fetch(force_refresh=False)
    
    env = FastIntradayEnvV456(
        data=df,
        initial_balance=100000,
        max_position=0.01,
        fee_rate=0.001,
        slippage_rate=0.0005,
        drawdown_limit=0.3,
    )
    
    print("=" * 80)
    print("Advanced Environment Diagnostics")
    print("=" * 80)
    print()
    
    all_rewards = []
    all_fees = []
    all_balances = []
    episode_data = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        ep_rewards = []
        ep_fees = []
        ep_balances = [env.balance]
        
        step = 0
        while not (done or truncated) and step < 100:  # Cap at 100 steps for analysis
            action = np.random.randint(0, 3)
            obs, reward, done, truncated, info = env.step(action)
            
            ep_rewards.append(reward)
            all_rewards.append(reward)
            
            # Fee estimation (indirect)
            # We can estimate from balance change
            if step > 0:
                balance_change = env.balance - ep_balances[-1]
                # Negative change usually means fee
                if balance_change < 0:
                    all_fees.append(-balance_change)
                    ep_fees.append(-balance_change)
            
            ep_balances.append(env.balance)
            all_balances.append(env.balance)
            
            step += 1
        
        episode_data.append({
            "episode": ep,
            "length": len(ep_rewards),
            "mean_reward": np.mean(ep_rewards) if ep_rewards else 0,
            "std_reward": np.std(ep_rewards) if ep_rewards else 0,
            "min_reward": np.min(ep_rewards) if ep_rewards else 0,
            "max_reward": np.max(ep_rewards) if ep_rewards else 0,
            "total_fee": sum(ep_fees),
            "initial_balance": 100000,
            "final_balance": env.balance,
            "total_pnl": env.total_pnl,
        })
    
    # Analysis
    print("Reward Analysis:")
    print("-" * 80)
    all_rewards_arr = np.array(all_rewards)
    print(f"  Total samples: {len(all_rewards_arr)}")
    print(f"  Mean: {all_rewards_arr.mean():.4f}")
    print(f"  Std:  {all_rewards_arr.std():.4f}")
    print(f"  Min:  {all_rewards_arr.min():.4f}")
    print(f"  Max:  {all_rewards_arr.max():.4f}")
    print(f"  Median: {np.median(all_rewards_arr):.4f}")
    print(f"  Negative %: {(all_rewards_arr < 0).sum() / len(all_rewards_arr) * 100:.1f}%")
    print(f"  Positive %: {(all_rewards_arr > 0).sum() / len(all_rewards_arr) * 100:.1f}%")
    
    print()
    print("Fee Analysis:")
    print("-" * 80)
    if all_fees:
        all_fees_arr = np.array(all_fees)
        print(f"  Transaction count: {len(all_fees_arr)}")
        print(f"  Mean fee: {all_fees_arr.mean():.2f} JPY")
        print(f"  Max fee: {all_fees_arr.max():.2f} JPY")
        print(f"  Total fees: {all_fees_arr.sum():.2f} JPY")
    else:
        print(f"  No fees recorded (check fee detection logic)")
    
    print()
    print("Episode Summary:")
    print("-" * 80)
    print(f"{'Ep':<4} {'Len':<6} {'AvgR':<10} {'StdR':<10} {'MinR':<10} {'MaxR':<10} {'Fee':<10} {'PnL':<10} {'Balance':<12}")
    print("-" * 80)
    for data in episode_data:
        print(f"{data['episode']:<4} {data['length']:<6} "
              f"{data['mean_reward']:<10.4f} {data['std_reward']:<10.4f} "
              f"{data['min_reward']:<10.4f} {data['max_reward']:<10.4f} "
              f"{data['total_fee']:<10.2f} {data['total_pnl']:<10.2f} "
              f"{data['final_balance']:<12.2f}")
    
    print()
    print("=" * 80)
    print("INSIGHTS:")
    print("=" * 80)
    
    # Key findings
    mean_reward = all_rewards_arr.mean()
    if mean_reward < -0.2:
        print("⚠  Average reward is very negative (< -0.2)")
        print("    → Adjust reward function coefficients (especially alpha, beta, gamma)")
    elif mean_reward < 0:
        print("⚠  Average reward is negative")
        print("    → Fine-tune reward scaling or initial conditions")
    else:
        print("✓ Average reward is positive - good sign!")
    
    neg_pct = (all_rewards_arr < 0).sum() / len(all_rewards_arr) * 100
    if neg_pct > 80:
        print(f"⚠  {neg_pct:.0f}% of rewards are negative")
        print("    → Consider reward function redesign")
    
    mean_ep_len = np.mean([d['length'] for d in episode_data])
    if mean_ep_len < 10:
        print(f"⚠  Average episode length is short ({mean_ep_len:.1f} steps)")
        print("    → Adjust drawdown_limit or reward scaling")
    else:
        print(f"✓ Episode length is reasonable ({mean_ep_len:.1f} steps)")
    
    if all_fees:
        mean_fee = np.mean(all_fees)
        if mean_fee > 10000:
            print(f"⚠  Average fee is high ({mean_fee:.0f} JPY)")
            print("    → Check max_position and transaction size limits")
        else:
            print(f"✓ Fee levels are reasonable ({mean_fee:.0f} JPY per transaction)")


if __name__ == "__main__":
    run_detailed_analysis(episodes=5)
