#!/usr/bin/env python3
"""
Week 4: Episode Length Analysis
現在のEnvironmentでのエピソード長の分布を分析
"""

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.utils.data_manager import DataManager

def analyze_episodes(n_episodes: int = 20) -> dict:
    """分析"""
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
    
    results = {
        "episode_lengths": [],
        "max_balances": [],
        "min_balances": [],
        "pnls": [],
        "action_sequences": [],
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_length = 0
        initial_balance = env.balance
        actions = []
        
        while not (done or truncated):
            # Random policy for analysis
            action = np.random.randint(0, 3)  # 0=SELL, 1=HOLD, 2=BUY
            actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            ep_length += 1
        
        results["episode_lengths"].append(ep_length)
        results["max_balances"].append(env.max_balance)
        results["min_balances"].append(env.balance)
        results["pnls"].append(env.total_pnl)
        results["action_sequences"].append(actions)
    
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("Episode Length Distribution Analysis")
    print("=" * 80)
    
    results = analyze_episodes(n_episodes=20)
    
    ep_lens = np.array(results["episode_lengths"])
    print(f"\nEpisode Lengths (n={len(ep_lens)}):")
    print(f"  Mean: {ep_lens.mean():.2f} steps")
    print(f"  Std:  {ep_lens.std():.2f}")
    print(f"  Min:  {ep_lens.min()}")
    print(f"  Max:  {ep_lens.max()}")
    print(f"  Median: {np.median(ep_lens):.1f}")
    
    print(f"\nBalance Analysis:")
    max_bals = np.array(results["max_balances"])
    min_bals = np.array(results["min_balances"])
    pnls = np.array(results["pnls"])
    
    print(f"  Max Balance (mean): {max_bals.mean():.0f} JPY")
    print(f"  Final Balance (mean): {min_bals.mean():.0f} JPY")
    print(f"  PnL (mean): {pnls.mean():.0f} JPY")
    
    print(f"\nAction Distribution Analysis:")
    for ep, actions in enumerate(results["action_sequences"][:5]):
        sell_cnt = actions.count(0)
        hold_cnt = actions.count(1)
        buy_cnt = actions.count(2)
        total = len(actions)
        print(f"  Episode {ep+1}: SELL={sell_cnt/total:.1%}, HOLD={hold_cnt/total:.1%}, BUY={buy_cnt/total:.1%}")
    
    # Overall
    all_actions = [a for actions in results["action_sequences"] for a in actions]
    sell_cnt = all_actions.count(0)
    hold_cnt = all_actions.count(1)
    buy_cnt = all_actions.count(2)
    total = len(all_actions)
    print(f"  Overall: SELL={sell_cnt/total:.1%}, HOLD={hold_cnt/total:.1%}, BUY={buy_cnt/total:.1%}")
    
    print("\n" + "=" * 80)
    if ep_lens.mean() > 10:
        print("✓ Episode lengths are reasonable (>10 steps)")
    else:
        print("⚠ Episode lengths are still short (<10 steps)")
    
    if hold_cnt/total < 0.9:
        print("✓ Action diversity is good (<90% HOLD)")
    else:
        print("⚠ Action diversity is poor (>90% HOLD)")
