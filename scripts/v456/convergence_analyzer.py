#!/usr/bin/env python3
"""
Week 4: Convergence & Learning Analysis
訓練の収束特性と学習進度の詳細分析
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import deque
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from stable_baselines3 import SAC


def analyze_learning_dynamics(model_path: str, n_test_episodes: int = 100) -> dict:
    """学習ダイナミクスの分析"""
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'btc_jpy_1m_v454.csv'
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    env = FastIntradayEnvV456(
        data=df,
        initial_balance=100000,
        max_position=0.01,
        fee_rate=0.001,
        slippage_rate=0.0005,
        drawdown_limit=0.3,
    )
    
    try:
        model = SAC.load(model_path, env=env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return {}
    
    results = {
        "convergence_metrics": {},
        "policy_statistics": {},
        "environment_adaptation": {},
        "reward_trends": [],
    }
    
    # Sliding window analysis
    window_size = 10
    reward_window = deque(maxlen=window_size)
    length_window = deque(maxlen=window_size)
    action_windows = {"SELL": deque(maxlen=window_size), 
                      "HOLD": deque(maxlen=window_size),
                      "BUY": deque(maxlen=window_size)}
    
    print(f"Analyzing learning dynamics over {n_test_episodes} episodes...")
    print("-" * 80)
    
    all_rewards = []
    all_lengths = []
    all_pnls = []
    
    for ep in range(n_test_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        ep_length = 0
        ep_reward = 0.0
        initial_balance = env.balance
        action_counts = [0, 0, 0]
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1
            
            obs, reward, done, truncated, info = env.step(action)
            ep_length += 1
            ep_reward += reward
        
        final_balance = env.balance
        pnl = final_balance - initial_balance
        
        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)
        all_pnls.append(pnl)
        
        reward_window.append(ep_reward)
        length_window.append(ep_length)
        
        action_windows["SELL"].append(action_counts[0] / ep_length if ep_length > 0 else 0)
        action_windows["HOLD"].append(action_counts[1] / ep_length if ep_length > 0 else 0)
        action_windows["BUY"].append(action_counts[2] / ep_length if ep_length > 0 else 0)
        
        results["reward_trends"].append({
            "episode": ep,
            "reward": ep_reward,
            "length": ep_length,
            "rolling_reward_mean": np.mean(list(reward_window)),
            "rolling_length_mean": np.mean(list(length_window)),
            "pnl": pnl,
        })
        
        if (ep + 1) % 20 == 0:
            print(f"  Episodes {ep+1:3d}: Avg Reward={np.mean(list(reward_window)):.4f}, "
                  f"Avg Length={np.mean(list(length_window)):.1f}, "
                  f"HOLD%={np.mean(list(action_windows['HOLD'])):.1%}")
    
    # Convergence analysis
    rewards_arr = np.array(all_rewards)
    
    # Split into phases
    first_third = rewards_arr[:len(rewards_arr)//3]
    second_third = rewards_arr[len(rewards_arr)//3:2*len(rewards_arr)//3]
    last_third = rewards_arr[2*len(rewards_arr)//3:]
    
    results["convergence_metrics"] = {
        "phase1_mean": float(first_third.mean()),
        "phase2_mean": float(second_third.mean()),
        "phase3_mean": float(last_third.mean()),
        "phase1_std": float(first_third.std()),
        "phase2_std": float(second_third.std()),
        "phase3_std": float(last_third.std()),
        "trend": "improving" if last_third.mean() > first_third.mean() else "degrading",
        "improvement_pct": float((last_third.mean() - first_third.mean()) / (abs(first_third.mean()) + 1e-8) * 100),
    }
    
    # Policy statistics
    results["policy_statistics"] = {
        "reward_mean": float(rewards_arr.mean()),
        "reward_std": float(rewards_arr.std()),
        "reward_min": float(rewards_arr.min()),
        "reward_max": float(rewards_arr.max()),
        "reward_skewness": float(pd.Series(rewards_arr).skew()),
        "episode_length_mean": float(np.mean(all_lengths)),
        "episode_length_std": float(np.std(all_lengths)),
        "episode_length_min": float(np.min(all_lengths)),
        "episode_length_max": float(np.max(all_lengths)),
    }
    
    # Environment adaptation
    results["environment_adaptation"] = {
        "final_action_dist": {
            "SELL": float(np.mean(list(action_windows["SELL"]))),
            "HOLD": float(np.mean(list(action_windows["HOLD"]))),
            "BUY": float(np.mean(list(action_windows["BUY"]))),
        },
        "pnl_mean": float(np.mean(all_pnls)),
        "win_rate": float((np.array(all_pnls) > 0).sum() / len(all_pnls)),
    }
    
    return results


def print_convergence_report(results: dict):
    """収束レポート"""
    
    print("\n" + "=" * 80)
    print("CONVERGENCE & LEARNING ANALYSIS")
    print("=" * 80)
    
    conv = results.get("convergence_metrics", {})
    print("\n📈 Learning Phases (Reward)")
    print("-" * 80)
    print(f"  Phase 1 (Early):   {conv.get('phase1_mean', 0):>8.4f} ± {conv.get('phase1_std', 0):.4f}")
    print(f"  Phase 2 (Mid):     {conv.get('phase2_mean', 0):>8.4f} ± {conv.get('phase2_std', 0):.4f}")
    print(f"  Phase 3 (Late):    {conv.get('phase3_mean', 0):>8.4f} ± {conv.get('phase3_std', 0):.4f}")
    print(f"  Trend:             {conv.get('trend', 'unknown').upper()}")
    print(f"  Improvement:       {conv.get('improvement_pct', 0):+.1f}%")
    
    policy = results.get("policy_statistics", {})
    print("\n🎯 Policy Statistics")
    print("-" * 80)
    print(f"  Reward Mean:       {policy.get('reward_mean', 0):.4f}")
    print(f"  Reward Std:        {policy.get('reward_std', 0):.4f}")
    print(f"  Reward Range:      [{policy.get('reward_min', 0):.4f}, {policy.get('reward_max', 0):.4f}]")
    print(f"  Reward Skewness:   {policy.get('reward_skewness', 0):.2f}")
    
    print("\n  Episode Length Mean: {:.1f} steps".format(policy.get('episode_length_mean', 0)))
    print(f"  Episode Length Std:  {policy.get('episode_length_std', 0):.1f}")
    print(f"  Episode Length Range: [{policy.get('episode_length_min', 0)}, {policy.get('episode_length_max', 0)}]")
    
    adapt = results.get("environment_adaptation", {})
    print("\n🔄 Environment Adaptation")
    print("-" * 80)
    dist = adapt.get('final_action_dist', {})
    print(f"  SELL:  {dist.get('SELL', 0):.1%}")
    print(f"  HOLD:  {dist.get('HOLD', 0):.1%}")
    print(f"  BUY:   {dist.get('BUY', 0):.1%}")
    print(f"  PnL Mean:   {adapt.get('pnl_mean', 0):+.0f} JPY")
    print(f"  Win Rate:   {adapt.get('win_rate', 0):.1%}")
    
    print("\n" + "=" * 80)


def save_results(results: dict, output_file: str = "convergence_report.json"):
    """Save to JSON"""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    model_path = "models/week4_fixed/sac_model" if len(sys.argv) < 2 else sys.argv[1]
    
    print("=" * 80)
    print("Convergence & Learning Analysis")
    print("=" * 80)
    
    results = analyze_learning_dynamics(model_path, n_test_episodes=100)
    
    if results:
        print_convergence_report(results)
        save_results(results, "results/week4_convergence_analysis.json")
