#!/usr/bin/env python3
"""
Week 4: Aggressive Reward Function Optimization
試す複数の係数組み合わせによるスイープ
- alpha (churn penalty): HOLDへのバイアスを排除
- beta (holding time penalty): 長い保持への抑制
- gamma (inventory risk): 大きなポジション保有への抑制
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# 親ディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

RESULTS_DIR = Path("results/week4_reward_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 試験する係数組み合わせ
PARAMETER_SETS = [
    # Default (baseline)
    {"alpha": 0.2, "beta": 0.01, "gamma": 0.5, "desc": "baseline"},
    
    # Alpha variation: churn penaltyを緩和 (HOLDバイアス排除)
    {"alpha": 0.05, "beta": 0.01, "gamma": 0.5, "desc": "alpha_low"},
    {"alpha": 0.10, "beta": 0.01, "gamma": 0.5, "desc": "alpha_mid"},
    
    # Beta variation: holding time penaltyを調整
    {"alpha": 0.2, "beta": 0.001, "gamma": 0.5, "desc": "beta_low"},
    {"alpha": 0.2, "beta": 0.005, "gamma": 0.5, "desc": "beta_mid"},
    
    # Gamma variation: inventory risk調整
    {"alpha": 0.2, "beta": 0.01, "gamma": 0.1, "desc": "gamma_low"},
    {"alpha": 0.2, "beta": 0.01, "gamma": 1.0, "desc": "gamma_high"},
    
    # Combined: alpha低 + beta低
    {"alpha": 0.05, "beta": 0.001, "gamma": 0.5, "desc": "combined_aggressive"},
    
    # Combined: alpha非常に低い (チャーンペナルティ最小)
    {"alpha": 0.01, "beta": 0.01, "gamma": 0.5, "desc": "alpha_minimal"},
    {"alpha": 0.01, "beta": 0.001, "gamma": 0.5, "desc": "alpha_minimal_beta_low"},
]


def run_quick_test(params: dict, test_episodes: int = 5) -> dict:
    """
    Quick test with given parameters.
    Run a short training and measure action distribution + episode length.
    """
    test_script = f"""
import sys
sys.path.insert(0, '.')
import numpy as np
from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.utils.data_manager import DataManager

# Create environment
dm = DataManager()
df = dm.load_cached_or_fetch(force_refresh=False)
env = FastIntradayEnvV456(
    data=df,
    initial_balance=100000,
    max_position=0.01,
    fee_rate=0.001,
    slippage_rate=0.0005,
    drawdown_limit=0.3,
    alpha={params['alpha']},
    beta={params['beta']},
    gamma={params['gamma']},
)

# Simple policy: random actions
results = {{"episode_lengths": [], "avg_reward": 0.0, "action_dist": [0, 0, 0]}}
total_reward = 0.0
total_steps = 0

for ep in range({test_episodes}):
    obs, info = env.reset()
    done = False
    truncated = False
    ep_reward = 0.0
    ep_length = 0
    action_counts = [0, 0, 0]
    
    while not (done or truncated):
        action = np.random.randint(0, 3)  # 0=SELL, 1=HOLD, 2=BUY
        action_counts[action] += 1
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        ep_length += 1
        total_steps += 1
    
    results["episode_lengths"].append(ep_length)
    total_reward += ep_reward
    results["action_dist"][0] += action_counts[0]
    results["action_dist"][1] += action_counts[1]
    results["action_dist"][2] += action_counts[2]

results["avg_reward"] = total_reward / {test_episodes} if {test_episodes} > 0 else 0.0
results["avg_episode_length"] = sum(results["episode_lengths"]) / len(results["episode_lengths"])
results["total_steps"] = total_steps

# Normalize action distribution
if total_steps > 0:
    results["action_dist"] = [x / total_steps for x in results["action_dist"]]

import json
print(json.dumps(results))
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return {
                "status": "FAILED",
                "error": result.stderr,
                "params": params
            }
        
        # Extract JSON from output
        output_lines = result.stdout.strip().split('\n')
        json_line = output_lines[-1] if output_lines else "{}"
        
        test_result = json.loads(json_line)
        test_result["params"] = params
        test_result["status"] = "OK"
        
        return test_result
        
    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT",
            "params": params,
            "error": "Test took too long"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "params": params,
            "error": str(e)
        }


def main():
    print("=" * 80)
    print("Week 4: Aggressive Reward Parameter Sweep")
    print("=" * 80)
    print()
    
    start_time = datetime.now()
    all_results = []
    
    for i, params in enumerate(PARAMETER_SETS, 1):
        desc = params.pop("desc")
        print(f"[{i}/{len(PARAMETER_SETS)}] Testing {desc}...")
        print(f"  α={params['alpha']}, β={params['beta']}, γ={params['gamma']}")
        
        result = run_quick_test(params, test_episodes=3)
        result["desc"] = desc
        all_results.append(result)
        
        if result["status"] == "OK":
            print(f"  ✓ Avg Episode Length: {result.get('avg_episode_length', 'N/A'):.1f}")
            print(f"  ✓ Avg Reward: {result.get('avg_reward', 'N/A'):.4f}")
            print(f"  ✓ Action Dist: SELL={result.get('action_dist', [0,0,0])[0]:.2%}, "
                  f"HOLD={result.get('action_dist', [0,0,0])[1]:.2%}, "
                  f"BUY={result.get('action_dist', [0,0,0])[2]:.2%}")
        else:
            print(f"  ✗ {result['status']}: {result.get('error', 'Unknown error')[:100]}")
        
        print()
    
    # Save results
    results_file = RESULTS_DIR / f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 80)
    print(f"Results saved to: {results_file}")
    print("=" * 80)
    
    # Summary table
    print("\nSummary of Successful Tests:")
    print("-" * 80)
    print(f"{'Desc':<25} {'α':<6} {'β':<8} {'γ':<6} {'Avg Len':<10} {'Avg Reward':<12} {'%HOLD':<8}")
    print("-" * 80)
    
    successful = [r for r in all_results if r.get("status") == "OK"]
    for r in successful:
        hold_pct = r.get("action_dist", [0, 0, 0])[1] * 100
        print(f"{r['desc']:<25} {r['params']['alpha']:<6.2f} {r['params']['beta']:<8.4f} "
              f"{r['params']['gamma']:<6.1f} {r.get('avg_episode_length', 0):<10.1f} "
              f"{r.get('avg_reward', 0):<12.4f} {hold_pct:<8.1f}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if successful:
        # Best by episode length
        best_length = max(successful, key=lambda r: r.get("avg_episode_length", 0))
        print(f"\n✓ Longest Episodes: {best_length['desc']} ({best_length.get('avg_episode_length', 0):.1f} steps)")
        
        # Best by reward
        best_reward = max(successful, key=lambda r: r.get("avg_reward", -float('inf')))
        print(f"✓ Best Reward: {best_reward['desc']} ({best_reward.get('avg_reward', 0):.4f})")
        
        # Lowest HOLD %
        best_hold = min(successful, key=lambda r: r.get("action_dist", [0, 1, 0])[1])
        hold_pct = best_hold.get("action_dist", [0, 1, 0])[1] * 100
        print(f"✓ Lowest HOLD Action: {best_hold['desc']} ({hold_pct:.1f}% HOLD)")


if __name__ == "__main__":
    main()
