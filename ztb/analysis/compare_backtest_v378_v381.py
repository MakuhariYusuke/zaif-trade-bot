#!/usr/bin/env python3
"""
Compare v378 vs v381 models using backtest
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
from ztb.utils.data_utils import load_csv_data_optimized
from sb3_contrib import MaskablePPO
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.policies.policy_utils import predict_with_masks


def run_backtest_for_model(
    model_path: str,
    model_name: str,
    data_path: str = "ml-dataset-enhanced.csv",
    episodes: int = 10
) -> Dict[str, Any]:
    """Run backtest for a single model"""
    
    print(f"\n{'='*60}")
    print(f"Running backtest for: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    # Load data
    df = load_csv_data_optimized(data_path)
    print(f"Loaded {len(df)} rows of data")
    
    # Create environment with NO feature filtering (same as training)
    # Training used curated_features but still had 110 dimensions
    config = {
        "transaction_cost": 0.0005,
        "max_position_size": 0.5,
        "enable_correlation_reduction": False,
        "enable_feature_filtering": False,  # Disable filtering to keep all 110 features
    }
    
    env = HeavyTradingEnv(df=df, config=config)
    
    # Load model
    try:
        model = MaskablePPO.load(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Run episodes
    total_reward = 0.0
    total_pnl = 0.0
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    episode_rewards = []
    episode_pnls = []
    episode_steps = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        episode_reward = 0.0
        episode_pnl = 0.0
        
        while not (done or truncated):
            action, _states = predict_with_masks(model, obs, env, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()
            
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_pnl += info.get('pnl', 0.0)
            
            # Count actions
            if action == 0:
                action_counts["HOLD"] += 1
            elif action == 1:
                action_counts["BUY"] += 1
            else:
                action_counts["SELL"] += 1
            
            steps += 1
            
            # Safety limit
            if steps > 10000:
                print(f"  ⚠️  Episode {episode + 1} exceeded 10000 steps, terminating")
                break
        
        total_reward += episode_reward
        total_pnl += episode_pnl
        episode_rewards.append(episode_reward)
        episode_pnls.append(episode_pnl)
        episode_steps.append(steps)
        
        print(f"  Episode {episode + 1:2d}: Reward={episode_reward:8.2f}, PnL={episode_pnl:10.6f}, Steps={steps:4d}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_pnl = np.mean(episode_pnls)
    std_pnl = np.std(episode_pnls)
    avg_steps = np.mean(episode_steps)
    
    total_actions = sum(action_counts.values())
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "episodes": episodes,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_pnl": avg_pnl,
        "std_pnl": std_pnl,
        "total_pnl": sum(episode_pnls),
        "avg_steps": avg_steps,
        "action_counts": action_counts,
        "action_distribution": {
            "HOLD": (action_counts["HOLD"] / total_actions * 100) if total_actions > 0 else 0,
            "BUY": (action_counts["BUY"] / total_actions * 100) if total_actions > 0 else 0,
            "SELL": (action_counts["SELL"] / total_actions * 100) if total_actions > 0 else 0,
        },
        "episode_rewards": episode_rewards,
        "episode_pnls": episode_pnls,
    }
    
    # Print summary
    print(f"\n📊 Results for {model_name}:")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average PnL: {avg_pnl:.6f} ± {std_pnl:.6f}")
    print(f"  Total PnL: {sum(episode_pnls):.6f}")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"\n  Action Distribution:")
    for action, pct in results["action_distribution"].items():
        count = action_counts[action]
        print(f"    {action}: {count:5d} ({pct:5.1f}%)")
    
    return results


def compare_models(results1: Dict[str, Any], results2: Dict[str, Any]) -> None:
    """Compare two model results"""
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    model1_name = results1["model_name"]
    model2_name = results2["model_name"]
    
    # Reward comparison
    print(f"\n📊 Average Reward:")
    print(f"  {model1_name:20s}: {results1['avg_reward']:8.2f} ± {results1['std_reward']:.2f}")
    print(f"  {model2_name:20s}: {results2['avg_reward']:8.2f} ± {results2['std_reward']:.2f}")
    reward_diff = results2['avg_reward'] - results1['avg_reward']
    reward_pct = (reward_diff / abs(results1['avg_reward']) * 100) if results1['avg_reward'] != 0 else 0
    print(f"  Difference: {reward_diff:+.2f} ({reward_pct:+.1f}%)")
    
    # PnL comparison
    print(f"\n💰 Average PnL:")
    print(f"  {model1_name:20s}: {results1['avg_pnl']:10.6f} ± {results1['std_pnl']:.6f}")
    print(f"  {model2_name:20s}: {results2['avg_pnl']:10.6f} ± {results2['std_pnl']:.6f}")
    pnl_diff = results2['avg_pnl'] - results1['avg_pnl']
    print(f"  Difference: {pnl_diff:+.6f}")
    
    # Total PnL comparison
    print(f"\n💰 Total PnL ({results1['episodes']} episodes):")
    print(f"  {model1_name:20s}: {results1['total_pnl']:10.6f}")
    print(f"  {model2_name:20s}: {results2['total_pnl']:10.6f}")
    total_diff = results2['total_pnl'] - results1['total_pnl']
    print(f"  Difference: {total_diff:+.6f}")
    
    # Action distribution comparison
    print(f"\n🎯 Action Distribution:")
    print(f"  {'Action':<10s} {model1_name:>15s} {model2_name:>15s} {'Difference':>12s}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*12}")
    for action in ["HOLD", "BUY", "SELL"]:
        pct1 = results1['action_distribution'][action]
        pct2 = results2['action_distribution'][action]
        diff = pct2 - pct1
        print(f"  {action:<10s} {pct1:14.1f}% {pct2:14.1f}% {diff:+11.1f}%")
    
    # Winner determination
    print(f"\n🏆 WINNER:")
    
    criteria_scores = {model1_name: 0, model2_name: 0}
    
    # Criterion 1: Higher average reward
    if results2['avg_reward'] > results1['avg_reward']:
        criteria_scores[model2_name] += 1
        print(f"  ✅ Avg Reward: {model2_name}")
    else:
        criteria_scores[model1_name] += 1
        print(f"  ✅ Avg Reward: {model1_name}")
    
    # Criterion 2: Higher total PnL
    if results2['total_pnl'] > results1['total_pnl']:
        criteria_scores[model2_name] += 1
        print(f"  ✅ Total PnL: {model2_name}")
    else:
        criteria_scores[model1_name] += 1
        print(f"  ✅ Total PnL: {model1_name}")
    
    # Criterion 3: Lower HOLD percentage
    if results2['action_distribution']['HOLD'] < results1['action_distribution']['HOLD']:
        criteria_scores[model2_name] += 1
        print(f"  ✅ Active Trading (Lower HOLD): {model2_name}")
    else:
        criteria_scores[model1_name] += 1
        print(f"  ✅ Active Trading (Lower HOLD): {model1_name}")
    
    # Criterion 4: Lower variance (more stable)
    if results2['std_reward'] < results1['std_reward']:
        criteria_scores[model2_name] += 1
        print(f"  ✅ Stability (Lower Std): {model2_name}")
    else:
        criteria_scores[model1_name] += 1
        print(f"  ✅ Stability (Lower Std): {model1_name}")
    
    print(f"\n  Final Score:")
    print(f"    {model1_name}: {criteria_scores[model1_name]}/4")
    print(f"    {model2_name}: {criteria_scores[model2_name]}/4")
    
    if criteria_scores[model2_name] > criteria_scores[model1_name]:
        print(f"\n  🎉 Overall Winner: {model2_name}")
    elif criteria_scores[model1_name] > criteria_scores[model2_name]:
        print(f"\n  🎉 Overall Winner: {model1_name}")
    else:
        print(f"\n  🤝 Tie!")


def main() -> None:
    """Main comparison function"""
    
    models = [
        {
            "name": "v378_scale",
            "path": "models/ppo_reward_v378_scale.zip",
        },
        {
            "name": "v381_revised",
            "path": "models/ppo_reward_v381_revised_profit_focused.zip",
        }
    ]
    
    data_path = "ml-dataset-enhanced.csv"
    episodes = 10
    
    print(f"\n{'#'*60}")
    print("# v378 vs v381 Backtest Comparison")
    print(f"{'#'*60}")
    print(f"Data: {data_path}")
    print(f"Episodes per model: {episodes}")
    
    results_list = []
    
    for model_info in models:
        result = run_backtest_for_model(
            model_path=model_info["path"],
            model_name=model_info["name"],
            data_path=data_path,
            episodes=episodes
        )
        if result:
            results_list.append(result)
        else:
            print(f"❌ Failed to run backtest for {model_info['name']}")
            return
    
    # Compare results
    if len(results_list) == 2:
        compare_models(results_list[0], results_list[1])
        
        # Save results to JSON
        output_file = "backtest_v378_v381_comparison.json"
        with open(output_file, 'w') as f:
            json.dump({
                "models": results_list,
                "data_path": data_path,
                "episodes": episodes,
            }, f, indent=2, default=str)
        print(f"\n✅ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
