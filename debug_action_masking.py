#!/usr/bin/env python3
"""
Debug Action Masking

Check what actions are being masked during evaluation.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.environment import HeavyTradingEnv


def debug_action_masking(model_path: str, data_path: str, num_steps: int = 100):
    """Debug action masking behavior."""
    print(f"Loading model from {model_path}")
    
    # Try loading as MaskablePPO first
    try:
        model = MaskablePPO.load(model_path)
        model_type = "MaskablePPO"
    except Exception:
        model = PPO.load(model_path)
        model_type = "PPO"
    
    print(f"Model type: {model_type}")
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data: {len(df)} rows, {len(df.columns)} columns")
    
    # Create environment with DEBUG
    config = {
        "transaction_cost": 0.0005,
        "max_position_size": 0.5,
        "curriculum_stage": "forced_balance",
        "exchange": "coincheck",  # No trade frequency limits
    }
    
    env = HeavyTradingEnv(df=df, config=config, random_start=False)
    
    # Collect mask statistics
    mask_stats = {
        "hold_masked": 0,
        "buy_masked": 0,
        "sell_masked": 0,
        "all_allowed": 0,
    }
    
    action_predictions = []
    action_taken = []
    
    obs, _ = env.reset()
    
    for step in range(num_steps):
        # Get action mask
        legal_actions = env.get_legal_actions()
        
        # Track masking
        if legal_actions[0] == 0:
            mask_stats["hold_masked"] += 1
        if legal_actions[1] == 0:
            mask_stats["buy_masked"] += 1
        if legal_actions[2] == 0:
            mask_stats["sell_masked"] += 1
        if all(legal_actions == 1):
            mask_stats["all_allowed"] += 1
        
        # Print detailed info for first 10 steps
        if step < 10:
            print(f"\nStep {step}:")
            print(f"  Position: {env.position}")
            print(f"  Portfolio value: {env.initial_portfolio_value + env.total_pnl:.2f}")
            print(f"  Legal actions: HOLD={legal_actions[0]}, BUY={legal_actions[1]}, SELL={legal_actions[2]}")
            
            # Check volatility
            if hasattr(env, "df") and env.current_step > 20:
                recent_prices = env.df.iloc[max(0, env.current_step - 20):env.current_step]["close"]
                if len(recent_prices) > 1:
                    returns = recent_prices.pct_change().dropna()
                    current_volatility = returns.std()
                    print(f"  Volatility: {current_volatility:.4f}")
        
        # Predict action
        if model_type == "MaskablePPO":
            action, _ = model.predict(obs, action_masks=legal_actions, deterministic=False)
        else:
            action, _ = model.predict(obs, deterministic=False)
        
        action_int = int(action.item() if hasattr(action, "item") else action)
        action_predictions.append(action_int)
        
        # Check if predicted action is legal
        if legal_actions[action_int] == 0:
            print(f"  ⚠️  WARNING: Model predicted ILLEGAL action {action_int}!")
            print(f"     Legal actions: {legal_actions}")
        
        # Take action
        obs, reward, terminated, truncated, _ = env.step(action_int)
        action_taken.append(action_int)
        
        if terminated or truncated:
            break
    
    # Summary
    print("\n" + "=" * 60)
    print("MASKING STATISTICS")
    print("=" * 60)
    print(f"Total steps: {num_steps}")
    print(f"HOLD masked:  {mask_stats['hold_masked']:4d} ({mask_stats['hold_masked']/num_steps*100:5.1f}%)")
    print(f"BUY masked:   {mask_stats['buy_masked']:4d} ({mask_stats['buy_masked']/num_steps*100:5.1f}%)")
    print(f"SELL masked:  {mask_stats['sell_masked']:4d} ({mask_stats['sell_masked']/num_steps*100:5.1f}%)")
    print(f"All allowed:  {mask_stats['all_allowed']:4d} ({mask_stats['all_allowed']/num_steps*100:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("ACTION DISTRIBUTION")
    print("=" * 60)
    
    pred_counts = Counter(action_predictions)
    taken_counts = Counter(action_taken)
    
    print("Predicted actions (by model):")
    print(f"  HOLD:  {pred_counts.get(0, 0):4d} ({pred_counts.get(0, 0)/len(action_predictions)*100:5.1f}%)")
    print(f"  BUY:   {pred_counts.get(1, 0):4d} ({pred_counts.get(1, 0)/len(action_predictions)*100:5.1f}%)")
    print(f"  SELL:  {pred_counts.get(2, 0):4d} ({pred_counts.get(2, 0)/len(action_predictions)*100:5.1f}%)")
    
    print("\nActual actions taken:")
    print(f"  HOLD:  {taken_counts.get(0, 0):4d} ({taken_counts.get(0, 0)/len(action_taken)*100:5.1f}%)")
    print(f"  BUY:   {taken_counts.get(1, 0):4d} ({taken_counts.get(1, 0)/len(action_taken)*100:5.1f}%)")
    print(f"  SELL:  {taken_counts.get(2, 0):4d} ({taken_counts.get(2, 0)/len(action_taken)*100:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug action masking")
    parser.add_argument(
        "--model-path",
        default="models/ppo_memory_optimized.zip",
        help="Path to model",
    )
    parser.add_argument(
        "--data-path",
        default="ml-dataset-enhanced.csv",
        help="Path to data",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps to run",
    )
    
    args = parser.parse_args()
    debug_action_masking(args.model_path, args.data_path, args.steps)


if __name__ == "__main__":
    main()
