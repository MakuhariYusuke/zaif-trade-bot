"""
Phase 3: Execution Realism Comparison.

This script compares the performance of a v450 SAC model under:
1. Ideal Execution (No slippage, instant execution)
2. Realistic Execution (ATR-based slippage, latency, fees)

It trains a model on the 'Ideal' environment (standard practice) and then
evaluates it on both environments to quantify the 'Realism Gap'.
"""

import os
import sys
from pathlib import Path

# Pre-import torch to avoid DLL loading issues on Windows
try:
    import torch
except ImportError:
    pass


import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def run_comparison():
    print("========================================================")
    print("🚀 Phase 3: Execution Realism Comparison (v450)")
    print("========================================================")

    # 1. Setup Configurations
    # -----------------------
    data_path = os.path.join(project_root, "data", "range_medium_featured.csv")

    # Base Config (Ideal)
    base_env_config = {
        "initial_portfolio_value": 100000.0,
        "transaction_cost": 0.000,
        "slippage": 0.0,
        "execution_model": None,  # Ideal
        "adaptive_threshold_mode": True,
        "threshold_volatility_multiplier": 1.0,
        "use_continuous_actions": True,
        "action_space_type": "continuous",
        "feature_set": "full",
    }

    # Realistic Config Overrides (applied to environment section)
    realistic_env_overrides = {
        "execution_model": {
            "base_slippage": 0.0005,  # 0.05%
            "atr_slippage_factor": 0.5,
            "base_latency_ms": 50.0,
            "latency_jitter_ms": 20.0,
        },
        "transaction_cost": 0.001,  # 0.1% fee
    }

    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    # Helper to extract config
    def create_env_config(base_env_config, overrides=None):
        config_dict = base_env_config.copy()
        if overrides:
            config_dict.update(overrides)

        # Filter for EnvironmentConfig fields
        env_config_params = {
            "transaction_cost": config_dict.get("transaction_cost", 0.0),
            "slippage": config_dict.get("slippage", 0.0),
            "execution_model": config_dict.get("execution_model", None),
            "feature_set": config_dict.get("feature_set", "full"),
            "reward_scaling": 1.0,
            "use_continuous_actions": True,
            "action_space_type": "continuous",
        }
        return EnvironmentConfig(**env_config_params)

    # Helper for env kwargs
    def get_env_kwargs(base_env_config):
        return {
            "initial_balance": base_env_config.get("initial_portfolio_value", 100000.0),
            "use_continuous_actions": base_env_config.get(
                "use_continuous_actions", True
            ),
            "action_space_type": base_env_config.get("action_space_type", "continuous"),
        }

    # 2. Train Model on Ideal Environment
    # -----------------------------------
    print("\n[Step 1] Training model on Ideal Environment (Direct SAC)...")

    # Create Training Env
    train_env_config = create_env_config(base_env_config)
    train_env = HeavyTradingEnv(
        config=train_env_config, df=df, **get_env_kwargs(base_env_config)
    )

    # Train Model
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_starts=100,
        buffer_size=10000,
        batch_size=256,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
    )
    model.learn(total_timesteps=5000)

    # Save model
    model_path = os.path.join(project_root, "models", "sac_model_direct.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Load the trained model
    print(f"\n[Step 2] Loading trained model from {model_path}...")
    model = SAC.load(model_path)
    print(f"DEBUG: Model Observation Space: {model.observation_space}")

    # 3. Evaluate on Ideal Environment
    # --------------------------------
    print("\n[Step 3] Evaluating on Ideal Environment...")

    # Load data manually for env
    df = pd.read_csv(data_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    # Ideal Env
    ideal_env_config = create_env_config(base_env_config)
    ideal_env = HeavyTradingEnv(
        config=ideal_env_config, df=df, **get_env_kwargs(base_env_config)
    )

    # DEBUG: Check observation space and reset
    print(f"DEBUG: Observation Space: {ideal_env.observation_space}")
    obs, _ = ideal_env.reset()
    print(f"DEBUG: Reset Observation Shape: {obs.shape}")
    print(f"DEBUG: Reset Observation Type: {type(obs)}")

    # DEBUG: Check step observation
    action = ideal_env.action_space.sample()
    obs, reward, terminated, truncated, info = ideal_env.step(action)
    print(f"DEBUG: Step Observation Shape: {obs.shape}")

    # Reset again before evaluation
    ideal_env.reset()

    mean_reward_ideal, std_reward_ideal = evaluate_policy(
        model, ideal_env, n_eval_episodes=3
    )
    print(f"Ideal Reward: {mean_reward_ideal:.4f} +/- {std_reward_ideal:.4f}")

    # Get detailed stats from the last episode of ideal env
    ideal_stats = ideal_env.get_statistics()
    ideal_pnl = ideal_stats.get("total_pnl", 0.0)
    ideal_trades = ideal_stats.get("total_trades", 0)

    # 4. Evaluate on Realistic Environment
    # ------------------------------------
    print("\n[Step 4] Evaluating on Realistic Environment...")

    # Realistic Env
    realistic_env_config = create_env_config(
        base_env_config, overrides=realistic_env_overrides
    )
    realistic_env = HeavyTradingEnv(
        config=realistic_env_config, df=df, **get_env_kwargs(base_env_config)
    )

    mean_reward_real, std_reward_real = evaluate_policy(
        model, realistic_env, n_eval_episodes=3
    )
    print(f"Realistic Reward: {mean_reward_real:.4f} +/- {std_reward_real:.4f}")

    # Get detailed stats
    real_stats = realistic_env.get_statistics()
    real_pnl = real_stats.get("total_pnl", 0.0)
    real_trades = real_stats.get("total_trades", 0)

    # 5. Report Results
    # -----------------
    print("\n========================================================")
    print("📊 COMPARISON RESULTS")
    print("========================================================")
    print(f"{'Metric':<20} | {'Ideal':<15} | {'Realistic':<15} | {'Gap':<15}")
    print("-" * 75)
    print(
        f"{'Mean Reward':<20} | {mean_reward_ideal:<15.4f} | {mean_reward_real:<15.4f} | {mean_reward_real - mean_reward_ideal:<15.4f}"
    )
    print(
        f"{'Total PnL':<20} | {ideal_pnl:<15.2f} | {real_pnl:<15.2f} | {real_pnl - ideal_pnl:<15.2f}"
    )
    print(
        f"{'Total Trades':<20} | {ideal_trades:<15} | {real_trades:<15} | {real_trades - ideal_trades:<15}"
    )
    print("========================================================")

    # Save report
    report_path = os.path.join(
        project_root, "docs", "v450", "reports", "execution_comparison_results.md"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# v450 Phase 3: Execution Realism Comparison\n\n")
        f.write(f"**Date:** {pd.Timestamp.now()}\n\n")
        f.write("## Overview\n")
        f.write(
            "Comparison of model performance between Ideal (Zero Slippage/Latency) and Realistic (ATR Slippage + Latency) environments.\n\n"
        )
        f.write("## Results\n\n")
        f.write("| Metric | Ideal | Realistic | Gap |\n")
        f.write("|---|---|---|---|\n")
        f.write(
            f"| Mean Reward | {mean_reward_ideal:.4f} | {mean_reward_real:.4f} | {mean_reward_real - mean_reward_ideal:.4f} |\n"
        )
        f.write(
            f"| Total PnL | {ideal_pnl:.2f} | {real_pnl:.2f} | {real_pnl - ideal_pnl:.2f} |\n"
        )
        f.write(
            f"| Total Trades | {ideal_trades} | {real_trades} | {real_trades - ideal_trades} |\n"
        )
        f.write("\n## Analysis\n")
        f.write(f"- **Realism Gap (PnL):** {real_pnl - ideal_pnl:.2f}\n")
        f.write(
            "- **Impact:** The realistic execution model simulates slippage based on volatility. A negative gap indicates the strategy is sensitive to execution costs.\n"
        )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_comparison()
