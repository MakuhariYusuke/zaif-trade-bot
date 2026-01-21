#!/usr/bin/env python3
"""
v457 Backtest Script
Loads a trained SAC model and evaluates it on a specified dataset.
Uses EnvironmentFactory for consistent feature engineering.
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from stable_baselines3 import SAC

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(workspace_root))

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)
from ztb.training.utils.v457_config_utils import extract_env_config, load_config_dict, extract_seed
from ztb.utils.seed_manager import set_global_seed
from utils.results_utils import save_backtest_results

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("v457_backtest")

def bucket_ttl_action(ttl_value: float) -> str:
    ttl_value = max(0.0, min(ttl_value, 1.0))
    bucket_index = min(int(ttl_value * 5), 4)
    low = bucket_index * 0.2
    high = low + 0.2
    return f"{low:.1f}-{high:.1f}"

def main():
    parser = argparse.ArgumentParser(description="v457 Backtest")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data csv")
    parser.add_argument("--config", type=str, default="config/v457/base/config.yaml", help="Path to config yaml")
    parser.add_argument("--output", type=str, default="backtest_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("v457 Backtest")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data:  {args.data}")
    logger.info("=" * 60)

    # 1. Load Data
    data_path = Path(workspace_root / args.data)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col=0)
    logger.info(f"Loaded {len(df)} rows.")

    # Calculate Base Features
    logger.info("Calculating features...")
    df = calculate_base_features(df, copy=False)

    # 2. Config
    config_file = Path(workspace_root / args.config)
    env_config_dict = {}
    seed = args.seed
    if config_file.exists():
        try:
            full_config = load_config_dict(config_file)
            env_config_dict = extract_env_config(full_config)
            if seed is None:
                seed = extract_seed(full_config)
        except Exception as e:
            logger.warning(f"Failed to load config {config_file}: {e}. Using defaults.")
    if seed is not None:
        set_global_seed(seed)

    # 3. Environment
    env = create_fast_intraday_env_v456(df=df, env_config=env_config_dict)
    if env is None:
        logger.error("Failed to create environment.")
        sys.exit(1)
    del df
    
    # 4. Load Model
    logger.info("Loading SAC model...")
    model = SAC.load(args.model, env=env)
    
    # 5. Run Backtest Loop
    logger.info("Running backtest loop...")
    if seed is not None:
        obs, info = env.reset(seed=seed)
        logger.info(f"Env reset: start_index={info.get('start_index')}")
    else:
        obs, info = env.reset()
    done = False
    
    portfolio_history = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # FastIntradayEnv exposes balance/pnl in info or attributes?
        # Standard env attributes usually:
        # env.balance, env.position, env.total_pnl
        # But FastIntradayEnvV456 might be different. 
        # Checking factory_v456.py -> creates FastIntradayEnvV456
        # Let's check environment attributes from the instance
        
        step_price = info.get("current_price")
        if step_price is None:
            if env.current_step < len(env.close_prices):
                step_price = float(env.close_prices[env.current_step])
            else:
                step_price = float("nan")

        portfolio_value = info.get("portfolio_value", env.balance)
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        action_value = float(action_array[0]) if action_array.size > 0 else 0.0
        action_ttl = float(action_array[1]) if action_array.size > 1 else None

        step_data = {
            "step": env.current_step,
            "price": step_price,
            "position": env.position,
            "balance": portfolio_value,
            "gross_pnl": getattr(env, "gross_pnl", env.total_pnl),
            "net_pnl": getattr(env, "net_pnl", env.total_pnl),
            "reward": reward,
            "action": action_value,
            "action_ttl": action_ttl,
            "ttl_forced_exits": info.get("ttl_forced_exits"),
            "cooldown_triggers": info.get("cooldown_triggers"),
        }
        portfolio_history.append(step_data)

    # 6. Analysis
    res_df = pd.DataFrame(portfolio_history)
    total_pnl = getattr(env, "net_pnl", env.total_pnl)
    gross_pnl = getattr(env, "gross_pnl", env.total_pnl)
    final_balance = env.balance
    trades = (res_df["position"].diff() != 0).sum()
    
    logger.info("-" * 40)
    logger.info(f"Final Balance: {final_balance:.2f}")
    logger.info(f"Net PnL:       {total_pnl:.2f}")
    logger.info(f"Gross PnL:     {gross_pnl:.2f}")
    logger.info(f"Trade Actions: {trades}")
    logger.info("-" * 40)
    
    # Save Results
    out_dir = Path(workspace_root / args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "backtest_log.csv")
    portfolio_values = res_df["balance"].tolist() if "balance" in res_df.columns else []
    action_distribution = {}
    if "action" in res_df.columns:
        for action in res_df["action"]:
            action_key = "hold"
            try:
                action_val = float(action)
                if action_val > 0.3:
                    action_key = "buy"
                elif action_val < -0.3:
                    action_key = "sell"
            except (TypeError, ValueError):
                action_key = "hold"
            action_distribution[action_key] = action_distribution.get(action_key, 0) + 1

    ttl_action_distribution = {}
    ttl_action_values = []
    if "action_ttl" in res_df.columns:
        for ttl_val in res_df["action_ttl"].dropna():
            try:
                ttl_float = float(ttl_val)
            except (TypeError, ValueError):
                continue
            ttl_action_values.append(ttl_float)
            bucket = bucket_ttl_action(ttl_float)
            ttl_action_distribution[bucket] = ttl_action_distribution.get(bucket, 0) + 1

    avg_ttl_action = sum(ttl_action_values) / len(ttl_action_values) if ttl_action_values else 0.0
    ttl_forced_exits = None
    cooldown_triggers = None
    if "ttl_forced_exits" in res_df.columns:
        ttl_forced_series = res_df["ttl_forced_exits"].dropna()
        if not ttl_forced_series.empty:
            ttl_forced_exits = int(ttl_forced_series.iloc[-1])
    if "cooldown_triggers" in res_df.columns:
        cooldown_series = res_df["cooldown_triggers"].dropna()
        if not cooldown_series.empty:
            cooldown_triggers = int(cooldown_series.iloc[-1])

    metrics = {
        "total_steps": len(res_df),
        "total_trades": int(trades),
        "net_pnl": float(total_pnl),
        "gross_pnl": float(gross_pnl),
        "final_balance": float(final_balance),
        "total_fees": float(getattr(env, "total_fees", 0.0)),
        "total_slippage": float(getattr(env, "total_slippage", 0.0)),
        "reward_scale": getattr(env, "reward_scale", None),
        "reward_clip": getattr(env, "reward_clip", None),
        "action_distribution": action_distribution,
        "ttl_action_distribution": ttl_action_distribution,
        "avg_ttl_action": float(avg_ttl_action),
        "ttl_forced_exits": ttl_forced_exits,
        "cooldown_triggers": cooldown_triggers,
    }
    save_backtest_results(
        portfolio_values=portfolio_values,
        trade_history=[],
        metrics=metrics,
        output_dir=out_dir,
        filename_prefix="backtest",
        metadata={
            "seed": seed,
            "model_path": str(args.model),
            "data_path": str(args.data),
            "config_path": str(args.config),
        },
    )
    logger.info(f"Saved results to {out_dir}")

if __name__ == "__main__":
    main()
