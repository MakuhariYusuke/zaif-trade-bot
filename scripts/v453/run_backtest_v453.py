#!/usr/bin/env python3
"""
Backtest script for v453 Hybrid Strategy (v452 Model + Heuristic Filters).
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.analysis.regime.v444_regime_classifier import V444RegimeClassifier
from ztb.metrics.metrics import calculate_all_metrics

# Configure logging to reduce spam
logging.basicConfig(level=logging.WARNING)
logging.getLogger("ztb").setLevel(logging.WARNING)

# Configuration
# Using v452 model as the base
DEFAULT_MODEL_PATH = "models/sac_v452_optimized_10k.zip"
DEFAULT_DATA_PATH = "data/btc_jpy_1m_merged.csv"
DEFAULT_THRESHOLDS_PATH = "config/v452/threshold_optimized.json"
DEFAULT_HYBRID_CONFIG_PATH = "config/v453/hybrid_config_v3.json"
DEFAULT_RESULTS_DIR = "backtest_results/v453_hybrid_v3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v453 hybrid backtest")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--thresholds-path", default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--hybrid-config", default=DEFAULT_HYBRID_CONFIG_PATH)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--use-last-fraction",
        type=float,
        default=0.2,
        help="Use only last fraction of data for speed (0<frac<=1)",
    )
    return parser.parse_args()


def _derive_discrete_action_from_continuous(act_val: float) -> int:
    # Must match the runner's own fallback mapping for consistency
    if act_val > 0.33:
        return 1
    if act_val < -0.33:
        return 2
    return 0


def _is_filter_active(
    hybrid_config: dict,
    current_hour: int | None,
    current_volatility: float | None,
    market_regime: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not hybrid_config or not hybrid_config.get("enabled", False):
        return False, reasons

    time_filter = hybrid_config.get("time_filter", {})
    if time_filter.get("enabled", False):
        excluded = time_filter.get("excluded_hours", [])
        if current_hour is not None and current_hour in excluded:
            reasons.append("time")

    vol_filter = hybrid_config.get("volatility_filter", {})
    if vol_filter.get("enabled", False):
        v_min = vol_filter.get("danger_zone_min", 0.0)
        v_max = vol_filter.get("danger_zone_max", 1.0)
        if current_volatility is not None and v_min <= current_volatility <= v_max:
            reasons.append("volatility")

    regime_filter = hybrid_config.get("regime_filter", {})
    if regime_filter.get("enabled", False):
        excluded_regimes = regime_filter.get("excluded_regimes", [])
        if market_regime in excluded_regimes:
            reasons.append("regime")

    return len(reasons) > 0, reasons

def run_backtest():
    args = _parse_args()

    print("Starting backtest for v453 Hybrid Strategy")
    print(f"Base Model: {args.model_path}")
    print(f"Hybrid Config: {args.hybrid_config}")
    print(f"Results Dir: {args.results_dir}")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        return

    # Load optimized thresholds (v452)
    with open(args.thresholds_path, 'r') as f:
        optimized_thresholds = json.load(f)
    print(f"Loaded thresholds: {optimized_thresholds}")

    # Load Hybrid Config (v453)
    with open(args.hybrid_config, 'r') as f:
        hybrid_config = json.load(f)
    print(f"Loaded hybrid config: {hybrid_config}")

    # Load Data
    print("Loading data...")
    print(f"Using merged data: {args.data_path}")
    df = pd.read_csv(args.data_path, parse_dates=["timestamp"], index_col="timestamp")
    
    # For speed, use last fraction
    if not (0 < args.use_last_fraction <= 1.0):
        raise ValueError("--use-last-fraction must be 0 < frac <= 1")
    start_idx = int(len(df) * (1.0 - args.use_last_fraction))
    df = df.iloc[start_idx:]
    print(f"Data truncated for speed: {len(df)} rows (Last {args.use_last_fraction:.0%})")
    
    # Create Environment Config
    env_config = EnvironmentConfig(
        feature_set="default",
        use_continuous_actions=True,
        target_feature_count=138,
        correlation_reduction=True,
        adaptive_threshold_mode=True,  # Enable adaptive thresholds for HFT logic
        threshold_volatility_multiplier=1.0,
    )
    
    # Inject optimized thresholds
    env_config.regime_threshold_multipliers = optimized_thresholds
    
    # Inject Hybrid Config (v453)
    env_config.hybrid_config = hybrid_config
    
    # Create Environment
    print("Creating environment...")
    env = HeavyTradingEnv(df=df, config=env_config)
    
    # Enable Regime Adaptation
    print("Enabling regime adaptation...")
    regime_classifier = V444RegimeClassifier()
    env.enable_market_regime_adaptation(regime_classifier=regime_classifier)
    
    # Load Model
    print("Loading model...")
    model = SAC.load(args.model_path, env=env)
    
    # Run Backtest
    print("Running backtest loop...")
    obs, _ = env.reset()
    done = False
    
    history = []
    prev_position = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Collect data for analysis
        # Note: _resolve_price might not be public, use data_manager if needed
        # But env._resolve_price works if it exists
        if hasattr(env, "_resolve_price"):
             current_price = env._resolve_price()
        else:
             current_price = env.data_manager.get_price_at_step(env.current_step)
        
        # Handle regime enum
        regime_val = info.get("market_regime", "unknown")
        if hasattr(regime_val, "value"):
            regime_val = regime_val.value

        # v453: effective action (reflects filters/no-op detection)
        effective_action = info.get("effective_action")

        # Attempted discrete action inferred from continuous value (runner view)
        cont_val = action[0] if isinstance(action, (list, np.ndarray)) else float(action)
        attempted_discrete_action = _derive_discrete_action_from_continuous(cont_val)

        # Hour (UTC) derived from timestamp
        ts_val = info.get("timestamp", df.index[env.current_step - 1])
        ts_dt = pd.to_datetime(ts_val)
        current_hour = int(ts_dt.hour)

        # Volatility proxy used in env is ATR; log it for analysis
        try:
            current_atr = float(env.data_manager.get_atr_at_step(env.current_step))
        except Exception:
            current_atr = None

        filter_active, filter_reasons = _is_filter_active(
            hybrid_config=hybrid_config,
            current_hour=current_hour,
            current_volatility=current_atr,
            market_regime=str(regime_val),
        )

        position = int(getattr(env, "position", 0))
        blocked_entry = (
            prev_position == 0
            and position == 0
            and attempted_discrete_action in (1, 2)
            and effective_action == 0
        )
        
        step_data = {
            "timestamp": ts_dt,
            "portfolio_value": env.portfolio_value,
            "price": current_price,
            "action": cont_val,
            "attempted_discrete_action": attempted_discrete_action,
            "effective_action": effective_action,
            "position": position,
            "pnl": env.total_pnl,
            "reward": reward,
            "regime": str(regime_val),
            "action_type": info.get("action_type"),  # None if missing
            "hour": current_hour,
            "atr": current_atr,
            "filter_active": filter_active,
            "filter_reasons": ",".join(filter_reasons),
            "blocked_entry": blocked_entry,
        }
        
        # v453: Use effective_action from info if available (reflects filters)
        if "effective_action" in info:
            eff_act = info["effective_action"]
            if eff_act == 1:
                step_data["action_type"] = "BUY"
            elif eff_act == 2:
                step_data["action_type"] = "SELL"
            else:
                step_data["action_type"] = "HOLD"
        # If action_type is not in info, derive it (assuming continuous action [-1, 1])
        elif step_data["action_type"] is None:
            act_val = step_data["action"]
            if act_val > 0.33:
                step_data["action_type"] = "BUY"
            elif act_val < -0.33:
                step_data["action_type"] = "SELL"
            else:
                step_data["action_type"] = "HOLD"

        prev_position = position

        history.append(step_data)
        
        if env.current_step % 1000 == 0:
            print(f"Step {env.current_step}, PnL: {env.total_pnl:.2f}")
            
    print(f"Backtest finished. Final PnL: {env.total_pnl:.2f}")
    
    # Save Results
    os.makedirs(args.results_dir, exist_ok=True)
    results_df = pd.DataFrame(history)
    
    csv_path = os.path.join(args.results_dir, "backtest_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Calculate Metrics
    print("\nCalculating performance metrics...")
    # Calculate returns (percentage change of portfolio value)
    results_df["return"] = results_df["portfolio_value"].pct_change().fillna(0)
    
    # Calculate comprehensive metrics
    # period_per_year for 1m data: 365 days * 24 hours * 60 minutes = 525600
    metrics = calculate_all_metrics(results_df["return"], period_per_year=525600)
    
    # Print Metrics
    print("\n=== Performance Metrics ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"Volatility: {metrics['volatility']:.4f}")
    
    # Save summary JSON
    summary = {
        "initial_balance": env.initial_portfolio_value,
        "final_balance": env.portfolio_value,
        "total_pnl": env.total_pnl,
        "total_steps": env.current_step,
        "metrics": metrics
    }
    
    # Handle non-serializable types in metrics (like numpy types)

    with open(os.path.join(args.results_dir, "backtest_results.json"), "w") as f:
        json.dump(summary, f, indent=4, default=default_converter)

if __name__ == "__main__":
    run_backtest()
