import os
import sys
from pathlib import Path

# Set environment variable to fix potential DLL issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import json

import pandas as pd

# Import torch first to avoid DLL issues

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env import HeavyTradingEnv
from ztb.utils.analysis_formatters import print_formatted_metrics
from ztb.utils.data_utils import load_csv_data
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.training_utils import load_model


def run_backtest_v451():
    print("Starting Chronos (v451) Backtest...")

    # 1. Load Data
    data_path = os.path.join(project_root, "data", "btc_jpy_1m_v451.csv")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = load_csv_data(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Use the last 20% for backtesting (or a specific period if needed)
    # For consistency with training (which likely used the beginning), we should test on unseen data if possible.
    # But since we just trained on 20k steps, we probably just used the beginning of the file.
    # Let's use the last 10000 rows for backtest to be safe/consistent with the evaluation attempt.
    backtest_df = df.iloc[-20000:].copy()

    print(
        f"Backtest data: {len(backtest_df)} rows from {backtest_df.index[0]} to {backtest_df.index[-1]}"
    )
    print(f"Data columns ({len(backtest_df.columns)}): {backtest_df.columns.tolist()}")

    # 2. Setup Environment
    env_config = {
        "initial_portfolio_value": 1000000.0,  # Correct key
        "leverage": 1.0,
        "min_trade_size": 0.001,
        "trading_fee": 0.001,
        "window_size": 60,
        "reward_type": "hybrid",
        "feature_set": "v451",
        # Enable advanced regime classifier to fix "stuck" regime reporting
        "advanced_market_regime": {
            "enabled": True,
            "regime_classifier_config": {
                "lookback_periods": {"short": 20, "medium": 50, "long": 100},
                "regime_scheme": "comprehensive",
            },
        },
    }

    env = HeavyTradingEnv(df=backtest_df, config=env_config)

    # DEBUG: Check features
    print(f"Environment feature count: {len(env.feature_names or [])}")
    # print(f"Environment feature names: {env.feature_names}")

    # Fix feature mismatch (143 vs 138)
    # The model expects 138 features, but the environment produces 143.
    # We suspect the extra features are the duplicate regime features and vol_rank.
    if len(env.feature_names or []) == 143:
        print("Detected 143 features. Attempting to align with model (138 features)...")

        # Suspected extra features
        extra_features = [
            "regime_low",
            "regime_med_low",
            "regime_med_high",
            "regime_high",
            "vol_rank",
        ]

        # Create corrected feature list
        if env.feature_names:
            corrected_features = [f for f in env.feature_names if f not in extra_features]
        else:
            corrected_features = []

        if len(corrected_features) == 138:
            print("Aligned feature count to 138. Re-initializing environment...")

            # Update config with explicit feature names
            env_config["feature_names"] = corrected_features

            # Re-create environment
            env = HeavyTradingEnv(df=backtest_df, config=env_config)
            print(f"New environment feature count: {len(env.feature_names or [])}")
        else:
            print(
                f"Warning: Could not align features. Count after removal: {len(corrected_features)}"
            )
            print(f"Extra features tried to remove: {extra_features}")

    # 3. Load Model
    model_path = os.path.join(
        project_root, "models", "sac_v451_phase7_regime_aware.zip"
    )
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # 4. Run Backtest Loop
    print("Running backtest loop...")
    obs, _ = env.reset()
    done = False

    results = []

    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # DEBUG: Check info keys if price is missing
        current_idx = env.current_step
        if current_idx >= len(env.df):
            current_idx = len(env.df) - 1

        # Use backtest_df for timestamp as env.df has reset index
        timestamp = backtest_df.index[current_idx]
        price = env.df.iloc[current_idx]["close"]  # Assuming 'close' exists

        # Determine action type
        action_val = float(action[0])
        if action_val > 0.01:  # Threshold
            action_type = "BUY"
        elif action_val < -0.01:
            action_type = "SELL"
        else:
            action_type = "HOLD"

        # Capture step data
        current_step_data = {
            "timestamp": timestamp,
            "action_type": action_type,
            "action_val": action_val,
            "reward": float(reward),
            "portfolio_value": float(info.get("portfolio_value", 0.0)),
            "price": float(price),
            "hour": timestamp.hour,
            "regime": info.get("market_regime", "UNKNOWN"),
        }
        results.append(current_step_data)

        step_count += 1
        if step_count % 1000 == 0:
            print(f"Step {step_count}/{len(backtest_df)}...")

        if done or truncated:
            break

    results_df = pd.DataFrame(results)
    results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])
    results_df.set_index("timestamp", inplace=True)

    # Calculate PnL
    results_df["pnl"] = results_df["portfolio_value"].diff()

    # 5. Save Results
    output_dir = os.path.join(project_root, "backtest_results", "v451")
    os.makedirs(output_dir, exist_ok=True)

    results_json_path = os.path.join(output_dir, "backtest_results.json")
    results_csv_path = os.path.join(output_dir, "backtest_results.csv")

    # Save detailed CSV
    results_df.to_csv(results_csv_path, index=False)

    # Save Summary JSON
    regime_counts = {}
    if "regime" in results_df.columns:
        regime_counts = {
            str(k): int(v)
            for k, v in results_df["regime"].value_counts().to_dict().items()
        }

    summary = {
        "model_name": "sac_v451_phase7_regime_aware",
        "total_steps": len(results_df),
        "initial_balance": env_config["initial_portfolio_value"],
        "final_balance": float(results_df["portfolio_value"].iloc[-1]),
        "total_pnl": float(
            results_df["portfolio_value"].iloc[-1]
            - env_config["initial_portfolio_value"]
        ),
        "return_pct": float(
            (
                results_df["portfolio_value"].iloc[-1]
                / env_config["initial_portfolio_value"]
            )
            - 1
        )
        * 100,
        "action_counts": {
            str(k): int(v)
            for k, v in results_df["action_type"].value_counts().to_dict().items()
        },
        "regime_counts": regime_counts,
    }

    safe_json_dump(summary, results_json_path, indent=4)

    print("Backtest complete.")
    print(f"Results saved to {output_dir}")
    print(
        f"Final Balance: {summary['final_balance']:.2f} (Return: {summary['return_pct']:.2f}%)"
    )

    # Print formatted metrics
    print_formatted_metrics(summary, "SAC v451 Backtest Results")


if __name__ == "__main__":
    run_backtest_v451()
