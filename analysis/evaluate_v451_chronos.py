import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.trading.environment.heavy_env.heavy_env import HeavyTradingEnv


def evaluate_chronos_model():
    print("Starting Chronos (v451) Model Evaluation...")

    # 1. Load Data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "btc_jpy_1m_v451.csv",
    )
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Use a specific validation period (e.g., last 20% or a specific known difficult period)
    # For now, let's look at a recent chunk to see behavior
    eval_df = df.iloc[-10000:].copy()

    print(
        f"Loaded data: {len(eval_df)} rows from {eval_df.index[0]} to {eval_df.index[-1]}"
    )

    # 2. Setup Environment
    env_config = {
        "initial_balance": 10000.0,
        "leverage": 1.0,
        "min_trade_size": 0.001,
        "trading_fee": 0.001,
        "window_size": 60,
        "reward_type": "hybrid",
        "feature_set": "v451",  # CRITICAL: Use the Chronos feature set
    }

    env = HeavyTradingEnv(df=eval_df, **env_config)

    # 3. Load Model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "sac_v451_phase7_regime_aware.zip",
    )
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}")

    # 4. Run Evaluation Loop
    obs, _ = env.reset()
    done = False

    results = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # Capture step data
        current_step_data = {
            "timestamp": info.get("timestamp"),
            "action_type": info.get("action_type"),  # BUY/SELL/HOLD
            "action_val": action[0],
            "reward": reward,
            "portfolio_value": info.get("portfolio_value"),
            "price": info.get("current_price"),
            "hour": pd.to_datetime(info.get("timestamp")).hour,
            "regime": info.get(
                "regime", "UNKNOWN"
            ),  # Assuming env passes this through info, if not we derive
        }
        results.append(current_step_data)

        if done or truncated:
            break

    results_df = pd.DataFrame(results)
    results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])
    results_df.set_index("timestamp", inplace=True)

    # Calculate PnL per step
    results_df["pnl"] = results_df["portfolio_value"].diff()

    # 5. Analysis: Hourly Performance
    hourly_pnl = results_df.groupby("hour")["pnl"].sum()
    hourly_actions = (
        results_df.groupby(["hour", "action_type"]).size().unstack(fill_value=0)
    )

    print("\n=== Hourly Performance Analysis ===")
    print(hourly_pnl)

    # 6. Analysis: Regime Performance
    # Note: If regime isn't in info, we might need to reconstruct it, but v451 env should have it
    if "regime" in results_df.columns:
        regime_pnl = results_df.groupby("regime")["pnl"].sum()
        regime_counts = results_df["regime"].value_counts()

        print("\n=== Regime Performance Analysis ===")
        print(regime_pnl)
        print("\nRegime Distribution:")
        print(regime_counts)

    # 7. Save Report
    report = {
        "total_pnl": results_df["pnl"].sum(),
        "final_portfolio_value": results_df["portfolio_value"].iloc[-1],
        "hourly_pnl": hourly_pnl.to_dict(),
        "action_counts": results_df["action_type"].value_counts().to_dict(),
    }

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "analysis",
        "chronos_v451_eval_report.json",
    )
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nEvaluation complete. Report saved to {output_path}")

    # Simple Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df["portfolio_value"], label="Portfolio Value")
    plt.title("Chronos v451 Evaluation: Portfolio Value")
    plt.xlabel("Time")
    plt.ylabel("Value (JPY)")
    plt.legend()
    plot_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "analysis",
        "chronos_v451_eval_plot.png",
    )
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    evaluate_chronos_model()
