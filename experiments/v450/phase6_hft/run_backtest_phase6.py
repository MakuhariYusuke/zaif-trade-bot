import json
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def main():
    model_path = project_root / "models" / "sac_v450_phase6_hft.zip"
    data_path = project_root / "data" / "btc_jpy_1m_dataset.csv"
    output_path = project_root / "backtest_results" / "phase6_hft_backtest.json"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}")
    # Load model with custom objects to ensure compatibility if needed
    # We don't need to override hyperparameters for inference, just load weights
    model = SAC.load(str(model_path))

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Ensure timestamp is parsed
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Use a subset for quicker testing if needed
    # df = df.tail(20000)

    print("Creating environment...")
    # Use Phase 6 settings: 0.0 fees to encourage HFT
    env_config = EnvironmentConfig(
        max_position_size=1.0,
        transaction_cost=0.0,  # 0.0% fee for HFT simulation
        reward_scaling=1.0,
        initial_portfolio_value=1000000.0,  # 1M JPY
        timeframe="1m",
        feature_set="full",
    )

    env = HeavyTradingEnv(df=df, config=env_config)

    print("Starting backtest...")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    done = False

    portfolio_history = []
    price_history = []
    actions_history = []
    timestamps = []
    trade_pnls = []
    prev_total_pnl = 0.0

    # Initial state
    portfolio_history.append(env.portfolio_value)
    # Add initial timestamp
    if "timestamp" in df.columns and len(df) > 0:
        timestamps.append(str(df.iloc[0]["timestamp"]))
    else:
        timestamps.append("")

    step_count = 0
    while not done:
        # Predict action
        # Use stochastic=False (deterministic) for evaluation to use the mean of the distribution
        # BUT, if the model learned to be stochastic (high entropy) to satisfy the penalty,
        # deterministic=True might collapse to 0 (HOLD).
        # Let's try stochastic=True to see if it reproduces the training behavior.
        action, _ = model.predict(obs, deterministic=False)

        # Execute step

        step_result = env.step(action)

        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        # Record data
        portfolio_history.append(info.get("portfolio_value", env.portfolio_value))

        # Get price correctly
        try:
            current_price = env.data_manager.get_price_at_step(env.current_step - 1)
        except:
            current_price = 0.0
        price_history.append(current_price)

        # Handle action format
        if hasattr(action, "__getitem__"):
            act_val = float(action[0])
        else:
            act_val = float(action)

        # Convert to discrete for analysis compatibility
        discrete_act = continuous_to_discrete_action(act_val)
        actions_history.append(discrete_act)

        # Track realized PnL for win rate
        current_total_pnl = env.total_pnl
        if step_count > 0:
            realized_pnl_change = current_total_pnl - prev_total_pnl
            if abs(realized_pnl_change) > 1e-6:  # Filter noise
                trade_pnls.append(realized_pnl_change)

        prev_total_pnl = current_total_pnl

        # Get timestamp
        ts = info.get("timestamp")
        if ts is None or ts == "":
            try:
                idx = env.current_step - 1
                if 0 <= idx < len(df):
                    ts = df.iloc[idx]["timestamp"]
            except:
                pass

        timestamps.append(str(ts) if ts is not None else "")

        obs = next_obs
        step_count += 1

        if step_count % 5000 == 0:
            print(
                f"Step {step_count}, Portfolio: {portfolio_history[-1]:.2f}, Total PnL: {env.total_pnl:.2f}"
            )

    # Force close at the end to realize PnL
    if env.position != 0:
        print(f"Force closing position at end. Position: {env.position}")
        try:
            current_price = price_history[-1]
            entry_price = env.position_manager.entry_price
            position_size = env.position

            gross_pnl = (current_price - entry_price) * position_size
            cost = abs(position_size) * current_price * env.config.transaction_cost
            net_pnl = gross_pnl - cost

            print(f"Final Close PnL: {net_pnl:.2f}")
            trade_pnls.append(net_pnl)

        except Exception as e:
            print(f"Error calculating final close PnL: {e}")

    print("Backtest finished.")

    # Prepare results for analyze_backtest.py
    results = {
        "portfolio_history": portfolio_history,
        "price_history": price_history,
        "actions": actions_history,
        "timestamps": timestamps,
        "initial_balance": portfolio_history[0],
        "final_balance": portfolio_history[-1],
        "total_steps": step_count,
        "initial_btc": 0.0,
        "final_btc": env.position,
        "trade_pnls": trade_pnls,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saving results to {output_path}")


if __name__ == "__main__":
    main()
