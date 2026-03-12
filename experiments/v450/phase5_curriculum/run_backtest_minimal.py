import json
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def main():
    model_path = project_root / "models" / "sac_v450_phase5_stage4_pnl_focused.zip"
    data_path = project_root / "data" / "btc_jpy_1m_dataset.csv"
    output_path = project_root / "backtest_results" / "phase5_stage4_backtest.json"

    print(f"Loading model from {model_path}")
    model = SAC.load(str(model_path))

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Ensure timestamp is parsed
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Use a subset for quicker testing if needed, but let's try full first or a reasonable chunk
    # df = df.tail(10000)

    print("Creating environment...")
    env_config = EnvironmentConfig(
        max_position_size=1.0,
        transaction_cost=0.001,
        reward_scaling=1.0,
        initial_portfolio_value=200000.0,
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
    # Add initial timestamp (approximate as start of data or first step)
    if "timestamp" in df.columns and len(df) > 0:
        timestamps.append(str(df.iloc[0]["timestamp"]))
    else:
        timestamps.append("")

    # We don't have price/timestamp easily before first step unless we peek, but let's just record during loop

    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)

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
            # current_step is already incremented in step()
            current_price = env.data_manager.get_price_at_step(env.current_step - 1)
        except Exception:
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

        # Track trades for win rate calculation
        # If trade_executed is true, we assume a trade happened.
        # But HeavyTradingEnv doesn't give us PnL of that specific trade easily in info.
        # We can try to infer from pnl history if we had it.
        # Alternatively, we can just use the 'pnl' from info which is step pnl.
        # If step pnl is non-zero and action was not HOLD, it's likely a trade impact (or unrealized pnl change).
        # Actually, realized pnl is what matters for win rate.
        # HeavyTradingEnv updates total_pnl.
        # Let's track realized pnl changes.

        current_total_pnl = env.total_pnl
        if step_count > 0:
            realized_pnl_change = current_total_pnl - prev_total_pnl
            # If there was a realized pnl change, record it as a trade result
            # Note: This might aggregate multiple trades if they happen in same step (unlikely here)
            # or if funding fees etc apply.
            # But mostly it captures closed trades.
            if abs(realized_pnl_change) > 0:
                trade_pnls.append(realized_pnl_change)

        prev_total_pnl = current_total_pnl

        # Get timestamp from info or dataframe
        ts = info.get("timestamp")
        if ts is None or ts == "":
            # Try to get from dataframe if index matches
            # HeavyTradingEnv usually tracks current_step
            try:
                # env.current_step might be ahead by 1
                idx = env.current_step - 1
                if 0 <= idx < len(df):
                    ts = df.iloc[idx]["timestamp"]
            except Exception:
                pass

        timestamps.append(str(ts) if ts is not None else "")

        obs = next_obs
        step_count += 1

        if step_count % 1000 == 0:
            print(
                f"Step {step_count}, Portfolio: {portfolio_history[-1]:.2f}, Total PnL: {env.total_pnl:.2f}"
            )

    # Force close at the end to realize PnL
    if env.position != 0:
        print(f"Force closing position at end. Position: {env.position}")
        # Calculate final close PnL manually or use env method if available
        # HeavyTradingEnv doesn't have a simple 'close_position' method exposed easily without action
        # But we can simulate a SELL action of the full size

        # Or just calculate it:
        current_price = price_history[-1]
        # We need average entry price.
        # env.position_manager.entry_price
        try:
            entry_price = env.position_manager.entry_price
            position_size = env.position

            # PnL = (Price - Entry) * Size (for Long)
            # PnL = (Entry - Price) * Size (for Short) -> Wait, Size is negative for Short?
            # Usually: (Exit - Entry) * Size

            gross_pnl = (current_price - entry_price) * position_size

            # Transaction cost
            cost = abs(position_size) * current_price * env.config.transaction_cost

            net_pnl = gross_pnl - cost

            print(f"Final Close PnL: {net_pnl:.2f}")
            trade_pnls.append(net_pnl)

            # Update final balance to reflect this realization (if not already in portfolio value)
            # Portfolio value usually includes unrealized PnL, so it shouldn't change much except for the fee.
            # But for 'win_rate' calculation which uses 'trade_pnls', we need this entry.

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

    print(f"Saving results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
