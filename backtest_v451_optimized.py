import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import torch first to avoid DLL initialization errors on Windows
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


def run_backtest():
    # Load optimized config
    config_path = os.path.join(
        project_root, "config", "v451", "sac_v451_optimized.json"
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    # Setup environment config
    env_config = config["training"]["environment"]["config"]

    # Ensure adaptive threshold mode is enabled to use the new logic
    env_config["adaptive_threshold_mode"] = True
    env_config["threshold_volatility_multiplier"] = 1.0

    # Load data
    data_path = os.path.join(project_root, "data", "btc_jpy_1m_v451.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Create environment
    env = HeavyTradingEnv(df, env_config)

    # Load model
    model_path = os.path.join(
        project_root, "models", "sac_v451_phase7_regime_aware.zip"
    )
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = os.path.join(
            project_root, "checkpoints", "v451", "phase7", "best_model.zip"
        )

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    model = SAC.load(model_path, env=env)

    # Run backtest
    obs, _ = env.reset()
    done = False

    # Manual history tracking to avoid deque truncation
    full_portfolio_history = []
    full_action_history = []

    print("Starting backtest...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        full_portfolio_history.append(env.portfolio_value)
        # Action might be an array, take the first element or the value itself
        if isinstance(action, (np.ndarray, list)):
            full_action_history.append(action[0] if len(action) > 0 else 0)
        else:
            full_action_history.append(action)

        if env.current_step % 1000 == 0:
            print(f"Step {env.current_step}, PnL: {env.total_pnl:.2f}")

    print(f"Backtest finished. Total PnL: {env.total_pnl:.2f}")

    # Save results manually
    results_dir = os.path.join(project_root, "backtest_results", "v451_optimized")
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame(
        {"portfolio_value": full_portfolio_history, "action": full_action_history}
    )

    # Try to get more detailed history if available
    if hasattr(env, "trade_history"):
        trade_df = pd.DataFrame(env.trade_history)
        trade_df.to_csv(os.path.join(results_dir, "trades.csv"), index=False)

    # Save main results
    results_df.to_csv(os.path.join(results_dir, "backtest_results.csv"), index=False)
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    run_backtest()
