#!/usr/bin/env python3
"""Run action confidence diagnostics end-to-end.

If `--model` is provided, it will use SACBacktester to run the model and obtain a trade log.
Otherwise it will run the `HeavyTradingEnv` with a random policy for one episode to produce a sample trade log.

Outputs a CSV summary via the diagnostics module.
"""
import argparse
import json
import tempfile
from pathlib import Path

from ztb.analysis.action_confidence_diagnostics import (
    bin_and_aggregate,
    compute_trade_metrics,
    extract_trades_from_step_logs,
)


def run_with_model(model_path: str, data_path: str):
    from ztb.analysis.sac_backtester import SACBacktester

    bt = SACBacktester(model_path=model_path)
    bt.load_config()  # try to load config if exists
    result = bt.run_backtest(data_path, num_episodes=1, deterministic=True)
    return result.trade_log


def run_random_policy(data_path: str):
    # Create environment and step with random actions to collect trade info
    import numpy as np
    import pandas as pd

    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig

    df = pd.read_csv(data_path)
    env = HeavyTradingEnv(df=df, config=EnvironmentConfig())

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False
    trade_log = []
    step = 0
    while not done and step < env.n_steps - 1:
        action = np.random.uniform(-1, 1, size=(1,))
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result
        if info.get("trade_executed", False):
            try:
                action_value = float(action[0])
            except Exception:
                action_value = float(action)
            trade_info = {
                "step": step,
                "timestamp": info.get("timestamp"),
                "action": action_value,
                "price": info.get("price", 0),
                "position": info.get("position", 0),
                "pnl": info.get("pnl", 0),
                "step_pnl": info.get("step_pnl", 0),
                "reward": float(reward),
            }
            trade_log.append(trade_info)
        step += 1
    return trade_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False, help="CSV time series data path")
    parser.add_argument("--model", required=False, help="SAC model path (optional)")
    parser.add_argument("--out-csv", default="action_confidence_summary.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_path = args.data or (project_root / "data" / "btc_jpy_1m_v454.csv")

    if args.model:
        trade_log = run_with_model(args.model, str(data_path))
    else:
        trade_log = run_random_policy(str(data_path))

    # save trade log
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
        json.dump(trade_log, tf)
        tmp_path = tf.name

    # run diagnostics
    from ztb.analysis.action_confidence_diagnostics import load_trade_log

    trades = load_trade_log(tmp_path)
    windows = extract_trades_from_step_logs(trades)
    metrics = [compute_trade_metrics(w) for w in windows if compute_trade_metrics(w)]
    bins = [0.0, 0.005, 0.01, 0.015, 0.03, 1.0]
    summary = bin_and_aggregate(metrics, bins)
    summary.to_csv(args.out_csv, index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
