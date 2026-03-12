"""Lightweight SAC v438 quick backtest shim used by unit tests."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import SAC
except Exception:  # pragma: no cover - patched in tests
    class SAC:  # type: ignore[no-redef]
        @staticmethod
        def load(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("SAC backend unavailable")

try:
    from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer
except Exception:  # pragma: no cover - patched in tests
    class SACv427FeatureEngineer:  # type: ignore[no-redef]
        def generate_v427_features(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

try:
    from ztb.trading.environment.environment import HeavyTradingEnv
except Exception:  # pragma: no cover - patched in tests
    HeavyTradingEnv = object  # type: ignore[assignment]


def _calculate_max_drawdown(portfolio_df: pd.DataFrame) -> float:
    portfolio_values = portfolio_df["portfolio_value"].astype(float)
    running_peak = portfolio_values.cummax()
    drawdowns = (portfolio_values - running_peak) / running_peak.replace(0, pd.NA)
    return float(drawdowns.min() * -1.0)


def _calculate_sharpe_ratio(portfolio_df: pd.DataFrame) -> float:
    portfolio_values = portfolio_df["portfolio_value"].astype(float)
    returns = portfolio_values.pct_change().dropna()
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std)


def calculate_backtest_summary(
    results_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> dict[str, Any]:
    if results_df.empty:
        return {}

    summary: dict[str, Any] = {
        "total_episodes": len(results_df),
        "avg_total_reward": float(results_df["total_reward"].mean()),
        "total_trades_all_episodes": int(trades_df.shape[0]),
        "best_episode_reward": float(results_df["total_reward"].max()),
        "worst_episode_reward": float(results_df["total_reward"].min()),
    }

    if not portfolio_df.empty and "portfolio_value" in portfolio_df:
        summary["sharpe_ratio"] = _calculate_sharpe_ratio(portfolio_df)
        summary["max_drawdown"] = _calculate_max_drawdown(portfolio_df)

    return summary


def backtest_sac_v438_quick(
    model_path: str,
    data_path: str,
    output_dir: str = "backtest_experiments/v438.1",
    n_episodes: int = 3,
    deterministic: bool = True,
) -> dict[str, Any] | None:
    model_file = Path(model_path)
    data_file = Path(data_path)

    if not model_file.exists():
        logger.error("Model file not found: %s", model_path)
        return None
    if not data_file.exists():
        logger.error("Data file not found: %s", data_path)
        return None

    os.makedirs(output_dir, exist_ok=True)

    raw_df = pd.read_csv(data_file)
    feature_df = SACv427FeatureEngineer().generate_v427_features(raw_df)
    env = HeavyTradingEnv(feature_df)
    model = SAC.load(str(model_file))

    results_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for episode in range(n_episodes):
        obs, _info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        total_steps = 0
        total_trades = 0

        while not done and not truncated:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            total_steps += 1

            portfolio_value = info.get("portfolio_value")
            if portfolio_value is not None:
                portfolio_rows.append(
                    {"episode": episode + 1, "step": total_steps, "portfolio_value": portfolio_value}
                )

            if info.get("trade_executed"):
                total_trades += 1
                trade_rows.append(
                    {
                        "episode": episode + 1,
                        "step": total_steps,
                        "action": info.get("action", action),
                    }
                )

        results_rows.append(
            {
                "total_reward": total_reward,
                "total_trades": total_trades,
                "final_portfolio_value": portfolio_rows[-1]["portfolio_value"]
                if portfolio_rows
                else None,
                "total_steps": total_steps,
                "avg_reward_per_step": total_reward / total_steps if total_steps else 0.0,
                "trades_per_step": total_trades / total_steps if total_steps else 0.0,
            }
        )

    summary = calculate_backtest_summary(
        pd.DataFrame(results_rows),
        pd.DataFrame(portfolio_rows),
        pd.DataFrame(trade_rows),
    )
    logger.info("Quick backtest complete: %s", summary)
    return summary


def run_quick_backtest() -> dict[str, Any] | None:
    return backtest_sac_v438_quick(
        model_path="checkpoints/sac_v438_production_150000_steps.zip",
        data_path="data/btc_jpy_real_dataset.csv",
        output_dir="backtest_experiments/v438.1",
        n_episodes=3,
        deterministic=True,
    )


__all__ = ["calculate_backtest_summary", "backtest_sac_v438_quick", "run_quick_backtest"]
