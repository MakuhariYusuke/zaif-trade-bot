#!/usr/bin/env python3
"""Compare single PPO model against ensemble across paper trading runs."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

TRADING_DAYS_PER_YEAR = 252

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment import HeavyTradingEnv  # noqa: E402  pylint: disable=wrong-import-position

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    model_name: str
    avg_reward: float
    avg_return: float
    total_return: float
    win_rate: float
    sharpe_ratio: float
    buy_actions: int
    sell_actions: int
    hold_actions: int


def load_models(paths: Iterable[Path]) -> List[PPO]:
    models: List[PPO] = []
    for path in paths:
        LOGGER.info("Loading model %s", path)
        models.append(PPO.load(str(path)))
    return models


def select_action(models: List[PPO], observation: np.ndarray) -> Dict[str, np.ndarray]:
    obs_tensor = torch.as_tensor(observation[None, :], dtype=torch.float32)
    probs: List[np.ndarray] = []
    for model in models:
        obs_device = obs_tensor.to(model.device)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_device)
            dist = distribution.distribution  # type: ignore[attr-defined]
            probs.append(dist.probs.detach().cpu().numpy()[0])
    mean_probs = np.mean(probs, axis=0)
    action = int(np.argmax(mean_probs))
    return {"action": action, "probs": mean_probs}


def evaluate(models: List[PPO], df: pd.DataFrame, args: argparse.Namespace, model_name: str) -> EvaluationResult:
    env_config = {
        "reward_scaling": 1.0,
        "transaction_cost": 0.0,
        "max_position_size": 1.0,
        "feature_set": args.feature_set,
        "timeframe": args.timeframe,
    }

    episode_rewards: List[float] = []
    episode_returns: List[float] = []
    buy_actions = sell_actions = hold_actions = 0

    for episode in range(args.episodes):
        env = HeavyTradingEnv(df=df, config=env_config, random_start=True)
        obs, _ = env.reset()
        balance = args.initial_balance
        reward_sum = 0.0

        for step in range(args.max_steps):
            selection = select_action(models, obs)
            action = selection["action"]
            LOGGER.debug("%s - Episode %d Step %d -> action %d", model_name, episode + 1, step, action)

            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward

            if action == 0:
                hold_actions += 1
            elif action == 1:
                buy_actions += 1
            elif action == 2:
                sell_actions += 1

            balance += info.get("pnl", 0.0)
            if terminated or truncated:
                break

        env.close()
        episode_rewards.append(reward_sum)
        episode_returns.append((balance - args.initial_balance) / args.initial_balance)

    total_return = float(np.prod([1 + r for r in episode_returns]) - 1) if episode_returns else 0.0
    avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    sharpe = 0.0
    if len(episode_returns) > 1 and np.std(episode_returns) > 0:
        sharpe = (
            np.mean(episode_returns) / np.std(episode_returns)
        ) * np.sqrt(TRADING_DAYS_PER_YEAR)

    win_rate = sum(1 for r in episode_returns if r > 0) / len(episode_returns) if episode_returns else 0.0

    return EvaluationResult(
        model_name=model_name,
        avg_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        avg_return=avg_return,
        total_return=total_return,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        buy_actions=buy_actions,
        sell_actions=sell_actions,
        hold_actions=hold_actions,
    )


def write_report(results: List[EvaluationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model",
            "avg_reward",
            "avg_return",
            "total_return",
            "win_rate",
            "sharpe_ratio",
            "buy_actions",
            "sell_actions",
            "hold_actions",
        ])
        for result in results:
            writer.writerow([
                result.model_name,
                f"{result.avg_reward:.6f}",
                f"{result.avg_return:.6f}",
                f"{result.total_return:.6f}",
                f"{result.win_rate:.6f}",
                f"{result.sharpe_ratio:.6f}",
                result.buy_actions,
                result.sell_actions,
                result.hold_actions,
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ensemble PPO against single model")
    parser.add_argument(
        "--single-model",
        type=Path,
        default=Path("models/scalping_15s_balance_test12_balanced_data.zip"),
        help="Path to baseline single PPO model",
    )
    parser.add_argument(
        "--ensemble-models",
        nargs="+",
        default=[
            "models/ensemble_model_1.zip",
            "models/ensemble_model_2.zip",
            "models/ensemble_model_3.zip",
        ],
        help="Paths to ensemble member models",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("ml-dataset-enhanced-balanced.csv"),
        help="Dataset used for evaluation",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of paper trading episodes per comparison",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=100000.0,
        help="Initial account balance for evaluation",
    )
    parser.add_argument(
        "--feature-set",
        default="full",
        help="Feature set used by the evaluation environment",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Environment timeframe",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/ensemble_comparison.csv"),
        help="Where to write the CSV summary",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    LOGGER.info("Loading dataset %s", args.data_path)
    data_frame = pd.read_csv(args.data_path)

    single_models = load_models([args.single_model])
    ensemble_models = load_models([Path(p) for p in args.ensemble_models])

    single_result = evaluate(single_models, data_frame, args, model_name="single")
    ensemble_result = evaluate(ensemble_models, data_frame, args, model_name="ensemble_mean_prob")

    results = [single_result, ensemble_result]
    for result in results:
        LOGGER.info(
            "%s summary: avg_return=%.4f total_return=%.4f win_rate=%.4f sharpe=%.4f",
            result.model_name,
            result.avg_return,
            result.total_return,
            result.win_rate,
            result.sharpe_ratio,
        )

    write_report(results, args.report_path)
    LOGGER.info("Wrote comparison report to %s", args.report_path)
    print("\nComparison report saved to", args.report_path)
    for result in results:
        print(
            f"{result.model_name}: avg_return={result.avg_return:.4f}, total_return={result.total_return:.4f}, "
            f"win_rate={result.win_rate:.4f}, sharpe={result.sharpe_ratio:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
