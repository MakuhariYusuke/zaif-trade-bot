#!/usr/bin/env python3
"""Shared runner for 5000-step SAC experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

# Ensure project root is importable when executed from arbitrary CWD.
project_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(project_root))

from ztb.training.constants import (
    BATCH_SIZE_SMALL,
    DEFAULT_BUFFER_SIZE_AGGRESSIVE,
    DEFAULT_ENT_COEF_SAC,
    DEFAULT_GAMMA,
    DEFAULT_LEARNING_RATE_SAC,
    DEFAULT_LEARNING_STARTS_MINIMAL,
    DEFAULT_TARGET_UPDATE_INTERVAL,
    DEFAULT_TAU,
    DEFAULT_VERBOSE,
)
from ztb.utils.constants import (
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_PROGRESS_BAR,
    DEFAULT_SEED,
    DEFAULT_TOTAL_TIMESTEPS,
)
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.logging_utils import setup_logging
from ztb.utils.training_utils import (
    create_checkpoint_callback,
    display_training_complete,
    save_model,
)

setup_logging()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Train5000Profile:
    name: str
    title: str
    model_path: str
    stats_path: str
    checkpoint_dir: str
    checkpoint_prefix: str
    threshold: float
    no_action_penalty: float = 0.0
    action_bonus: float = 0.0
    learning_rate: float = DEFAULT_LEARNING_RATE_SAC
    buffer_size: int = DEFAULT_BUFFER_SIZE_AGGRESSIVE
    learning_starts: int = DEFAULT_LEARNING_STARTS_MINIMAL
    batch_size: int = BATCH_SIZE_SMALL
    ent_coef: float = DEFAULT_ENT_COEF_SAC
    net_arch: tuple[int, int] = (64, 64)
    timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ
    improvements: tuple[str, ...] = ()


def _create_trading_env(profile: Train5000Profile) -> gym.Env:
    np.random.seed(DEFAULT_SEED)
    n_steps = 1000
    t = np.linspace(0, 4 * np.pi, n_steps)
    trend = 0.1 * np.sin(t * 0.1)
    noise = np.random.normal(0, 0.005, n_steps)
    prices = 5_000_000 * (1 + np.cumsum(trend + noise))

    class TradingEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
            self.prices = prices
            self.initial_balance = 200_000.0
            self.trade_size = 0.1
            self.reset()

        def reset(self, seed=None, options=None):
            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0.0
            self.prev_portfolio_value = self.initial_balance
            return self._get_observation(), {}

        def step(self, action):
            action_value = float(action[0])
            price = float(self.prices[self.current_step])
            trade_executed = False

            if action_value > profile.threshold and self.balance > price * self.trade_size:
                self.position += self.trade_size
                self.balance -= price * self.trade_size
                trade_executed = True
            elif action_value < -profile.threshold and self.position >= self.trade_size:
                self.position -= self.trade_size
                self.balance += price * self.trade_size
                trade_executed = True

            portfolio_value = self.balance + self.position * price
            reward = (portfolio_value - self.prev_portfolio_value) / max(self.prev_portfolio_value, 1e-9)
            if not trade_executed:
                reward += profile.no_action_penalty
            if trade_executed:
                reward += profile.action_bonus
            self.prev_portfolio_value = portfolio_value

            self.current_step += 1
            done = self.current_step >= len(self.prices) - 1
            return self._get_observation(), float(reward), done, False, {}

        def _get_observation(self):
            price = float(self.prices[self.current_step])
            prev = float(self.prices[max(self.current_step - 1, 0)])
            trend_now = (price - prev) / max(prev, 1e-9)
            return np.array(
                [
                    price / 10_000_000.0,
                    trend_now,
                    self.position,
                    self.balance / self.initial_balance,
                    self.current_step / len(self.prices),
                ],
                dtype=np.float32,
            )

    return TradingEnv()


def run_training(profile: Train5000Profile) -> dict[str, Any]:
    logger.info(f"Starting {profile.name} training...")
    env = _create_trading_env(profile)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=profile.learning_rate,
        buffer_size=profile.buffer_size,
        learning_starts=profile.learning_starts,
        batch_size=profile.batch_size,
        tau=DEFAULT_TAU,
        gamma=DEFAULT_GAMMA,
        ent_coef=profile.ent_coef,
        target_update_interval=DEFAULT_TARGET_UPDATE_INTERVAL,
        verbose=DEFAULT_VERBOSE,
        policy_kwargs={"net_arch": list(profile.net_arch)},
    )

    checkpoint_callback = create_checkpoint_callback(
        save_freq=profile.checkpoint_freq,
        save_path=profile.checkpoint_dir,
        name_prefix=profile.checkpoint_prefix,
    )

    stats: dict[str, Any] = {
        "total_timesteps": profile.timesteps,
        "environment": f"{profile.name}_trading_env",
        "model_config": {
            "learning_rate": profile.learning_rate,
            "buffer_size": profile.buffer_size,
            "batch_size": profile.batch_size,
            "learning_starts": profile.learning_starts,
            "ent_coef": profile.ent_coef,
            "net_arch": list(profile.net_arch),
        },
        "improvements": list(profile.improvements),
    }

    start = time.time()
    try:
        model.learn(
            total_timesteps=profile.timesteps,
            callback=checkpoint_callback,
            progress_bar=DEFAULT_PROGRESS_BAR,
        )
        duration = time.time() - start

        saved = save_model(model, profile.model_path)
        stats.update(
            {
                "training_completed": True,
                "final_status": "success",
                "model_path": profile.model_path if saved else None,
            }
        )
        display_training_complete(
            {
                "total_timesteps": profile.timesteps,
                "model_path": profile.model_path,
                "final_status": "success",
            },
            duration,
        )
    except Exception as exc:
        stats.update(
            {
                "training_completed": False,
                "final_status": "failed",
                "error": str(exc),
            }
        )
        logger.exception("Training failed")

    safe_json_dump(stats, profile.stats_path, indent=2)
    logger.info(f"Training stats saved to {profile.stats_path}")

    print("\n" + "=" * 50)
    print(profile.title)
    print("=" * 50)
    print(f"Status: {stats['final_status']}")
    print(f"Timesteps: {stats['total_timesteps']}")
    print(f"Model saved: {stats.get('model_path', 'N/A')}")
    if profile.improvements:
        print("\nImprovements:")
        for item in profile.improvements:
            print(f"  - {item}")
    print("=" * 50)

    return stats
