#!/usr/bin/env python3
"""Shared wrapper/callback utilities for v456 MLP training scripts."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WarmupDrawdownWrapper(gym.Env):
    """Generic environment wrapper with warmup drawdown control hooks."""

    def __init__(
        self,
        base_env,
        warmup_steps: int = 10,
        initial_drawdown_limit: float = 0.5,
        final_drawdown_limit: float = 0.3,
        drawdown_decay_steps: Optional[int] = None,
        reward_transform: Optional[Callable[[float, dict], float]] = None,
        action_observer: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        super().__init__()
        self.env = base_env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.warmup_steps = warmup_steps
        self.warmup_counter = 0
        self.initial_drawdown_limit = initial_drawdown_limit
        self.final_drawdown_limit = final_drawdown_limit
        self.drawdown_decay_steps = drawdown_decay_steps
        self.reward_transform = reward_transform
        self.action_observer = action_observer
        self.episode_steps = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.warmup_counter = 0
        self.episode_steps = 0
        return obs, info

    def _update_drawdown_limit(self) -> None:
        if self.warmup_counter < self.warmup_steps:
            progress = self.warmup_counter / max(self.warmup_steps, 1)
            self.env.drawdown_limit = (
                self.initial_drawdown_limit
                + progress * (self.final_drawdown_limit - self.initial_drawdown_limit)
            )
            self.warmup_counter += 1
            return

        if self.drawdown_decay_steps and self.drawdown_decay_steps > 0:
            progress = min(1.0, self.episode_steps / self.drawdown_decay_steps)
            self.env.drawdown_limit = (
                self.initial_drawdown_limit * (1 - progress)
                + self.final_drawdown_limit * progress
            )

    def step(self, action: np.ndarray):
        self._update_drawdown_limit()

        if self.action_observer is not None:
            self.action_observer(action)

        result = self.env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            is_five = True
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False
            is_five = False

        if self.reward_transform is not None:
            reward = self.reward_transform(float(reward), info)

        self.episode_steps += 1

        if is_five:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class MilestoneRecorderCallback(BaseCallback):
    """Shared milestone logging callback with optional action stats."""

    def __init__(
        self,
        log_interval: int = 100,
        action_stats_provider: Optional[Callable[[], dict]] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__()
        self.log_interval = log_interval
        self.milestones = {}
        self.action_stats_provider = action_stats_provider
        self._logger = logging.getLogger(logger_name or __name__)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval != 0:
            return True

        if not (hasattr(self.model, "logger") and self.model.logger):
            return True

        ep_rew = self.model.logger.name_to_value.get("rollout/ep_rew_mean", np.nan)
        ep_len = self.model.logger.name_to_value.get("rollout/ep_len_mean", np.nan)
        milestone_key = self.n_calls // self.log_interval

        payload = {
            "timestamp": datetime.now().isoformat(),
            "episode_reward_mean": float(ep_rew) if not np.isnan(ep_rew) else None,
            "episode_length_mean": float(ep_len) if not np.isnan(ep_len) else None,
        }
        if self.action_stats_provider is not None:
            payload["action_stats"] = self.action_stats_provider()
        self.milestones[self.n_calls] = payload

        self._logger.info(f"\n{'='*70}")
        self._logger.info(f"📊 Milestone #{milestone_key} ({self.n_calls:,} steps)")
        self._logger.info(f"{'='*70}")
        self._logger.info(f"  Episode Reward Mean: {ep_rew:.6f}")
        self._logger.info(f"  Episode Length Mean: {ep_len:.1f}")
        if self.action_stats_provider is not None:
            stats = self.action_stats_provider()
            self._logger.info(f"  Action Mean: {stats.get('action_mean', 0):.4f}")
            self._logger.info(f"  BUY Rate: {stats.get('buy_ratio', 0):.2%}")
            self._logger.info(f"  SELL Rate: {stats.get('sell_ratio', 0):.2%}")
            self._logger.info(f"  HOLD Rate: {stats.get('hold_ratio', 0):.2%}")
        self._logger.info("")

        return True

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.milestones, f, indent=2, default=float)
