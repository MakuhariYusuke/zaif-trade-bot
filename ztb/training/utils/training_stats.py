#!/usr/bin/env python3
"""
Training statistics tracking.
"""

import time
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np


class TrainingStats:
    """
    Tracks training statistics and metrics.
    """

    def __init__(self):
        """Initialize training statistics."""
        self.start_time = time.time()
        self.episode_count = 0
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf
        self.best_episode = 0

    def update(
        self,
        reward: float,
        episode_length: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update statistics with new episode data.

        Args:
            reward: Episode reward
            episode_length: Episode length
            metrics: Additional metrics
        """
        self.episode_count += 1
        self.step_count += episode_length
        self.episode_rewards.append(reward)
        self.episode_lengths.append(episode_length)

        if metrics:
            for key, value in metrics.items():
                self.metrics[key].append(value)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_episode = self.episode_count

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        elapsed_time = time.time() - self.start_time

        summary = {
            "elapsed_time": elapsed_time,
            "episodes": self.episode_count,
            "total_steps": self.step_count,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "steps_per_second": self.step_count / elapsed_time
            if elapsed_time > 0
            else 0,
        }

        # Add custom metrics
        for key, values in self.metrics.items():
            if values:
                summary[f"mean_{key}"] = np.mean(values)
                summary[f"std_{key}"] = np.std(values)

        return summary

    def reset(self) -> None:
        """Reset all statistics."""
        self.__init__()

    def get_recent_stats(self, window: int = 10) -> Dict[str, Any]:
        """
        Get statistics for recent episodes.

        Args:
            window: Number of recent episodes to consider

        Returns:
            Recent statistics
        """
        if len(self.episode_rewards) < window:
            return self.get_summary()

        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]

        return {
            "recent_mean_reward": np.mean(recent_rewards),
            "recent_std_reward": np.std(recent_rewards),
            "recent_mean_length": np.mean(recent_lengths),
            "recent_best_reward": max(recent_rewards),
        }
