#!/usr/bin/env python3
"""
SAC-Specific Callbacks for Soft Actor-Critic Training.

Callbacks focused on SAC dynamics such as entropy temperature adaptation,
value-function stability monitoring, target-network update tuning, and
exploration diagnostics.
"""

from __future__ import annotations

import logging

import numpy as np

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    NoOpMemoryOptimizedCallback,
)
from ztb.training.callbacks.shared.utils.value_utils import (
    append_bounded as _append_bounded_value,
    as_optional_float as _as_float,
)
from ztb.types.common import ObjectMap

_HISTORY_LIMIT = 1_000

def _append_bounded(history: list[float], value: float, max_len: int) -> None:
    _append_bounded_value(history, value, max_len)

class SACTemperatureScheduler(NoOpMemoryOptimizedCallback):
    """Adaptive entropy-temperature scheduler for SAC."""

    def __init__(
        self,
        initial_temp: float = 1.0,
        min_temp: float = 0.1,
        max_temp: float = 2.0,
        decay_rate: float = 0.995,
        adaptive: bool = True,
        final_temp: float | None = None,
        window_size: int = 100,
    ):
        super().__init__()
        self.initial_temp = initial_temp
        self.min_temp = final_temp if final_temp is not None else min_temp
        self.max_temp = max_temp
        self.decay_rate = decay_rate
        self.adaptive = adaptive
        self.window_size = max(10, window_size)

        self.current_temp = initial_temp
        self.reward_history: list[float] = []
        self.entropy_history: list[float] = []
        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.current_temp = self.initial_temp
        self.reward_history.clear()
        self.entropy_history.clear()
        self.logger.info(
            "SAC temperature scheduler initialized (temp=%.4f)",
            self.current_temp,
        )

    def on_training_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.logger.info(
            "SAC temperature scheduler finished (final_temp=%.4f)",
            self.current_temp,
        )

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if logs is None:
            return

        reward = _as_float(logs.get("episode_reward", logs.get("reward")))
        entropy = _as_float(logs.get("entropy", logs.get("policy_entropy")))

        if reward is not None:
            _append_bounded(self.reward_history, reward, self.window_size)
        if entropy is not None:
            _append_bounded(self.entropy_history, entropy, self.window_size)

        if self.adaptive:
            self._adaptive_update()
        else:
            self._decay_update()

        self.current_temp = float(np.clip(self.current_temp, self.min_temp, self.max_temp))
        logs["temperature"] = self.current_temp

        if context.epoch % 10 == 0:
            self.logger.debug("SAC temperature updated: %.4f", self.current_temp)

    def _adaptive_update(self) -> None:
        if len(self.reward_history) < 10 or len(self.entropy_history) < 10:
            return

        recent_rewards = self.reward_history[-10:]
        recent_entropies = self.entropy_history[-10:]

        reward_std = float(np.std(recent_rewards))
        avg_entropy = float(np.mean(recent_entropies))
        reward_trend = float(np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0])

        if reward_trend > 0.01 and avg_entropy < 0.5:
            self.current_temp *= 1.05
        elif reward_trend < -0.01 and avg_entropy > 1.0:
            self.current_temp *= 0.95
        elif reward_std > 0.5:
            self.current_temp *= 0.98

    def _decay_update(self) -> None:
        self.current_temp *= self.decay_rate

    def get_current_temperature(self) -> float:
        return self.current_temp

    def get_temperature_stats(self) -> ObjectMap:
        return {
            "current_temp": self.current_temp,
            "initial_temp": self.initial_temp,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "adaptive": self.adaptive,
            "history_size": len(self.reward_history),
        }

class SACValueFunctionMonitor(NoOpMemoryOptimizedCallback):
    """Monitor SAC Q/value stability and divergence signals."""

    def __init__(
        self,
        monitor_frequency: int = 50,
        convergence_threshold: float = 0.01,
        divergence_threshold: float = 10.0,
    ):
        super().__init__()
        self.monitor_frequency = max(1, monitor_frequency)
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold

        self.q_values_history: list[float] = []
        self.value_history: list[float] = []
        self.q_value_gaps: list[float] = []

        self.convergence_epochs = 0
        self.last_q_value: float | None = None
        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.q_values_history.clear()
        self.value_history.clear()
        self.q_value_gaps.clear()
        self.convergence_epochs = 0
        self.last_q_value = None
        self.logger.info("SAC value function monitor initialized")

    def on_training_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.logger.info(
            "SAC value function monitor finished (convergence_epochs=%s)",
            self.convergence_epochs,
        )

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if context.epoch % self.monitor_frequency != 0:
            return
        if logs is None:
            return

        q1_value = _as_float(logs.get("q1_value", logs.get("critic_1_value")))
        q2_value = _as_float(logs.get("q2_value", logs.get("critic_2_value")))
        value = _as_float(
            logs.get("value", logs.get("value_function", logs.get("value_mean")))
        )

        if q1_value is not None and q2_value is not None:
            avg_q = (q1_value + q2_value) / 2.0
            q_gap = abs(q1_value - q2_value)

            _append_bounded(self.q_values_history, avg_q, _HISTORY_LIMIT)
            _append_bounded(self.q_value_gaps, q_gap, _HISTORY_LIMIT)

            if self.last_q_value is not None:
                q_change = abs(avg_q - self.last_q_value)
                if q_change < self.convergence_threshold:
                    self.convergence_epochs += 1
                else:
                    self.convergence_epochs = 0
            self.last_q_value = avg_q

            if q_gap > self.divergence_threshold:
                self.logger.warning("Large Q-value gap detected: %.4f", q_gap)

        if value is not None:
            _append_bounded(self.value_history, value, _HISTORY_LIMIT)

    def get_value_function_stats(self) -> ObjectMap:
        stats: ObjectMap = {
            "q_values_count": len(self.q_values_history),
            "value_count": len(self.value_history),
            "q_gaps_count": len(self.q_value_gaps),
            "convergence_epochs": self.convergence_epochs,
        }

        if self.q_values_history:
            stats.update(
                {
                    "q_values_mean": float(np.mean(self.q_values_history)),
                    "q_values_std": float(np.std(self.q_values_history)),
                    "q_values_latest": self.q_values_history[-1],
                }
            )

        if self.q_value_gaps:
            stats.update(
                {
                    "q_gap_mean": float(np.mean(self.q_value_gaps)),
                    "q_gap_max": float(np.max(self.q_value_gaps)),
                    "q_gap_latest": self.q_value_gaps[-1],
                }
            )

        if self.value_history:
            stats.update(
                {
                    "value_mean": float(np.mean(self.value_history)),
                    "value_std": float(np.std(self.value_history)),
                    "value_latest": self.value_history[-1],
                }
            )

        return stats

class SACTargetNetworkUpdater(NoOpMemoryOptimizedCallback):
    """Adaptive target-network update-rate controller for SAC."""

    def __init__(
        self,
        initial_tau: float = 0.005,
        min_tau: float = 0.001,
        max_tau: float = 0.01,
        adaptive: bool = True,
        stability_window: int = 50,
        update_frequency: int = 1,
    ):
        super().__init__()
        self.initial_tau = initial_tau
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.adaptive = adaptive
        self.stability_window = max(10, stability_window)
        self.update_frequency = max(1, update_frequency)

        self.current_tau = initial_tau
        self.q_loss_history: list[float] = []
        self.policy_loss_history: list[float] = []
        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.current_tau = self.initial_tau
        self.q_loss_history.clear()
        self.policy_loss_history.clear()
        self.logger.info(
            "SAC target updater initialized (tau=%.6f)",
            self.current_tau,
        )

    def on_training_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.logger.info(
            "SAC target updater finished (final_tau=%.6f)",
            self.current_tau,
        )

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if context.epoch % self.update_frequency != 0:
            return
        if logs is None:
            return

        q_loss = _as_float(logs.get("q_loss", logs.get("critic_loss")))
        policy_loss = _as_float(logs.get("policy_loss", logs.get("actor_loss")))

        if q_loss is not None:
            _append_bounded(self.q_loss_history, q_loss, self.stability_window)
        if policy_loss is not None:
            _append_bounded(self.policy_loss_history, policy_loss, self.stability_window)

        if self.adaptive and len(self.q_loss_history) >= 2:
            q_stability = float(np.std(self.q_loss_history[-10:]))
            policy_stability = (
                float(np.std(self.policy_loss_history[-10:]))
                if self.policy_loss_history
                else 0.0
            )
            avg_stability = (q_stability + policy_stability) / 2.0

            if avg_stability < 0.1:
                self.current_tau = min(self.current_tau * 1.1, self.max_tau)
            elif avg_stability > 1.0:
                self.current_tau = max(self.current_tau * 0.9, self.min_tau)
            else:
                self.current_tau = self.initial_tau

        logs["target_updated"] = True
        logs["current_tau"] = self.current_tau

    def get_current_tau(self) -> float:
        return self.current_tau

    def get_target_update_stats(self) -> ObjectMap:
        return {
            "current_tau": self.current_tau,
            "initial_tau": self.initial_tau,
            "min_tau": self.min_tau,
            "max_tau": self.max_tau,
            "adaptive": self.adaptive,
            "update_frequency": self.update_frequency,
            "q_stability_count": len(self.q_loss_history),
            "policy_stability_count": len(self.policy_loss_history),
        }

class SACExplorationMonitor(NoOpMemoryOptimizedCallback):
    """Monitor SAC exploration quality and entropy/reward coupling."""

    def __init__(self, monitor_frequency: int = 25):
        super().__init__()
        self.monitor_frequency = max(1, monitor_frequency)

        self.action_entropy_history: list[float] = []
        self.action_diversity_history: list[float] = []
        self.action_std_history: list[float] = []
        self.entropy_reward_correlation: list[tuple[float, float]] = []

        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.action_entropy_history.clear()
        self.action_diversity_history.clear()
        self.action_std_history.clear()
        self.entropy_reward_correlation.clear()

    def on_training_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        self.logger.info(
            "SAC exploration monitor finished (epochs=%s)",
            context.epoch,
        )

    def on_epoch_end(
        self, context: LearningContext, logs: ObjectMap | None = None
    ) -> None:
        if context.epoch % self.monitor_frequency != 0:
            return
        if logs is None:
            return

        action_entropy = _as_float(
            logs.get("action_entropy", logs.get("entropy", logs.get("policy_entropy")))
        )
        action_std = _as_float(logs.get("action_std", logs.get("policy_std")))
        reward = _as_float(logs.get("episode_reward", logs.get("reward")))

        if action_entropy is not None:
            _append_bounded(self.action_entropy_history, action_entropy, _HISTORY_LIMIT)
            if reward is not None:
                self.entropy_reward_correlation.append((action_entropy, reward))
                if len(self.entropy_reward_correlation) > _HISTORY_LIMIT:
                    del self.entropy_reward_correlation[: len(self.entropy_reward_correlation) - _HISTORY_LIMIT]

        if action_std is not None:
            _append_bounded(self.action_std_history, action_std, _HISTORY_LIMIT)

        diversity = self._calculate_action_diversity(logs.get("actions"))
        if diversity is not None:
            _append_bounded(self.action_diversity_history, diversity, _HISTORY_LIMIT)

    def _calculate_action_diversity(self, actions: object) -> float | None:
        if actions is None:
            return None

        try:
            action_array = np.asarray(actions)
        except Exception:
            return None

        if action_array.size == 0:
            return None

        if action_array.ndim == 1:
            return float(np.std(action_array))

        return float(np.mean(np.std(action_array, axis=0)))

    def get_exploration_stats(self) -> ObjectMap:
        stats: ObjectMap = {
            "entropy_count": len(self.action_entropy_history),
            "diversity_count": len(self.action_diversity_history),
            "correlation_count": len(self.entropy_reward_correlation),
            "action_std_count": len(self.action_std_history),
        }

        if self.action_entropy_history:
            stats.update(
                {
                    "entropy_mean": float(np.mean(self.action_entropy_history)),
                    "entropy_std": float(np.std(self.action_entropy_history)),
                    "entropy_latest": self.action_entropy_history[-1],
                }
            )

        if self.action_diversity_history:
            stats.update(
                {
                    "diversity_mean": float(np.mean(self.action_diversity_history)),
                    "diversity_std": float(np.std(self.action_diversity_history)),
                    "diversity_latest": self.action_diversity_history[-1],
                }
            )

        if self.action_std_history:
            stats.update(
                {
                    "action_std_mean": float(np.mean(self.action_std_history)),
                    "action_std_std": float(np.std(self.action_std_history)),
                    "action_std_latest": self.action_std_history[-1],
                }
            )

        if len(self.entropy_reward_correlation) > 10:
            entropies, rewards = zip(*self.entropy_reward_correlation[-50:])
            corr = float(np.corrcoef(entropies, rewards)[0, 1])
            if np.isfinite(corr):
                stats["entropy_reward_correlation"] = corr

        return stats

# Factory functions for easy instantiation

def create_sac_temperature_scheduler(**kwargs) -> SACTemperatureScheduler:
    """Create SAC temperature scheduler with default settings."""
    defaults: ObjectMap = {
        "initial_temp": 1.0,
        "min_temp": 0.1,
        "max_temp": 2.0,
        "decay_rate": 0.995,
        "adaptive": True,
    }
    defaults.update(kwargs)
    return SACTemperatureScheduler(**defaults)

def create_sac_value_monitor(**kwargs) -> SACValueFunctionMonitor:
    """Create SAC value function monitor with default settings."""
    defaults: ObjectMap = {
        "monitor_frequency": 50,
        "convergence_threshold": 0.01,
        "divergence_threshold": 10.0,
    }
    defaults.update(kwargs)
    return SACValueFunctionMonitor(**defaults)

def create_sac_target_updater(**kwargs) -> SACTargetNetworkUpdater:
    """Create SAC target network updater with default settings."""
    defaults: ObjectMap = {
        "initial_tau": 0.005,
        "min_tau": 0.001,
        "max_tau": 0.01,
        "adaptive": True,
        "stability_window": 50,
        "update_frequency": 1,
    }
    defaults.update(kwargs)
    return SACTargetNetworkUpdater(**defaults)

def create_sac_exploration_monitor(**kwargs) -> SACExplorationMonitor:
    """Create SAC exploration monitor with default settings."""
    defaults: ObjectMap = {"monitor_frequency": 25}
    defaults.update(kwargs)
    return SACExplorationMonitor(**defaults)

__all__ = [
    "SACTemperatureScheduler",
    "SACValueFunctionMonitor",
    "SACTargetNetworkUpdater",
    "SACExplorationMonitor",
    "create_sac_temperature_scheduler",
    "create_sac_value_monitor",
    "create_sac_target_updater",
    "create_sac_exploration_monitor",
]
