#!/usr/bin/env python3
"""
SAC-Specific Callbacks for Soft Actor-Critic Training.

This module provides SAC-specific callbacks that optimize training
for Soft Actor-Critic reinforcement learning algorithms.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ztb.training.callbacks.shared.base.learning_callback import (
    LearningContext,
    MemoryOptimizedCallback,
)


class SACTemperatureScheduler(MemoryOptimizedCallback):
    """
    SAC-specific temperature (entropy) scheduler.

    Dynamically adjusts the entropy temperature during training
    to balance exploration and exploitation.
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        min_temp: float = 0.1,
        max_temp: float = 2.0,
        decay_rate: float = 0.995,
        adaptive: bool = True,
    ):
        super().__init__()
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.decay_rate = decay_rate
        self.adaptive = adaptive
        self.current_temp = initial_temp

        # Adaptive parameters
        self.reward_history: List[float] = []
        self.entropy_history: List[float] = []
        self.window_size = 100

        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize temperature scheduling."""
        self.current_temp = self.initial_temp
        self.reward_history.clear()
        self.entropy_history.clear()
        self.logger.info(
            f"SAC temperature scheduler initialized with temp={self.current_temp}"
        )

    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Finalize temperature scheduling."""
        pass

    def on_epoch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update temperature based on training progress with enhanced error handling."""

        def _update_temperature():
            if logs is None:
                return

            # Extract relevant metrics
            reward = logs.get("episode_reward", logs.get("reward", 0))
            entropy = logs.get("entropy", logs.get("policy_entropy", 0))

            # Update history
            self.reward_history.append(reward)
            self.entropy_history.append(entropy)

            # Keep only recent history
            if len(self.reward_history) > self.window_size:
                self.reward_history.pop(0)
                self.entropy_history.pop(0)

            # Update temperature
            if self.adaptive:
                self._adaptive_update(context.epoch, logs)
            else:
                self._decay_update(context.epoch)

            # Ensure bounds
            self.current_temp = np.clip(self.current_temp, self.min_temp, self.max_temp)

            # Add current temperature to logs
            if logs is not None:
                logs["temperature"] = self.current_temp

            # Log temperature update
            if context.epoch % 10 == 0:  # Log every 10 epochs
                self.logger.debug(f"SAC temperature updated: {self.current_temp:.4f}")

        self.safe_execute(_update_temperature)

    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of each batch."""
        pass
        """Update temperature based on training progress."""
        if logs is None:
            return

        # Extract relevant metrics
        reward = logs.get("episode_reward", logs.get("reward", 0))
        entropy = logs.get("entropy", logs.get("policy_entropy", 0))

        # Update history
        self.reward_history.append(reward)
        self.entropy_history.append(entropy)

        # Keep only recent history
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            self.entropy_history.pop(0)

        # Update temperature
        if self.adaptive:
            self._adaptive_update(context.epoch, logs)
        else:
            self._decay_update(context.epoch)

        # Ensure bounds
        self.current_temp = np.clip(self.current_temp, self.min_temp, self.max_temp)

        # Add current temperature to logs
        if logs is not None:
            logs["temperature"] = self.current_temp

        # Log temperature update
        if context.epoch % 10 == 0:  # Log every 10 epochs
            self.logger.debug(f"SAC temperature updated: {self.current_temp:.4f}")

    def _adaptive_update(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Adaptive temperature update based on training dynamics."""
        if len(self.reward_history) < 10:
            return  # Need minimum history

        # Calculate recent performance metrics
        recent_rewards = self.reward_history[-10:]
        recent_entropies = self.entropy_history[-10:]

        avg_reward = np.mean(recent_rewards)
        avg_entropy = np.mean(recent_entropies)
        reward_std = np.std(recent_rewards)

        # Adaptive logic:
        # - If rewards are improving but entropy is too low -> increase temp
        # - If rewards are plateauing and entropy is high -> decrease temp
        # - If rewards are volatile -> stabilize temp

        reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]

        if reward_trend > 0.01 and avg_entropy < 0.5:  # Improving but low entropy
            self.current_temp *= 1.05  # Increase exploration
        elif reward_trend < -0.01 and avg_entropy > 1.0:  # Declining and high entropy
            self.current_temp *= 0.95  # Reduce exploration
        elif reward_std > 0.5:  # High volatility
            self.current_temp = np.clip(
                self.current_temp * 0.98, self.min_temp, self.max_temp
            )

    def _decay_update(self, epoch: int) -> None:
        """Simple exponential decay update."""
        self.current_temp *= self.decay_rate

    def get_current_temperature(self) -> float:
        """Get current temperature value."""
        return self.current_temp

    def get_temperature_stats(self) -> Dict[str, Any]:
        """Get temperature scheduling statistics."""
        return {
            "current_temp": self.current_temp,
            "initial_temp": self.initial_temp,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "adaptive": self.adaptive,
            "history_size": len(self.reward_history),
        }


class SACValueFunctionMonitor(MemoryOptimizedCallback):
    """
    SAC-specific value function monitoring.

    Monitors Q-value functions and value function for convergence
    and potential issues during training.
    """

    def __init__(
        self,
        monitor_frequency: int = 50,
        convergence_threshold: float = 0.01,
        divergence_threshold: float = 10.0,
    ):
        super().__init__()
        self.monitor_frequency = monitor_frequency
        self.convergence_threshold = convergence_threshold
        self.divergence_threshold = divergence_threshold

        # Monitoring data
        self.q_values_history: List[float] = []
        self.value_history: List[float] = []
        self.q_value_gaps: List[float] = []  # Difference between Q1 and Q2

        # Convergence tracking
        self.convergence_epochs = 0
        self.last_q_values = None

        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize value function monitoring."""
        self.q_values_history.clear()
        self.value_history.clear()
        self.q_value_gaps.clear()
        self.convergence_epochs = 0
        self.last_q_values = None
        self.logger.info("SAC value function monitoring initialized")

    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Finalize value function monitoring."""
        self.logger.info(
            f"SAC value function monitoring completed. Convergence epochs: {self.convergence_epochs}"
        )

    def on_epoch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor value functions at specified frequency."""
        if context.epoch % self.monitor_frequency != 0:
            return

        if logs is None:
            return

        # Extract value function metrics
        q1_value = logs.get("q1_value", logs.get("critic_1_value"))
        q2_value = logs.get("q2_value", logs.get("critic_2_value"))
        value = logs.get("value", logs.get("value_function", logs.get("value_mean")))

        if q1_value is not None and q2_value is not None:
            avg_q = (q1_value + q2_value) / 2
            q_gap = abs(q1_value - q2_value)

            self.q_values_history.append(avg_q)
            self.q_value_gaps.append(q_gap)

            # Check for convergence
            if self.last_q_values is not None:
                q_change = abs(avg_q - self.last_q_values)
                if q_change < self.convergence_threshold:
                    self.convergence_epochs += 1
                else:
                    self.convergence_epochs = 0

            self.last_q_values = avg_q

            # Check for divergence
            if q_gap > self.divergence_threshold:
                self.logger.warning(f"Large Q-value gap detected: {q_gap:.4f}")

        if value is not None:
            self.value_history.append(value)

        # Keep history bounded
        max_history = 1000
        if len(self.q_values_history) > max_history:
            self.q_values_history.pop(0)
        if len(self.value_history) > max_history:
            self.value_history.pop(0)
        if len(self.q_value_gaps) > max_history:
            self.q_value_gaps.pop(0)

    def get_value_function_stats(self) -> Dict[str, Any]:
        """Get value function monitoring statistics."""
        stats = {
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

    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of each batch."""
        pass


class SACTargetNetworkUpdater(MemoryOptimizedCallback):
    """
    SAC-specific target network update manager.

    Manages soft updates of target networks with adaptive update rates
    based on training stability.
    """

    def __init__(
        self,
        initial_tau: float = 0.005,
        min_tau: float = 0.001,
        max_tau: float = 0.01,
        adaptive: bool = True,
        stability_window: int = 50,
    ):
        super().__init__()
        self.initial_tau = initial_tau
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.adaptive = adaptive
        self.stability_window = stability_window
        self.current_tau = initial_tau

        # Stability tracking
        self.q_value_stability: List[float] = []
        self.policy_loss_stability: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize target network updater."""
        self.current_tau = self.initial_tau
        self.q_value_stability.clear()
        self.policy_loss_stability.clear()
        self.logger.info(
            f"SAC target network updater initialized with tau={self.current_tau}"
        )

    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Finalize target network updater."""
        self.logger.info(
            f"SAC target network updater completed. Final tau: {self.current_tau}"
        )

    def on_epoch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update target network tau based on training stability."""
        if not self.adaptive or logs is None:
            return

        # Track stability metrics
        q_loss = logs.get("q_loss", logs.get("critic_loss"))
        policy_loss = logs.get("policy_loss", logs.get("actor_loss"))

        if q_loss is not None:
            self.q_value_stability.append(q_loss)
        if policy_loss is not None:
            self.policy_loss_stability.append(policy_loss)

        # Keep stability history bounded
        if len(self.q_value_stability) > self.stability_window:
            self.q_value_stability.pop(0)
        if len(self.policy_loss_stability) > self.stability_window:
            self.policy_loss_stability.pop(0)

        # Update tau based on stability
        if len(self.q_value_stability) >= 2:
            q_stability = np.std(self.q_value_stability[-10:])
            policy_stability = (
                np.std(self.policy_loss_stability[-10:])
                if self.policy_loss_stability
                else 0
            )

            # Adaptive tau logic:
            # - High stability (low variance) -> faster updates (higher tau)
            # - Low stability (high variance) -> slower updates (lower tau)
            avg_stability = (q_stability + policy_stability) / 2

            if avg_stability < 0.1:  # Very stable
                self.current_tau = min(self.current_tau * 1.1, self.max_tau)
            elif avg_stability > 1.0:  # Unstable
                self.current_tau = max(self.current_tau * 0.9, self.min_tau)
            else:  # Moderate stability
                self.current_tau = self.initial_tau

        # Log target network update only if adaptive updates occurred
        if logs is not None and len(self.q_value_stability) >= 2:
            logs["target_updated"] = True
            logs["current_tau"] = self.current_tau

    def get_current_tau(self) -> float:
        """Get current target network update rate."""
        return self.current_tau

    def get_target_update_stats(self) -> Dict[str, Any]:
        """Get target network update statistics."""
        return {
            "current_tau": self.current_tau,
            "initial_tau": self.initial_tau,
            "min_tau": self.min_tau,
            "max_tau": self.max_tau,
            "adaptive": self.adaptive,
            "q_stability_count": len(self.q_value_stability),
            "policy_stability_count": len(self.policy_loss_stability),
        }

    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of each batch."""
        pass


class SACExplorationMonitor(MemoryOptimizedCallback):
    """
    SAC-specific exploration monitoring.

    Monitors exploration behavior and provides insights into
    the balance between exploration and exploitation.
    """

    def __init__(self, monitor_frequency: int = 25):
        super().__init__()
        self.monitor_frequency = monitor_frequency

        # Exploration metrics
        self.action_entropy_history: List[float] = []
        self.state_visit_counts: Dict[str, int] = {}
        self.action_diversity_history: List[float] = []
        self.action_std_history: List[float] = []

        # Performance correlation
        self.entropy_reward_correlation: List[float] = []

        self.logger = logging.getLogger(__name__)

    def on_training_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize exploration monitoring."""
        self.action_entropy_history.clear()
        self.state_visit_counts.clear()
        self.action_diversity_history.clear()
        self.action_std_history.clear()
        self.entropy_reward_correlation.clear()
        self.logger.info("SAC exploration monitoring initialized")

    def on_training_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Finalize exploration monitoring."""
        self.logger.info(
            f"SAC exploration monitoring completed. Total epochs: {context.epoch}"
        )

    def on_epoch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Monitor exploration metrics."""
        if context.epoch % self.monitor_frequency != 0:
            return

        if logs is None:
            return

        # Extract exploration metrics
        action_entropy = logs.get("action_entropy", logs.get("policy_entropy"))
        action_std = logs.get("action_std")
        reward = logs.get("episode_reward", logs.get("reward"))

        if action_entropy is not None:
            self.action_entropy_history.append(action_entropy)

            # Track state visits (simplified - would need actual state data)
            if reward is not None:
                self.entropy_reward_correlation.append((action_entropy, reward))

        if action_std is not None:
            self.action_std_history.append(action_std)

        # Calculate action diversity (simplified)
        if "actions" in logs:
            actions = logs["actions"]
            if isinstance(actions, (list, np.ndarray)):
                diversity = self._calculate_action_diversity(actions)
                self.action_diversity_history.append(diversity)

        # Keep history bounded
        max_history = 500
        for history in [
            self.action_entropy_history,
            self.action_diversity_history,
            self.action_std_history,
        ]:
            if len(history) > max_history:
                history.pop(0)

        if len(self.entropy_reward_correlation) > max_history:
            self.entropy_reward_correlation.pop(0)

    def _calculate_action_diversity(self, actions) -> float:
        """Calculate action diversity metric."""
        if isinstance(actions, list):
            actions = np.array(actions)

        if len(actions.shape) == 1:
            # 1D actions
            return float(np.std(actions))
        else:
            # Multi-dimensional actions
            return float(np.mean(np.std(actions, axis=0)))

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get exploration monitoring statistics."""
        stats = {
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

        # Calculate entropy-reward correlation
        if len(self.entropy_reward_correlation) > 10:
            entropies, rewards = zip(
                *self.entropy_reward_correlation[-50:]
            )  # Last 50 points
            correlation = np.corrcoef(entropies, rewards)[0, 1]
            stats["entropy_reward_correlation"] = float(correlation)

        return stats

    def on_batch_start(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(
        self, context: LearningContext, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called at the end of each batch."""
        pass


# Factory functions for easy instantiation
def create_sac_temperature_scheduler(**kwargs) -> SACTemperatureScheduler:
    """Create SAC temperature scheduler with default settings."""
    defaults = {
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
    defaults = {
        "monitor_frequency": 50,
        "convergence_threshold": 0.01,
        "divergence_threshold": 10.0,
    }
    defaults.update(kwargs)
    return SACValueFunctionMonitor(**defaults)


def create_sac_target_updater(**kwargs) -> SACTargetNetworkUpdater:
    """Create SAC target network updater with default settings."""
    defaults = {
        "initial_tau": 0.005,
        "min_tau": 0.001,
        "max_tau": 0.01,
        "adaptive": True,
        "stability_window": 50,
    }
    defaults.update(kwargs)
    return SACTargetNetworkUpdater(**defaults)


def create_sac_exploration_monitor(**kwargs) -> SACExplorationMonitor:
    """Create SAC exploration monitor with default settings."""
    defaults = {"monitor_frequency": 25}
    defaults.update(kwargs)
    return SACExplorationMonitor(**defaults)
