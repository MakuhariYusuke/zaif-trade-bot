#!/usr/bin/env python3
"""
Training callbacks for unified training system.
"""

import logging
import time
from typing import Any, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ztb.trading.constants import SAC_CONTINUOUS_THRESHOLD, SAC_CONTINUOUS_THRESHOLD_NEG
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.training.sac_v430_training_optimizations import DynamicLRScheduler
from ztb.training.system_optimizer import SystemOptimizer


class TrainingProgressCallback(BaseCallback):
    """Enhanced callback for monitoring training progress and action distribution."""

    def __init__(
        self,
        check_freq: int = 1000,
        verbose: int = 1,
        system_optimizer: Optional[SystemOptimizer] = None,
        metrics_csv_writer: Optional[Any] = None,
        lr_scheduler: Optional[DynamicLRScheduler] = None,
        early_stopping: Optional[Any] = None,
        trainer_ref: Optional[Any] = None,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.system_optimizer = system_optimizer
        self.metrics_csv_writer = metrics_csv_writer
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.trainer_ref = trainer_ref
        self.trainer = trainer_ref  # Set trainer attribute for compatibility

        # Provide explicit type annotations so mypy can reason about these lists
        self.continuous_actions: List[float] = []
        self.discrete_actions: List[int] = []
        self.reward_history: List[float] = []
        self.episode_rewards: List[float] = []
        self.start_time = time.time()
        self.last_log_time = self.start_time

        # Additional metrics tracking
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.ent_coefs: List[float] = []
        self.learning_rates: List[float] = []
        self.episode_lengths: List[int] = []

        # Per-regime action tracking for debugging market regime adaptation
        self.regime_action_counts: dict = {}  # regime -> [hold_count, buy_count, sell_count]

        # Initialize optimizer feature tracker from trainer if available
        self.optimizer_tracker = None
        if (
            trainer_ref
            and hasattr(trainer_ref, "optimizer_tracker")
            and trainer_ref.optimizer_tracker is not None
        ):
            self.optimizer_tracker = trainer_ref.optimizer_tracker
        # Note: If optimizer_tracker is None, optimizer features are disabled for this training run

        # Early stopping parameters
        self.early_stopping_enabled = False
        self.early_stopping_patience = 10
        self.early_stopping_min_delta = 0.001
        self.best_reward = -float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False

        # Initialize early stopping if configured
        if early_stopping and isinstance(early_stopping, dict):
            self.early_stopping_enabled = early_stopping.get("enabled", False)
            self.early_stopping_patience = early_stopping.get("patience", 10)
            self.early_stopping_min_delta = early_stopping.get("min_delta", 0.001)

    def _on_step(self) -> bool:
        # Record action taken (handle both PPO discrete and SAC continuous actions)
        try:
            actions = self.locals.get("actions")
            if actions is not None:
                action_value = actions[0]
                if isinstance(action_value, np.ndarray):
                    action_value = action_value.item()

                # Check if this is PPO (discrete actions) or SAC (continuous actions)
                is_ppo = (
                    hasattr(self.trainer, "policy")
                    and hasattr(self.trainer.policy, "action_space")
                    and hasattr(self.trainer.policy.action_space, "n")
                    and self.trainer.policy.action_space.n == 3
                )

                if is_ppo:
                    # PPO: action is already discrete (0=HOLD, 1=BUY, 2=SELL)
                    discrete_action = int(action_value)
                    self.discrete_actions.append(discrete_action)
                    # Store as continuous for compatibility (map to -1, 0, 1)
                    continuous_equivalent = {0: 0.0, 1: 1.0, 2: -1.0}[discrete_action]
                    self.continuous_actions.append(continuous_equivalent)
                    logging.debug(
                        f"PPO action {action_value} -> discrete {discrete_action}"
                    )
                else:
                    # SAC: continuous action needs conversion
                    self.continuous_actions.append(action_value)
                    discrete_action = self._continuous_to_discrete_action(action_value)
                    self.discrete_actions.append(discrete_action)
                    logging.debug(
                        f"SAC action {action_value:.6f} -> discrete {discrete_action}"
                    )
            else:
                logging.debug("Actions not available - actions: %s", actions)
        except Exception as e:
            if hasattr(self, "logger") and self.logger is not None:
                logging.warning(f"Failed to record action: {e}")
            else:
                logging.warning(f"Failed to record action: {e}")

        # Record reward
        try:
            # locals is a dict-like mapping provided by SB3; guard access
            rewards = (
                self.locals.get("rewards") if isinstance(self.locals, dict) else None
            )
            if rewards:
                reward = rewards[0]
                self.reward_history.append(reward)

                # Record per-regime action counts for debugging
                try:
                    infos = self.locals.get("infos")
                    if infos and len(infos) > 0:
                        info = infos[0]
                        if isinstance(info, dict) and "market_regime" in info:
                            regime = info["market_regime"]
                            if regime not in self.regime_action_counts:
                                self.regime_action_counts[regime] = [
                                    0,
                                    0,
                                    0,
                                ]  # [HOLD, BUY, SELL]
                            if discrete_action >= 0 and discrete_action <= 2:
                                self.regime_action_counts[regime][discrete_action] += 1
                except Exception as e:
                    logging.debug(f"Failed to record regime action: {e}")

        except Exception as e:
            logging.warning("Failed to record reward: %s", e)

        # Record additional SAC-specific metrics
        try:
            # Extract SAC-specific metrics from the model/locals
            if hasattr(self, "model") and self.model is not None:
                # Actor loss
                if hasattr(self.model, "logger") and self.model.logger:
                    actor_loss = getattr(self.model.logger, "name_to_value", {}).get(
                        "train/actor_loss", None
                    )
                    if actor_loss is not None:
                        self.actor_losses.append(actor_loss)

                # Critic loss
                critic_loss = getattr(self.model.logger, "name_to_value", {}).get(
                    "train/critic_loss", None
                )
                if critic_loss is not None:
                    self.critic_losses.append(critic_loss)

                # Entropy coefficient
                ent_coef = getattr(self.model.logger, "name_to_value", {}).get(
                    "train/ent_coef", None
                )
                if ent_coef is not None:
                    self.ent_coefs.append(ent_coef)

                # Learning rate
                if hasattr(self.model, "policy") and hasattr(
                    self.model.policy, "optimizer"
                ):
                    lr = self.model.policy.optimizer.param_groups[0]["lr"]
                    self.learning_rates.append(lr)

        except Exception as e:
            # Don't fail training if metrics collection fails
            if self.verbose > 1:
                logging.debug(f"Failed to collect detailed metrics: {e}")

        # Apply dynamic LR scheduling
        if self.lr_scheduler and self.n_calls % self.check_freq == 0:
            try:
                # Use recent reward as loss proxy for LR scheduling
                recent_rewards = (
                    self.reward_history[-100:] if self.reward_history else [0]
                )
                avg_reward = (
                    sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                )
                # Convert reward to loss-like metric (negative reward)
                loss_proxy = -avg_reward
                lr_info = self.lr_scheduler.step(loss_proxy)
                if lr_info.get("lr_changed", False):
                    logging.info(
                        f"Learning rate adjusted: {lr_info['lr']:.6f} ({lr_info['action']})"
                    )
            except Exception as e:
                logging.debug(f"LR scheduling failed: {e}")

        # Log progress
        if self.n_calls % self.check_freq == 0:
            self._log_progress()

        # Update optimizer features for feature engineering (if enabled)
        if self.optimizer_tracker is not None:
            try:
                # Get current training metrics for optimizer features
                current_lr = None
                current_actor_loss = None
                current_critic_loss = None
                current_ent_coef = None

                # Extract current learning rate
                if hasattr(self.model, "policy") and hasattr(
                    self.model.policy, "optimizer"
                ):
                    current_lr = self.model.policy.optimizer.param_groups[0]["lr"]

                # Extract current losses from logger if available
                if hasattr(self.model, "logger") and self.model.logger:
                    logger_values = getattr(self.model.logger, "name_to_value", {})
                    current_actor_loss = logger_values.get("train/actor_loss")
                    current_critic_loss = logger_values.get("train/critic_loss")
                    current_ent_coef = logger_values.get("train/ent_coef")

                # Update optimizer features with current training state
                self.optimizer_tracker.update_optimizer_features(
                    step=self.n_calls,
                    learning_rate=current_lr,
                    actor_loss=current_actor_loss,
                    critic_loss=current_critic_loss,
                    entropy_coef=current_ent_coef,
                    reward=self.reward_history[-1] if self.reward_history else None,
                )
            except Exception as e:
                if self.verbose > 1:
                    logging.debug(f"Failed to update optimizer features: {e}")

        # Check early stopping conditions
        if self.early_stopping_enabled and self.n_calls % self.check_freq == 0:
            self._check_early_stopping()

        return True

    def _continuous_to_discrete_action(
        self,
        continuous_action: float,
        buy_threshold: float = SAC_CONTINUOUS_THRESHOLD,
        sell_threshold: float = SAC_CONTINUOUS_THRESHOLD_NEG,
    ) -> int:
        """Convert continuous action (-1 to 1) to discrete action (0=HOLD, 1=BUY, 2=SELL)."""
        # Use the centralized continuous_to_discrete_action function for consistency
        # continuous_to_discrete_action may be untyped to mypy; cast to int explicitly
        # Use centralized util; ensure input is a float
        try:
            ca = float(continuous_action)
        except Exception:
            ca = 0.0

        result = continuous_to_discrete_action(ca)
        try:
            return int(result)
        except Exception:
            # Fallback to HOLD when unexpected
            return 0

    def _log_progress(self) -> None:
        """Log training progress and action distribution."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0

        # Always show progress, even if no actions recorded yet
        if self.discrete_actions:
            total_actions = len(self.discrete_actions)
            # Shift discrete actions to non-negative range for bincount: SELL(-1)->0, HOLD(0)->1, BUY(1)->2
            shifted_actions = np.array(self.discrete_actions) + 1
            discrete_counts = np.bincount(shifted_actions, minlength=3)

            action_dist = {
                "SELL": discrete_counts[0] / total_actions,
                "HOLD": discrete_counts[1] / total_actions,
                "BUY": discrete_counts[2] / total_actions,
            }

            # Prefer the callback's logger when available
            # Prefer instance logger
            logging.info(
                "Step %6d | Elapsed: %6.1fs | SPS: %5.1f | HOLD: %.1f%% | BUY: %.1f%% | SELL: %.1f%% | Rewards: %d recorded",
                self.n_calls,
                elapsed,
                steps_per_sec,
                action_dist["HOLD"] * 100.0,
                action_dist["BUY"] * 100.0,
                action_dist["SELL"] * 100.0,
                len(self.reward_history),
            )

            # Log continuous action statistics for SAC models
            if self.continuous_actions and len(self.continuous_actions) > 100:
                recent_continuous = self.continuous_actions[-1000:]  # Last 1000 actions
                mean_action = np.mean(recent_continuous)
                std_action = np.std(recent_continuous)
                min_action = np.min(recent_continuous)
                max_action = np.max(recent_continuous)
                logging.info(
                    "Continuous Action Stats - Mean: %.3f, Std: %.3f, Min: %.3f, Max: %.3f, Range: %.3f",
                    mean_action,
                    std_action,
                    min_action,
                    max_action,
                    max_action - min_action,
                )

                # Log action distribution in ranges
                ranges = [
                    (-1.0, -0.8),
                    (-0.8, -0.6),
                    (-0.6, -0.4),
                    (-0.4, -0.2),
                    (-0.2, 0.0),
                    (0.0, 0.2),
                    (0.2, 0.4),
                    (0.4, 0.6),
                    (0.6, 0.8),
                    (0.8, 1.0),
                ]
                range_counts = []
                for r_min, r_max in ranges:
                    count = sum(1 for x in recent_continuous if r_min <= x < r_max)
                    range_counts.append(count)
                total_in_ranges = sum(range_counts)
                if total_in_ranges > 0:
                    range_dist = [
                        count / total_in_ranges * 100 for count in range_counts
                    ]
                    logging.info(
                        "Action Range Distribution (%%): [-1,-0.8]: %.1f, [-0.8,-0.6]: %.1f, [-0.6,-0.4]: %.1f, [-0.4,-0.2]: %.1f, [-0.2,0]: %.1f, [0,0.2]: %.1f, [0.2,0.4]: %.1f, [0.4,0.6]: %.1f, [0.6,0.8]: %.1f, [0.8,1]: %.1f",
                        *range_dist,
                    )

            # Log per-regime action distribution for debugging
            if self.regime_action_counts:
                regime_info = []
                for regime, counts in self.regime_action_counts.items():
                    total_regime_actions = sum(counts)
                    if total_regime_actions > 0:
                        hold_pct = counts[0] / total_regime_actions * 100
                        buy_pct = counts[1] / total_regime_actions * 100
                        sell_pct = counts[2] / total_regime_actions * 100
                        regime_info.append(
                            f"{regime}: H{hold_pct:.1f}%/B{buy_pct:.1f}%/S{sell_pct:.1f}%"
                        )
                if regime_info:
                    logging.info(
                        "Regime action distributions: %s", " | ".join(regime_info)
                    )
        else:
            # Show progress even when no actions recorded yet
            logging.info(
                "Step %6d | Elapsed: %6.1fs | SPS: %5.1f | No actions recorded yet | Rewards: %d recorded",
                self.n_calls,
                elapsed,
                steps_per_sec,
                len(self.reward_history),
            )

        # Apply system optimizations during training
        if self.system_optimizer:
            try:
                with self.system_optimizer.optimize_training_step(
                    f"training_step_{self.n_calls}"
                ):
                    pass  # System optimization is applied in the context manager
            except Exception as e:
                logging.warning("System optimization failed: %s", e)

        # Log detailed metrics to CSV
        if self.metrics_csv_writer:
            try:
                metrics = {
                    "episode_reward": self.episode_rewards[-1]
                    if self.episode_rewards
                    else 0,
                    "actor_loss": self.actor_losses[-1] if self.actor_losses else None,
                    "critic_loss": self.critic_losses[-1]
                    if self.critic_losses
                    else None,
                    "ent_coef": self.ent_coefs[-1] if self.ent_coefs else None,
                    "learning_rate": self.learning_rates[-1]
                    if self.learning_rates
                    else None,
                    "fps": steps_per_sec,
                    "total_episodes": len(self.episode_rewards),
                }
                # Call the SACTrainer's CSV logging method if available
                if hasattr(self, "trainer_ref") and self.trainer_ref:
                    self.trainer_ref._log_metrics_to_csv(self.n_calls, metrics)
            except Exception as e:
                logging.debug(f"CSV metrics logging failed: {e}")

        # Log to TensorBoard if model has logger
        try:
            if (
                hasattr(self, "model")
                and self.model
                and hasattr(self.model, "logger")
                and self.model.logger
            ):
                tb_writer = (
                    self.model.logger.output_formats[0]
                    if self.model.logger.output_formats
                    else None
                )
                if tb_writer and hasattr(tb_writer, "writer"):
                    # Log action distribution
                    if self.discrete_actions:
                        total_actions = len(self.discrete_actions)
                        shifted_actions = np.array(self.discrete_actions) + 1
                        discrete_counts = np.bincount(shifted_actions, minlength=3)
                        tb_writer.writer.add_scalar(
                            "actions/hold_ratio",
                            discrete_counts[1] / total_actions,
                            self.n_calls,
                        )
                        tb_writer.writer.add_scalar(
                            "actions/buy_ratio",
                            discrete_counts[2] / total_actions,
                            self.n_calls,
                        )
                        tb_writer.writer.add_scalar(
                            "actions/sell_ratio",
                            discrete_counts[0] / total_actions,
                            self.n_calls,
                        )

                    # Log recent metrics
                    if self.actor_losses:
                        tb_writer.writer.add_scalar(
                            "train/actor_loss_detailed",
                            self.actor_losses[-1],
                            self.n_calls,
                        )
                    if self.critic_losses:
                        tb_writer.writer.add_scalar(
                            "train/critic_loss_detailed",
                            self.critic_losses[-1],
                            self.n_calls,
                        )
                    if self.ent_coefs:
                        tb_writer.writer.add_scalar(
                            "train/ent_coef_detailed", self.ent_coefs[-1], self.n_calls
                        )
                    if self.learning_rates:
                        tb_writer.writer.add_scalar(
                            "train/learning_rate_detailed",
                            self.learning_rates[-1],
                            self.n_calls,
                        )

                    tb_writer.writer.flush()
        except Exception as e:
            logging.debug(f"TensorBoard logging failed: {e}")

    def _check_early_stopping(self):
        """Check if early stopping conditions are met."""
        if not self.early_stopping_enabled or not self.reward_history:
            return

        # Calculate recent average reward
        recent_rewards = self.reward_history[-100:]  # Last 100 rewards
        if len(recent_rewards) < 10:  # Need minimum samples
            return

        current_avg_reward = sum(recent_rewards) / len(recent_rewards)

        # Check if improvement is significant
        if current_avg_reward > self.best_reward + self.early_stopping_min_delta:
            self.best_reward = current_avg_reward
            self.early_stopping_counter = 0
            logging.info(f"Early stopping: New best reward {current_avg_reward:.4f}")
        else:
            self.early_stopping_counter += 1
            logging.debug(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")

        # Trigger early stopping if patience exceeded
        if self.early_stopping_counter >= self.early_stopping_patience:
            self.early_stopping_triggered = True
            logging.info(f"Early stopping triggered after {self.early_stopping_patience} steps without improvement")
            # Note: We can't directly stop training from callback, but we can set a flag
            # The training loop should check this flag
