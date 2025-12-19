#!/usr/bin/env python3
"""
Training callbacks for unified training system.
"""

import logging
import time
from typing import Any, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ztb.metrics.metrics import kurtosis, skewness
from ztb.trading.constants import (
    MULTIPLIER_INDEX_BUY,
    MULTIPLIER_INDEX_HOLD,
    MULTIPLIER_INDEX_SELL,
    SAC_CONTINUOUS_THRESHOLD,
    SAC_CONTINUOUS_THRESHOLD_NEG,
    get_action_count_index,
)
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.training.constants import ENV_EVAL_FREQUENCY
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
        checkpoint_manager: Optional[Any] = None,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.system_optimizer = system_optimizer
        self.metrics_csv_writer = metrics_csv_writer
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.trainer_ref = trainer_ref
        self.trainer = trainer_ref  # Set trainer attribute for compatibility
        self.checkpoint_manager = checkpoint_manager

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
        self.regime_action_counts: dict = {}  # regime -> [BUY, SELL, HOLD]

        # Reward components tracking for AB analysis
        self.reward_components_history: List[dict] = []

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
        self.best_reward = -float("inf")
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False

        # Initialize early stopping if configured
        if early_stopping and isinstance(early_stopping, dict):
            self.early_stopping_enabled = early_stopping.get("enabled", False)
            self.early_stopping_patience = early_stopping.get("patience", 10)
            self.early_stopping_min_delta = early_stopping.get("min_delta", 0.001)

    def _on_step(self) -> bool:
        # Initialize variables for the entire method scope
        discrete_action = 0  # Default HOLD
        action_value = 0.0  # Default value

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
                    ) if self.n_calls % 50 == 0 else None
                else:
                    # SAC: continuous action needs conversion
                    self.continuous_actions.append(action_value)
                    discrete_action = self._continuous_to_discrete_action(action_value)
                    self.discrete_actions.append(discrete_action)
                    logging.debug(
                        f"SAC action {action_value:.6f} -> discrete {discrete_action}"
                    ) if self.n_calls % 50 == 0 else None
            else:
                logging.debug("Actions not available - actions: %s", actions)
        except Exception as e:
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

                # DEBUG: Log detailed reward and state information
                try:
                    infos = self.locals.get("infos")
                    if infos and len(infos) > 0:
                        info = infos[0]
                        if isinstance(info, dict):
                            portfolio_value = info.get("portfolio_value", 0)
                            position = info.get("position", 0)
                            pnl = info.get("pnl", 0)
                            info.get("market_regime", "unknown")

                            # Collect reward_components if available for AB analysis
                            if "reward_components" in info:
                                self.reward_components_history.append(
                                    info["reward_components"].copy()
                                )

                            # Compact INFO log with key metrics (every 10 steps to reduce verbosity)
                            if self.n_calls % 10 == 0:
                                logging.info(
                                    f"Step {self.n_calls}: Action={discrete_action}({action_value:.3f}) | "
                                    f"Reward={reward:.4f} | PnL={pnl:.2f} | Portfolio={portfolio_value:.2f} | "
                                    f"Position={position:.4f}"
                                )
                except Exception as e:
                    logging.debug(f"Failed to log detailed reward info: {e}")

                # Record per-regime action counts for debugging
                try:
                    infos = self.locals.get("infos")
                    if infos and len(infos) > 0:
                        info = infos[0]
                        if isinstance(info, dict):
                            if "market_regime" in info:
                                regime = info["market_regime"]
                                if regime not in self.regime_action_counts:
                                    logging.warning(
                                        f"DEBUG: New regime detected: {regime}"
                                    )
                                    self.regime_action_counts[regime] = [
                                        0,
                                        0,
                                        0,
                                    ]  # [BUY, SELL, HOLD]
                                if discrete_action >= -1 and discrete_action <= 2:
                                    self.regime_action_counts[regime][
                                        get_action_count_index(discrete_action)
                                    ] += 1
                            else:
                                logging.warning(
                                    f"DEBUG: market_regime MISSING. Keys: {list(info.keys())}"
                                )
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

        # Periodic checkpoint saving (Week 9-10 requirement: save every 1000 steps)
        if (
            self.checkpoint_manager is not None
            and self.checkpoint_manager.should_checkpoint(self.n_calls)
        ):
            try:
                # Get current training metrics
                current_metrics = {}
                if hasattr(self.model, "logger") and self.model.logger:
                    logger_values = getattr(self.model.logger, "name_to_value", {})
                    current_metrics.update(
                        {
                            "actor_loss": logger_values.get("train/actor_loss"),
                            "critic_loss": logger_values.get("train/critic_loss"),
                            "ent_coef": logger_values.get("train/ent_coef"),
                            "learning_rate": logger_values.get("train/learning_rate"),
                        }
                    )

                # Save checkpoint with current training state
                self.checkpoint_manager.save(
                    step=self.n_calls,
                    model=self.model,
                    metrics=current_metrics,
                    extra={
                        "training_time": time.time() - self.start_time,
                        "episodes_completed": len(self.episode_rewards),
                    },
                )

                if self.verbose > 0:
                    logging.info(f"Checkpoint saved at step {self.n_calls}")

            except Exception as e:
                logging.warning(
                    f"Failed to save checkpoint at step {self.n_calls}: {e}"
                )

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

        # DEBUG: Log detailed training metrics
        if (
            self.actor_losses
            or self.critic_losses
            or self.ent_coefs
            or self.learning_rates
        ):
            recent_actor = self.actor_losses[-10:] if self.actor_losses else []
            recent_critic = self.critic_losses[-10:] if self.critic_losses else []
            recent_ent = self.ent_coefs[-10:] if self.ent_coefs else []
            recent_lr = self.learning_rates[-10:] if self.learning_rates else []

            avg_actor = sum(recent_actor) / len(recent_actor) if recent_actor else 0
            avg_critic = sum(recent_critic) / len(recent_critic) if recent_critic else 0
            avg_ent = sum(recent_ent) / len(recent_ent) if recent_ent else 0
            current_lr = recent_lr[-1] if recent_lr else 0

            logging.debug(
                f"Training metrics [Step {self.n_calls}]: ActorLoss={avg_actor:.4f} | "
                f"CriticLoss={avg_critic:.4f} | EntCoef={avg_ent:.4f} | LR={current_lr:.6f} | "
                f"SPS={steps_per_sec:.1f}"
            )

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

            # Save action distribution to TrainingReporter (if configured) at eval checkpoints
            try:
                if hasattr(self, "trainer_ref") and self.trainer_ref:
                    # Trainer config may specify eval frequency
                    try:
                        total_steps = int(
                            self.trainer_ref.config.get("training", {}).get(
                                "total_timesteps", 0
                            )
                        )
                        eval_freq = int(
                            self.trainer_ref.config.get("training", {}).get(
                                "eval_freq", ENV_EVAL_FREQUENCY
                            )
                        )
                    except Exception:
                        total_steps = 0
                        eval_freq = ENV_EVAL_FREQUENCY

                    if eval_freq and self.n_calls % eval_freq == 0:
                        stats = {
                            "action_distribution": action_dist,
                            "step": self.n_calls,
                        }
                        # Add reward component debugging info when present in env info
                        try:
                            infos = self.locals.get("infos")
                            if infos and len(infos) > 0 and isinstance(infos[0], dict):
                                info = infos[0]
                                # reward components are added to info by environment: e.g. skew_penalty, balance_penalty
                                components = {}
                                for k, v in info.items():
                                    # only include reward-related metrics to keep reports compact
                                    if (
                                        k.endswith("_penalty")
                                        or k.endswith("_shaping")
                                        or k == "action_bonus"
                                    ):
                                        components[k] = v
                                if components:
                                    stats["reward_components"] = components
                        except Exception:
                            # Don't fail training if we can't collect reward components
                            pass
                        reporter = getattr(self.trainer_ref, "reporter", None)
                        if reporter and hasattr(reporter, "log_training_progress"):
                            reporter.log_training_progress(
                                self.n_calls, total_steps, stats
                            )
            except Exception as e:
                logging.debug(f"Failed to record action_distribution to reporter: {e}")

            # Log continuous action statistics for SAC models
            if self.continuous_actions and len(self.continuous_actions) > 100:
                recent_continuous = self.continuous_actions[-1000:]  # Last 1000 actions
                mean_action = np.mean(recent_continuous)
                std_action = np.std(recent_continuous)
                min_action = np.min(recent_continuous)
                max_action = np.max(recent_continuous)

                # Calculate percentiles
                p25 = np.percentile(recent_continuous, 25)
                p50 = np.percentile(recent_continuous, 50)
                p75 = np.percentile(recent_continuous, 75)

                # Calculate action distribution metrics
                near_zero_count = sum(1 for x in recent_continuous if -0.1 <= x <= 0.1)
                extreme_negative_count = sum(1 for x in recent_continuous if x <= -0.8)
                extreme_positive_count = sum(1 for x in recent_continuous if x >= 0.8)

                near_zero_pct = near_zero_count / len(recent_continuous) * 100
                extreme_negative_pct = (
                    extreme_negative_count / len(recent_continuous) * 100
                )
                extreme_positive_pct = (
                    extreme_positive_count / len(recent_continuous) * 100
                )

                # Calculate action entropy (distribution diversity)
                hist, _ = np.histogram(
                    recent_continuous, bins=20, range=(-1, 1), density=True
                )
                hist = hist[hist > 0]  # Remove zero probabilities
                action_entropy = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0

                logging.warning(
                    "CONTINUOUS ACTION STATS [Step %d] - Mean: %.3f, Std: %.3f, Min: %.3f, Max: %.3f, Range: %.3f",
                    self.n_calls,
                    mean_action,
                    std_action,
                    min_action,
                    max_action,
                    max_action - min_action,
                )
                logging.warning(
                    "ACTION PERCENTILES [Step %d] - 25%%: %.3f, 50%%: %.3f, 75%%: %.3f | Near Zero (±0.1): %.1f%% | Extreme (≤-0.8): %.1f%% | Extreme (≥0.8): %.1f%%",
                    self.n_calls,
                    p25,
                    p50,
                    p75,
                    near_zero_pct,
                    extreme_negative_pct,
                    extreme_positive_pct,
                )
                logging.warning(
                    "ACTION DISTRIBUTION [Step %d] - Entropy: %.3f | Skewness: %.3f | Kurtosis: %.3f",
                    self.n_calls,
                    action_entropy,
                    np.mean(((recent_continuous - mean_action) / std_action) ** 3)
                    if std_action > 0
                    else 0,
                    np.mean(((recent_continuous - mean_action) / std_action) ** 4) - 3
                    if std_action > 0
                    else 0,
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
                    logging.warning(
                        "ACTION RANGE DISTRIBUTION [Step %d] (%%): [-1,-0.8]: %.1f, [-0.8,-0.6]: %.1f, [-0.6,-0.4]: %.1f, [-0.4,-0.2]: %.1f, [-0.2,0]: %.1f, [0,0.2]: %.1f, [0.2,0.4]: %.1f, [0.4,0.6]: %.1f, [0.6,0.8]: %.1f, [0.8,1]: %.1f",
                        self.n_calls,
                        *range_dist,
                    )

            # Log per-regime action distribution for debugging
            if self.regime_action_counts:
                regime_info = []
                for regime, counts in self.regime_action_counts.items():
                    total_regime_actions = sum(counts)
                    if total_regime_actions > 0:
                        buy_pct = (
                            counts[MULTIPLIER_INDEX_BUY] / total_regime_actions * 100
                        )
                        sell_pct = (
                            counts[MULTIPLIER_INDEX_SELL] / total_regime_actions * 100
                        )
                        hold_pct = (
                            counts[MULTIPLIER_INDEX_HOLD] / total_regime_actions * 100
                        )
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
            logging.debug(
                f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}"
            )

        # Trigger early stopping if patience exceeded
        if self.early_stopping_counter >= self.early_stopping_patience:
            self.early_stopping_triggered = True
            logging.info(
                f"Early stopping triggered after {self.early_stopping_patience} steps without improvement"
            )
            # Note: We can't directly stop training from callback, but we can set a flag
            # The training loop should check this flag

    def on_training_end(self) -> None:
        """Log final training statistics when training ends."""
        logging.warning("=" * 80)
        logging.warning("TRAINING COMPLETED - FINAL STATISTICS")
        logging.warning("=" * 80)

        # Log final discrete action distribution
        if self.discrete_actions:
            total_actions = len(self.discrete_actions)
            shifted_actions = np.array(self.discrete_actions) + 1
            discrete_counts = np.bincount(shifted_actions, minlength=3)

            action_dist = {
                "SELL": discrete_counts[0] / total_actions,
                "HOLD": discrete_counts[1] / total_actions,
                "BUY": discrete_counts[2] / total_actions,
            }

            logging.warning(
                "Final Discrete Action Distribution (Total: %d actions):", total_actions
            )
            logging.warning(
                "  HOLD: %.2f%% (%d)", action_dist["HOLD"] * 100, discrete_counts[1]
            )
            logging.warning(
                "  BUY:  %.2f%% (%d)", action_dist["BUY"] * 100, discrete_counts[2]
            )
            logging.warning(
                "  SELL: %.2f%% (%d)", action_dist["SELL"] * 100, discrete_counts[0]
            )

        # Log final continuous action statistics
        if self.continuous_actions:
            all_continuous = np.array(self.continuous_actions)
            total_continuous = len(all_continuous)

            if total_continuous > 0:
                mean_action = np.mean(all_continuous)
                std_action = np.std(all_continuous)
                min_action = np.min(all_continuous)
                max_action = np.max(all_continuous)

                # Calculate comprehensive statistics
                p10 = np.percentile(all_continuous, 10)
                p25 = np.percentile(all_continuous, 25)
                p50 = np.percentile(all_continuous, 50)
                p75 = np.percentile(all_continuous, 75)
                p90 = np.percentile(all_continuous, 90)

                # Action distribution analysis
                near_zero_count = np.sum(
                    (all_continuous >= -0.1) & (all_continuous <= 0.1)
                )
                extreme_negative_count = np.sum(all_continuous <= -0.8)
                extreme_positive_count = np.sum(all_continuous >= 0.8)
                strong_buy_count = np.sum(all_continuous >= 0.6)
                strong_sell_count = np.sum(all_continuous <= -0.6)

                # Action stability metrics
                if len(all_continuous) > 100:
                    # Calculate rolling means for stability analysis
                    window_size = min(100, len(all_continuous) // 10)
                    rolling_means = []
                    for i in range(window_size, len(all_continuous), window_size):
                        rolling_means.append(
                            np.mean(all_continuous[i - window_size : i])
                        )
                    action_stability = np.std(rolling_means) if rolling_means else 0
                else:
                    action_stability = 0

                logging.warning(
                    "Final Continuous Action Statistics (Total: %d actions):",
                    total_continuous,
                )
                logging.warning(
                    "  Basic Stats - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f",
                    mean_action,
                    std_action,
                    min_action,
                    max_action,
                )
                logging.warning(
                    "  Percentiles - 10%%: %.4f, 25%%: %.4f, 50%%: %.4f, 75%%: %.4f, 90%%: %.4f",
                    p10,
                    p25,
                    p50,
                    p75,
                    p90,
                )
                logging.warning("  Distribution Analysis:")
                logging.warning(
                    "    Near Zero (±0.1): %.2f%% (%d)",
                    near_zero_count / total_continuous * 100,
                    near_zero_count,
                )
                logging.warning(
                    "    Extreme Negative (≤-0.8): %.2f%% (%d)",
                    extreme_negative_count / total_continuous * 100,
                    extreme_negative_count,
                )
                logging.warning(
                    "    Extreme Positive (≥0.8): %.2f%% (%d)",
                    extreme_positive_count / total_continuous * 100,
                    extreme_positive_count,
                )
                logging.warning(
                    "    Strong Buy (≥0.6): %.2f%% (%d)",
                    strong_buy_count / total_continuous * 100,
                    strong_buy_count,
                )
                logging.warning(
                    "    Strong Sell (≤-0.6): %.2f%% (%d)",
                    strong_sell_count / total_continuous * 100,
                    strong_sell_count,
                )
                logging.warning(
                    "  Action Stability - Rolling Mean Std: %.4f", action_stability
                )

                # Action entropy and distribution shape
                hist, _ = np.histogram(
                    all_continuous, bins=20, range=(-1, 1), density=True
                )
                hist = hist[hist > 0]
                action_entropy = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
                skewness_val = skewness(all_continuous)
                kurtosis_val = kurtosis(all_continuous)

                logging.warning(
                    "  Distribution Shape - Entropy: %.4f, Skewness: %.4f, Kurtosis: %.4f",
                    action_entropy,
                    skewness_val,
                    kurtosis_val,
                )

        # Log reward statistics
        if self.reward_history:
            rewards = np.array(self.reward_history)
            logging.warning(
                "Final Reward Statistics (Total: %d rewards):", len(rewards)
            )
            logging.warning(
                "  Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f",
                np.mean(rewards),
                np.std(rewards),
                np.min(rewards),
                np.max(rewards),
            )
            logging.warning(
                "  Positive Rewards: %.2f%% (%d), Negative: %.2f%% (%d), Zero: %.2f%% (%d)",
                np.sum(rewards > 0) / len(rewards) * 100,
                np.sum(rewards > 0),
                np.sum(rewards < 0) / len(rewards) * 100,
                np.sum(rewards < 0),
                np.sum(rewards == 0) / len(rewards) * 100,
                np.sum(rewards == 0),
            )

        # Log regime-specific statistics
        if self.regime_action_counts:
            logging.warning("Final Regime-Specific Action Distributions:")
            for regime, counts in self.regime_action_counts.items():
                total_regime_actions = sum(counts)
                if total_regime_actions > 0:
                    buy_pct = counts[MULTIPLIER_INDEX_BUY] / total_regime_actions * 100
                    sell_pct = (
                        counts[MULTIPLIER_INDEX_SELL] / total_regime_actions * 100
                    )
                    hold_pct = (
                        counts[MULTIPLIER_INDEX_HOLD] / total_regime_actions * 100
                    )
                    logging.warning(
                        "  %s: BUY %.1f%% (%d), SELL %.1f%% (%d), HOLD %.1f%% (%d)",
                        regime,
                        buy_pct,
                        counts[MULTIPLIER_INDEX_BUY],
                        sell_pct,
                        counts[MULTIPLIER_INDEX_SELL],
                        hold_pct,
                        counts[MULTIPLIER_INDEX_HOLD],
                    )

        # Log training time and performance
        total_time = time.time() - self.start_time
        steps_per_sec = self.n_calls / total_time if total_time > 0 else 0
        logging.warning("Training Performance:")
        logging.warning("  Total Steps: %d", self.n_calls)
        logging.warning(
            "  Total Time: %.1f seconds (%.1f minutes)", total_time, total_time / 60
        )
        logging.warning("  Average Steps/Second: %.2f", steps_per_sec)

        logging.info("=" * 80)
