"""Training risk management component."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ztb.utils.exceptions.custom_exceptions import EarlyStoppingError, OverfittingError, TrainingInstabilityError
from ztb.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ztb.training.unified_trainer.trainer import UnifiedTrainer


class TrainingRiskManager:
    """Manages training risks including early stopping and overfitting detection."""

    def __init__(self, trainer: "UnifiedTrainer"):
        """Initialize training risk manager with reference to trainer."""
        self.trainer = trainer
        self.logger = get_logger(__name__)

        # Early stopping parameters
        self.early_stopping_patience = 20
        self.early_stopping_min_delta = 1e-4
        self.best_reward = float('-inf')
        self.epochs_without_improvement = 0
        self.early_stopping_triggered = False

        # Overfitting detection
        self.overfitting_threshold = 0.1  # 10% performance drop
        self.validation_window = 50
        self.overfitting_detected = False

        # Training stability
        self.loss_explosion_threshold = 10.0
        self.reward_explosion_threshold = 1000.0
        self.nan_loss_threshold = 5  # Consecutive NaN losses before stopping

        # Risk tracking
        self.consecutive_nan_losses = 0
        self.consecutive_high_losses = 0
        self.reward_history: List[float] = []
        self.validation_rewards: List[float] = []

    def check_early_stopping(self, current_reward: float) -> bool:
        """Check if early stopping should be triggered.

        Args:
            current_reward: Current average reward

        Returns:
            True if training should stop early

        Raises:
            EarlyStoppingError: When early stopping is triggered
        """
        if current_reward > self.best_reward + self.early_stopping_min_delta:
            self.best_reward = current_reward
            self.epochs_without_improvement = 0
            self.logger.debug(f"New best reward: {current_reward:.4f}")
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.early_stopping_patience:
            self.early_stopping_triggered = True
            error_msg = (f"Early stopping triggered after {self.epochs_without_improvement} "
                        f"epochs without improvement (best: {self.best_reward:.4f})")
            self.logger.warning(error_msg)
            raise EarlyStoppingError(
                error_msg,
                details={
                    "epochs_without_improvement": self.epochs_without_improvement,
                    "patience": self.early_stopping_patience,
                    "best_reward": self.best_reward,
                    "current_reward": current_reward
                }
            )

        return False

    def check_overfitting(self, training_reward: float, validation_reward: Optional[float] = None) -> bool:
        """Check for overfitting.

        Args:
            training_reward: Current training reward
            validation_reward: Validation reward (if available)

        Returns:
            True if overfitting detected

        Raises:
            OverfittingError: When overfitting is detected
        """
        self.reward_history.append(training_reward)

        if validation_reward is not None:
            self.validation_rewards.append(validation_reward)

        # Need sufficient history for detection
        if len(self.reward_history) < self.validation_window:
            return False

        # Check training reward stability (should not oscillate wildly)
        recent_rewards = self.reward_history[-self.validation_window:]
        reward_std = np.std(recent_rewards)
        reward_mean = np.mean(recent_rewards)

        # High variance indicates unstable training
        if reward_std > abs(reward_mean) * 0.5:  # 50% of mean
            self.logger.warning(f"High reward variance detected: std={reward_std:.4f}, mean={reward_mean:.4f}")
            return True

        # Check for performance degradation if validation data available
        if len(self.validation_rewards) >= self.validation_window:
            np.mean(self.reward_history[-self.validation_window//2:])
            recent_val = np.mean(self.validation_rewards[-self.validation_window//2:])
            older_val = np.mean(self.validation_rewards[-self.validation_window:-self.validation_window//2])

            if older_val > 0 and (recent_val - older_val) / older_val < -self.overfitting_threshold:
                self.overfitting_detected = True
                error_msg = (f"Overfitting detected: validation performance dropped by "
                           f"{(recent_val - older_val) / older_val:.1%}")
                self.logger.warning(error_msg)
                raise OverfittingError(
                    error_msg,
                    details={
                        "validation_drop": (recent_val - older_val) / older_val,
                        "recent_validation": recent_val,
                        "older_validation": older_val,
                        "threshold": self.overfitting_threshold
                    }
                )

        return False

    def check_training_stability(self, loss: Optional[float] = None, reward: Optional[float] = None) -> bool:
        """Check training stability and detect critical issues.

        Args:
            loss: Current loss value
            reward: Current reward value

        Returns:
            True if training should continue, False if critical issues detected

        Raises:
            TrainingInstabilityError: When training instability is detected
        """
        # Check for NaN loss
        if loss is not None:
            if not np.isfinite(loss):
                self.consecutive_nan_losses += 1
                if self.consecutive_nan_losses >= self.nan_loss_threshold:
                    error_msg = f"Too many consecutive NaN losses ({self.consecutive_nan_losses})"
                    self.logger.critical(error_msg)
                    raise TrainingInstabilityError(
                        error_msg,
                        details={
                            "consecutive_nan_losses": self.consecutive_nan_losses,
                            "threshold": self.nan_loss_threshold,
                            "current_loss": loss
                        }
                    )
            else:
                self.consecutive_nan_losses = 0

        # Check for exploding loss
        if loss is not None and loss > self.loss_explosion_threshold:
            self.consecutive_high_losses += 1
            if self.consecutive_high_losses >= 3:
                error_msg = f"Loss explosion detected: {loss:.4f}"
                self.logger.critical(error_msg)
                raise TrainingInstabilityError(
                    error_msg,
                    details={
                        "loss_value": loss,
                        "threshold": self.loss_explosion_threshold,
                        "consecutive_high_losses": self.consecutive_high_losses
                    }
                )
        else:
            self.consecutive_high_losses = 0

        # Check for reward explosion
        if reward is not None and abs(reward) > self.reward_explosion_threshold:
            error_msg = f"Reward explosion detected: {reward:.4f}"
            self.logger.critical(error_msg)
            raise TrainingInstabilityError(
                error_msg,
                details={
                    "reward_value": reward,
                    "threshold": self.reward_explosion_threshold
                }
            )

        return True

    def should_pause_training(self, current_stats: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if training should be paused due to risk factors.

        Args:
            current_stats: Current training statistics

        Returns:
            Tuple of (should_pause, reason)
        """
        # Check early stopping
        avg_reward = current_stats.get("average_reward", 0)
        try:
            self.check_early_stopping(avg_reward)
        except EarlyStoppingError:
            return True, "early_stopping"

        # Check overfitting
        try:
            self.check_overfitting(avg_reward)
        except OverfittingError:
            return True, "overfitting"

        # Check stability
        avg_loss = current_stats.get("average_loss", None)
        try:
            self.check_training_stability(avg_loss, avg_reward)
        except TrainingInstabilityError:
            return True, "instability"

        # Check time-based limits
        elapsed_hours = current_stats.get("elapsed_time_seconds", 0) / 3600
        if elapsed_hours > 24:  # 24 hour limit
            return True, "time_limit"

        return False, ""

    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive risk assessment.

        Returns:
            Dictionary with risk metrics
        """
        return {
            "early_stopping": {
                "triggered": self.early_stopping_triggered,
                "epochs_without_improvement": self.epochs_without_improvement,
                "patience": self.early_stopping_patience,
                "best_reward": self.best_reward,
            },
            "overfitting": {
                "detected": self.overfitting_detected,
                "threshold": self.overfitting_threshold,
                "validation_window": self.validation_window,
            },
            "stability": {
                "consecutive_nan_losses": self.consecutive_nan_losses,
                "loss_explosion_threshold": self.loss_explosion_threshold,
                "reward_explosion_threshold": self.reward_explosion_threshold,
            },
            "training_progress": {
                "total_rewards": len(self.reward_history),
                "reward_std": np.std(self.reward_history[-100:]) if len(self.reward_history) >= 100 else 0.0,
                "reward_trend": self._calculate_reward_trend(),
            },
        }

    def reset_risk_state(self) -> None:
        """Reset risk monitoring state (useful for restarting training)."""
        self.best_reward = float('-inf')
        self.epochs_without_improvement = 0
        self.early_stopping_triggered = False
        self.overfitting_detected = False
        self.consecutive_nan_losses = 0
        self.consecutive_high_losses = 0
        self.reward_history.clear()
        self.validation_rewards.clear()
        self.logger.info("Risk monitoring state reset")

    def _calculate_reward_trend(self) -> float:
        """Calculate recent reward trend.

        Returns:
            Trend value (positive = improving, negative = worsening)
        """
        if len(self.reward_history) < 10:
            return 0.0

        recent = self.reward_history[-50:] if len(self.reward_history) >= 50 else self.reward_history
        if len(recent) < 2:
            return 0.0

        # Simple trend calculation
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
