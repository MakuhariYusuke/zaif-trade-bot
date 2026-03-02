"""Training validation component."""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ztb.utils.exceptions.custom_exceptions import ActionValidationError
from ztb.utils.logging_utils import get_logger
from ztb.utils.type_guards import (
    is_valid_action
)

if TYPE_CHECKING:
    from ztb.training.unified_trainer.trainer import UnifiedTrainer

if TYPE_CHECKING:
    from ztb.training.unified_trainer.trainer import UnifiedTrainer

class TrainingValidationManager:
    """Handles validation of training data, models, and training process."""

    def __init__(self, trainer: "UnifiedTrainer"):
        """Initialize training validation manager with reference to trainer."""
        self.trainer = trainer
        self.logger = get_logger(__name__)

        # Validation thresholds
        self.min_reward_threshold = -1000.0  # Minimum acceptable reward
        self.max_reward_threshold = 1000.0   # Maximum acceptable reward
        self.min_loss_threshold = 0.0        # Minimum loss (must be non-negative)
        self.max_loss_threshold = 1000.0     # Maximum acceptable loss
        self.nan_tolerance = 0.01            # Maximum fraction of NaN values allowed

    def validate_training_data(self, observations: NDArray[np.float32],
                             actions: NDArray[np.int64], rewards: NDArray[np.float32],
                             next_observations: NDArray[np.float32]) -> list[str]:
        """Validate training data batch.

        Args:
            observations: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_observations: Next observations

        Returns:
            list of validation issues (empty if all valid)
        """
        issues = []

        # Check shapes
        if len(observations) != len(actions) or len(actions) != len(rewards):
            issues.append("Inconsistent batch sizes between observations, actions, and rewards")

        if len(next_observations) != len(observations):
            issues.append("Next observations length doesn't match observations")

        # Check for NaN/Inf values
        if np.any(~np.isfinite(observations)):
            nan_fraction = np.mean(~np.isfinite(observations))
            if nan_fraction > self.nan_tolerance:
                issues.append(f"Too many NaN/Inf values in observations: {nan_fraction:.1%}")

        if np.any(~np.isfinite(rewards)):
            nan_fraction = np.mean(~np.isfinite(rewards))
            if nan_fraction > self.nan_tolerance:
                issues.append(f"Too many NaN/Inf values in rewards: {nan_fraction:.1%}")

        # Check reward bounds
        if np.any(rewards < self.min_reward_threshold) or np.any(rewards > self.max_reward_threshold):
            issues.append(f"Rewards outside acceptable range [{self.min_reward_threshold}, {self.max_reward_threshold}]")

        # Check action validity
        valid_actions = [0, 1, 2]  # HOLD, BUY, SELL
        invalid_actions = ~np.isin(actions, valid_actions)
        if np.any(invalid_actions):
            issues.append(f"Invalid actions found: {np.unique(actions[invalid_actions])}")

        # Check observation bounds (should be normalized)
        if np.any(np.abs(observations) > 10.0):  # Allow some margin for unnormalized data
            issues.append("Observations may not be properly normalized")

        return issues

    def validate_model_output(self, action_probs: NDArray[np.float32] | None = None,
                            values: NDArray[np.float32] | None = None) -> list[str]:
        """Validate model output.

        Args:
            action_probs: Action probabilities
            values: Value estimates

        Returns:
            list of validation issues (empty if all valid)
        """
        issues = []

        # Validate action probabilities
        if action_probs is not None:
            if np.any(~np.isfinite(action_probs)):
                issues.append("NaN/Inf values in action probabilities")

            if np.any(action_probs < 0) or np.any(action_probs > 1):
                issues.append("Action probabilities outside [0, 1] range")

            # Check probability sums (should be close to 1)
            prob_sums = np.sum(action_probs, axis=-1)
            if not np.allclose(prob_sums, 1.0, atol=1e-6):
                issues.append("Action probabilities don't sum to 1")

        # Validate value estimates
        if values is not None:
            if np.any(~np.isfinite(values)):
                issues.append("NaN/Inf values in value estimates")

            # Check for reasonable value ranges (should be in reasonable bounds)
            if np.any(np.abs(values) > 1e6):
                issues.append("Value estimates seem unreasonably large")

        return issues

    def validate_training_progress(self, stats: dict[str, Any]) -> list[str]:
        """Validate training progress and detect potential issues.

        Args:
            stats: Training statistics

        Returns:
            list of validation issues (empty if all valid)
        """
        issues = []

        # Check for training collapse (rewards consistently negative)
        recent_rewards = stats.get("recent_avg_reward", 0)
        if recent_rewards < -100 and stats.get("total_timesteps", 0) > 10000:
            issues.append(f"Training may have collapsed: recent average reward {recent_rewards:.2f}")

        # Check for exploding gradients (loss too high)
        avg_loss = stats.get("average_loss", 0)
        if avg_loss > self.max_loss_threshold:
            issues.append(f"Loss too high: {avg_loss:.2f} > {self.max_loss_threshold}")

        # Check for NaN loss
        if not np.isfinite(avg_loss):
            issues.append("Training loss is NaN or infinite")

        # Check learning progress (should improve over time)
        if stats.get("total_timesteps", 0) > 50000:  # After sufficient training
            reward_trend = self._calculate_reward_trend(stats)
            if reward_trend < -0.01:  # Consistently decreasing
                issues.append("Rewards are consistently decreasing - possible training instability")

        return issues

    def validate_environment_reset(self, observation: NDArray[np.float32],
                                 info: dict[str, Any]) -> list[str]:
        """Validate environment reset.

        Args:
            observation: Initial observation
            info: Reset info

        Returns:
            list of validation issues (empty if all valid)
        """
        issues = []

        # Check observation validity
        if np.any(~np.isfinite(observation)):
            issues.append("Invalid observation after reset (NaN/Inf)")

        # Check required info fields
        required_fields = ["position", "portfolio_value"]
        for field in required_fields:
            if field not in info:
                issues.append(f"Missing required info field: {field}")

        # Check portfolio value
        portfolio_value = info.get("portfolio_value", 0)
        if portfolio_value <= 0:
            issues.append(f"Invalid portfolio value after reset: {portfolio_value}")

        return issues

    def validate_action_selection(self, action: int, legal_actions: list[int] | None = None) -> None:
        """Validate selected action.

        Args:
            action: Selected action
            legal_actions: list of legal actions (if available)

        Raises:
            ActionValidationError: If action is invalid
        """
        # Use type guard for validation
        if not is_valid_action(action):
            raise ActionValidationError(
                f"Invalid action: {action}",
                details={"action": action, "expected_range": "[0, 1, 2]"}
            )

        # Check against legal actions if provided
        if legal_actions is not None and action not in legal_actions:
            raise ActionValidationError(
                f"Action {action} not in legal actions",
                details={"action": action, "legal_actions": legal_actions}
            )

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of validation state.

        Returns:
            Dictionary with validation configuration
        """
        return {
            "reward_bounds": [self.min_reward_threshold, self.max_reward_threshold],
            "loss_bounds": [self.min_loss_threshold, self.max_loss_threshold],
            "nan_tolerance": self.nan_tolerance,
            "validation_enabled": True,
        }

    def _calculate_reward_trend(self, stats: dict[str, Any]) -> float:
        """Calculate reward trend over recent episodes.

        Args:
            stats: Training statistics

        Returns:
            Trend coefficient (positive = improving, negative = worsening)
        """
        # Simple linear trend calculation
        rewards = stats.get("episode_rewards", [])
        if len(rewards) < 10:
            return 0.0

        recent_rewards = rewards[-100:]  # Last 100 episodes
        if len(recent_rewards) < 10:
            return 0.0

        # Calculate slope using linear regression
        x = np.arange(len(recent_rewards))
        y = np.array(recent_rewards)

        # Simple slope calculation
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0.0
