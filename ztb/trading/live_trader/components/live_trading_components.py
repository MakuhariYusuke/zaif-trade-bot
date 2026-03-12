"""
Live Trading Components - Core components for live trading.

This module separates core live trading logic from the main LiveTrader class,
including model management, feature computation, and trading loop components.
"""

from typing import Any

from ztb.trading.environment.constants import (
    DEFAULT_MAX_ACTION_HISTORY,
)
from ztb.utils.exceptions.custom_exceptions import ModelError
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelManager:
    """
    Manages model loading and prediction for live trading.

    This class handles:
    - Model loading and validation
    - Prediction execution
    - Model state management
    """

    def __init__(self):
        """Initialize ModelManager."""
        self.logger = get_logger(__name__)
        self.model = None
        self.model_path = None
        self.algorithm = None
        self.expected_features = None
        self.feature_names = None

    def load_model(self, model_path: str, algorithm: str) -> bool:
        """
        Load trading model.

        Args:
            model_path: Path to model file
            algorithm: Algorithm type ('ppo' or 'sac')

        Returns:
            True if loaded successfully, False otherwise

        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model loading fails
        """
        try:
            from pathlib import Path

            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model_path = model_path
            self.algorithm = algorithm

            if algorithm.lower() == "ppo":
                from sb3_contrib import MaskablePPO
                self.model = MaskablePPO.load(model_path)
            elif algorithm.lower() == "sac":
                from stable_baselines3 import SAC
                self.model = SAC.load(model_path)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            self.logger.info(f"Model loaded successfully: {algorithm} from {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict_action(
        self,
        observation: Any,
        action_masks: Any | None = None
    ) -> Any:
        """
        Predict trading action.

        Args:
            observation: Current market observation
            action_masks: Action masks for constrained prediction

        Returns:
            Predicted action

        Raises:
            RuntimeError: If model not loaded or prediction fails
        """
        if self.model is None:
            raise ModelError("Model not loaded")

        try:
            if self.algorithm == "ppo" and action_masks is not None:
                action, _ = self.model.predict(observation, action_masks=action_masks)
            else:
                action, _ = self.model.predict(observation)

            return action

        except Exception as e:
            self.logger.error(f"Failed to predict action: {e}")
            raise ModelError(f"Action prediction failed: {e}") from e

    def validate_model(self) -> dict[str, Any]:
        """
        Validate loaded model.

        Returns:
            Validation results dictionary

        Raises:
            RuntimeError: If model validation fails
        """
        if self.model is None:
            raise ModelError("No model loaded for validation")

        try:
            validation_results = {
                "model_loaded": True,
                "algorithm": self.algorithm,
                "model_path": self.model_path,
                "has_features": self.expected_features is not None,
                "feature_count": len(self.expected_features) if self.expected_features else 0,
            }

            # Additional validation checks
            if hasattr(self.model, 'policy'):
                validation_results["has_policy"] = True
            else:
                validation_results["has_policy"] = False
                self.logger.warning("Model missing policy component")

            return validation_results

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            raise ModelError(f"Model validation failed: {e}") from e

class FeatureComputer:
    """
    Handles feature computation for live trading.

    This class manages:
    - Real-time feature calculation
    - Feature normalization
    - Feature validation
    """

    def __init__(self):
        """Initialize FeatureComputer."""
        self.logger = get_logger(__name__)
        self.price_history = None
        self.feature_computer = None

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize feature computation.

        Args:
            config: Feature computation configuration

        Raises:
            ValueError: If initialization fails
        """
        try:
            from collections import deque
            from ztb.trading.live_trader.feature_computation import FeatureComputation

            self.price_history = deque(maxlen=config.get('price_history_size', DEFAULT_MAX_ACTION_HISTORY))
            self.feature_computer = FeatureComputation()

            self.logger.info("Feature computer initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize feature computer: {e}")
            raise

    def compute_features(self, market_data: dict[str, Any]) -> Any | None:
        """
        Compute features from market data.

        Args:
            market_data: Current market data

        Returns:
            Computed features or None if computation fails
        """
        try:
            if self.feature_computer is None:
                raise ModelError("Feature computer not initialized")

            # Update price history
            if 'price' in market_data and self.price_history is not None:
                self.price_history.append(market_data['price'])

            # Compute features
            features = self.feature_computer.compute_features(market_data)
            return features

        except Exception as e:
            self.logger.error(f"Failed to compute features: {e}")
            return None

    def validate_features(self, features: Any) -> bool:
        """
        Validate computed features.

        Args:
            features: Features to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            if features is None:
                return False

            # Check for NaN/inf values
            import numpy as np
            if hasattr(features, 'flatten'):
                flat_features = features.flatten()
                if np.any(~np.isfinite(flat_features)):
                    self.logger.warning("Features contain non-finite values")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Feature validation failed: {e}")
            return False

class TradingLoopManager:
    """
    Manages the main trading loop execution.

    This class handles:
    - Trading loop state management
    - Execution timing
    - Error recovery
    - Performance monitoring
    """

    def __init__(self):
        """Initialize TradingLoopManager."""
        self.logger = get_logger(__name__)
        self.is_running = False
        self.loop_count = 0
        self.error_count = 0
        self.last_execution_time = None

    def start_loop(self) -> None:
        """Start the trading loop."""
        self.is_running = True
        self.loop_count = 0
        self.error_count = 0
        self.logger.info("Trading loop started")

    def stop_loop(self) -> None:
        """Stop the trading loop."""
        self.is_running = False
        self.logger.info(f"Trading loop stopped after {self.loop_count} iterations")

    def execute_iteration(self, iteration_func: callable) -> bool:
        """
        Execute a single trading loop iteration.

        Args:
            iteration_func: Function to execute for this iteration

        Returns:
            True if successful, False if failed
        """
        if not self.is_running:
            return False

        try:
            import time
            start_time = time.time()

            iteration_func()

            execution_time = time.time() - start_time
            self.last_execution_time = execution_time
            self.loop_count += 1

            # Log performance occasionally
            if self.loop_count % 100 == 0:
                self.logger.info(f"Trading loop iteration {self.loop_count}, execution time: {execution_time:.3f}s")

            return True

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Trading loop iteration failed: {e}")

            # Stop loop if too many errors
            if self.error_count > 10:
                self.logger.error("Too many errors, stopping trading loop")
                self.stop_loop()

            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get trading loop statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "is_running": self.is_running,
            "loop_count": self.loop_count,
            "error_count": self.error_count,
            "last_execution_time": self.last_execution_time,
            "error_rate": self.error_count / max(1, self.loop_count),
        }
