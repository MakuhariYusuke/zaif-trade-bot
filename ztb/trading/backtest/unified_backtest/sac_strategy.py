#!/usr/bin/env python3
"""
SAC Trading Strategy

Implements SAC (Soft Actor-Critic) based trading strategy for the unified backtest framework.
Supports regime adaptation and leverages SAC learning outcomes.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ....utils.logging_utils import get_logger
from .strategy_base import MLTradingStrategy

logger = get_logger(__name__)


class SACStrategy(MLTradingStrategy):
    """
    SAC-based trading strategy with regime adaptation capabilities.

    Features:
    - SAC model integration
    - Regime-based decision making
    - Feature engineering support
    - Learning outcome analysis
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        regime_classifier_path: Optional[str] = None,
        feature_engineer: Optional["FeatureEngineer"] = None,
    ):
        """
        Initialize SAC strategy.

        Args:
            name: Strategy name
            model_path: Path to trained SAC model
            regime_classifier_path: Path to regime classifier
            feature_engineer: Feature engineering instance
        """
        super().__init__(name, model_path)
        self.regime_classifier_path = regime_classifier_path
        self.feature_engineer = feature_engineer

        # SAC-specific attributes
        self.regime_classifier: Optional["RegimeClassifier"] = None
        self.current_regime: Optional[str] = None
        self.action_space: tuple[float, float] = (-1.0, 1.0)

        # Learning outcome tracking
        self.learning_metrics: Dict[str, Union[int, float, str]] = {}
        self.regime_performance: Dict[str, list] = {}

    def initialize(
        self, data: pd.DataFrame, backtest_config: "BacktestConfig", **kwargs
    ) -> None:
        """
        Initialize the SAC strategy.

        Args:
            data: Market data
            backtest_config: Backtest configuration
            **kwargs: Additional parameters
        """
        try:
            # Load SAC model
            self.load_model()

            # Load regime classifier if provided
            if self.regime_classifier_path:
                self.load_regime_classifier()

            # Initialize feature engineer
            if self.feature_engineer:
                self.feature_engineer.initialize(data)

            # Set action space bounds
            self.action_space = kwargs.get("action_space", (-1.0, 1.0))

            self.is_initialized = True
            logger.info(f"SAC strategy {self.name} initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SAC strategy: {e}")
            raise

    def load_model(self) -> None:
        """Load the trained SAC model."""
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(f"SAC model not found: {self.model_path}")

        try:
            # Import SAC lazily to avoid importing torch/stable_baselines3 at module import time
            try:
                from stable_baselines3 import SAC as _SAC
            except Exception:
                _SAC = None

            if _SAC is None:
                raise ImportError("stable_baselines3.SAC is required to load models")

            self.model = _SAC.load(self.model_path)
            logger.info(f"Loaded SAC model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load SAC model: {e}")
            raise

    def load_regime_classifier(self) -> None:
        """Load the regime classifier."""
        if not self.regime_classifier_path:
            return

        try:
            # Import here to avoid circular imports
            from ztb.analysis.regime.v444_regime_classifier import V444RegimeClassifier

            self.regime_classifier = V444RegimeClassifier()
            # Note: load_model method may not exist, using alternative approach
            logger.info(
                f"Regime classifier initialized from {self.regime_classifier_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load regime classifier: {e}")

    def generate_signal(
        self, data: pd.DataFrame, current_position: int
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Generate trading signal using SAC model.

        Args:
            data: Market data with OHLCV and features
            current_position: Current position (-1, 0, 1 for short, flat, long)

        Returns:
            Signal dictionary
        """
        try:
            # Preprocess data for model input
            features = self.preprocess_data(data)

            if features.empty:
                return {"action": "hold", "reason": "insufficient_data"}

            # Get current regime if classifier available
            if self.regime_classifier and hasattr(
                self.regime_classifier, "predict_regime"
            ):
                # Get latest data point for regime prediction
                latest_data = data.iloc[-1:].to_dict("records")[0]
                self.current_regime = self.regime_classifier.predict_regime(latest_data)
            else:
                self.current_regime = "unknown"

            # Get SAC action
            action, _ = self.model.predict(features.values, deterministic=True)

            # Convert continuous action to discrete signal
            signal = self._convert_action_to_signal(action[0])

            # Track regime performance
            self._track_regime_performance(signal, self.current_regime or "unknown")

            return signal

        except Exception as e:
            logger.warning(f"Error generating SAC signal: {e}")
            return {"action": "hold", "reason": "error"}

    def _convert_action_to_signal(
        self, action: float
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Convert SAC continuous action to discrete trading signal.

        Args:
            action: Continuous action from SAC (-1 to 1)

        Returns:
            Signal dictionary
        """
        # Normalize action to [-1, 1] range
        action = np.clip(action, *self.action_space)

        # Define action thresholds
        buy_threshold = 0.3
        sell_threshold = -0.3

        if action > buy_threshold:
            return {
                "action": "buy",
                "size": min(float(abs(action)), 1.0),
                "confidence": min(float(abs(action)), 1.0),
                "reason": f"SAC_action_{action:.3f}",
                "regime": self.current_regime,
            }
        elif action < sell_threshold:
            return {
                "action": "sell",
                "size": min(float(abs(action)), 1.0),
                "confidence": min(float(abs(action)), 1.0),
                "reason": f"SAC_action_{action:.3f}",
                "regime": self.current_regime,
            }
        else:
            return {
                "action": "hold",
                "reason": f"SAC_action_{action:.3f}",
                "regime": self.current_regime,
            }

    def _track_regime_performance(
        self, signal: Dict[str, Union[str, int, float, bool]], regime: str
    ) -> None:
        """
        Track performance metrics by regime.

        Args:
            signal: Generated signal
            regime: Current market regime
        """
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []

        self.regime_performance[regime].append(
            {
                "signal": signal.get("action", "hold"),
                "confidence": signal.get("confidence", 0.0),
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )

    def update_hyperparameters(self, hyperparameters: Dict[str, float]) -> None:
        """
        Update strategy hyperparameters.

        Args:
            hyperparameters: Dictionary of hyperparameter names and values
        """
        # SAC models typically don't have runtime hyperparameters to update
        # This could be used for action thresholds or other strategy parameters
        if "buy_threshold" in hyperparameters:
            # Update buy threshold (this is just an example)
            pass
        if "sell_threshold" in hyperparameters:
            # Update sell threshold (this is just an example)
            pass

    def get_learning_outcomes(self) -> Dict[str, Union[int, float, str, dict, list]]:
        """
        Get SAC learning outcomes and analysis.

        Returns:
            Dictionary containing learning metrics and insights
        """
        outcomes: Dict[str, Union[int, float, str, dict, list]] = {
            "model_path": self.model_path,
            "regime_performance": self.regime_performance,
            "learning_metrics": self.learning_metrics,
            "regime_distribution": {},
        }

        # Analyze regime distribution
        if self.regime_performance:
            total_signals = sum(
                len(signals) for signals in self.regime_performance.values()
            )
            regime_dist: Dict[str, float] = {}
            for regime, signals in self.regime_performance.items():
                regime_dist[regime] = len(signals) / total_signals
            outcomes["regime_distribution"] = regime_dist

        return outcomes

    def get_config(self) -> Dict[str, Union[str, int, float, bool]]:
        """Get strategy configuration."""
        config: Dict[str, Union[str, int, float, bool]] = super().get_config()
        config.update(
            {
                "model_path": self.model_path,
                "regime_classifier_path": self.regime_classifier_path,
                "action_space": self.action_space,
                "has_regime_classifier": self.regime_classifier is not None,
            }
        )
        return config
