"""
Regime Adaptive Training Mixin

This module provides a mixin class that adds market regime adaptation
capabilities to SAC training classes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ztb.analysis.market_regime_classifier import (
    MarketRegimeClassifier,
    RegimeDetectionResult,
    RegimeType,
)

logger = logging.getLogger(__name__)


class RegimeAdaptiveTrainerMixin(ABC):
    """
    Mixin class that adds market regime adaptation capabilities to trainers

    This mixin provides:
    - Dynamic hyperparameter adjustment based on market regime
    - Regime-aware reward scaling
    - Adaptive exploration strategies
    - Performance tracking per regime
    """

    def __init__(self, regime_config: Optional[Dict[str, Any]] = None):
        """
        Initialize regime adaptive capabilities

        Args:
            regime_config: Configuration for regime adaptation
        """
        self.regime_config = regime_config or self._get_default_regime_config()
        self.regime_classifier = None
        self.current_regime = None
        self.regime_history = []
        self.regime_performance = {}
        self.regime_adaptation_enabled = self.regime_config.get("enabled", True)

        if self.regime_adaptation_enabled:
            self._initialize_regime_classifier()

    def _get_default_regime_config(self) -> Dict[str, Any]:
        """Get default regime adaptation configuration"""
        return {
            "enabled": True,
            "regime_classifier_config": {
                "regime_scheme": "comprehensive",
                "use_multi_timeframe": True,
                "confidence_threshold": 0.6,
            },
            "adaptation_rules": {
                "hyperparameter_adjustment": True,
                "reward_scaling": True,
                "exploration_adjustment": True,
            },
            "regime_specific_params": {
                "strong_bull": {
                    "ent_coef": 0.01,
                    "learning_rate": 0.0003,
                    "reward_scale": 1.2,
                },
                "strong_bear": {
                    "ent_coef": 0.01,
                    "learning_rate": 0.0003,
                    "reward_scale": 1.2,
                },
                "high_volatility_range": {
                    "ent_coef": 0.05,
                    "learning_rate": 0.0005,
                    "reward_scale": 0.8,
                },
                "consolidation": {
                    "ent_coef": 0.02,
                    "learning_rate": 0.0001,
                    "reward_scale": 1.5,
                },
            },
            "adaptation_frequency": 100,  # steps
            "performance_tracking_window": 1000,
        }

    def _initialize_regime_classifier(self):
        """Initialize the market regime classifier"""
        try:
            classifier_config = self.regime_config.get("regime_classifier_config", {})
            self.regime_classifier = MarketRegimeClassifier(classifier_config)
            logger.info("Regime classifier initialized for adaptive training")
        except Exception as e:
            logger.error(f"Failed to initialize regime classifier: {e}")
            self.regime_adaptation_enabled = False

    def detect_market_regime(
        self, data: pd.DataFrame, current_index: int = -1
    ) -> Optional[RegimeDetectionResult]:
        """
        Detect current market regime

        Args:
            data: Market data DataFrame
            current_index: Current data index

        Returns:
            RegimeDetectionResult or None if detection fails
        """
        if not self.regime_adaptation_enabled or self.regime_classifier is None:
            return None

        try:
            result = self.regime_classifier.detect_regime(data, current_index)
            self.current_regime = result.primary_regime
            self.regime_history.append(result)

            # Keep history limited
            max_history = self.regime_config.get("max_history", 1000)
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]

            return result

        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return None

    def get_regime_specific_parameters(self, regime: RegimeType) -> Dict[str, Any]:
        """
        Get regime-specific training parameters

        Args:
            regime: Current market regime

        Returns:
            Dictionary of regime-specific parameters
        """
        regime_key = regime.value

        # Get base parameters
        base_params = self.regime_config.get("regime_specific_params", {})

        # Return regime-specific params or defaults
        return base_params.get(
            regime_key, {"ent_coef": 0.02, "learning_rate": 0.0003, "reward_scale": 1.0}
        )

    def adapt_hyperparameters_for_regime(self, regime: RegimeType) -> Dict[str, Any]:
        """
        Adapt hyperparameters based on current market regime

        Args:
            regime: Current market regime

        Returns:
            Dictionary of adapted hyperparameters
        """
        if not self.regime_config.get("adaptation_rules", {}).get(
            "hyperparameter_adjustment", True
        ):
            return {}

        regime_params = self.get_regime_specific_parameters(regime)

        adapted_params = {}

        # Adapt entropy coefficient for exploration
        if "ent_coef" in regime_params:
            adapted_params["ent_coef"] = regime_params["ent_coef"]

        # Adapt learning rate
        if "learning_rate" in regime_params:
            adapted_params["learning_rate"] = regime_params["learning_rate"]

        logger.info(
            f"Adapted hyperparameters for regime {regime.value}: {adapted_params}"
        )
        return adapted_params

    def get_regime_aware_reward_scale(self, regime: RegimeType) -> float:
        """
        Get reward scaling factor based on market regime

        Args:
            regime: Current market regime

        Returns:
            Reward scaling factor
        """
        if not self.regime_config.get("adaptation_rules", {}).get(
            "reward_scaling", True
        ):
            return 1.0

        regime_params = self.get_regime_specific_parameters(regime)
        return regime_params.get("reward_scale", 1.0)

    def should_adapt_now(self, step_count: int) -> bool:
        """
        Check if adaptation should occur at current step

        Args:
            step_count: Current training step count

        Returns:
            True if adaptation should occur
        """
        frequency = self.regime_config.get("adaptation_frequency", 100)
        return step_count % frequency == 0

    def update_regime_performance(
        self, regime: RegimeType, reward: float, step_count: int
    ):
        """
        Update performance tracking for current regime

        Args:
            regime: Current market regime
            reward: Recent reward value
            step_count: Current step count
        """
        regime_key = regime.value

        if regime_key not in self.regime_performance:
            self.regime_performance[regime_key] = {
                "rewards": [],
                "step_counts": [],
                "avg_reward": 0.0,
                "total_steps": 0,
            }

        perf_data = self.regime_performance[regime_key]

        # Add new data point
        perf_data["rewards"].append(reward)
        perf_data["step_counts"].append(step_count)

        # Maintain window size
        window_size = self.regime_config.get("performance_tracking_window", 1000)
        if len(perf_data["rewards"]) > window_size:
            perf_data["rewards"] = perf_data["rewards"][-window_size:]
            perf_data["step_counts"] = perf_data["step_counts"][-window_size:]

        # Update statistics
        perf_data["avg_reward"] = np.mean(perf_data["rewards"])
        perf_data["total_steps"] = len(perf_data["step_counts"])

    def get_regime_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of performance across all regimes

        Returns:
            Dictionary with regime performance statistics
        """
        summary = {}

        for regime_key, perf_data in self.regime_performance.items():
            summary[regime_key] = {
                "average_reward": perf_data["avg_reward"],
                "total_steps": perf_data["total_steps"],
                "sample_count": len(perf_data["rewards"]),
            }

        return summary

    def get_adaptation_suggestions(self) -> List[str]:
        """
        Get suggestions for improving regime adaptation

        Returns:
            List of suggestion strings
        """
        suggestions = []

        if not self.regime_adaptation_enabled:
            suggestions.append("Enable regime adaptation for better performance")
            return suggestions

        perf_summary = self.get_regime_performance_summary()

        if not perf_summary:
            suggestions.append("Collect more regime performance data for analysis")
            return suggestions

        # Analyze performance patterns
        avg_rewards = {k: v["average_reward"] for k, v in perf_summary.items()}

        if avg_rewards:
            best_regime = max(avg_rewards, key=avg_rewards.get)
            worst_regime = min(avg_rewards, key=avg_rewards.get)

            if avg_rewards[best_regime] > 0 and avg_rewards[worst_regime] < 0:
                suggestions.append(
                    f"Consider adjusting parameters for {worst_regime} regime"
                )
                suggestions.append(f"Use {best_regime} regime parameters as reference")

        # Check regime distribution
        regime_counts = {}
        for result in self.regime_history[-100:]:  # Last 100 detections
            regime_key = result.primary_regime.value
            regime_counts[regime_key] = regime_counts.get(regime_key, 0) + 1

        if len(regime_counts) < 3:
            suggestions.append(
                "Limited regime diversity detected - consider adjusting detection thresholds"
            )

        return suggestions

    def export_regime_data(self, filepath: str):
        """
        Export regime detection and performance data

        Args:
            filepath: Path to export data
        """
        try:
            export_data = {
                "regime_history": [
                    {
                        "timestamp": result.detection_timestamp.isoformat(),
                        "regime": result.primary_regime.value,
                        "confidence": result.confidence,
                        "secondary_regimes": [
                            {"regime": r.value, "confidence": c}
                            for r, c in result.secondary_regimes
                        ],
                    }
                    for result in self.regime_history
                ],
                "regime_performance": self.regime_performance,
                "current_regime": self.current_regime.value
                if self.current_regime
                else None,
                "config": self.regime_config,
            }

            import json

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Regime data exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export regime data: {e}")

    def load_regime_data(self, filepath: str):
        """
        Load regime detection and performance data

        Args:
            filepath: Path to load data from
        """
        try:
            import json

            with open(filepath, "r") as f:
                data = json.load(f)

            # Restore performance data
            self.regime_performance = data.get("regime_performance", {})

            # Restore current regime if available
            if data.get("current_regime"):
                self.current_regime = RegimeType(data["current_regime"])

            logger.info(f"Regime data loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load regime data: {e}")

    @abstractmethod
    def apply_hyperparameter_adaptation(self, adapted_params: Dict[str, Any]):
        """
        Apply adapted hyperparameters to the training process

        Args:
            adapted_params: Dictionary of parameters to apply
        """
        pass

    @abstractmethod
    def get_current_market_data(self) -> Optional[pd.DataFrame]:
        """
        Get current market data for regime detection

        Returns:
            DataFrame with market data or None
        """
        pass

    @abstractmethod
    def get_current_step_count(self) -> int:
        """
        Get current training step count

        Returns:
            Current step count
        """
        pass
