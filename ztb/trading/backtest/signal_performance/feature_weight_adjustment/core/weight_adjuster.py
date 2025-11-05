"""
Dynamic Weight Adjuster

Core component that orchestrates feature weight adjustments based on
SAC learning and signal performance data.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from ztb.utils.logging_utils import get_logger

from ..interfaces.adjustment_interface import WeightAdjustmentInterface
from ..interfaces.data_provider_interface import DataProviderInterface
from ..config.adjustment_config import AdjustmentConfig, AdjustmentStrategyType
from ..utils.validation_utils import ValidationUtils
from ..utils.data_processor import DataProcessor


logger = get_logger(__name__)


class DynamicWeightAdjuster:
    """
    Main orchestrator for dynamic feature weight adjustment.

    This class coordinates data collection, validation, and weight adjustment
    using configurable strategies.
    """

    def __init__(
        self,
        data_provider: DataProviderInterface,
        adjustment_strategy: WeightAdjustmentInterface,
        config: Optional[AdjustmentConfig] = None
    ):
        """
        Initialize DynamicWeightAdjuster.

        Args:
            data_provider: Provider of SAC learning and signal performance data
            adjustment_strategy: Strategy for adjusting weights
            config: Configuration for the adjustment system
        """
        self.data_provider = data_provider
        self.adjustment_strategy = adjustment_strategy
        self.config = config or AdjustmentConfig()

        # Initialize utilities
        self.validator = ValidationUtils()
        self.data_processor = DataProcessor()

        # State tracking
        self.current_weights: Dict[str, float] = {}
        self.adjustment_history: List[Dict[str, Any]] = []
        self.last_adjustment_time: Optional[datetime] = None

        # Performance tracking
        self.consecutive_adjustments = 0
        self.total_adjustments = 0

        logger.info("DynamicWeightAdjuster initialized")

    def set_weights(self, weights: Dict[str, float]) -> bool:
        """
        Set initial or current feature weights.

        Args:
            weights: Feature weights to set

        Returns:
            True if weights were set successfully, False otherwise
        """
        if not self.validator.validate_weight_dict(weights):
            logger.error("Invalid weights provided")
            return False

        # Sanitize weights according to config
        constraints = {
            'min_weight': self.config.min_weight,
            'max_weight': self.config.max_weight
        }
        sanitized_weights = self.validator.sanitize_weights(weights, constraints)

        # Normalize if required
        if self.config.weight_normalization:
            sanitized_weights = self.data_processor.normalize_weights(sanitized_weights)

        self.current_weights = sanitized_weights
        logger.info(f"Set {len(sanitized_weights)} feature weights")
        return True

    def get_weights(self) -> Dict[str, float]:
        """
        Get current feature weights.

        Returns:
            Current feature weights
        """
        return self.current_weights.copy()

    def should_adjust_weights(self) -> bool:
        """
        Determine if weights should be adjusted based on configuration and state.

        Returns:
            True if adjustment should proceed, False otherwise
        """
        if not self.config.enabled:
            return False

        # Check data availability
        if not self.data_provider.is_data_available():
            logger.debug("Insufficient data for weight adjustment")
            return False

        # Check data quality
        quality_metrics = self.data_provider.get_data_quality_metrics()
        if quality_metrics.get('overall_quality', 1.0) < self.config.data_quality_threshold:
            logger.debug("Data quality below threshold for adjustment")
            return False

        # Check consecutive adjustment limit
        if self.consecutive_adjustments >= self.config.max_consecutive_adjustments:
            logger.warning("Maximum consecutive adjustments reached")
            return False

        # Check minimum data points
        performance_data = self.data_provider.get_signal_performance_data()
        trade_count = performance_data.get('trade_count', 0)
        if trade_count < self.config.min_data_points:
            logger.debug(f"Insufficient data points: {trade_count} < {self.config.min_data_points}")
            return False

        return True

    def adjust_weights(self) -> Dict[str, float]:
        """
        Perform weight adjustment based on current data and strategy.

        Returns:
            Adjusted feature weights
        """
        if not self.should_adjust_weights():
            logger.debug("Weight adjustment skipped")
            return self.current_weights.copy()

        try:
            # Collect data for adjustment
            performance_data = self.data_provider.get_signal_performance_data()
            feature_importance = self.data_provider.get_feature_importance_data()
            market_conditions = self.data_provider.get_market_conditions()

            # Validate data
            if not self.validator.validate_performance_data(performance_data):
                logger.error("Invalid performance data")
                return self.current_weights.copy()

            if not self.validator.validate_feature_importance(feature_importance):
                logger.error("Invalid feature importance data")
                return self.current_weights.copy()

            if not self.validator.validate_market_conditions(market_conditions):
                logger.warning("Invalid market conditions data, proceeding without it")
                market_conditions = {}

            # Perform adjustment
            adjusted_weights = self.adjustment_strategy.adjust_weights(
                current_weights=self.current_weights,
                performance_data=performance_data,
                feature_importance=feature_importance,
                market_conditions=market_conditions
            )

            # Validate adjusted weights
            if not self.adjustment_strategy.validate_weights(adjusted_weights):
                logger.error("Adjustment strategy produced invalid weights")
                if self.config.rollback_on_failure:
                    logger.info("Rolling back to previous weights")
                    return self.current_weights.copy()
                else:
                    raise ValueError("Invalid weights produced by adjustment strategy")

            # Sanitize and normalize
            constraints = {
                'min_weight': self.config.min_weight,
                'max_weight': self.config.max_weight
            }
            sanitized_weights = self.validator.sanitize_weights(adjusted_weights, constraints)

            if self.config.weight_normalization:
                sanitized_weights = self.data_processor.normalize_weights(sanitized_weights)

            # Record adjustment
            self._record_adjustment(
                old_weights=self.current_weights,
                new_weights=sanitized_weights,
                performance_data=performance_data,
                feature_importance=feature_importance,
                market_conditions=market_conditions
            )

            # Update state
            self.current_weights = sanitized_weights
            self.last_adjustment_time = datetime.now()
            self.consecutive_adjustments += 1
            self.total_adjustments += 1

            logger.info(f"Successfully adjusted {len(sanitized_weights)} feature weights")
            return sanitized_weights.copy()

        except Exception as e:
            logger.error(f"Weight adjustment failed: {e}")
            if self.config.rollback_on_failure:
                logger.info("Rolling back to previous weights due to error")
                return self.current_weights.copy()
            else:
                raise

    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """
        Get history of weight adjustments.

        Returns:
            List of adjustment records
        """
        return self.adjustment_history.copy()

    def get_adjustment_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about weight adjustments.

        Returns:
            Dictionary of adjustment statistics
        """
        if not self.adjustment_history:
            return {"total_adjustments": 0}

        # Calculate statistics
        weight_changes = []
        performance_changes = []

        for adjustment in self.adjustment_history:
            # Calculate average weight change
            old_weights = adjustment.get('old_weights', {})
            new_weights = adjustment.get('new_weights', {})

            if old_weights and new_weights:
                changes = []
                for feature in old_weights:
                    if feature in new_weights:
                        change = abs(new_weights[feature] - old_weights[feature])
                        changes.append(change)
                if changes:
                    weight_changes.append(np.mean(changes))

            # Track performance changes
            performance = adjustment.get('performance_data', {})
            if 'total_return' in performance:
                performance_changes.append(performance['total_return'])

        return {
            "total_adjustments": self.total_adjustments,
            "consecutive_adjustments": self.consecutive_adjustments,
            "average_weight_change": np.mean(weight_changes) if weight_changes else 0.0,
            "max_weight_change": np.max(weight_changes) if weight_changes else 0.0,
            "performance_trend": np.mean(performance_changes) if performance_changes else 0.0,
            "last_adjustment_time": self.last_adjustment_time.isoformat() if self.last_adjustment_time else None,
        }

    def reset(self) -> None:
        """
        Reset the adjuster to initial state.
        """
        self.current_weights = {}
        self.adjustment_history.clear()
        self.last_adjustment_time = None
        self.consecutive_adjustments = 0
        self.total_adjustments = 0

        # Reset strategy
        self.adjustment_strategy.reset()

        logger.info("DynamicWeightAdjuster reset to initial state")

    def _record_adjustment(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        performance_data: Dict[str, Any],
        feature_importance: Dict[str, float],
        market_conditions: Dict[str, Any]
    ) -> None:
        """
        Record a weight adjustment in history.

        Args:
            old_weights: Weights before adjustment
            new_weights: Weights after adjustment
            performance_data: Performance data used for adjustment
            feature_importance: Feature importance data used
            market_conditions: Market conditions at adjustment time
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "old_weights": old_weights.copy(),
            "new_weights": new_weights.copy(),
            "performance_data": performance_data.copy(),
            "feature_importance": feature_importance.copy(),
            "market_conditions": market_conditions.copy(),
            "strategy_metadata": self.adjustment_strategy.get_adjustment_metadata(),
            "config_snapshot": self.config.to_dict(),
        }

        self.adjustment_history.append(record)

        # Maintain history size
        if len(self.adjustment_history) > self.config.max_history_length:
            self.adjustment_history = self.adjustment_history[-self.config.max_history_length:]

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.

        Returns:
            Dictionary containing system status information
        """
        return {
            "enabled": self.config.enabled,
            "current_weights_count": len(self.current_weights),
            "total_adjustments": self.total_adjustments,
            "consecutive_adjustments": self.consecutive_adjustments,
            "last_adjustment_time": self.last_adjustment_time.isoformat() if self.last_adjustment_time else None,
            "data_available": self.data_provider.is_data_available(),
            "data_quality": self.data_provider.get_data_quality_metrics(),
            "strategy_type": self.config.strategy_type.value,
            "adjustment_frequency": self.config.adjustment_frequency.value,
        }