"""
Validation Utilities

Provides validation functions for weight adjustment system components.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ValidationUtils:
    """
    Utility class for validating weight adjustment system inputs and outputs.
    """

    @staticmethod
    def validate_weight_dict(weights: Dict[str, float]) -> bool:
        """
        Validate a weight dictionary.

        Args:
            weights: Dictionary of feature weights

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(weights, dict):
            logger.error("Weights must be a dictionary")
            return False

        if not weights:
            logger.error("Weights dictionary cannot be empty")
            return False

        for feature, weight in weights.items():
            if not isinstance(feature, str):
                logger.error(f"Feature name must be string, got {type(feature)}")
                return False

            if not isinstance(weight, (int, float)):
                logger.error(f"Weight must be numeric, got {type(weight)} for feature {feature}")
                return False

            if not np.isfinite(weight):
                logger.error(f"Weight must be finite, got {weight} for feature {feature}")
                return False

        return True

    @staticmethod
    def validate_performance_data(data: Dict[str, Any]) -> bool:
        """
        Validate performance data structure.

        Args:
            data: Performance data dictionary

        Returns:
            True if valid, False otherwise
        """
        required_keys = ['total_return', 'win_rate', 'trade_count']

        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key: {key}")
                return False

        # Validate win_rate range
        win_rate = data.get('win_rate', 0)
        if not (0.0 <= win_rate <= 1.0):
            logger.error(f"Win rate must be between 0 and 1, got {win_rate}")
            return False

        # Validate trade_count
        trade_count = data.get('trade_count', 0)
        if not isinstance(trade_count, (int, float)) or trade_count < 0:
            logger.error(f"Trade count must be non-negative number, got {trade_count}")
            return False

        return True

    @staticmethod
    def validate_feature_importance(importance: Dict[str, float]) -> bool:
        """
        Validate feature importance scores.

        Args:
            importance: Feature importance dictionary

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(importance, dict):
            logger.error("Feature importance must be a dictionary")
            return False

        for feature, score in importance.items():
            if not isinstance(feature, str):
                logger.error(f"Feature name must be string, got {type(feature)}")
                return False

            if not isinstance(score, (int, float)):
                logger.error(f"Importance score must be numeric, got {type(score)} for feature {feature}")
                return False

            if not (0.0 <= score <= 1.0):
                logger.error(f"Importance score must be between 0 and 1, got {score} for feature {feature}")
                return False

        return True

    @staticmethod
    def validate_market_conditions(conditions: Dict[str, Any]) -> bool:
        """
        Validate market conditions data.

        Args:
            conditions: Market conditions dictionary

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(conditions, dict):
            logger.error("Market conditions must be a dictionary")
            return False

        # Check for common market condition indicators
        valid_keys = ['volatility', 'trend', 'volume', 'regime', 'momentum']

        if not any(key in conditions for key in valid_keys):
            logger.warning("No recognized market condition indicators found")

        # Validate numeric values
        for key, value in conditions.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    logger.error(f"Market condition {key} must be finite, got {value}")
                    return False
            elif isinstance(value, str):
                # String values should be non-empty
                if not value.strip():
                    logger.error(f"Market condition {key} string value cannot be empty")
                    return False
            # Other types are allowed (booleans, etc.)

        return True

    @staticmethod
    def validate_adjustment_config(config: Dict[str, Any]) -> bool:
        """
        Validate adjustment configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(config, dict):
            logger.error("Configuration must be a dictionary")
            return False

        # Check required configuration keys
        required_keys = ['enabled', 'learning_rate', 'max_adjustment_rate']

        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required configuration key: {key}")
                return False

        # Validate learning rate
        learning_rate = config.get('learning_rate', 0)
        if not (0.0 < learning_rate <= 1.0):
            logger.error(f"Learning rate must be between 0 and 1, got {learning_rate}")
            return False

        # Validate max adjustment rate
        max_rate = config.get('max_adjustment_rate', 0)
        if not (0.0 < max_rate <= 1.0):
            logger.error(f"Max adjustment rate must be between 0 and 1, got {max_rate}")
            return False

        # Validate weight constraints
        min_weight = config.get('min_weight', 0.0)
        max_weight = config.get('max_weight', 1.0)

        if not (0.0 <= min_weight <= max_weight <= 1.0):
            logger.error(f"Weight constraints invalid: min={min_weight}, max={max_weight}")
            return False

        return True

    @staticmethod
    def check_data_quality(data: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, bool]:
        """
        Check data quality against thresholds.

        Args:
            data: Data to check
            thresholds: Quality thresholds

        Returns:
            Dictionary of quality check results
        """
        results = {}

        # Check data completeness
        total_points = data.get('total_points', 0)
        valid_points = data.get('valid_points', 0)
        completeness_threshold = thresholds.get('completeness', 0.8)

        results['completeness'] = (valid_points / total_points) >= completeness_threshold if total_points > 0 else False

        # Check data recency
        latest_timestamp = data.get('latest_timestamp')
        current_time = data.get('current_time')

        if latest_timestamp and current_time:
            age_hours = (current_time - latest_timestamp).total_seconds() / 3600
            max_age_hours = thresholds.get('max_age_hours', 24)
            results['recency'] = age_hours <= max_age_hours
        else:
            results['recency'] = False

        # Check statistical quality
        std_dev = data.get('std_dev', 0)
        min_std_dev = thresholds.get('min_std_dev', 0.01)
        results['variability'] = std_dev >= min_std_dev

        # Check for outliers
        outlier_ratio = data.get('outlier_ratio', 0)
        max_outlier_ratio = thresholds.get('max_outlier_ratio', 0.1)
        results['outlier_ratio'] = outlier_ratio <= max_outlier_ratio

        return results

    @staticmethod
    def calculate_data_quality_score(data: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score.

        Args:
            data: Data quality metrics

        Returns:
            Quality score between 0 and 1
        """
        quality_checks = ValidationUtils.check_data_quality(data, {})

        # Weight the quality checks
        weights = {
            'completeness': 0.4,
            'recency': 0.3,
            'variability': 0.2,
            'outlier_ratio': 0.1
        }

        score = 0.0
        total_weight = 0.0

        for check, passed in quality_checks.items():
            if check in weights:
                score += weights[check] if passed else 0.0
                total_weight += weights[check]

        return score / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def sanitize_weights(weights: Dict[str, float], constraints: Dict[str, float]) -> Dict[str, float]:
        """
        Sanitize weights according to constraints.

        Args:
            weights: Raw weights
            constraints: Weight constraints

        Returns:
            Sanitized weights
        """
        sanitized = {}

        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)

        for feature, weight in weights.items():
            # Clamp to valid range
            sanitized_weight = max(min_weight, min(max_weight, weight))

            # Ensure finite
            if not np.isfinite(sanitized_weight):
                sanitized_weight = (min_weight + max_weight) / 2  # Use midpoint

            sanitized[feature] = sanitized_weight

        return sanitized