"""
Unit Tests for Dynamic Weight Adjustment System

Tests for the core weight adjustment functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from ztb.trading.backtest.signal_performance.feature_weight_adjustment import (
    DynamicWeightAdjuster,
    PerformanceDrivenStrategy,
    CorrelationBasedStrategy,
    ReinforcementLearningStrategy,
    AdjustmentStrategyRegistry,
    PerformanceEvaluator,
    AdjustmentConfig,
    AdjustmentStrategyType,
)


class TestDynamicWeightAdjuster:
    """Test cases for DynamicWeightAdjuster."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AdjustmentConfig(
            strategy_type=AdjustmentStrategyType.PERFORMANCE_DRIVEN,
            max_adjustment_rate=0.05,
            min_weight=0.01,
            max_weight=1.0,
        )

        # Mock data provider and strategy
        self.mock_data_provider = Mock()
        self.mock_data_provider.get_signal_performance_data.return_value = {
            "trade_count": 100,
            "win_rate": 0.6,
            "total_return": 0.05,
            "feature_performance": {
                "feature1": {"win_rate": 0.7, "return_contribution": 0.03},
                "feature2": {"win_rate": 0.5, "return_contribution": 0.02},
            }
        }
        self.mock_data_provider.get_feature_importance_data.return_value = {
            "feature1": 0.8,
            "feature2": 0.6,
        }
        self.mock_data_provider.get_market_conditions.return_value = {}
        self.mock_data_provider.is_data_available.return_value = True
        self.mock_data_provider.get_data_quality_metrics.return_value = {"overall_quality": 0.9}

        self.mock_strategy = Mock()
        self.mock_strategy.adjust_weights.return_value = {"feature1": 0.6, "feature2": 0.4}
        self.mock_strategy.validate_weights.return_value = True

        self.adjuster = DynamicWeightAdjuster(
            data_provider=self.mock_data_provider,
            adjustment_strategy=self.mock_strategy,
            config=self.config
        )

        # Set initial weights
        self.adjuster.set_weights({"feature1": 0.5, "feature2": 0.5})

    def test_initialization(self):
        """Test proper initialization."""
        assert self.adjuster.config == self.config
        assert self.adjuster.total_adjustments == 0
        assert isinstance(self.adjuster.adjustment_strategy, Mock)

    def test_adjust_weights_success(self):
        """Test successful weight adjustment."""
        result = self.adjuster.adjust_weights()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"feature1", "feature2"}
        assert abs(sum(result.values()) - 1.0) < 1e-6  # Should sum to 1.0

    def test_adjust_weights_invalid_input(self):
        """Test handling of invalid inputs."""
        # Test with invalid data from provider
        self.mock_data_provider.get_signal_performance_data.return_value = {}
        result = self.adjuster.adjust_weights()
        # Should return current weights when data is invalid
        assert result == self.adjuster.current_weights

    def test_get_adjustment_history(self):
        """Test retrieval of adjustment history."""
        history = self.adjuster.get_adjustment_history()
        assert isinstance(history, list)
        assert len(history) == 0  # Initially empty

    def test_reset(self):
        """Test reset functionality."""
        # Make some adjustments first
        self.adjuster.adjust_weights()

        # Reset
        self.adjuster.reset()
        assert self.adjuster.total_adjustments == 0


class TestPerformanceDrivenStrategy:
    """Test cases for PerformanceDrivenStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"adjustment_rate": 0.05}
        self.strategy = PerformanceDrivenStrategy(self.config)

    def test_initialization(self):
        """Test proper initialization."""
        assert self.strategy.config["adjustment_rate"] == 0.05
        assert self.strategy.adjustment_count == 0

    def test_adjust_weights_good_performance(self):
        """Test adjustment with good performance."""
        current_weights = {"feature1": 0.5, "feature2": 0.5}
        performance_data = {
            "win_rate": 0.6,  # Added required field
            "total_return": 0.05,  # Added required field
            "trade_count": 100,  # Added required field
            "feature_performance": {
                "feature1": {"win_rate": 0.8, "return_contribution": 0.05, "sharpe_ratio": 1.2},
                "feature2": {"win_rate": 0.4, "return_contribution": -0.01, "sharpe_ratio": 0.3},
            }
        }
        feature_importance = {"feature1": 0.8, "feature2": 0.4}

        result = self.strategy.adjust_weights(
            current_weights, performance_data, feature_importance
        )

        # Feature1 should get higher weight due to better performance
        assert result["feature1"] > result["feature2"]

    def test_market_condition_adjustment(self):
        """Test market condition adjustments."""
        current_weights = {"feature1": 0.5, "feature2": 0.5}
        performance_data = {
            "feature_performance": {
                "feature1": {"win_rate": 0.6},
                "feature2": {"win_rate": 0.6},
            }
        }
        feature_importance = {"feature1": 0.5, "feature2": 0.5}

        # High volatility should reduce adjustment magnitude
        market_conditions = {"volatility": 0.8, "trend_strength": 0.5}

        result = self.strategy.adjust_weights(
            current_weights, performance_data, feature_importance, market_conditions
        )

        assert isinstance(result, dict)


class TestCorrelationBasedStrategy:
    """Test cases for CorrelationBasedStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"correlation_threshold": 0.8}
        self.strategy = CorrelationBasedStrategy(self.config)

    def test_adjust_weights_high_correlation(self):
        """Test adjustment with highly correlated features."""
        current_weights = {"feature1": 0.5, "feature2": 0.5}
        # High correlation between features
        correlation_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
        performance_data = {"feature_correlations": correlation_matrix}
        feature_importance = {"feature1": 0.6, "feature2": 0.8}  # feature2 is more important

        result = self.strategy.adjust_weights(
            current_weights, performance_data, feature_importance
        )

        # Less important correlated feature should get penalty
        assert result["feature1"] < current_weights["feature1"]

    def test_adjust_weights_no_correlation_data(self):
        """Test handling when no correlation data is available."""
        current_weights = {"feature1": 0.5, "feature2": 0.5}
        performance_data = {}  # No correlation data
        feature_importance = {"feature1": 0.5, "feature2": 0.5}

        result = self.strategy.adjust_weights(
            current_weights, performance_data, feature_importance
        )

        # Should return unchanged weights
        assert result == current_weights


class TestReinforcementLearningStrategy:
    """Test cases for ReinforcementLearningStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"learning_rate": 0.01, "exploration_rate": 0.1}
        self.strategy = ReinforcementLearningStrategy(self.config)

    def test_initialization(self):
        """Test proper initialization."""
        assert len(self.strategy.q_table) == 0
        assert len(self.strategy.state_history) == 0

    def test_adjust_weights_basic(self):
        """Test basic RL adjustment."""
        current_weights = {"feature1": 0.5, "feature2": 0.5}
        performance_data = {"win_rate": 0.6, "total_return": 0.03}
        feature_importance = {"feature1": 0.7, "feature2": 0.5}

        result = self.strategy.adjust_weights(
            current_weights, performance_data, feature_importance
        )

        assert isinstance(result, dict)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_q_table_update(self):
        """Test Q-table learning."""
        # Make multiple adjustments to build history
        current_weights = {"feature1": 0.5, "feature2": 0.5}
        performance_data = {"win_rate": 0.6, "total_return": 0.03}
        feature_importance = {"feature1": 0.7, "feature2": 0.5}

        for _ in range(5):
            result = self.strategy.adjust_weights(
                current_weights, performance_data, feature_importance
            )
            current_weights = result

        # Q-table should have been populated
        assert len(self.strategy.q_table) > 0


class TestAdjustmentStrategyRegistry:
    """Test cases for AdjustmentStrategyRegistry."""

    def test_register_and_get_strategy(self):
        """Test strategy registration and retrieval."""
        # Register a test strategy
        AdjustmentStrategyRegistry.register("test_strategy", PerformanceDrivenStrategy)

        strategy = AdjustmentStrategyRegistry.get_strategy("test_strategy")
        assert isinstance(strategy, PerformanceDrivenStrategy)

    def test_get_unknown_strategy(self):
        """Test error handling for unknown strategies."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            AdjustmentStrategyRegistry.get_strategy("unknown_strategy")

    def test_list_strategies(self):
        """Test listing registered strategies."""
        strategies = AdjustmentStrategyRegistry.list_strategies()
        assert isinstance(strategies, list)
        assert "performance_driven" in strategies

    def test_clear_registry(self):
        """Test registry clearing."""
        initial_count = len(AdjustmentStrategyRegistry.list_strategies())
        AdjustmentStrategyRegistry.clear_registry()
        assert len(AdjustmentStrategyRegistry.list_strategies()) == 0


class TestPerformanceEvaluator:
    """Test cases for PerformanceEvaluator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = PerformanceEvaluator()

    def test_evaluate_adjustment_impact(self):
        """Test impact evaluation."""
        before_weights = {"feature1": 0.4, "feature2": 0.6}
        after_weights = {"feature1": 0.6, "feature2": 0.4}
        performance_data = {"win_rate": 0.65, "total_return": 0.04}
        feature_importance = {"feature1": 0.8, "feature2": 0.5}

        result = self.evaluator.evaluate_adjustment_impact(
            before_weights, after_weights, performance_data, feature_importance
        )

        assert "impact_score" in result
        assert "recommendation" in result
        assert isinstance(result["impact_score"], float)

    def test_get_performance_trends(self):
        """Test trend analysis."""
        # Need some history first
        self.evaluator.performance_history = [
            {"impact_score": 0.1, "evaluation_timestamp": "2024-01-01T00:00:00"},
            {"impact_score": 0.2, "evaluation_timestamp": "2024-01-02T00:00:00"},
            {"impact_score": 0.15, "evaluation_timestamp": "2024-01-03T00:00:00"},
        ]

        trends = self.evaluator.get_performance_trends()
        assert "trend_analysis" in trends
        assert "improving" in trends["trend_analysis"]["trend_direction"]

    def test_get_adjustment_effectiveness(self):
        """Test effectiveness metrics."""
        # Add some history
        self.evaluator.performance_history = [
            {"impact_score": 0.1},
            {"impact_score": -0.05},
            {"impact_score": 0.2},
            {"impact_score": 0.15},
            {"impact_score": -0.1},
        ]

        effectiveness = self.evaluator.get_adjustment_effectiveness()
        assert "success_rate" in effectiveness
        assert "average_impact" in effectiveness
        assert effectiveness["total_adjustments"] == 5


if __name__ == "__main__":
    pytest.main([__file__])