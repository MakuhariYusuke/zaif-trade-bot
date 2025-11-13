"""
Unit tests for signal common metrics

Tests the standardized metric calculation functions used across
signal processing components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from ztb.trading.signal.common.metrics import (
    calculate_trend_metrics, calculate_volatility_metrics,
    calculate_momentum_metrics, calculate_volume_metrics,
    calculate_support_resistance_metrics, calculate_composite_score
)


class TestCalculateTrendMetrics:
    """Test calculate_trend_metrics function"""

    def test_calculate_trend_metrics_uptrend(self):
        """Test trend metrics calculation for uptrend"""
        # Create uptrend data
        prices = [100, 102, 105, 108, 110, 112, 115, 118, 120, 122]
        data = pd.DataFrame({'close': prices})

        metrics = calculate_trend_metrics(data)

        assert 'trend_strength' in metrics
        assert 'bull_strength' in metrics
        assert 'bear_strength' in metrics
        assert metrics['trend_strength'] > 0  # Positive trend
        assert metrics['bull_strength'] > metrics['bear_strength']

    def test_calculate_trend_metrics_downtrend(self):
        """Test trend metrics calculation for downtrend"""
        # Create downtrend data
        prices = [120, 118, 115, 112, 110, 108, 105, 102, 100, 98]
        data = pd.DataFrame({'close': prices})

        metrics = calculate_trend_metrics(data)

        assert metrics['trend_strength'] < 0  # Negative trend
        assert metrics['bear_strength'] > metrics['bull_strength']

    def test_calculate_trend_metrics_sideways(self):
        """Test trend metrics calculation for sideways movement"""
        # Create sideways data
        prices = [100, 101, 99, 100, 101, 99, 100, 101, 99, 100]
        data = pd.DataFrame({'close': prices})

        metrics = calculate_trend_metrics(data)

        assert abs(metrics['trend_strength']) < 1.0  # Weak trend
        assert abs(metrics['bull_strength'] - metrics['bear_strength']) < 0.5

    def test_calculate_trend_metrics_insufficient_data(self):
        """Test trend metrics with insufficient data"""
        data = pd.DataFrame({'close': [100, 101]})  # Only 2 points

        metrics = calculate_trend_metrics(data)

        # Should return default values
        assert isinstance(metrics, dict)
        assert 'trend_strength' in metrics

    def test_calculate_trend_metrics_empty_data(self):
        """Test trend metrics with empty data"""
        data = pd.DataFrame()

        metrics = calculate_trend_metrics(data)

        assert isinstance(metrics, dict)


class TestCalculateVolatilityMetrics:
    """Test calculate_volatility_metrics function"""

    def test_calculate_volatility_metrics_high_volatility(self):
        """Test volatility metrics for high volatility data"""
        # Create high volatility data
        prices = [100, 110, 95, 120, 80, 105, 115, 90, 125, 85]
        data = pd.DataFrame({'close': prices})

        metrics = calculate_volatility_metrics(data)

        assert 'volatility' in metrics
        assert metrics['volatility'] > 0.05  # High volatility

    def test_calculate_volatility_metrics_low_volatility(self):
        """Test volatility metrics for low volatility data"""
        # Create low volatility data
        prices = [100, 101, 99, 100, 101, 99, 100, 101, 99, 100]
        data = pd.DataFrame({'close': prices})

        metrics = calculate_volatility_metrics(data)

        assert metrics['volatility'] < 0.02  # Low volatility

    def test_calculate_volatility_metrics_constant_price(self):
        """Test volatility metrics for constant price"""
        prices = [100] * 10
        data = pd.DataFrame({'close': prices})

        metrics = calculate_volatility_metrics(data)

        assert metrics['volatility'] == 0.0


class TestCalculateMomentumMetrics:
    """Test calculate_momentum_metrics function"""

    def test_calculate_momentum_metrics_positive(self):
        """Test momentum metrics for positive momentum"""
        # Create accelerating uptrend
        prices = [100, 102, 105, 109, 114, 120, 127, 135, 144, 154]
        data = pd.DataFrame({'close': prices})

        metrics = calculate_momentum_metrics(data)

        assert 'momentum' in metrics
        assert metrics['momentum'] > 0

    def test_calculate_momentum_metrics_negative(self):
        """Test momentum metrics for negative momentum"""
        # Create accelerating downtrend
        prices = [154, 144, 135, 127, 120, 114, 109, 105, 102, 100]
        data = pd.DataFrame({'close': prices})

        metrics = calculate_momentum_metrics(data)

        assert metrics['momentum'] < 0


class TestCalculateVolumeMetrics:
    """Test calculate_volume_metrics function"""

    def test_calculate_volume_metrics_increasing(self):
        """Test volume metrics for increasing volume"""
        volume = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        data = pd.DataFrame({'volume': volume})

        metrics = calculate_volume_metrics(data)

        assert 'volume_trend' in metrics
        assert metrics['volume_trend'] > 0

    def test_calculate_volume_metrics_decreasing(self):
        """Test volume metrics for decreasing volume"""
        volume = [1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]
        data = pd.DataFrame({'volume': volume})

        metrics = calculate_volume_metrics(data)

        assert metrics['volume_trend'] < 0

    def test_calculate_volume_metrics_missing_volume(self):
        """Test volume metrics when volume column is missing"""
        data = pd.DataFrame({'close': [100, 101, 102]})

        metrics = calculate_volume_metrics(data)

        # Should handle missing volume gracefully
        assert isinstance(metrics, dict)


class TestCalculateSupportResistanceMetrics:
    """Test calculate_support_resistance_metrics function"""

    def test_calculate_support_resistance_metrics(self):
        """Test support/resistance metrics calculation"""
        # Create data with clear support/resistance levels
        high = [110, 115, 112, 118, 116, 120, 117, 122, 119, 125]
        low = [95, 98, 97, 102, 100, 105, 103, 108, 106, 110]
        data = pd.DataFrame({'high': high, 'low': low})

        metrics = calculate_support_resistance_metrics(data)

        assert 'support_resistance_strength' in metrics
        assert isinstance(metrics['support_resistance_strength'], (int, float))

    def test_calculate_support_resistance_metrics_missing_columns(self):
        """Test support/resistance metrics with missing OHLC columns"""
        data = pd.DataFrame({'close': [100, 101, 102]})

        metrics = calculate_support_resistance_metrics(data)

        # Should handle missing columns gracefully
        assert isinstance(metrics, dict)


class TestCalculateCompositeScore:
    """Test calculate_composite_score function"""

    def test_calculate_composite_score_balanced(self):
        """Test composite score calculation with balanced components"""
        components = {
            'trend': 0.7,
            'momentum': 0.6,
            'volatility': 0.5,
            'volume': 0.8
        }
        weights = {
            'trend': 0.25,
            'momentum': 0.25,
            'volatility': 0.25,
            'volume': 0.25
        }

        score = calculate_composite_score(components, weights)

        assert 0 <= score <= 100
        # Expected: (0.7*0.25 + 0.6*0.25 + 0.5*0.25 + 0.8*0.25) * 100 = 65.0
        assert score == 65.0

    def test_calculate_composite_score_weighted(self):
        """Test composite score with different weights"""
        components = {
            'trend': 0.8,
            'momentum': 0.4
        }
        weights = {
            'trend': 0.7,  # Higher weight for trend
            'momentum': 0.3
        }

        score = calculate_composite_score(components, weights)

        # Expected: (0.8*0.7 + 0.4*0.3) * 100 = 64.0
        assert score == 64.0

    def test_calculate_composite_score_empty_components(self):
        """Test composite score with empty components"""
        components = {}
        weights = {'trend': 0.5, 'momentum': 0.5}

        score = calculate_composite_score(components, weights)

        assert score == 50.0  # Default neutral score

    def test_calculate_composite_score_missing_weights(self):
        """Test composite score when some weights are missing"""
        components = {
            'trend': 0.7,
            'momentum': 0.6,
            'volatility': 0.5
        }
        weights = {
            'trend': 0.5,
            'momentum': 0.5
            # Missing volatility weight
        }

        score = calculate_composite_score(components, weights)

        # Should only use available weights, renormalize
        # trend: 0.7 * (0.5/1.0) = 0.35
        # momentum: 0.6 * (0.5/1.0) = 0.3
        # Total: 0.65 * 100 = 65.0
        assert score == 65.0

    def test_calculate_composite_score_zero_weights(self):
        """Test composite score with zero weights"""
        components = {
            'trend': 0.8,
            'momentum': 0.4
        }
        weights = {
            'trend': 0.0,
            'momentum': 0.0
        }

        score = calculate_composite_score(components, weights)

        assert score == 50.0  # Default when no valid weights

    def test_calculate_composite_score_none_weights(self):
        """Test composite score with None weights"""
        components = {'trend': 0.7}
        weights = None

        score = calculate_composite_score(components, weights)

        assert score == 50.0  # Default when weights is None

    @pytest.mark.parametrize("components,weights,expected", [
        ({'a': 1.0}, {'a': 1.0}, 100.0),
        ({'a': 0.0}, {'a': 1.0}, 0.0),
        ({'a': 0.5, 'b': 0.5}, {'a': 0.5, 'b': 0.5}, 50.0),
        ({'a': 0.8, 'b': 0.4}, {'a': 0.3, 'b': 0.7}, 56.0)
    ])
    def test_calculate_composite_score_parametrized(self, components, weights, expected):
        """Test composite score with various inputs"""
        score = calculate_composite_score(components, weights)
        assert score == expected


class TestMetricsIntegration:
    """Test integration of multiple metrics functions"""

    def test_all_metrics_integration(self):
        """Test that all metric functions work together"""
        # Create comprehensive test data
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
            'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })

        # Calculate all metrics
        trend_metrics = calculate_trend_metrics(data)
        volatility_metrics = calculate_volatility_metrics(data)
        momentum_metrics = calculate_momentum_metrics(data)
        volume_metrics = calculate_volume_metrics(data)
        sr_metrics = calculate_support_resistance_metrics(data)

        # All should return dictionaries
        assert isinstance(trend_metrics, dict)
        assert isinstance(volatility_metrics, dict)
        assert isinstance(momentum_metrics, dict)
        assert isinstance(volume_metrics, dict)
        assert isinstance(sr_metrics, dict)

        # All should have expected keys
        assert 'trend_strength' in trend_metrics
        assert 'volatility' in volatility_metrics
        assert 'momentum' in momentum_metrics
        assert 'volume_trend' in volume_metrics
        assert 'support_resistance_strength' in sr_metrics

    def test_metrics_with_realistic_data(self):
        """Test metrics with more realistic market data"""
        np.random.seed(42)
        n = 50

        # Generate realistic price series
        returns = np.random.normal(0.001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n)),
            'high': prices * (1 + abs(np.random.normal(0, 0.01, n))),
            'low': prices * (1 - abs(np.random.normal(0, 0.01, n))),
            'close': prices,
            'volume': np.random.uniform(1000, 2000, n)
        })

        # Should not raise exceptions
        trend_metrics = calculate_trend_metrics(data)
        volatility_metrics = calculate_volatility_metrics(data)
        momentum_metrics = calculate_momentum_metrics(data)
        volume_metrics = calculate_volume_metrics(data)
        sr_metrics = calculate_support_resistance_metrics(data)

        # All values should be numeric
        for metrics in [trend_metrics, volatility_metrics, momentum_metrics,
                       volume_metrics, sr_metrics]:
            for key, value in metrics.items():
                assert isinstance(value, (int, float, np.number)), f"{key} is not numeric"