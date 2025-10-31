"""
Unit tests for Market Regime Classifier

Tests the generic market regime classification system to ensure
proper regime detection and adaptation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from ztb.analysis.market_regime_classifier import (
    MarketRegimeClassifier,
    RegimeType,
    RegimeMetrics,
    RegimeDetectionResult
)


class TestMarketRegimeClassifier:
    """Test suite for MarketRegimeClassifier"""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        np.random.seed(42)

        # Create trending data (bull trend)
        base_price = 100
        trend = np.linspace(0, 20, 100)  # Upward trend
        noise = np.random.normal(0, 2, 100)
        close = base_price + trend + noise

        # Create OHLC data
        high = close + np.abs(np.random.normal(0, 1, 100))
        low = close - np.abs(np.random.normal(0, 1, 100))
        open_price = close + np.random.normal(0, 0.5, 100)
        volume = np.random.uniform(1000, 10000, 100)

        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        return df

    @pytest.fixture
    def classifier(self):
        """Create a basic classifier instance"""
        config = {
            'lookback_periods': {'short': 10, 'medium': 20, 'long': 50},
            'regime_scheme': 'basic'
        }
        return MarketRegimeClassifier(config)

    def test_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier.config is not None
        assert classifier.lookback_periods['short'] == 10
        assert len(classifier.regime_definitions) > 0

    def test_detect_regime_basic(self, classifier, sample_price_data):
        """Test basic regime detection"""
        result = classifier.detect_regime(sample_price_data)

        assert isinstance(result, RegimeDetectionResult)
        assert isinstance(result.primary_regime, RegimeType)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.metrics, RegimeMetrics)

    def test_regime_metrics_calculation(self, classifier, sample_price_data):
        """Test regime metrics calculation"""
        metrics = classifier._calculate_regime_metrics(sample_price_data, 50)

        assert isinstance(metrics, RegimeMetrics)
        assert isinstance(metrics.trend_strength, (int, float))
        assert isinstance(metrics.volatility, (int, float))
        assert isinstance(metrics.momentum, (int, float))

    def test_regime_multiplier(self, classifier):
        """Test regime multiplier functionality"""
        # Test with default config (no adaptation)
        multiplier = classifier.get_regime_multiplier(RegimeType.STRONG_BULL, 'reward')
        assert multiplier == 1.0

        # Test with adaptation config
        classifier.config['adaptation'] = {
            'enabled': True,
            'regime_reward_multipliers': {
                RegimeType.STRONG_BULL: 1.5
            }
        }

        multiplier = classifier.get_regime_multiplier(RegimeType.STRONG_BULL, 'reward')
        assert multiplier == 1.5

        # Test penalty multiplier
        classifier.config['adaptation']['regime_penalty_multipliers'] = {
            RegimeType.STRONG_BEAR: 0.8
        }

        multiplier = classifier.get_regime_multiplier(RegimeType.STRONG_BEAR, 'penalty')
        assert multiplier == 0.8

    def test_insufficient_data_handling(self, classifier):
        """Test handling of insufficient data"""
        small_df = pd.DataFrame({
            'close': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'volume': [1000, 1100, 1200]
        })

        result = classifier.detect_regime(small_df, 2)
        assert result.primary_regime == RegimeType.CONSOLIDATION  # Default fallback

    def test_custom_regime_definitions(self):
        """Test custom regime definitions"""
        custom_config = {
            'regime_scheme': 'custom',
            'custom_regime_definitions': [
                {
                    'name': 'Custom Bull',
                    'regime_type': 'strong_bull',
                    'conditions': {'trend_strength': {'min': 2.0}},
                    'priority': 10
                }
            ]
        }

        classifier = MarketRegimeClassifier(custom_config)
        assert len(classifier.regime_definitions) == 1
        assert classifier.regime_definitions[0].name == 'Custom Bull'

    def test_regime_classification_logic(self, classifier):
        """Test regime classification logic with mock metrics"""
        # Test strong bull trend
        strong_bull_metrics = RegimeMetrics(
            trend_strength=4.0,
            bull_strength=3.0,
            bear_strength=0.5,
            volatility=0.05,
            momentum=5.0,
            volume_trend=10.0,
            price_range_ratio=2.0,
            adx=35.0,
            rsi=70.0,
            macd_signal=0.5,
            bollinger_position=0.8,
            support_resistance_strength=0.7
        )

        regime, confidence = classifier._classify_regime(strong_bull_metrics)
        assert regime == RegimeType.STRONG_BULL
        assert confidence > 0.8

        # Test high volatility
        high_vol_metrics = RegimeMetrics(
            trend_strength=0.5,
            bull_strength=1.0,
            bear_strength=1.0,
            volatility=0.25,
            momentum=0.0,
            volume_trend=0.0,
            price_range_ratio=5.0,
            adx=15.0,
            rsi=50.0,
            macd_signal=0.0,
            bollinger_position=0.5,
            support_resistance_strength=0.5
        )

        regime, confidence = classifier._classify_regime(high_vol_metrics)
        assert regime == RegimeType.HIGH_VOLATILITY_RANGE

    def test_secondary_regimes(self, classifier):
        """Test secondary regime calculation"""
        metrics = RegimeMetrics(
            trend_strength=3.0,
            bull_strength=2.5,
            bear_strength=0.5,
            volatility=0.18,
            momentum=3.0,
            volume_trend=5.0,
            price_range_ratio=3.0,
            adx=30.0,
            rsi=65.0,
            macd_signal=0.3,
            bollinger_position=0.7,
            support_resistance_strength=0.6
        )

        primary_regime = RegimeType.STRONG_BULL
        secondary_regimes = classifier._calculate_secondary_regimes(metrics, primary_regime)

        assert isinstance(secondary_regimes, list)
        if secondary_regimes:
            assert len(secondary_regimes[0]) == 2  # (regime, confidence) tuple

    @patch('ztb.analysis.market_regime_classifier.logger')
    def test_error_handling(self, mock_logger, classifier, sample_price_data):
        """Test error handling in regime detection"""
        # Test with corrupted data
        bad_data = sample_price_data.copy()
        bad_data['close'] = np.nan

        result = classifier.detect_regime(bad_data)
        # Should not crash and return some result
        assert isinstance(result, RegimeDetectionResult)

        # Note: Current implementation handles NaN gracefully without warnings
        # If warnings are added later, this test should be updated

    def test_config_validation(self):
        """Test configuration validation"""
        # Test with minimal config
        min_config = {}
        classifier = MarketRegimeClassifier(min_config)
        assert classifier.config is not None

        # Test with full config
        full_config = {
            'lookback_periods': {'short': 5, 'medium': 15, 'long': 30},
            'thresholds': {
                'strong_trend_threshold': 2.0,
                'high_volatility_threshold': 0.20
            },
            'regime_scheme': 'comprehensive',
            'adaptation': {
                'enabled': True,
                'regime_reward_multipliers': {
                    RegimeType.STRONG_BULL: 1.2
                }
            }
        }

        classifier = MarketRegimeClassifier(full_config)
        assert classifier.lookback_periods['short'] == 5
        assert classifier.thresholds['strong_trend_threshold'] == 2.0


class TestRegimeMetrics:
    """Test suite for RegimeMetrics dataclass"""

    def test_regime_metrics_creation(self):
        """Test RegimeMetrics creation and validation"""
        metrics = RegimeMetrics(
            trend_strength=2.5,
            bull_strength=2.0,
            bear_strength=0.5,
            volatility=0.15,
            momentum=1.5,
            volume_trend=3.0,
            price_range_ratio=2.5,
            adx=25.0,
            rsi=60.0,
            macd_signal=0.2,
            bollinger_position=0.6,
            support_resistance_strength=0.7
        )

        assert metrics.trend_strength == 2.5
        assert metrics.volatility == 0.15
        assert metrics.rsi == 60.0

    def test_regime_metrics_defaults(self):
        """Test RegimeMetrics with default values"""
        # All fields should be properly initialized
        metrics = RegimeMetrics(
            trend_strength=0.0,
            bull_strength=0.0,
            bear_strength=0.0,
            volatility=0.0,
            momentum=0.0,
            volume_trend=0.0,
            price_range_ratio=0.0,
            adx=0.0,
            rsi=0.0,
            macd_signal=0.0,
            bollinger_position=0.0,
            support_resistance_strength=0.0
        )

        assert metrics.trend_strength == 0.0
        assert metrics.bollinger_position == 0.0


class TestRegimeDetectionResult:
    """Test suite for RegimeDetectionResult dataclass"""

    def test_detection_result_creation(self):
        """Test RegimeDetectionResult creation"""
        metrics = RegimeMetrics(
            trend_strength=1.0,
            bull_strength=0.8,
            bear_strength=0.2,
            volatility=0.1,
            momentum=0.5,
            volume_trend=1.0,
            price_range_ratio=1.5,
            adx=20.0,
            rsi=55.0,
            macd_signal=0.1,
            bollinger_position=0.5,
            support_resistance_strength=0.6
        )

        result = RegimeDetectionResult(
            primary_regime=RegimeType.MODERATE_BULL,
            confidence=0.85,
            secondary_regimes=[(RegimeType.LOW_VOLATILITY_RANGE, 0.3)],
            metrics=metrics,
            detection_timestamp=pd.Timestamp('2023-01-01'),
            lookback_period=20
        )

        assert result.primary_regime == RegimeType.MODERATE_BULL
        assert result.confidence == 0.85
        assert len(result.secondary_regimes) == 1
        assert result.lookback_period == 20