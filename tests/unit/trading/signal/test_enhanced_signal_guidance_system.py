"""
Unit tests for EnhancedSignalGuidanceSystem

Tests the enhanced signal guidance system with regime adaptation
and strategic guidance generation.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ztb.trading.signal.common.base_classes import SignalContext, SignalResult
from ztb.trading.signal.guidance.enhanced_system import EnhancedSignalGuidanceSystem
from ztb.trading.signal.regime.classifier import RegimeType


class TestEnhancedSignalGuidanceSystem:
    """Test suite for EnhancedSignalGuidanceSystem"""

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        np.random.seed(42)

        # Create realistic price data with some trend
        close_prices = []
        base_price = 100.0
        for i in range(100):
            trend = 0.0005 * i  # Slight uptrend
            volatility = 0.01 * np.random.normal(0, 1)
            price = base_price * (1 + trend + volatility)
            close_prices.append(price)
            base_price = price

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": close_prices,
                "high": [
                    p * (1 + abs(np.random.normal(0, 0.005))) for p in close_prices
                ],
                "low": [
                    p * (1 - abs(np.random.normal(0, 0.005))) for p in close_prices
                ],
                "close": close_prices,
                "volume": np.random.uniform(1000, 2000, 100),
            }
        )
        data.set_index("timestamp", inplace=True)
        return data

    @pytest.fixture
    def system(self):
        """Create EnhancedSignalGuidanceSystem instance"""
        config = {
            "regime_config": {
                "lookback_periods": {"short": 10, "medium": 20, "long": 50},
                "confidence_threshold": 0.6,
            },
            "quality_config": {
                "weights": {"trend": 0.25, "momentum": 0.20, "regime": 0.25},
                "thresholds": {"buy": 75, "sell": 25, "hold": 50},
            },
            "adaptation_config": {
                "learning_rate": 0.1,
                "performance_window": 50,
                "regime_memory": 25,
            },
        }
        return EnhancedSignalGuidanceSystem(config)

    def test_initialization(self, system):
        """Test system initialization"""
        assert system is not None
        assert hasattr(system, "regime_classifier")
        assert hasattr(system, "quality_scorer")
        assert hasattr(system, "regime_adaptation_params")
        assert (
            len(system.regime_adaptation_params) >= 17
        )  # At least 17 regimes supported
        assert len(system.performance_history) == 0

    def test_regime_adaptation_params_structure(self, system):
        """Test regime adaptation parameters structure"""
        for regime_type, params in system.regime_adaptation_params.items():
            assert "signal_bias" in params
            assert "confidence_multiplier" in params
            assert "threshold_adjustment" in params
            assert "description" in params
            assert isinstance(params["signal_bias"], (int, float))
            assert isinstance(params["confidence_multiplier"], (int, float))
            assert isinstance(params["threshold_adjustment"], (int, float))

    def test_sell_regime_bias(self, system):
        """Test SELL specialized regimes have negative bias"""
        sell_regimes = [
            RegimeType.SELL_BREAKDOWN,
            RegimeType.SELL_DIVERGENCE,
            RegimeType.SELL_MOMENTUM_WEAK,
            RegimeType.SELL_VOLUME_SURGE,
        ]

        for regime in sell_regimes:
            params = system.regime_adaptation_params[regime]
            assert params["signal_bias"] < 0, f"{regime} should have negative bias"
            assert (
                params["confidence_multiplier"] >= 1.0
            ), f"{regime} should have confidence boost"

    def test_bull_regime_bias(self, system):
        """Test bull trend regimes have positive bias"""
        bull_regimes = [
            RegimeType.STRONG_BULL_TREND,
            RegimeType.MODERATE_BULL_TREND,
            RegimeType.WEAK_BULL_TREND,
        ]

        for regime in bull_regimes:
            params = system.regime_adaptation_params[regime]
            assert params["signal_bias"] > 0, f"{regime} should have positive bias"

    def test_process_signal_basic(self, system, sample_market_data):
        """Test basic signal processing"""
        context = SignalContext(
            market_data=sample_market_data,
            position_context={"size": 0.0, "entry_price": None},
            portfolio_state={"cash": 10000.0, "total_value": 10000.0},
            timestamp=sample_market_data.index[-1],
        )

        result = system.process_signal(context)

        assert isinstance(result, SignalResult)
        assert hasattr(result, "discrete_action")
        assert hasattr(result, "quality_score")
        assert hasattr(result, "confidence")
        assert hasattr(result, "metadata")

        # Check metadata contains regime information
        assert "regime" in result.metadata
        assert "regime_confidence" in result.metadata
        assert "base_quality_score" in result.metadata
        assert "regime_adaptation" in result.metadata
        assert "strategic_guidance" in result.metadata

    def test_regime_adaptation_application(self, system, sample_market_data):
        """Test regime adaptation is applied to signals"""
        context = SignalContext(
            market_data=sample_market_data,
            position_context={"size": 0.0},
            portfolio_state={"cash": 10000.0, "total_value": 10000.0},
            timestamp=sample_market_data.index[-1],
        )

        # Mock regime classifier to return specific regime
        with patch.object(system.regime_classifier, "process_signal") as mock_regime:
            mock_regime.return_value = SignalResult(
                discrete_action=0,
                quality_score=50.0,
                confidence=0.8,
                metadata={"regime_type": RegimeType.SELL_BREAKDOWN},
            )

            # Mock quality scorer
            with patch.object(system.quality_scorer, "process_signal") as mock_quality:
                mock_quality.return_value = SignalResult(
                    discrete_action=1,
                    quality_score=70.0,
                    confidence=0.7,
                    metadata={},  # BUY signal
                )

                result = system.process_signal(context)

                # SELL_BREAKDOWN should apply negative bias
                adaptation = result.metadata["regime_adaptation"]
                assert "regime_bias_applied" in adaptation
                assert (
                    adaptation["regime_bias_applied"] < 0
                )  # Negative bias for SELL regime

                # Original score should be adjusted
                assert "original_score" in adaptation
                assert adaptation["original_score"] == 70.0

    def test_strategic_guidance_generation(self, system, sample_market_data):
        """Test strategic guidance generation"""
        context = SignalContext(
            market_data=sample_market_data,
            position_context={"size": 0.0},
            portfolio_state={"cash": 10000.0, "total_value": 10000.0},
            timestamp=sample_market_data.index[-1],
        )

        result = system.process_signal(context)

        guidance = result.metadata["strategic_guidance"]
        required_fields = [
            "primary_action",
            "regime_context",
            "confidence_level",
            "risk_assessment",
            "position_sizing",
            "time_horizon",
            "stop_loss_guidance",
            "take_profit_guidance",
        ]

        for field in required_fields:
            assert field in guidance, f"Missing guidance field: {field}"

    def test_action_string_conversion(self, system):
        """Test action to string conversion"""
        test_cases = [
            (2, "STRONG_BUY"),
            (1, "BUY"),
            (0, "HOLD"),
            (-1, "SELL"),
            (-2, "STRONG_SELL"),
        ]

        for action, expected in test_cases:
            result = system._action_to_string(action)
            assert result == expected

    def test_confidence_level_conversion(self, system):
        """Test confidence to level conversion"""
        test_cases = [
            (0.9, "VERY_HIGH"),
            (0.7, "HIGH"),
            (0.5, "MODERATE"),
            (0.3, "LOW"),
            (0.1, "VERY_LOW"),
        ]

        for confidence, expected in test_cases:
            result = system._confidence_to_level(confidence)
            assert result == expected

    def test_risk_assessment(self, system):
        """Test risk assessment logic"""
        test_cases = [
            (RegimeType.EXTREME_VOLATILITY, "HIGH"),
            (RegimeType.SELL_BREAKDOWN, "HIGH"),
            (RegimeType.CONSOLIDATION, "LOW"),
            (RegimeType.MODERATE_BULL_TREND, "LOW"),
        ]

        for regime, expected_risk in test_cases:
            result = system._assess_risk(regime, Mock(confidence=0.8))
            assert result == expected_risk

    def test_position_sizing_recommendation(self, system):
        """Test position sizing recommendations"""
        test_cases = [
            (RegimeType.EXTREME_VOLATILITY, "confidence", "REDUCED"),
            (RegimeType.STRONG_BULL_TREND, "confidence", "INCREASED"),
            (0.9, "high_confidence", "LARGE"),
            (0.4, "low_confidence", "SMALL"),
        ]

        # Test regime-based sizing
        for regime, _, expected in test_cases[:2]:
            result = system._recommend_position_sizing(regime, Mock(confidence=0.7))
            assert expected in result

        # Test confidence-based sizing
        result_high = system._recommend_position_sizing(
            RegimeType.CONSOLIDATION, Mock(confidence=0.9)
        )
        assert "LARGE" in result_high

        result_low = system._recommend_position_sizing(
            RegimeType.CONSOLIDATION, Mock(confidence=0.3)
        )
        assert "SMALL" in result_low

    def test_time_horizon_recommendation(self, system):
        """Test time horizon recommendations"""
        short_term_regimes = [
            RegimeType.EXTREME_VOLATILITY,
            RegimeType.HIGH_VOLATILITY_RANGE,
            RegimeType.SELL_BREAKDOWN,
        ]

        for regime in short_term_regimes:
            result = system._recommend_time_horizon(regime)
            assert result == "SHORT_TERM"

        # Test medium term regime
        result = system._recommend_time_horizon(RegimeType.MODERATE_BULL_TREND)
        assert result == "MEDIUM_TERM"

    def test_stop_loss_guidance(self, system, sample_market_data):
        """Test stop loss guidance generation"""
        context = SignalContext(
            market_data=sample_market_data,
            position_context={"size": 0.0},
            portfolio_state={"cash": 10000.0, "total_value": 10000.0},
            timestamp=sample_market_data.index[-1],
        )

        # Test extreme volatility stop loss
        guidance = system._generate_stop_loss_guidance(
            RegimeType.EXTREME_VOLATILITY, context
        )
        assert guidance["type"] == "PERCENTAGE"
        assert guidance["value"] == 0.05

        # Test normal regime stop loss
        guidance = system._generate_stop_loss_guidance(
            RegimeType.MODERATE_BULL_TREND, context
        )
        assert guidance["type"] == "PERCENTAGE"
        assert guidance["value"] == 0.03

    def test_take_profit_guidance(self, system, sample_market_data):
        """Test take profit guidance generation"""
        context = SignalContext(
            market_data=sample_market_data,
            position_context={"size": 0.0},
            portfolio_state={"cash": 10000.0, "total_value": 10000.0},
            timestamp=sample_market_data.index[-1],
        )

        # Test strong trend take profit
        guidance = system._generate_take_profit_guidance(
            RegimeType.STRONG_BULL_TREND, context
        )
        assert guidance["type"] == "PERCENTAGE"
        assert guidance["value"] == 0.08

        # Test consolidation take profit
        guidance = system._generate_take_profit_guidance(
            RegimeType.CONSOLIDATION, context
        )
        assert guidance["type"] == "PERCENTAGE"
        assert guidance["value"] == 0.03

    def test_performance_tracking(self, system, sample_market_data):
        """Test performance tracking"""
        context = SignalContext(
            market_data=sample_market_data,
            position_context={"size": 0.0},
            portfolio_state={"cash": 10000.0, "total_value": 10000.0},
            timestamp=sample_market_data.index[-1],
        )

        initial_history_length = len(system.performance_history)

        # Process several signals
        for i in range(3):
            system.process_signal(context)

        # History should have grown
        assert len(system.performance_history) == initial_history_length + 3

        # Check history structure
        for entry in system.performance_history[-3:]:
            assert "regime" in entry
            assert "action" in entry
            assert "confidence" in entry
            assert "quality_score" in entry

    def test_get_performance_metrics(self, system, sample_market_data):
        """Test performance metrics retrieval"""
        context = SignalContext(
            market_data=sample_market_data,
            position_context={"size": 0.0},
            portfolio_state={"cash": 10000.0, "total_value": 10000.0},
            timestamp=sample_market_data.index[-1],
        )

        # Add some performance data
        system.process_signal(context)

        # Mock regime-specific performance
        system.regime_performance[RegimeType.MODERATE_BULL_TREND] = [
            {"confidence": 0.8, "quality_score": 75.0},
            {"confidence": 0.7, "quality_score": 72.0},
        ]

        metrics = system._get_performance_metrics(RegimeType.MODERATE_BULL_TREND)

        assert "regime_signal_count" in metrics
        assert "recent_avg_confidence" in metrics
        assert "recent_avg_quality" in metrics
        assert metrics["regime_signal_count"] == 2

    def test_get_system_status(self, system, sample_market_data):
        """Test system status retrieval"""
        context = SignalContext(
            market_data=sample_market_data,
            position_context={"size": 0.0},
            portfolio_state={"cash": 10000.0, "total_value": 10000.0},
            timestamp=sample_market_data.index[-1],
        )

        # Add some processing history
        system.process_signal(context)

        status = system.get_system_status()

        required_fields = [
            "regime_classifier_status",
            "performance_history_length",
            "active_regimes",
            "total_regime_adaptations",
            "system_config",
        ]

        for field in required_fields:
            assert field in status

        assert status["performance_history_length"] >= 1
        assert isinstance(status["active_regimes"], list)

    def test_error_handling(self, system):
        """Test error handling in signal processing"""
        # Create invalid context (missing required fields)
        invalid_context = Mock()
        del invalid_context.market_data  # Remove required attribute

        result = system.process_signal(invalid_context)

        assert isinstance(result, SignalResult)
        assert result.discrete_action == 0
        assert result.quality_score == 50.0
        assert "error" in result.metadata

    def test_adapted_action_determination(self, system):
        """Test adapted action determination with thresholds"""
        # Test strong BUY signal
        action = system._determine_adapted_action(
            85, {"strong_buy": 80, "buy": 65, "sell": 35, "strong_sell": 20}
        )
        assert action == 2

        # Test BUY signal
        action = system._determine_adapted_action(
            70, {"strong_buy": 80, "buy": 65, "sell": 35, "strong_sell": 20}
        )
        assert action == 1

        # Test SELL signal
        action = system._determine_adapted_action(
            30, {"strong_buy": 80, "buy": 65, "sell": 35, "strong_sell": 20}
        )
        assert action == -1

        # Test strong SELL signal
        action = system._determine_adapted_action(
            15, {"strong_buy": 80, "buy": 65, "sell": 35, "strong_sell": 20}
        )
        assert action == -2

        # Test HOLD signal
        action = system._determine_adapted_action(
            50, {"strong_buy": 80, "buy": 65, "sell": 35, "strong_sell": 20}
        )
        assert action == 0

    def test_threshold_adjustment_application(self, system):
        """Test threshold adjustment application"""
        base_thresholds = {"strong_buy": 80, "buy": 65, "sell": 35, "strong_sell": 20}

        # Test positive adjustment (bull regime)
        adapted = system._apply_regime_adaptation(
            Mock(discrete_action=1, quality_score=70.0, confidence=0.8, metadata={}),
            RegimeType.STRONG_BULL_TREND,
            Mock(
                confidence=0.8, metadata={"regime_type": RegimeType.STRONG_BULL_TREND}
            ),
        )

        # Thresholds should be adjusted for bull regime
        used_thresholds = adapted.metadata["thresholds_used"]
        assert used_thresholds["buy"] > base_thresholds["buy"]  # Higher buy threshold

        # Test negative adjustment (sell regime)
        adapted = system._apply_regime_adaptation(
            Mock(discrete_action=-1, quality_score=30.0, confidence=0.8, metadata={}),
            RegimeType.SELL_BREAKDOWN,
            Mock(confidence=0.8, metadata={"regime_type": RegimeType.SELL_BREAKDOWN}),
        )

        used_thresholds = adapted.metadata["thresholds_used"]
        assert used_thresholds["sell"] < base_thresholds["sell"]  # Lower sell threshold
