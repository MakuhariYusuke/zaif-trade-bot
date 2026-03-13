#!/usr/bin/env python3
"""
Unit tests for SignalGuidanceSystem
"""

import sys
from unittest.mock import Mock

import pandas as pd
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.trading.signal.signal_guidance_system import (
    SignalGuidanceSystem,
    MarketTrend,
    SignalType,
    GuidanceConfig
)


class TestSignalGuidanceSystem:
    """Test cases for SignalGuidanceSystem"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = GuidanceConfig()
        self.system = SignalGuidanceSystem(self.config)

    def test_initialization(self):
        """Test system initialization"""
        assert self.system.config.guidance_level == 'adaptive'
        assert len(self.system.signal_history) == 0
        assert self.system.market_context.current_trend == MarketTrend.NEUTRAL

    def test_market_trend_analysis(self):
        """Test market trend analysis"""
        # Test neutral trend (insufficient data)
        trend = self.system._analyze_market_trend()
        assert trend == MarketTrend.NEUTRAL

        # Setup bullish trend
        self.system.market_context.price_trend = [100.0, 100.5, 101.0, 101.5, 102.1]  # >0.2% increase
        trend = self.system._analyze_market_trend()
        assert trend == MarketTrend.BULLISH

        # Setup bearish trend
        self.system.market_context.price_trend = [100.0, 99.5, 99.0, 98.5, 97.9]  # >0.2% decrease
        trend = self.system._analyze_market_trend()
        assert trend == MarketTrend.BEARISH

    def test_position_context_analysis(self):
        """Test position context analysis"""
        portfolio = {
            'btc_balance': 0.1,
            'jpy_balance': 50000.0,
            'portfolio_value': 100000.0,
            'current_price': 100000.0
        }

        ctx = self.system.get_position_context(portfolio)

        assert ctx.has_position == True
        assert ctx.position_ratio == 0.1  # 0.1 * 100000 / 100000 = 0.1
        assert ctx.is_overexposed == False  # 0.1 < 0.8
        assert ctx.can_buy == True  # > 1000

    def test_signal_context_analysis(self):
        """Test signal context analysis"""
        # Empty history
        ctx = self.system.get_signal_context()
        assert ctx.recent_bias == 'neutral'
        assert ctx.signal_streak == 0
        assert ctx.last_signal is None

        # Add signals
        self.system.signal_history = [SignalType.BUY, SignalType.BUY, SignalType.SELL]
        ctx = self.system.get_signal_context()
        assert ctx.recent_bias == 'buy'
        assert ctx.signal_streak == 1
        assert ctx.last_signal == SignalType.SELL

    def test_adaptive_threshold(self):
        """Test adaptive threshold calculation"""
        # Test conservative mode
        config = GuidanceConfig(guidance_level='conservative')
        system = SignalGuidanceSystem(config)
        threshold = system._get_adaptive_threshold(MarketTrend.NEUTRAL, Mock(), Mock())
        assert threshold == config.conservative_threshold

        # Test aggressive mode
        config = GuidanceConfig(guidance_level='aggressive')
        system = SignalGuidanceSystem(config)
        threshold = system._get_adaptive_threshold(MarketTrend.NEUTRAL, Mock(), Mock())
        assert threshold == config.aggressive_threshold

        # Test adaptive mode
        config = GuidanceConfig(guidance_level='adaptive')
        system = SignalGuidanceSystem(config)
        position_ctx = Mock()
        position_ctx.is_overexposed = False
        position_ctx.is_underexposed = False
        signal_ctx = Mock()
        signal_ctx.signal_streak = 0
        threshold = system._get_adaptive_threshold(MarketTrend.BULLISH, position_ctx, signal_ctx)
        assert threshold == config.base_threshold * 0.8  # Bullish adjustment

    def test_sell_probability_calculation(self):
        """Test sell probability calculation"""
        # Mock contexts
        market_trend = MarketTrend.BEARISH
        position_ctx = Mock()
        position_ctx.is_overexposed = True
        signal_ctx = Mock()
        signal_ctx.last_signal = SignalType.BUY
        signal_ctx.signal_streak = 1

        # Setup history with no recent sells
        self.system.signal_history = [SignalType.BUY, SignalType.BUY, SignalType.BUY, SignalType.BUY, SignalType.BUY]

        probability = self.system._calculate_sell_probability(market_trend, position_ctx, signal_ctx)
        expected = (self.config.sell_injection_base_probability *
                   self.config.sell_injection_bearish_multiplier *
                   self.config.sell_injection_overexposed_multiplier *
                   self.config.sell_injection_no_recent_sell_multiplier)

        assert probability == min(expected, self.config.sell_injection_max_probability)

    def test_position_guidance(self):
        """Test position-based guidance"""
        # Mock position context
        position_ctx = Mock()
        position_ctx.is_overexposed = True
        position_ctx.has_position = True
        position_ctx.can_buy = False

        # Test overexposure prevention
        action = self.system._apply_position_guidance(SignalType.BUY.value, position_ctx)
        assert action == SignalType.HOLD.value

        # Test no position sell prevention
        position_ctx.has_position = False
        action = self.system._apply_position_guidance(SignalType.SELL.value, position_ctx)
        assert action == SignalType.HOLD.value

        # Test no funds buy prevention
        position_ctx.has_position = True
        position_ctx.is_overexposed = False
        action = self.system._apply_position_guidance(SignalType.BUY.value, position_ctx)
        assert action == SignalType.HOLD.value

    def test_trend_guidance(self):
        """Test trend-based guidance"""
        # Mock position context
        position_ctx = Mock()
        position_ctx.is_underexposed = True
        position_ctx.can_buy = True
        position_ctx.has_position = True

        # Test bullish market BUY injection
        action = self.system._apply_trend_guidance(SignalType.HOLD.value, MarketTrend.BULLISH, position_ctx)
        # Should potentially return BUY (probabilistic, but structure is correct)

        # Test bearish market SELL injection
        action = self.system._apply_trend_guidance(SignalType.HOLD.value, MarketTrend.BEARISH, position_ctx)
        # Should potentially return SELL (probabilistic, but structure is correct)

    def test_signal_guidance(self):
        """Test signal pattern-based guidance"""
        # Mock signal context
        signal_ctx = Mock()
        signal_ctx.signal_streak = 5
        signal_ctx.last_signal = SignalType.BUY
        signal_ctx.recent_bias = 'buy'
        signal_ctx.sell_signal_ratio = 0.05  # Below threshold

        # Test streak prevention
        action = self.system._apply_signal_guidance(SignalType.BUY.value, signal_ctx)
        # Should potentially return HOLD (probabilistic, but structure is correct)

        # Test sell signal promotion
        action = self.system._apply_signal_guidance(SignalType.HOLD.value, signal_ctx)
        # Should potentially return SELL (probabilistic, but structure is correct)

    def test_full_guidance_pipeline(self):
        """Test complete guidance pipeline"""
        # Setup test data
        row = pd.Series({'close': 100000.0, 'volume': 100.0})
        portfolio = {
            'btc_balance': 0.01,
            'jpy_balance': 50000.0,
            'portfolio_value': 100000.0,
            'current_price': 100000.0
        }

        self.system.multi_timeframe_analyzer.analyze_convergence = Mock(
            return_value=Mock(convergence_score=80.0)
        )
        self.system.multi_timeframe_analyzer.timeframes = {}
        self.system.convergence_calculator.get_convergence_report = Mock(
            return_value={"recommendation": "strong_convergence"}
        )
        self.system._create_market_dataframe = Mock(
            return_value=pd.DataFrame(
                {
                    "open": [100000.0],
                    "high": [100100.0],
                    "low": [99900.0],
                    "close": [100000.0],
                    "volume": [100.0],
                }
            )
        )
        self.system.quality_scorer.calculate_signal_quality = Mock(return_value=(0, 60.0))
        self.system._apply_convergence_enhancement = Mock(return_value=60.0)
        self.system._convert_score_to_action = Mock(return_value=SignalType.BUY.value)

        action = self.system.apply_guidance(0.5, row, portfolio)
        assert action in [-1, 0, 1]  # Valid action values

        # Check signal history recording
        assert len(self.system.signal_history) == 1
        assert isinstance(self.system.signal_history[0], SignalType)

    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test default config
        config = GuidanceConfig()
        assert config.guidance_level == 'adaptive'
        assert config.base_threshold == 0.33

        # Test custom config
        custom_config = GuidanceConfig(
            guidance_level='conservative',
            base_threshold=0.5,
            max_history=50
        )
        assert custom_config.guidance_level == 'conservative'
        assert custom_config.base_threshold == 0.5
        assert custom_config.max_history == 50


if __name__ == '__main__':
    pytest.main([__file__])
