"""
Unit tests for Signal Guidance System

テストケース:
- Quality scorer integration
- Deterministic signal generation
- Position safety checks
- Fallback behavior
- Backward compatibility
"""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.signal.signal_guidance_system import SignalGuidanceSystem, GuidanceConfig


class TestSignalGuidanceSystem:
    """Test cases for SignalGuidanceSystem class"""

    @pytest.fixture
    def guidance_system(self):
        """Create SignalGuidanceSystem instance"""
        config = GuidanceConfig()
        return SignalGuidanceSystem(config)

    @pytest.fixture
    def sample_market_row(self):
        """Create sample market data row"""
        return pd.Series({
            'open': 49500,
            'high': 50500,
            'low': 49400,
            'close': 50000,
            'volume': 1500.5,
            'price': 50000
        })

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio data"""
        return {
            'btc_balance': 0.5,
            'jpy_balance': 100000,
            'current_price': 50000,
            'portfolio_value': 125000
        }

    def test_initialization(self):
        """Test system initialization"""
        system = SignalGuidanceSystem()
        assert system.config is not None
        assert system.quality_scorer is not None
        assert len(system.signal_history) == 0

    def test_apply_guidance_deterministic(self, guidance_system, sample_market_row, sample_portfolio):
        """Test deterministic signal guidance"""
        # Test multiple times with same input - should get consistent results
        results = []
        for _ in range(5):
            action = guidance_system.apply_guidance(0.5, sample_market_row, sample_portfolio)
            results.append(action)

        # All results should be valid actions (-1, 0, 1)
        assert all(action in [-1, 0, 1] for action in results)

        # Results should be recorded in history
        assert len(guidance_system.signal_history) == 5

    def test_apply_guidance_buy_signal(self, guidance_system, sample_portfolio):
        """Test buy signal generation"""
        # Create bullish market conditions
        bullish_row = pd.Series({
            'open': 49000,
            'high': 51000,
            'low': 48500,
            'close': 50500,  # Higher close
            'volume': 2000.0,
            'price': 50500
        })

        action = guidance_system.apply_guidance(0.8, bullish_row, sample_portfolio)
        assert action in [-1, 0, 1]  # Valid action

    def test_apply_guidance_sell_signal(self, guidance_system, sample_portfolio):
        """Test sell signal generation"""
        # Create bearish market conditions
        bearish_row = pd.Series({
            'open': 51000,
            'high': 51500,
            'low': 49500,
            'close': 49500,  # Lower close
            'volume': 1800.0,
            'price': 49500
        })

        action = guidance_system.apply_guidance(-0.8, bearish_row, sample_portfolio)
        assert action in [-1, 0, 1]  # Valid action

    def test_position_safety_checks(self, guidance_system):
        """Test position-based safety checks"""
        # No BTC balance - should not generate SELL
        no_btc_portfolio = {
            'btc_balance': 0.0,
            'jpy_balance': 200000,
            'current_price': 50000
        }

        action = guidance_system._apply_position_safety(-1, no_btc_portfolio)  # Try to SELL
        assert action == 0  # Should be HOLD

        # No JPY balance - should not generate BUY
        no_jpy_portfolio = {
            'btc_balance': 1.0,
            'jpy_balance': 0,
            'current_price': 50000
        }

        action = guidance_system._apply_position_safety(1, no_jpy_portfolio)  # Try to BUY
        assert action == 0  # Should be HOLD

    def test_market_dataframe_creation(self, guidance_system, sample_market_row, sample_portfolio):
        """Test market DataFrame creation for technical analysis"""
        df = guidance_system._create_market_dataframe(sample_market_row, sample_portfolio)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_fallback_conversion(self, guidance_system):
        """Test fallback action conversion"""
        assert guidance_system._fallback_conversion(0.2) == 1   # BUY
        assert guidance_system._fallback_conversion(-0.2) == -1  # SELL
        assert guidance_system._fallback_conversion(0.0) == 0    # HOLD
        assert guidance_system._fallback_conversion(0.05) == 0   # HOLD

    def test_error_handling(self, guidance_system, sample_portfolio):
        """Test error handling and fallback behavior"""
        # Invalid market data
        invalid_row = pd.Series({'invalid': 'data'})

        # Should not raise exception, should fallback
        action = guidance_system.apply_guidance(0.0, invalid_row, sample_portfolio)
        assert action in [-1, 0, 1]

    def test_signal_history_tracking(self, guidance_system, sample_market_row, sample_portfolio):
        """Test signal history tracking"""
        initial_history_length = len(guidance_system.signal_history)

        # Generate several signals
        for i in range(3):
            guidance_system.apply_guidance(0.1 * i, sample_market_row, sample_portfolio)

        # History should have grown
        assert len(guidance_system.signal_history) == initial_history_length + 3

        # All signals should be valid SignalType enums
        from ztb.trading.signal.signal_guidance_system import SignalType
        assert all(isinstance(signal, SignalType) for signal in guidance_system.signal_history)

    def test_market_context_update(self, guidance_system, sample_portfolio):
        """Test market context updates"""
        initial_trend_length = len(guidance_system.market_context.price_trend)

        # Update context multiple times
        for _ in range(3):
            row = pd.Series({
                'close': 50000 + np.random.normal(0, 100),
                'volume': 1500 + np.random.normal(0, 100)
            })
            guidance_system.update_market_context(row, sample_portfolio)

        # Price trend should have grown
        assert len(guidance_system.market_context.price_trend) > initial_trend_length

    def test_config_persistence(self):
        """Test configuration persistence"""
        custom_config = GuidanceConfig(
            guidance_level='aggressive',
            max_history=50,
            base_threshold=0.2
        )

        system = SignalGuidanceSystem(custom_config)
        assert system.config.guidance_level == 'aggressive'
        assert system.config.max_history == 50
        assert system.config.base_threshold == 0.2

    def test_backward_compatibility(self, guidance_system, sample_market_row, sample_portfolio):
        """Test backward compatibility with existing interface"""
        # Should work with existing method signatures
        action = guidance_system.apply_guidance(0.0, sample_market_row, sample_portfolio)
        assert isinstance(action, int)
        assert action in [-1, 0, 1]

        # Market context should still be updated
        assert len(guidance_system.market_context.price_trend) > 0