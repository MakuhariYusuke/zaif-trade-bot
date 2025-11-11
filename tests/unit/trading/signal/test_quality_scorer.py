"""
Unit tests for Signal Quality Scorer

テストケース:
- Quality score calculation accuracy
- Action determination from scores
- Position context adjustments
- Technical indicator integration
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.signal.quality_scorer import SignalQualityScorer


class TestSignalQualityScorer:
    """Test cases for SignalQualityScorer class"""

    @pytest.fixture
    def scorer(self):
        """Create SignalQualityScorer instance"""
        return SignalQualityScorer()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        np.random.seed(42)
        n_points = 50

        # Generate price data
        base_price = 50000
        prices = [base_price]
        for i in range(n_points - 1):
            change = np.random.normal(0, 0.01)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        # Create OHLCV DataFrame
        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, 0.005)))
            low = close * (1 - abs(np.random.normal(0, 0.005)))
            open_price = data[-1]['close'] if data else close
            volume = np.random.lognormal(10, 0.5)

            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

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
        """Test scorer initialization"""
        scorer = SignalQualityScorer()
        assert scorer.buy_threshold == 70
        assert scorer.sell_threshold == 30
        assert scorer.hold_threshold == 45

        # Test custom config
        custom_config = {
            'buy_threshold': 80,
            'sell_threshold': 20,
            'weights': {'rsi': 0.5, 'macd': 0.3, 'bollinger': 0.2}
        }
        scorer_custom = SignalQualityScorer(custom_config)
        assert scorer_custom.buy_threshold == 80
        assert scorer_custom.sell_threshold == 20

    def test_calculate_signal_quality_buy_signal(self, scorer, sample_market_data, sample_portfolio):
        """Test quality scoring for buy signal"""
        # Create bullish market conditions
        bullish_data = sample_market_data.copy()
        # Make RSI oversold
        bullish_data.loc[len(bullish_data)-1, 'close'] = bullish_data['close'].iloc[0] * 0.95

        action, score = scorer.calculate_signal_quality(bullish_data, 0.8, sample_portfolio)

        assert isinstance(action, int)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_calculate_signal_quality_sell_signal(self, scorer, sample_market_data, sample_portfolio):
        """Test quality scoring for sell signal"""
        # Create bearish market conditions
        bearish_data = sample_market_data.copy()
        # Make RSI overbought
        bearish_data.loc[len(bearish_data)-1, 'close'] = bearish_data['close'].iloc[0] * 1.05

        action, score = scorer.calculate_signal_quality(bearish_data, -0.8, sample_portfolio)

        assert isinstance(action, int)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_score_to_action_conversion(self, scorer):
        """Test score to action conversion"""
        # Buy signals
        assert scorer._score_to_action(75) == 1  # BUY
        assert scorer._score_to_action(85) == 1  # BUY

        # Sell signals
        assert scorer._score_to_action(25) == -1  # SELL
        assert scorer._score_to_action(15) == -1  # SELL

        # Hold signals
        assert scorer._score_to_action(50) == 0   # HOLD
        assert scorer._score_to_action(45) == 0   # HOLD
        assert scorer._score_to_action(60) == 0   # HOLD

    def test_position_adjustments(self, scorer):
        """Test position-based score adjustments"""
        base_score = 60

        # Overexposed position (should reduce buy signals)
        overexposed_portfolio = {
            'btc_balance': 1.0,
            'jpy_balance': 10000,
            'current_price': 50000
        }
        adjusted_score = scorer._apply_position_adjustments(base_score, overexposed_portfolio)
        assert adjusted_score < base_score  # Should be reduced

        # Underexposed position (should increase buy signals)
        underexposed_portfolio = {
            'btc_balance': 0.1,
            'jpy_balance': 200000,
            'current_price': 50000
        }
        adjusted_score = scorer._apply_position_adjustments(base_score, underexposed_portfolio)
        assert adjusted_score > base_score  # Should be increased

    def test_rsi_score_calculation(self, scorer):
        """Test RSI-based score calculation"""
        # Oversold RSI
        oversold_signals = {'rsi': 25}
        rsi_score = scorer._calculate_rsi_score(oversold_signals)
        assert rsi_score > 50  # Should be bullish

        # Overbought RSI
        overbought_signals = {'rsi': 75}
        rsi_score = scorer._calculate_rsi_score(overbought_signals)
        assert rsi_score < 50  # Should be bearish

        # Neutral RSI
        neutral_signals = {'rsi': 50}
        rsi_score = scorer._calculate_rsi_score(neutral_signals)
        assert rsi_score == 50  # Should be neutral

    def test_macd_score_calculation(self, scorer):
        """Test MACD-based score calculation"""
        # Bullish MACD crossover
        bullish_signals = {
            'macd_line': 100,
            'macd_signal': 95,
            'macd_histogram': 5
        }
        macd_score = scorer._calculate_macd_score(bullish_signals)
        assert macd_score > 50  # Should be bullish

        # Bearish MACD crossover
        bearish_signals = {
            'macd_line': 95,
            'macd_signal': 100,
            'macd_histogram': -5
        }
        macd_score = scorer._calculate_macd_score(bearish_signals)
        assert macd_score < 50  # Should be bearish

    def test_bollinger_score_calculation(self, scorer):
        """Test Bollinger Bands-based score calculation"""
        # Price near lower band (bullish)
        lower_band_signals = {'bb_position': 0.1}
        bb_score = scorer._calculate_bollinger_score(lower_band_signals, pd.DataFrame({'close': [50000]}))
        assert bb_score > 50  # Should be bullish

        # Price near upper band (bearish)
        upper_band_signals = {'bb_position': 0.9}
        bb_score = scorer._calculate_bollinger_score(upper_band_signals, pd.DataFrame({'close': [50000]}))
        assert bb_score < 50  # Should be bearish

    def test_trend_score_calculation(self, scorer):
        """Test trend-based score calculation"""
        # Uptrend
        uptrend_data = pd.DataFrame({'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]})
        trend_score = scorer._calculate_trend_score(uptrend_data)
        assert trend_score > 50  # Should be bullish

        # Downtrend
        downtrend_data = pd.DataFrame({'close': [110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100]})
        trend_score = scorer._calculate_trend_score(downtrend_data)
        assert trend_score < 50  # Should be bearish

    def test_continuous_action_blending(self, scorer):
        """Test blending of quality score with continuous action"""
        quality_score = 60

        # Positive continuous action should increase score
        blended = scorer._blend_continuous_action(quality_score, 0.8)
        assert blended > quality_score

        # Negative continuous action should decrease score
        blended = scorer._blend_continuous_action(quality_score, -0.8)
        assert blended < quality_score

        # Neutral continuous action should keep score similar
        blended = scorer._blend_continuous_action(quality_score, 0.0)
        assert abs(blended - quality_score) < 10

    def test_error_handling(self, scorer, sample_portfolio):
        """Test error handling and fallback behavior"""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        action, score = scorer.calculate_signal_quality(empty_df, 0.0, sample_portfolio)
        assert isinstance(action, int)
        assert isinstance(score, float)

        # Invalid portfolio data
        invalid_portfolio = {}
        action, score = scorer.calculate_signal_quality(pd.DataFrame(), 0.0, invalid_portfolio)
        assert isinstance(action, int)
        assert isinstance(score, float)

    def test_fallback_conversion(self, scorer):
        """Test fallback action conversion"""
        assert scorer._fallback_action(0.2) == 1   # BUY
        assert scorer._fallback_action(-0.2) == -1  # SELL
        assert scorer._fallback_action(0.0) == 0    # HOLD
        assert scorer._fallback_action(0.05) == 0   # HOLD (below threshold)