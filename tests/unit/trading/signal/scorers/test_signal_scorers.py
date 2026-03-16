"""
Unit tests for individual SignalScorers

Tests cover TechnicalSignalScorer, PatternRecognitionScorer, SentimentSignalScorer,
and VolumeProfileScorer with various market conditions and edge cases.
"""

import unittest
import pandas as pd
import numpy as np

from ztb.trading.signal.ensemble_signal_generator import (
    TechnicalSignalScorer,
    PatternRecognitionScorer,
    SentimentSignalScorer,
    VolumeProfileScorer
)


class TestTechnicalSignalScorer(unittest.TestCase):
    """Test cases for TechnicalSignalScorer"""

    def setUp(self):
        """Set up test fixtures"""
        self.scorer = TechnicalSignalScorer()

        # Create sample market data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'open': 100 + np.random.randn(50) * 2,
            'high': 102 + np.random.randn(50) * 2,
            'low': 98 + np.random.randn(50) * 2,
            'close': 100 + np.random.randn(50) * 2,
            'volume': np.random.randint(1000, 10000, 50)
        })

        self.market_data = {
            'df': self.df,
            'continuous_action': 0.0,
            'portfolio': {'position': 0, 'cash': 10000, 'value': 10000}
        }

    def test_initialization(self):
        """Test TechnicalSignalScorer initialization"""
        self.assertIsInstance(self.scorer, TechnicalSignalScorer)
        self.assertTrue(hasattr(self.scorer, 'technical_indicators'))

    def test_calculate_score_normal_conditions(self):
        """Test score calculation under normal market conditions"""
        score = self.scorer.calculate_score(self.market_data)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_calculate_score_oversold_rsi(self):
        """Test score with oversold RSI conditions"""
        # Create oversold conditions (RSI < 30)
        oversold_df = self.df.copy()
        oversold_df['close'] = np.linspace(95, 85, 50)  # Declining prices

        market_data_oversold = self.market_data.copy()
        market_data_oversold['df'] = oversold_df

        score = self.scorer.calculate_score(market_data_oversold)
        self.assertIsInstance(score, float)

    def test_calculate_score_overbought_rsi(self):
        """Test score with overbought RSI conditions"""
        # Create overbought conditions (RSI > 70)
        overbought_df = self.df.copy()
        overbought_df['close'] = np.linspace(105, 115, 50)  # Rising prices

        market_data_overbought = self.market_data.copy()
        market_data_overbought['df'] = overbought_df

        score = self.scorer.calculate_score(market_data_overbought)
        self.assertIsInstance(score, float)

    def test_calculate_score_bullish_macd(self):
        """Test score with bullish MACD crossover"""
        score = self.scorer.calculate_score(self.market_data)
        self.assertIsInstance(score, float)

    def test_calculate_score_bearish_macd(self):
        """Test score with bearish MACD crossover"""
        # Create bearish conditions
        bearish_df = self.df.copy()
        bearish_df['close'] = np.linspace(110, 90, 50)  # Sharp decline

        market_data_bearish = self.market_data.copy()
        market_data_bearish['df'] = bearish_df

        score = self.scorer.calculate_score(market_data_bearish)
        self.assertIsInstance(score, float)

    def test_calculate_score_bollinger_lower_band(self):
        """Test score when price is near Bollinger lower band"""
        # Create conditions where price is near lower band
        low_volatility_df = self.df.copy()
        low_volatility_df['close'] = 100 + np.sin(np.linspace(0, 4*np.pi, 50)) * 1  # Low volatility

        market_data_low_vol = self.market_data.copy()
        market_data_low_vol['df'] = low_volatility_df

        score = self.scorer.calculate_score(market_data_low_vol)
        self.assertIsInstance(score, float)

    def test_calculate_score_bollinger_upper_band(self):
        """Test score when price is near Bollinger upper band"""
        score = self.scorer.calculate_score(self.market_data)
        self.assertIsInstance(score, float)

    def test_get_confidence(self):
        """Test confidence calculation"""
        confidence = self.scorer.get_confidence(self.market_data)

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_get_confidence_low_data_quality(self):
        """Test confidence with low data quality"""
        # Test with very short DataFrame
        short_df = self.df.head(5)
        market_data_short = self.market_data.copy()
        market_data_short['df'] = short_df

        confidence = self.scorer.get_confidence(market_data_short)
        self.assertLess(confidence, 0.5)  # Should be lower confidence

    def test_error_handling_empty_dataframe(self):
        """Test error handling with empty DataFrame"""
        empty_market_data = self.market_data.copy()
        empty_market_data['df'] = pd.DataFrame()

        score = self.scorer.calculate_score(empty_market_data)
        confidence = self.scorer.get_confidence(empty_market_data)

        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)

    def test_error_handling_none_dataframe(self):
        """Test error handling with None DataFrame"""
        none_market_data = self.market_data.copy()
        none_market_data['df'] = None

        score = self.scorer.calculate_score(none_market_data)
        confidence = self.scorer.get_confidence(none_market_data)

        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)


class TestPatternRecognitionScorer(unittest.TestCase):
    """Test cases for PatternRecognitionScorer"""

    def setUp(self):
        """Set up test fixtures"""
        self.scorer = PatternRecognitionScorer()

        # Create sample market data with trend
        np.random.seed(42)
        self.df = pd.DataFrame({
            'open': 100 + np.random.randn(50) * 2,
            'high': 102 + np.random.randn(50) * 2,
            'low': 98 + np.random.randn(50) * 2,
            'close': 100 + np.linspace(0, 10, 50),  # Upward trend
            'volume': np.random.randint(1000, 10000, 50)
        })

        self.market_data = {
            'df': self.df,
            'continuous_action': 0.0,
            'portfolio': {'position': 0, 'cash': 10000, 'value': 10000}
        }

    def test_initialization(self):
        """Test PatternRecognitionScorer initialization"""
        self.assertIsInstance(self.scorer, PatternRecognitionScorer)
        self.assertEqual(self.scorer.min_periods, 20)

    def test_calculate_score_uptrend_continuation(self):
        """Test score calculation for uptrend continuation"""
        score = self.scorer.calculate_score(self.market_data)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_calculate_score_downtrend_continuation(self):
        """Test score calculation for downtrend continuation"""
        # Create downtrend
        downtrend_df = self.df.copy()
        downtrend_df['close'] = 100 - np.linspace(0, 10, 50)

        market_data_down = self.market_data.copy()
        market_data_down['df'] = downtrend_df

        score = self.scorer.calculate_score(market_data_down)
        self.assertIsInstance(score, float)

    def test_calculate_score_reversal_pattern(self):
        """Test score calculation for reversal patterns"""
        # Create potential reversal (uptrend with recent decline)
        reversal_df = self.df.copy()
        reversal_df['close'] = 100 + np.linspace(0, 8, 50)
        reversal_df.loc[45:49, 'close'] = [108, 107, 106, 105, 104]  # Recent decline

        market_data_reversal = self.market_data.copy()
        market_data_reversal['df'] = reversal_df

        score = self.scorer.calculate_score(market_data_reversal)
        self.assertIsInstance(score, float)


    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        short_df = self.df.head(10)
        market_data_short = self.market_data.copy()
        market_data_short['df'] = short_df

        score = self.scorer.calculate_score(market_data_short)
        confidence = self.scorer.get_confidence(market_data_short)

        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)


class TestSentimentSignalScorer(unittest.TestCase):
    """Test cases for SentimentSignalScorer"""

    def setUp(self):
        """Set up test fixtures"""
        self.scorer = SentimentSignalScorer()

        # Create sample market data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'open': 100 + np.random.randn(50) * 2,
            'high': 102 + np.random.randn(50) * 2,
            'low': 98 + np.random.randn(50) * 2,
            'close': 100 + np.random.randn(50) * 2,
            'volume': np.random.randint(1000, 10000, 50)
        })

        self.market_data = {
            'df': self.df,
            'continuous_action': 0.0,
            'portfolio': {'position': 0, 'cash': 10000, 'value': 10000}
        }

    def test_initialization(self):
        """Test SentimentSignalScorer initialization"""
        self.assertIsInstance(self.scorer, SentimentSignalScorer)
        self.assertEqual(self.scorer.sentiment_window, 24)

    def test_calculate_score_positive_sentiment(self):
        """Test score with positive sentiment (rising short and long momentum)"""
        # Create positive momentum
        positive_df = self.df.copy()
        positive_df['close'] = 100 + np.linspace(0, 5, 50)  # Steady rise

        market_data_positive = self.market_data.copy()
        market_data_positive['df'] = positive_df

        score = self.scorer.calculate_score(market_data_positive)
        self.assertIsInstance(score, float)

    def test_calculate_score_negative_sentiment(self):
        """Test score with negative sentiment (falling short and long momentum)"""
        # Create negative momentum
        negative_df = self.df.copy()
        negative_df['close'] = 100 - np.linspace(0, 5, 50)  # Steady decline

        market_data_negative = self.market_data.copy()
        market_data_negative['df'] = negative_df

        score = self.scorer.calculate_score(market_data_negative)
        self.assertIsInstance(score, float)

    def test_calculate_score_mixed_sentiment(self):
        """Test score with mixed sentiment"""
        score = self.scorer.calculate_score(self.market_data)
        self.assertIsInstance(score, float)

    def test_get_confidence(self):
        """Test confidence calculation"""
        confidence = self.scorer.get_confidence(self.market_data)

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

        # Sentiment confidence should be moderate (around 0.4)
        self.assertAlmostEqual(confidence, 0.4, delta=0.1)


class TestVolumeProfileScorer(unittest.TestCase):
    """Test cases for VolumeProfileScorer"""

    def setUp(self):
        """Set up test fixtures"""
        self.scorer = VolumeProfileScorer()

        # Create sample market data with volume
        np.random.seed(42)
        self.df = pd.DataFrame({
            'open': 100 + np.random.randn(50) * 2,
            'high': 102 + np.random.randn(50) * 2,
            'low': 98 + np.random.randn(50) * 2,
            'close': 100 + np.random.randn(50) * 2,
            'volume': np.random.randint(1000, 10000, 50)
        })

        self.market_data = {
            'df': self.df,
            'continuous_action': 0.0,
            'portfolio': {'position': 0, 'cash': 10000, 'value': 10000}
        }

    def test_initialization(self):
        """Test VolumeProfileScorer initialization"""
        self.assertIsInstance(self.scorer, VolumeProfileScorer)
        self.assertEqual(self.scorer.volume_window, 20)

    def test_calculate_score_high_volume_confirmation(self):
        """Test score with high volume price confirmation"""
        # Create price rise with high volume
        high_volume_df = self.df.copy()
        high_volume_df['close'] = 100 + np.linspace(0, 3, 50)
        high_volume_df['volume'] = np.random.randint(8000, 15000, 50)  # High volume

        market_data_high_vol = self.market_data.copy()
        market_data_high_vol['df'] = high_volume_df

        score = self.scorer.calculate_score(market_data_high_vol)
        self.assertIsInstance(score, float)

    def test_calculate_score_low_volume_uncertainty(self):
        """Test score with low volume uncertainty"""
        # Create low volume conditions
        low_volume_df = self.df.copy()
        low_volume_df['volume'] = np.random.randint(100, 500, 50)  # Low volume

        market_data_low_vol = self.market_data.copy()
        market_data_low_vol['df'] = low_volume_df

        score = self.scorer.calculate_score(market_data_low_vol)
        self.assertIsInstance(score, float)

    def test_calculate_score_volume_divergence(self):
        """Test score with volume divergence patterns"""
        score = self.scorer.calculate_score(self.market_data)
        self.assertIsInstance(score, float)

    def test_get_confidence(self):
        """Test confidence calculation"""
        confidence = self.scorer.get_confidence(self.market_data)

        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

        no_volume_df = self.df.drop(columns=['volume'])
        market_data_no_vol = self.market_data.copy()
        market_data_no_vol['df'] = no_volume_df

        confidence = self.scorer.get_confidence(market_data_no_vol)
        self.assertLess(confidence, 0.3)  # Should be low confidence without volume

    def test_insufficient_volume_data(self):
        """Test behavior with insufficient volume data"""
        short_df = self.df.head(10)
        market_data_short = self.market_data.copy()
        market_data_short['df'] = short_df

        score = self.scorer.calculate_score(market_data_short)
        confidence = self.scorer.get_confidence(market_data_short)

        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)


if __name__ == '__main__':
    unittest.main()
