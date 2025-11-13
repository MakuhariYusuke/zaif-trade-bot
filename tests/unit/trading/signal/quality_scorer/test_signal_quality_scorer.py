"""
Unit tests for SignalQualityScorer - Phase 1 Technical Indicator Improvements
"""

import unittest
import pandas as pd
import numpy as np

from ztb.trading.signal.quality_scorer import SignalQualityScorer


class TestSignalQualityScorer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'buy_threshold': 85,
            'sell_threshold': 15,
            'enable_ensemble': False,
            'weights': {
                'rsi': 0.25,
                'macd': 0.25,
                'bollinger': 0.20,
                'atr': 0.15,
                'trend': 0.15,
                'momentum': 0.15,
                'stochastic': 0.10
            }
        }
        self.scorer = SignalQualityScorer(self.config)

        # Create sample market data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        self.df = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 2,
            'high': 102 + np.random.randn(100) * 2,
            'low': 98 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        self.portfolio = {'position': 0, 'cash': 10000, 'value': 10000}

    def test_initialization(self):
        """Test SignalQualityScorer initialization with Phase 1 weights"""
        self.assertIsInstance(self.scorer, SignalQualityScorer)
        self.assertEqual(self.scorer.weights['rsi'], 0.25)
        self.assertEqual(self.scorer.weights['macd'], 0.25)
        self.assertEqual(self.scorer.weights['momentum'], 0.15)
        self.assertEqual(self.scorer.weights['stochastic'], 0.10)

    def test_calculate_signal_quality(self):
        """Test basic signal quality calculation"""
        action, score = self.scorer.calculate_signal_quality(self.df, 0.0, self.portfolio)
        self.assertIsInstance(action, int)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIn(action, [-1, 0, 1])

    def test_rsi_score_improvements(self):
        """Test improved RSI scoring with enhanced zoning"""
        # Test extreme oversold (strong BUY signal)
        rsi_signals = {'rsi': 20.0}
        score = self.scorer._calculate_rsi_score(rsi_signals)
        self.assertGreaterEqual(score, 90)  # Should be in 90-100 range

        # Test normal oversold (moderate BUY signal)
        rsi_signals = {'rsi': 30.0}
        score = self.scorer._calculate_rsi_score(rsi_signals)
        self.assertGreaterEqual(score, 70)  # Should be in 70-80 range
        self.assertLessEqual(score, 80)

        # Test extreme overbought (strong SELL signal)
        rsi_signals = {'rsi': 80.0}
        score = self.scorer._calculate_rsi_score(rsi_signals)
        self.assertLessEqual(score, 10)  # Should be in 0-10 range

        # Test normal overbought (moderate SELL signal)
        rsi_signals = {'rsi': 70.0}
        score = self.scorer._calculate_rsi_score(rsi_signals)
        self.assertGreaterEqual(score, 20)  # Should be in 20-30 range
        self.assertLessEqual(score, 30)

        # Test neutral zone
        rsi_signals = {'rsi': 50.0}
        score = self.scorer._calculate_rsi_score(rsi_signals)
        self.assertGreaterEqual(score, 25)  # Should be in 25-55 range
        self.assertLessEqual(score, 55)

    def test_atr_score_contextual(self):
        """Test contextual ATR scoring based on market volatility"""
        # High volatility market (ATR > 0.8 * avg_atr)
        atr_signals = {'atr': 1.0, 'avg_atr': 1.0}  # market_volatility = 1.0
        score = self.scorer._calculate_atr_score(atr_signals)
        self.assertGreaterEqual(score, 50)  # Should be boosted for trend formation

        # Low volatility market (ATR < 0.3 * avg_atr)
        atr_signals = {'atr': 0.2, 'avg_atr': 1.0}  # market_volatility = 0.2
        score = self.scorer._calculate_atr_score(atr_signals)
        self.assertLessEqual(score, 80)  # Should be dampened for noise potential

        # Normal volatility market
        atr_signals = {'atr': 0.6, 'avg_atr': 1.0}  # market_volatility = 0.6
        score = self.scorer._calculate_atr_score(atr_signals)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_momentum_score_calculation(self):
        """Test momentum indicator scoring"""
        # Strong positive momentum
        momentum_signals = {'momentum': 2.0}
        score = self.scorer._calculate_momentum_score(momentum_signals)
        self.assertGreater(score, 65)  # Should be in bullish range

        # Strong negative momentum
        momentum_signals = {'momentum': -2.0}
        score = self.scorer._calculate_momentum_score(momentum_signals)
        self.assertLess(score, 35)  # Should be in bearish range

        # Neutral momentum
        momentum_signals = {'momentum': 0.0}
        score = self.scorer._calculate_momentum_score(momentum_signals)
        self.assertAlmostEqual(score, 50, delta=5)  # Should be around neutral

    def test_stochastic_score_calculation(self):
        """Test stochastic indicator scoring"""
        # Bullish crossover in overbought zone
        stoch_signals = {'stoch_k': 75.0, 'stoch_d': 70.0}
        score = self.scorer._calculate_stochastic_score(stoch_signals)
        self.assertGreater(score, 70)  # Should be in bullish range

        # Bearish crossover in oversold zone
        stoch_signals = {'stoch_k': 25.0, 'stoch_d': 30.0}
        score = self.scorer._calculate_stochastic_score(stoch_signals)
        self.assertLess(score, 30)  # Should be in bearish range

    def test_weight_balance_total(self):
        """Test that all weights sum to 1.0"""
        total_weight = sum(self.scorer.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

    def test_ensemble_integration_disabled(self):
        """Test ensemble integration when disabled"""
        config_no_ensemble = self.config.copy()
        config_no_ensemble['enable_ensemble'] = False
        scorer = SignalQualityScorer(config_no_ensemble)

        action, score = scorer.calculate_signal_quality(self.df, 0.0, self.portfolio)
        self.assertIsInstance(action, int)
        self.assertIsInstance(score, float)

    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Empty dataframe
        empty_df = pd.DataFrame()
        action, score = self.scorer.calculate_signal_quality(empty_df, 0.0, self.portfolio)
        self.assertIsInstance(action, int)
        self.assertIsInstance(score, float)

    def test_threshold_boundaries(self):
        """Test buy/sell threshold boundaries"""
        # Force high score for BUY signal
        high_score_signals = {
            'rsi': 20.0,  # Extreme oversold
            'macd_line': 2.0, 'macd_signal': 1.0, 'macd_histogram': 1.0,
            'bollinger_position': -0.5,
            'atr': 1.0, 'avg_atr': 1.0,
            'trend_strength': 0.8,
            'momentum': 2.0,
            'stoch_k': 75.0, 'stoch_d': 70.0
        }

        # Mock the _get_enhanced_signals method to return our test signals
        original_method = self.scorer._get_enhanced_signals
        self.scorer._get_enhanced_signals = lambda df: high_score_signals

        try:
            action, score = self.scorer.calculate_signal_quality(self.df, 0.0, self.portfolio)
            self.assertGreaterEqual(score, 85)  # Should trigger BUY signal
        finally:
            self.scorer._get_enhanced_signals = original_method


if __name__ == '__main__':
    unittest.main()</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\tests\unit\trading\signal\quality_scorer\test_signal_quality_scorer.py