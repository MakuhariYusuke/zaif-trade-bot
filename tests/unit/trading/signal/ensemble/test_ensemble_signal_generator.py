"""
Unit tests for EnsembleSignalGenerator

Tests cover initialization, ensemble signal generation, individual scorer testing,
dynamic weight adjustment, confidence calculation, and error handling.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from ztb.trading.signal.ensemble_signal_generator import (
    EnsembleSignalGenerator,
    TechnicalSignalScorer,
    PatternRecognitionScorer,
    SentimentSignalScorer,
    VolumeProfileScorer,
    BaseSignalScorer
)


class TestEnsembleSignalGenerator(unittest.TestCase):
    """Test cases for EnsembleSignalGenerator"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'signal_sources': ['technical', 'pattern', 'sentiment', 'volume'],
            'ensemble_weights': {
                'technical': 0.4,
                'pattern': 0.3,
                'sentiment': 0.2,
                'volume': 0.1
            },
            'enable_dynamic_weights': True,
            'confidence_threshold': 0.6
        }

        # Create sample market data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        self.df = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 2,
            'high': 102 + np.random.randn(100) * 2,
            'low': 98 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': np.random.randint(1000, 10000, 100)
        })

        self.market_data = {
            'df': self.df,
            'continuous_action': 0.0,
            'portfolio': {'position': 0, 'cash': 10000, 'value': 10000}
        }

        self.generator = EnsembleSignalGenerator(self.config)

    def test_initialization(self):
        """Test EnsembleSignalGenerator initialization"""
        self.assertIsInstance(self.generator, EnsembleSignalGenerator)
        self.assertIn('technical', self.generator.signal_sources)
        self.assertIn('pattern', self.generator.signal_sources)
        self.assertIn('sentiment', self.generator.signal_sources)
        self.assertIn('volume', self.generator.signal_sources)

    def test_default_config(self):
        """Test default configuration loading"""
        generator = EnsembleSignalGenerator()
        self.assertIsInstance(generator, EnsembleSignalGenerator)
        self.assertTrue(hasattr(generator, 'signal_sources'))

    def test_generate_ensemble_signal(self):
        """Test ensemble signal generation"""
        score, confidence = self.generator.generate_ensemble_signal(self.market_data)

        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_generate_ensemble_signal_empty_data(self):
        """Test ensemble signal generation with empty data"""
        empty_market_data = {
            'df': pd.DataFrame(),
            'continuous_action': 0.0,
            'portfolio': {}
        }

        score, confidence = self.generator.generate_ensemble_signal(empty_market_data)

        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)

    def test_adjust_weights_dynamically(self):
        """Test dynamic weight adjustment"""
        confidences = {
            'technical': 0.8,
            'pattern': 0.6,
            'sentiment': 0.4,
            'volume': 0.7
        }

        adjusted_weights = self.generator._adjust_weights_dynamically(confidences)

        self.assertIsInstance(adjusted_weights, dict)
        self.assertEqual(len(adjusted_weights), 4)
        self.assertAlmostEqual(sum(adjusted_weights.values()), 1.0, places=5)

    def test_adjust_weights_dynamically_low_confidence(self):
        """Test dynamic weight adjustment with low confidence"""
        confidences = {
            'technical': 0.2,
            'pattern': 0.1,
            'sentiment': 0.3,
            'volume': 0.2
        }

        adjusted_weights = self.generator._adjust_weights_dynamically(confidences)

        # Should fall back to base weights when confidence is low
        self.assertIsInstance(adjusted_weights, dict)
        self.assertEqual(len(adjusted_weights), 4)

    def test_get_signal_reliability(self):
        """Test signal reliability calculation"""
        reliability = self.generator.get_signal_reliability(self.market_data)

        self.assertIsInstance(reliability, dict)
        self.assertIn('overall_reliability', reliability)
        self.assertIn('source_reliabilities', reliability)
        self.assertGreaterEqual(reliability['overall_reliability'], 0)
        self.assertLessEqual(reliability['overall_reliability'], 1)

    def test_individual_scorer_types(self):
        """Test that all signal sources are correct types"""
        self.assertIsInstance(self.generator.signal_sources['technical'], TechnicalSignalScorer)
        self.assertIsInstance(self.generator.signal_sources['pattern'], PatternRecognitionScorer)
        self.assertIsInstance(self.generator.signal_sources['sentiment'], SentimentSignalScorer)
        self.assertIsInstance(self.generator.signal_sources['volume'], VolumeProfileScorer)

    def test_base_signal_scorer_interface(self):
        """Test BaseSignalScorer interface"""
        scorer = BaseSignalScorer()
        self.assertEqual(scorer.name, 'basesignalscorer')
        self.assertIsInstance(scorer.config, dict)

        # Test abstract methods
        with self.assertRaises(NotImplementedError):
            scorer.calculate_score({})

    def test_technical_signal_scorer(self):
        """Test TechnicalSignalScorer functionality"""
        scorer = TechnicalSignalScorer()
        score = scorer.calculate_score(self.market_data)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

        confidence = scorer.get_confidence(self.market_data)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_pattern_recognition_scorer(self):
        """Test PatternRecognitionScorer functionality"""
        scorer = PatternRecognitionScorer()
        score = scorer.calculate_score(self.market_data)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

        confidence = scorer.get_confidence(self.market_data)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_sentiment_signal_scorer(self):
        """Test SentimentSignalScorer functionality"""
        scorer = SentimentSignalScorer()
        score = scorer.calculate_score(self.market_data)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

        confidence = scorer.get_confidence(self.market_data)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_volume_profile_scorer(self):
        """Test VolumeProfileScorer functionality"""
        scorer = VolumeProfileScorer()
        score = scorer.calculate_score(self.market_data)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

        confidence = scorer.get_confidence(self.market_data)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_scorer_error_handling(self):
        """Test error handling in individual scorers"""
        # Test with invalid data
        invalid_market_data = {
            'df': None,
            'continuous_action': 0.0,
            'portfolio': {}
        }

        for scorer_name, scorer in self.generator.signal_sources.items():
            score = scorer.calculate_score(invalid_market_data)
            confidence = scorer.get_confidence(invalid_market_data)

            self.assertIsInstance(score, float)
            self.assertIsInstance(confidence, float)

    def test_ensemble_with_disabled_sources(self):
        """Test ensemble with some sources disabled"""
        config_partial = self.config.copy()
        config_partial['signal_sources'] = ['technical', 'pattern']

        generator = EnsembleSignalGenerator(config_partial)
        score, confidence = generator.generate_ensemble_signal(self.market_data)

        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)

    def test_weight_normalization(self):
        """Test weight normalization"""
        # Test with weights that don't sum to 1
        config_unnormalized = self.config.copy()
        config_unnormalized['ensemble_weights'] = {
            'technical': 0.5,
            'pattern': 0.5,
            'sentiment': 0.5,
            'volume': 0.5
        }

        generator = EnsembleSignalGenerator(config_unnormalized)
        score, confidence = generator.generate_ensemble_signal(self.market_data)

        # Should still work despite unnormalized weights
        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)

    @patch('ztb.trading.signal.ensemble_signal_generator.logger')
    def test_logging_integration(self, mock_logger):
        """Test logging integration"""
        self.generator.generate_ensemble_signal(self.market_data)

        # Verify logging was called during ensemble generation
        # Note: Individual scorers may also log
        self.assertTrue(mock_logger.debug.called or mock_logger.info.called)

    def test_configuration_validation(self):
        """Test configuration validation and defaults"""
        # Test with minimal config
        minimal_config = {}
        generator = EnsembleSignalGenerator(minimal_config)

        self.assertIsInstance(generator, EnsembleSignalGenerator)
        self.assertTrue(len(generator.signal_sources) > 0)

        # Test ensemble generation with minimal config
        score, confidence = generator.generate_ensemble_signal(self.market_data)
        self.assertIsInstance(score, float)
        self.assertIsInstance(confidence, float)


if __name__ == '__main__':
    unittest.main()