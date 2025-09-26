#!/usr/bin/env python3
"""
Unit tests for quality gates functionality
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from ztb.evaluation.auto_feature_generator import AutoFeatureGenerator
from ztb.evaluation.evaluate import TradingEvaluator


class TestQualityGates(unittest.TestCase):
    """Test cases for quality gates in auto feature generation"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock AutoFeatureGenerator
        self.generator = Mock(spec=AutoFeatureGenerator)
        self.generator.params = {
            'quality_gates': {
                'max_nan_rate_threshold': 0.5,
                'min_abs_correlation_threshold': 0.1
            }
        }

        # Create sample OHLC data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        self.ohlc_data = pd.DataFrame({
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 105 + np.random.randn(100).cumsum(),
            'low': 95 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_apply_quality_gates_good_features(self):
        """Test quality gates with good quality features"""
        # Create high-quality features
        good_features = {
            'test_feature_1': pd.DataFrame({
                'col1': np.random.randn(100) * 0.1 + self.ohlc_data['close'] * 0.05  # Correlated with close
            }, index=self.ohlc_data.index),
            'test_feature_2': pd.DataFrame({
                'col2': np.random.randn(100) * 0.05 + self.ohlc_data['close'] * 0.03  # Correlated with close
            }, index=self.ohlc_data.index)
        }

        # Mock the _apply_quality_gates method to use real implementation
        from ztb.evaluation.auto_feature_generator import AutoFeatureGenerator
        real_generator = AutoFeatureGenerator.__new__(AutoFeatureGenerator)
        real_generator.params = self.generator.params

        filtered = real_generator._apply_quality_gates(good_features, self.ohlc_data)

        # Good features should pass
        self.assertEqual(len(filtered), 2)
        self.assertIn('test_feature_1', filtered)
        self.assertIn('test_feature_2', filtered)

    def test_apply_quality_gates_high_nan_rate(self):
        """Test quality gates reject features with high NaN rate"""
        # Create feature with too many NaNs
        bad_features = {
            'high_nan_feature': pd.DataFrame({
                'col1': [np.nan] * 80 + list(np.random.randn(20))  # 80% NaN
            }, index=self.ohlc_data.index)
        }

        from ztb.evaluation.auto_feature_generator import AutoFeatureGenerator
        real_generator = AutoFeatureGenerator.__new__(AutoFeatureGenerator)
        real_generator.params = self.generator.params

        filtered = real_generator._apply_quality_gates(bad_features, self.ohlc_data)

        # High NaN feature should be rejected
        self.assertEqual(len(filtered), 0)

    def test_apply_quality_gates_low_correlation(self):
        """Test quality gates reject features with low correlation"""
        # Create feature with very low correlation to price
        bad_features = {
            'low_corr_feature': pd.DataFrame({
                'col1': np.random.randn(100)  # Uncorrelated noise
            }, index=self.ohlc_data.index)
        }

        from ztb.evaluation.auto_feature_generator import AutoFeatureGenerator
        real_generator = AutoFeatureGenerator.__new__(AutoFeatureGenerator)
        real_generator.params = self.generator.params

        filtered = real_generator._apply_quality_gates(bad_features, self.ohlc_data)

        # Low correlation feature should be rejected
        self.assertEqual(len(filtered), 0)

    def test_apply_quality_gates_mixed_features(self):
        """Test quality gates with mix of good and bad features"""
        mixed_features = {
            'good_feature': pd.DataFrame({
                'col1': np.random.randn(100) * 0.1 + self.ohlc_data['close'] * 0.05
            }, index=self.ohlc_data.index),
            'high_nan_feature': pd.DataFrame({
                'col1': [np.nan] * 80 + list(np.random.randn(20))
            }, index=self.ohlc_data.index),
            'low_corr_feature': pd.DataFrame({
                'col1': np.random.randn(100)
            }, index=self.ohlc_data.index)
        }

        from ztb.evaluation.auto_feature_generator import AutoFeatureGenerator
        real_generator = AutoFeatureGenerator.__new__(AutoFeatureGenerator)
        real_generator.params = self.generator.params

        filtered = real_generator._apply_quality_gates(mixed_features, self.ohlc_data)

        # Only good feature should pass
        self.assertEqual(len(filtered), 1)
        self.assertIn('good_feature', filtered)
        self.assertNotIn('high_nan_feature', filtered)
        self.assertNotIn('low_corr_feature', filtered)

    def test_apply_quality_gates_no_config(self):
        """Test quality gates with no configuration (should use defaults)"""
        self.generator.params = {}  # No quality_gates config

        good_features = {
            'test_feature': pd.DataFrame({
                'col1': np.random.randn(100) * 0.1 + self.ohlc_data['close'] * 0.05
            }, index=self.ohlc_data.index)
        }

        from ztb.evaluation.auto_feature_generator import AutoFeatureGenerator
        real_generator = AutoFeatureGenerator.__new__(AutoFeatureGenerator)
        real_generator.params = self.generator.params

        filtered = real_generator._apply_quality_gates(good_features, self.ohlc_data)

        # Should still work with defaults
        self.assertEqual(len(filtered), 1)


class TestTradingEvaluatorQualityGates(unittest.TestCase):
    """Test cases for quality gates in TradingEvaluator"""

    def test_calculate_outlier_rate_normal_distribution(self):
        """Test outlier rate calculation with normal-like data"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Create normal-like data
        rewards = np.random.normal(0, 1, 100)
        stats = {"episode_rewards": rewards.tolist()}

        outlier_rate = evaluator._calculate_outlier_rate(stats)
        self.assertGreaterEqual(outlier_rate, 0.0)
        self.assertLessEqual(outlier_rate, 0.1)  # Should be low for normal data

    def test_calculate_outlier_rate_with_outliers(self):
        """Test outlier rate calculation with clear outliers"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Create data with clear outliers
        rewards = [0.1] * 95 + [5.0, -3.0, 4.0, -4.0, 6.0]  # 5 outliers out of 100
        stats = {"episode_rewards": rewards}

        outlier_rate = evaluator._calculate_outlier_rate(stats)
        self.assertGreater(outlier_rate, 0.04)  # At least 4% outliers detected

    def test_calculate_distribution_quality_normal(self):
        """Test distribution quality with normal data"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Create approximately normal data
        rewards = np.random.normal(0, 1, 50)
        stats = {"episode_rewards": rewards.tolist()}

        quality = evaluator._calculate_distribution_quality(stats)
        self.assertGreaterEqual(quality, 0.7)
        self.assertLessEqual(quality, 1.0)  # Should be high for normal-like data

    def test_calculate_stability_trend_improving(self):
        """Test stability trend with improving (decreasing) variance"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Create data with decreasing variance over time
        rewards = []
        for i in range(10):
            # Variance decreases over time
            variance = 1.0 / (i + 1)
            episode = np.random.normal(0, variance, 20)
            rewards.extend(episode)

        stats = {"episode_rewards": rewards}

        stability = evaluator._calculate_stability_trend(stats)
        self.assertGreater(stability, 0.5)  # Should show positive stability trend

    def test_calculate_data_quality_score_comprehensive(self):
        """Test that data quality score includes all new components"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Create comprehensive test stats
        stats = {
            "reward_stats": {
                "mean_total_reward": 1.0,
                "std_total_reward": 0.5,
                "sharpe_ratio": 0.8
            },
            "trading_stats": {
                "win_rate": 0.65,
                "hold_ratio_penalty": 0.1,
                "profit_factor": 1.5
            },
            "episode_rewards": np.random.normal(0, 1, 100).tolist()
        }

        score = evaluator._calculate_data_quality_score(stats)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(score, float)

    def test_data_quality_score_extreme_values(self):
        """Test data quality score handles extreme values properly"""
        evaluator = TradingEvaluator.__new__(TradingEvaluator)

        # Test with very poor stats
        stats = {
            "reward_stats": {
                "mean_total_reward": -1.0,
                "std_total_reward": 2.0,
                "sharpe_ratio": -0.5
            },
            "trading_stats": {
                "win_rate": 0.3,
                "hold_ratio_penalty": 0.8,
                "profit_factor": 0.5
            },
            "episode_rewards": [10, -10, 15, -15, 20]  # Very volatile
        }

        score = evaluator._calculate_data_quality_score(stats)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertLess(score, 0.5)  # Should be low for poor stats


if __name__ == '__main__':
    unittest.main()