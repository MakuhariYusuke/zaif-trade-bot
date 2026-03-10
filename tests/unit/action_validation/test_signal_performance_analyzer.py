#!/usr/bin/env python3
"""
Unit tests for SignalPerformanceAnalyzer

This module contains comprehensive unit tests for the SignalPerformanceAnalyzer
component, including signal quality scoring, SAC correlation analysis,
and performance reporting functionality.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.analysis.signal_performance_analyzer import (
    SignalPerformanceAnalyzer,
)
from ztb.trading.strategies.action_signal_guide.components.performance_tracker import (
    PerformanceTracker,
)
from ztb.trading.strategies.action_signal_guide.components.pattern_statistics import (
    PatternStatistics,
)


class TestSignalPerformanceAnalyzer(unittest.TestCase):
    """Test cases for SignalPerformanceAnalyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_performance_tracker = MagicMock(spec=PerformanceTracker)
        self.mock_pattern_statistics = MagicMock(spec=PatternStatistics)

        # Create analyzer instance
        self.analyzer = SignalPerformanceAnalyzer(
            performance_tracker=self.mock_performance_tracker,
            pattern_statistics=self.mock_pattern_statistics,
            max_history_size=100
        )

    def test_initialization(self):
        """Test SignalPerformanceAnalyzer initialization."""
        self.assertIsInstance(self.analyzer, SignalPerformanceAnalyzer)
        self.assertEqual(self.analyzer.max_history_size, 100)
        self.assertIsInstance(self.analyzer.signal_quality_history, list)
        self.assertIsInstance(self.analyzer.signal_sac_correlations, list)
        self.assertEqual(len(self.analyzer.quality_weights), 4)
        self.assertIn('strength', self.analyzer.quality_weights)
        self.assertIn('confidence', self.analyzer.quality_weights)
        self.assertIn('success_rate', self.analyzer.quality_weights)
        self.assertIn('consistency', self.analyzer.quality_weights)

    def test_calculate_signal_quality_score_basic(self):
        """Test basic signal quality score calculation."""
        score = self.analyzer.calculate_signal_quality_score(
            signal_strength=0.8,
            signal_confidence=0.9,
            pattern_type='fibonacci',
            historical_success_rate=0.7,
            consistency_score=0.8
        )

        # Verify score is within valid range
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Verify calculation: (0.4*0.8 + 0.3*0.9 + 0.2*0.7 + 0.1*0.8) * 1.1
        expected_base = 0.4*0.8 + 0.3*0.9 + 0.2*0.7 + 0.1*0.8  # 0.79
        expected_score = min(1.0, expected_base * 1.1)  # 0.869
        self.assertAlmostEqual(score, expected_score, places=3)

    def test_calculate_signal_quality_score_pattern_adjustments(self):
        """Test pattern type adjustments in quality scoring."""
        # Test different pattern adjustments
        patterns_and_adjustments = {
            'fibonacci': 1.1,
            'harmonic': 1.05,
            'dow_theory': 1.0,
            'candlestick': 0.95,
            'oscillator': 0.9,
            'unknown': 1.0  # Default adjustment
        }

        base_params = {
            'signal_strength': 0.5,
            'signal_confidence': 0.5,
            'historical_success_rate': 0.5,
            'consistency_score': 0.5
        }

        base_score = 0.4*0.5 + 0.3*0.5 + 0.2*0.5 + 0.1*0.5  # 0.5

        for pattern, adjustment in patterns_and_adjustments.items():
            with self.subTest(pattern=pattern):
                score = self.analyzer.calculate_signal_quality_score(
                    pattern_type=pattern,
                    **base_params
                )
                expected = min(1.0, base_score * adjustment)
                self.assertAlmostEqual(score, expected, places=3)

    def test_calculate_signal_quality_score_edge_cases(self):
        """Test edge cases in signal quality scoring."""
        # Test with all zeros
        score = self.analyzer.calculate_signal_quality_score(
            signal_strength=0.0,
            signal_confidence=0.0,
            pattern_type='fibonacci',
            historical_success_rate=0.0,
            consistency_score=0.0
        )
        self.assertEqual(score, 0.0)

        # Test with all ones
        score = self.analyzer.calculate_signal_quality_score(
            signal_strength=1.0,
            signal_confidence=1.0,
            pattern_type='fibonacci',
            historical_success_rate=1.0,
            consistency_score=1.0
        )
        expected = min(1.0, 1.0 * 1.1)  # 1.1 clamped to 1.0
        self.assertEqual(score, expected)

    def test_analyze_sac_learning_correlation_no_data(self):
        """Test SAC correlation analysis with no data."""
        result = self.analyzer.analyze_sac_learning_correlation()

        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No SAC learning data available')

    def test_analyze_sac_learning_correlation_with_data(self):
        """Test SAC correlation analysis with mock data."""
        # Add mock signal quality data
        self.analyzer.signal_quality_history = [
            {'quality_score': 0.8, 'timestamp': pd.Timestamp('2024-01-01')},
            {'quality_score': 0.6, 'timestamp': pd.Timestamp('2024-01-02')},
            {'quality_score': 0.9, 'timestamp': pd.Timestamp('2024-01-03')},
        ]

        # Mock SAC learning logs
        sac_logs = [
            {'reward': 1.0, 'loss': 0.5, 'timestep': 0},
            {'reward': 0.5, 'loss': 0.3, 'timestep': 1},
            {'reward': 1.5, 'loss': 0.2, 'timestep': 2},
        ]

        result = self.analyzer.analyze_sac_learning_correlation(
            sac_learning_logs=sac_logs,
            correlation_window=50
        )

        # Verify result structure
        self.assertIn('correlation_coefficient', result)
        self.assertIn('p_value', result)
        self.assertIn('correlation_strength', result)
        self.assertIn('rolling_correlations', result)
        self.assertIn('contribution_analysis', result)
        self.assertIn('data_points', result)
        self.assertIn('correlation_trend', result)

        # Verify correlation coefficient is valid
        self.assertGreaterEqual(result['correlation_coefficient'], -1.0)
        self.assertLessEqual(result['correlation_coefficient'], 1.0)

        # Verify data points count
        self.assertEqual(result['data_points'], 3)

    def test_analyze_sac_learning_correlation_insufficient_data(self):
        """Test SAC correlation analysis with insufficient data."""
        # Add only one data point
        self.analyzer.signal_quality_history = [
            {'quality_score': 0.8, 'timestamp': pd.Timestamp('2024-01-01')},
        ]

        sac_logs = [{'reward': 1.0, 'loss': 0.5, 'timestep': 0}]

        result = self.analyzer.analyze_sac_learning_correlation(
            sac_learning_logs=sac_logs
        )

        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Insufficient data for correlation analysis')

    def test_calculate_trend_with_sufficient_data(self):
        """Test trend calculation with sufficient data."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        trend = self.analyzer._calculate_trend(values)

        # Should return a float between -1 and 1
        self.assertIsInstance(trend, float)
        self.assertGreaterEqual(trend, -1.0)
        self.assertLessEqual(trend, 1.0)

        # With increasing values, trend should be positive
        self.assertGreater(trend, 0.0)

    def test_calculate_trend_with_insufficient_data(self):
        """Test trend calculation with insufficient data."""
        values = [0.5]  # Only 1 point
        trend = self.analyzer._calculate_trend(values)

        self.assertEqual(trend, 0.0)

    def test_calculate_trend_with_pandas_series(self):
        """Test trend calculation with pandas Series input."""
        values = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        trend = self.analyzer._calculate_trend(values)

        self.assertIsInstance(trend, float)
        self.assertGreaterEqual(trend, -1.0)
        self.assertLessEqual(trend, 1.0)

    def test_generate_performance_report_empty(self):
        """Test performance report generation with no data."""
        report = self.analyzer.generate_performance_report()

        # Should still generate a report structure
        self.assertIn('timestamp', report)
        self.assertIn('signal_quality_metrics', report)
        self.assertIn('sac_correlation_analysis', report)
        self.assertIn('pattern_effectiveness', report)
        self.assertIn('regime_performance', report)
        self.assertIn('recommendations', report)

    def test_generate_performance_report_with_data(self):
        """Test performance report generation with mock data."""
        # Add mock signal quality data
        self.analyzer.signal_quality_history = [
            {'quality_score': 0.8, 'pattern_type': 'fibonacci', 'timestamp': pd.Timestamp.now()},
            {'quality_score': 0.6, 'pattern_type': 'harmonic', 'timestamp': pd.Timestamp.now()},
            {'quality_score': 0.9, 'pattern_type': 'fibonacci', 'timestamp': pd.Timestamp.now()},
        ]

        # Add mock SAC correlation data
        self.analyzer.signal_sac_correlations = [
            {'correlation': 0.7, 'p_value': 0.01, 'overall_correlation': 0.7},
            {'correlation': 0.8, 'p_value': 0.005, 'overall_correlation': 0.8},
        ]

        report = self.analyzer.generate_performance_report()

        # Verify report structure
        self.assertIn('timestamp', report)
        self.assertIn('signal_quality_metrics', report)
        self.assertIn('sac_correlation_analysis', report)
        self.assertIn('pattern_effectiveness', report)
        self.assertIn('regime_performance', report)
        self.assertIn('recommendations', report)

        # Verify signal quality metrics
        quality_metrics = report['signal_quality_metrics']
        self.assertIn('average_quality_score', quality_metrics)
        self.assertIn('quality_score_std', quality_metrics)
        self.assertIn('quality_score_trend', quality_metrics)

        # Verify SAC correlation analysis
        correlation_analysis = report['sac_correlation_analysis']
        self.assertIn('average_correlation', correlation_analysis)
        self.assertIn('correlation_trend', correlation_analysis)

        # Verify recommendations is a list
        self.assertIsInstance(report['recommendations'], list)

    def test_interpret_correlation_strength(self):
        """Test correlation strength interpretation."""
        test_cases = [
            (0.9, 'very_strong'),
            (0.7, 'strong'),
            (0.3, 'moderate'),
            (0.1, 'weak'),
            (-0.1, 'weak'),
            (-0.3, 'moderate'),
            (-0.7, 'strong'),
            (-0.9, 'very_strong'),
            (0.0, 'very_weak'),
        ]

        for correlation, expected in test_cases:
            with self.subTest(correlation=correlation):
                strength = self.analyzer._interpret_correlation_strength(correlation)
                self.assertEqual(strength, expected)

    def test_calculate_signal_contribution_score(self):
        """Test signal contribution score calculation."""
        score = self.analyzer.calculate_signal_contribution_score(
            signal_quality=0.8,
            sac_action_alignment=0.7,
            market_regime='bull',
            pattern_type='fibonacci'
        )

        # Verify score is within valid range
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Verify contribution score is stored
        self.assertIn('fibonacci', self.analyzer.signal_contribution_scores)
        self.assertIn('bull', self.analyzer.regime_signal_effectiveness)

    def test_history_size_management(self):
        """Test that history size is properly managed."""
        # Add more data than max_history_size
        for i in range(150):  # More than max_history_size (100)
            self.analyzer.signal_quality_history.append({
                'quality_score': 0.5,
                'pattern_type': 'test',
                'timestamp': pd.Timestamp.now()
            })

        # Force cleanup by calling a method that manages history
        self.analyzer.calculate_signal_quality_score(
            signal_strength=0.5,
            signal_confidence=0.5,
            pattern_type='test',
            historical_success_rate=0.5,
            consistency_score=0.5
        )

        # History should be trimmed to max_history_size
        self.assertLessEqual(len(self.analyzer.signal_quality_history), self.analyzer.max_history_size)


if __name__ == '__main__':
    unittest.main()