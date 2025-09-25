#!/usr/bin/env python3
"""
Unit tests for evaluator and analysis components.
"""

import pytest
import json
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from tools.evaluation.re_evaluate_features import ComprehensiveFeatureReEvaluator, generate_benchmark_output


class TestEvaluator:
    """Test the feature evaluator."""

    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly."""
        evaluator = ComprehensiveFeatureReEvaluator()
        assert hasattr(evaluator, 'price_data')
        assert isinstance(evaluator.price_data, pd.DataFrame)
        assert len(evaluator.price_data) > 0

    def test_evaluate_all_experimental(self):
        """Test evaluation of all experimental features."""
        evaluator = ComprehensiveFeatureReEvaluator()
        results = evaluator.evaluate_all_experimental(collect_frames=False)

        # Should have results for features
        details = {k: v for k,v in results.items() if not k.startswith('_')}
        assert len(details) > 20  # At least 20 features

        # Count successes
        success_count = sum(1 for v in details.values() if isinstance(v, dict) and v.get('status') == 'success')
        assert success_count >= 20  # At least 20 successful evaluations

        # Check insufficient count (should be 2: DOW, HourOfDay)
        insufficient_count = sum(1 for v in details.values() if isinstance(v, dict) and v.get('status') == 'insufficient')
        assert insufficient_count <= 3  # Allow some margin

        # Check error count (should be 0 after fixes)
        error_count = sum(1 for v in details.values() if isinstance(v, dict) and v.get('status') == 'error')
        assert error_count == 0  # Should be no errors

    def test_collect_frames(self):
        """Test frame collection for correlation analysis."""
        evaluator = ComprehensiveFeatureReEvaluator()
        results = evaluator.evaluate_all_experimental(collect_frames=True)

        assert '_success_frames' in results
        success_frames = results['_success_frames']
        assert isinstance(success_frames, dict)
        assert len(success_frames) >= 20  # At least 20 successful frames

        # Check frames are DataFrames
        for name, frame in success_frames.items():
            assert isinstance(frame, pd.DataFrame)
            assert len(frame) > 0

    def test_benchmark_generation(self):
        """Test benchmark output generation."""
        # Create mock details
        mock_details = {
            'test_feature_1': {
                'status': 'success',
                'computation_time_ms': 100.0,
                'nan_rate': 0.05,
                'sharpe_ratio': 0.1
            },
            'test_feature_2': {
                'status': 'success',
                'computation_time_ms': 200.0,
                'nan_rate': 0.02,
                'sharpe_ratio': 0.2
            }
        }

        # Generate benchmark
        generate_benchmark_output(mock_details)

        # Check files exist
        benchmark_csv = Path('reports/performance/benchmark_raw.csv')
        benchmark_json = Path('reports/performance/benchmark_summary.json')

        assert benchmark_csv.exists()
        assert benchmark_json.exists()

        # Check CSV content
        df = pd.read_csv(benchmark_csv)
        assert len(df) == 2
        assert 'feature_name' in df.columns
        assert 'computation_time_ms' in df.columns

        # Check JSON content
        with open(benchmark_json, 'r') as f:
            data = json.load(f)
        assert 'top5_slow' in data
        assert 'bottom5_slow' in data
        assert 'average_time_ms' in data

    def test_metrics_calculation(self):
        """Test that metrics are properly calculated."""
        evaluator = ComprehensiveFeatureReEvaluator()
        results = evaluator.evaluate_all_experimental(collect_frames=False)

        details = {k: v for k,v in results.items() if not k.startswith('_')}

        for name, result in details.items():
            if isinstance(result, dict) and result.get('status') == 'success':
                # Check required metrics
                assert 'sharpe_ratio' in result
                assert 'sortino_ratio' in result
                assert 'calmar_ratio' in result
                assert 'max_drawdown' in result
                assert 'num_periods' in result
                assert 'computation_time_ms' in result
                assert 'nan_rate' in result

                # Check types
                assert isinstance(result['sharpe_ratio'], (float, type(None)))
                assert isinstance(result['computation_time_ms'], (int, float))
                assert isinstance(result['nan_rate'], (int, float))