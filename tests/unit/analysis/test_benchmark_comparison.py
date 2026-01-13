"""
Unit tests for benchmark_comparison.py
"""

import pandas as pd
import pytest
from ztb.analysis.comparative.benchmark_comparison import BenchmarkComparisonAnalyzer


class TestBenchmarkComparisonAnalyzer:
    """Test cases for BenchmarkComparisonAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return BenchmarkComparisonAnalyzer()

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series"""
        import numpy as np
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.02, 100))

    def test_compare_with_benchmark(self, analyzer, sample_returns):
        """Test single benchmark comparison"""
        benchmark_returns = sample_returns * 0.9

        comparison = analyzer.compare_with_benchmark(
            sample_returns, benchmark_returns, "Test Benchmark"
        )

        assert comparison is not None
        assert comparison.benchmark_name == "Test Benchmark"
        assert hasattr(comparison, 'tracking_error')
        assert hasattr(comparison, 'information_ratio')

    def test_compare_with_benchmark(self, analyzer, sample_returns):
        """Test single benchmark comparison"""
        benchmark_returns = sample_returns * 0.9

        comparison = analyzer.compare_with_benchmark(
            sample_returns, benchmark_returns, "Test Benchmark"
        )

        assert comparison is not None
        assert comparison.benchmark_name == "Test Benchmark"
        assert hasattr(comparison, 'tracking_error')
        assert hasattr(comparison, 'information_ratio')