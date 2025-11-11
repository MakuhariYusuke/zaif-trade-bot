#!/usr/bin/env python3
"""
Unit Tests for Phase 2: Multi-Timeframe Trend Detection

Tests for MultiTimeframeAnalyzer and TrendConvergenceCalculator
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from ztb.trading.signal.multi_timeframe_analyzer import (
    MultiTimeframeAnalyzer, Timeframe, TrendDirection, TrendAnalysis
)
from ztb.trading.signal.trend_convergence_calculator import (
    TrendConvergenceCalculator, TrendConvergenceResult
)
from ztb.trading.signal.signal_guidance_system import SignalGuidanceSystem


@patch('ztb.trading.signal.multi_timeframe_analyzer.TechnicalIndicators')
class TestMultiTimeframeAnalyzer:
    """Test cases for MultiTimeframeAnalyzer"""

    def test_initialization(self, mock_technical_indicators):
        """Test analyzer initialization"""
        # Mock TechnicalIndicators instance
        mock_instance = Mock()
        mock_instance.calculate_rsi.return_value = 65.0
        mock_instance.calculate_macd.return_value = (0.5, 0.45, 0.05)  # macd, signal, histogram
        mock_instance.calculate_bollinger_bands.return_value = (50000, 51000, 49000)  # sma, upper, lower
        mock_technical_indicators.return_value = mock_instance
        
        analyzer = MultiTimeframeAnalyzer()
        assert analyzer is not None
        assert len(analyzer.timeframes) == 3
        assert all(tf in analyzer.timeframes for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15])

    def test_update_timeframe_data(self, mock_technical_indicators):
        """Test updating timeframe data"""
        # Mock TechnicalIndicators instance
        mock_instance = Mock()
        mock_instance.calculate_rsi.return_value = 65.0
        mock_instance.calculate_macd.return_value = (0.5, 0.45, 0.05)  # macd, signal, histogram
        mock_instance.calculate_bollinger_bands.return_value = (50000, 51000, 49000)  # sma, upper, lower
        mock_technical_indicators.return_value = mock_instance
        
        analyzer = MultiTimeframeAnalyzer()
        
        # Update M1 data
        analyzer.update_timeframe_data(Timeframe.M1, 50000.0, 100.0)

        assert len(analyzer.timeframes[Timeframe.M1].prices) == 1
        assert len(analyzer.timeframes[Timeframe.M1].volumes) == 1
        assert analyzer.timeframes[Timeframe.M1].prices[0] == 50000.0
        assert analyzer.timeframes[Timeframe.M1].volumes[0] == 100.0

    def test_analyze_timeframe_trend_insufficient_data(self, mock_technical_indicators):
        """Test trend analysis with insufficient data"""
        # Mock TechnicalIndicators instance
        mock_instance = Mock()
        mock_instance.calculate_rsi.return_value = 65.0
        mock_instance.calculate_macd.return_value = (0.5, 0.45, 0.05)  # macd, signal, histogram
        mock_instance.calculate_bollinger_bands.return_value = (50000, 51000, 49000)  # sma, upper, lower
        mock_technical_indicators.return_value = mock_instance
        
        analyzer = MultiTimeframeAnalyzer()
        
        # Add minimal data
        for i in range(5):
            analyzer.update_timeframe_data(Timeframe.M1, 50000.0 + i, 100.0)

        result = analyzer.analyze_timeframe_trend(Timeframe.M1)
        assert result is None  # Should return None for insufficient data

    def test_analyze_timeframe_trend_sufficient_data(self, mock_technical_indicators):
        """Test trend analysis with sufficient data"""
        # Mock TechnicalIndicators instance
        mock_instance = Mock()
        mock_instance.calculate_rsi.return_value = 65.0
        mock_instance.calculate_macd.return_value = (0.5, 0.45, 0.05)  # macd, signal, histogram
        mock_instance.calculate_bollinger_bands.return_value = (50000, 51000, 49000)  # sma, upper, lower
        mock_technical_indicators.return_value = mock_instance
        
        analyzer = MultiTimeframeAnalyzer()
        
        # Add sufficient data
        for i in range(25):
            price = 50000.0 + i * 10  # Upward trend
            analyzer.update_timeframe_data(Timeframe.M1, price, 100.0)

        result = analyzer.analyze_timeframe_trend(Timeframe.M1)

        assert result is not None
        assert isinstance(result, TrendAnalysis)
        assert result.rsi == 65.0
        assert result.macd_signal == "bullish"
        assert isinstance(result.direction, TrendDirection)

    def test_analyze_convergence_no_data(self, mock_technical_indicators):
        """Test convergence analysis with no data"""
        # Mock TechnicalIndicators instance
        mock_instance = Mock()
        mock_instance.calculate_rsi.return_value = 65.0
        mock_instance.calculate_macd.return_value = (0.5, 0.45, 0.05)  # macd, signal, histogram
        mock_instance.calculate_bollinger_bands.return_value = (50000, 51000, 49000)  # sma, upper, lower
        mock_technical_indicators.return_value = mock_instance
        
        analyzer = MultiTimeframeAnalyzer()
        
        result = analyzer.analyze_convergence()

        assert result.convergence_score == 50.0  # Neutral score
        assert result.dominant_trend == TrendDirection.NEUTRAL
        assert result.timeframe_agreement == 0.0

    def test_analyze_convergence_with_data(self, mock_technical_indicators):
        """Test convergence analysis with data"""
        # Mock TechnicalIndicators instance
        mock_instance = Mock()
        mock_instance.calculate_rsi.return_value = 65.0
        mock_instance.calculate_macd.return_value = (0.5, 0.45, 0.05)  # macd, signal, histogram
        mock_instance.calculate_bollinger_bands.return_value = (50000, 51000, 49000)  # sma, upper, lower
        mock_technical_indicators.return_value = mock_instance
        
        analyzer = MultiTimeframeAnalyzer()
        
        # Add data to create bullish trend across timeframes
        for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
            for i in range(25):
                price = 50000.0 + i * 10
                analyzer.update_timeframe_data(tf, price, 100.0)

        result = analyzer.analyze_convergence()

        assert result.convergence_score >= 0.0
        assert result.convergence_score <= 100.0
        assert isinstance(result.dominant_trend, TrendDirection)
        assert result.timeframe_agreement >= 0.0
        assert result.timeframe_agreement <= 1.0

    def test_get_analysis_summary(self, mock_technical_indicators):
        """Test getting comprehensive analysis summary"""
        # Mock TechnicalIndicators instance
        mock_instance = Mock()
        mock_instance.calculate_rsi.return_value = 65.0
        mock_instance.calculate_macd.return_value = (0.5, 0.45, 0.05)  # macd, signal, histogram
        mock_instance.calculate_bollinger_bands.return_value = (50000, 51000, 49000)  # sma, upper, lower
        mock_technical_indicators.return_value = mock_instance
        
        analyzer = MultiTimeframeAnalyzer()
        
        summary = analyzer.get_analysis_summary()

        assert "convergence" in summary
        assert "timeframe_analyses" in summary
        assert "data_points" in summary
        assert len(summary["data_points"]) == 3


class TestTrendConvergenceCalculator:
    """Test cases for TrendConvergenceCalculator"""

    def test_initialization(self):
        """Test calculator initialization"""
        calculator = TrendConvergenceCalculator()
        assert calculator is not None
        assert hasattr(calculator, 'weights')
        assert hasattr(calculator, 'recommendation_thresholds')

    def test_calculate_convergence_empty_data(self):
        """Test convergence calculation with empty data"""
        calculator = TrendConvergenceCalculator()
        result = calculator.calculate_convergence({})

        assert isinstance(result, TrendConvergenceResult)
        assert result.overall_score == 0.0
        assert result.recommendation == "insufficient_data"

    def test_calculate_convergence_single_timeframe(self):
        """Test convergence calculation with single timeframe"""
        calculator = TrendConvergenceCalculator()
        
        trend_analysis = TrendAnalysis(
            direction=TrendDirection.BULLISH,
            strength=80.0,
            momentum=20.0,
            rsi=65.0,
            macd_signal="bullish",
            bollinger_position="middle"
        )

        analyses = {Timeframe.M1: trend_analysis}
        result = calculator.calculate_convergence(analyses)

        assert isinstance(result, TrendConvergenceResult)
        assert result.overall_score >= 0.0
        assert result.overall_score <= 100.0

    def test_calculate_convergence_multiple_timeframes(self):
        """Test convergence calculation with multiple timeframes"""
        calculator = TrendConvergenceCalculator()
        
        # Create aligned bullish analyses
        analyses = {}
        for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
            analysis = TrendAnalysis(
                direction=TrendDirection.BULLISH,
                strength=75.0,
                momentum=15.0,
                rsi=62.0,
                macd_signal="bullish",
                bollinger_position="middle"
            )
            analyses[tf] = analysis

        result = calculator.calculate_convergence(analyses)

        assert isinstance(result, TrendConvergenceResult)
        assert result.overall_score > 0.0
        assert result.recommendation in ["strong_convergence", "moderate_convergence", "weak_convergence", "divergence"]
        assert result.metrics.alignment_score > 0.0
        assert result.metrics.strength_consistency > 0.0

    def test_calculate_convergence_divergence(self):
        """Test convergence calculation with divergent trends"""
        calculator = TrendConvergenceCalculator()
        
        # Create divergent analyses
        analyses = {
            Timeframe.M1: TrendAnalysis(
                direction=TrendDirection.BULLISH,
                strength=80.0,
                momentum=20.0,
                rsi=65.0,
                macd_signal="bullish",
                bollinger_position="middle"
            ),
            Timeframe.M5: TrendAnalysis(
                direction=TrendDirection.BEARISH,
                strength=70.0,
                momentum=-15.0,
                rsi=35.0,
                macd_signal="bearish",
                bollinger_position="middle"
            ),
            Timeframe.M15: TrendAnalysis(
                direction=TrendDirection.NEUTRAL,
                strength=50.0,
                momentum=0.0,
                rsi=50.0,
                macd_signal="neutral",
                bollinger_position="middle"
            )
        }

        result = calculator.calculate_convergence(analyses)

        assert isinstance(result, TrendConvergenceResult)
        assert result.metrics.divergence_penalty > 0.0  # Should have divergence penalty
        assert result.overall_score < 80.0  # Should be reduced due to divergence

    def test_get_convergence_report(self):
        """Test getting convergence report"""
        calculator = TrendConvergenceCalculator()
        
        analyses = {
            Timeframe.M1: TrendAnalysis(
                direction=TrendDirection.BULLISH,
                strength=80.0,
                momentum=20.0,
                rsi=65.0,
                macd_signal="bullish",
                bollinger_position="middle"
            )
        }

        report = calculator.get_convergence_report(analyses)

        assert isinstance(report, dict)
        assert "overall_score" in report
        assert "recommendation" in report
        assert "metrics" in report
        assert "trend_strength" in report


class TestSignalGuidanceSystemPhase2:
    """Test cases for Phase 2 SignalGuidanceSystem enhancements"""

    def test_phase_2_initialization(self):
        """Test Phase 2 component initialization"""
        guidance_system = SignalGuidanceSystem()
        
        assert hasattr(guidance_system, 'multi_timeframe_analyzer')
        assert hasattr(guidance_system, 'convergence_calculator')
        assert isinstance(guidance_system.multi_timeframe_analyzer, MultiTimeframeAnalyzer)
        assert isinstance(guidance_system.convergence_calculator, TrendConvergenceCalculator)

    def test_get_multi_timeframe_analysis(self):
        """Test getting multi-timeframe analysis"""
        guidance_system = SignalGuidanceSystem()
        
        analysis = guidance_system.get_multi_timeframe_analysis()

        assert isinstance(analysis, dict)
        assert "phase" in analysis
        assert "convergence" in analysis
        assert "timeframe_analyses" in analysis
        assert analysis["phase"] == "Phase 2 - Multi-timeframe Analysis"

    def test_get_phase_2_status(self):
        """Test getting Phase 2 status"""
        guidance_system = SignalGuidanceSystem()
        
        status = guidance_system.get_phase_2_status()

        assert isinstance(status, dict)
        assert "phase" in status
        assert "status" in status
        assert "components" in status
        assert "metrics" in status
        assert status["phase"] == "Phase 2 - Multi-timeframe Trend Detection"
        assert status["status"] in ["active", "error"]

    @patch('ztb.trading.signal.signal_guidance_system.SignalQualityScorer')
    def test_apply_guidance_with_phase_2(self, mock_scorer):
        """Test apply_guidance with Phase 2 enhancements"""
        guidance_system = SignalGuidanceSystem()
        
        # Mock quality scorer
        mock_instance = Mock()
        mock_instance.calculate_signal_quality.return_value = (1, 85.0)  # BUY action, 85 score
        mock_scorer.return_value = mock_instance

        # Create test market data
        row = pd.Series({
            'close': 50000.0,
            'volume': 100.0,
            'open': 49900.0,
            'high': 50100.0,
            'low': 49800.0
        })

        portfolio = {
            'btc_balance': 0.1,
            'jpy_balance': 500000.0,
            'portfolio_value': 550000.0,
            'current_price': 50000.0
        }

        # Add some historical data for multi-timeframe analysis
        for i in range(30):
            guidance_system.update_market_context(row, portfolio)

        # Apply guidance
        action = guidance_system.apply_guidance(0.8, row, portfolio)

        # Should return valid action
        assert action in [-1, 0, 1]

        # Verify Phase 2 components were used
        assert len(guidance_system.multi_timeframe_analyzer.timeframes[Timeframe.M1].prices) > 0


if __name__ == "__main__":
    pytest.main([__file__])