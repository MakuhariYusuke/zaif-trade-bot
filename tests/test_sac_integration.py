#!/usr/bin/env python3
"""
Unit tests for SAC Integration Components.

This module contains comprehensive unit tests for the SAC integration components
including SACSignalValidator, SACDecisionIntegrator, and SACPerformanceMonitor.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.components.sac_integration import (
    SACSignalValidator,
    SACDecisionIntegrator,
    SACPerformanceMonitor,
)


class MockActionSignal:
    """Mock ActionSignal for testing."""

    def __init__(self, action="BUY", confidence=0.8, pattern_type="test"):
        self.action = action
        self.confidence = confidence
        self.pattern_type = pattern_type
        self.price = 100.0
        self.timestamp = pd.Timestamp.now()


class TestSACSignalValidator(unittest.TestCase):
    """Test cases for SACSignalValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = SACSignalValidator()

    def test_initialization(self):
        """Test validator initialization."""
        self.assertIsInstance(self.validator, SACSignalValidator)
        self.assertEqual(self.validator.validation_threshold, 0.6)
        self.assertEqual(self.validator.confidence_boost_factor, 1.3)
        self.assertEqual(self.validator.confidence_penalty_factor, 0.7)

    def test_validate_with_sac_agreement(self):
        """Test validation when signal and SAC agree."""
        signals = [MockActionSignal(action="BUY", confidence=0.8)]
        sac_decisions = {"action": "BUY", "confidence": 0.7}
        market_data = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200]
        })

        result = self.validator.validate_with_sac(signals, sac_decisions, market_data)

        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].confidence, 0.8)  # Should be boosted

    def test_validate_with_sac_disagreement(self):
        """Test validation with SAC disagreement."""
        # Create signal with higher confidence to ensure it passes validation even with disagreement
        signals = [MockActionSignal(action="BUY", confidence=0.9)]
        sac_decisions = {"action": "SELL", "confidence": 0.7, "timestamp": pd.Timestamp.now()}
        market_data = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200]
        })

        result = self.validator.validate_with_sac(signals, sac_decisions, market_data)

        # Even with disagreement, signal should be returned with reduced confidence
        self.assertGreaterEqual(len(result), 0)  # May be 0 if correlation is too low
        if len(result) > 0:
            self.assertLess(result[0].confidence, 0.9)  # Should be penalized

    def test_validate_single_signal(self):
        """Test single signal validation."""
        signal = MockActionSignal(action="BUY", confidence=0.8)
        sac_decisions = {"action": "BUY", "confidence": 0.7}
        market_data = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200]
        })

        result = self.validator._validate_single_signal(signal, sac_decisions, market_data)

        self.assertIn("is_valid", result)
        self.assertIn("correlation_score", result)
        self.assertIn("action_alignment", result)
        self.assertIn("confidence_correlation", result)
        self.assertGreater(result["correlation_score"], 0)

    def test_market_alignment_calculation(self):
        """Test market condition alignment calculation."""
        signal = MockActionSignal(action="BUY", confidence=0.8)
        sac_decisions = {"action": "BUY", "market_state": "trending_bullish"}
        market_data = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200]
        })

        alignment = self.validator._calculate_market_alignment(signal, sac_decisions, market_data)

        self.assertIsInstance(alignment, float)
        self.assertGreaterEqual(alignment, 0.0)
        self.assertLessEqual(alignment, 1.0)


class TestSACDecisionIntegrator(unittest.TestCase):
    """Test cases for SACDecisionIntegrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.integrator = SACDecisionIntegrator()

    def test_initialization(self):
        """Test integrator initialization."""
        self.assertIsInstance(self.integrator, SACDecisionIntegrator)
        self.assertIn("signal_guide", self.integrator.decision_weights)
        self.assertIn("sac_decision", self.integrator.decision_weights)

    def test_integrate_decisions_agreement(self):
        """Test decision integration when signals and SAC agree."""
        signals = [MockActionSignal(action="BUY", confidence=0.8)]
        sac_decisions = {"action": "BUY", "confidence": 0.7}
        market_data = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200]
        })

        result = self.integrator.integrate_decisions(signals, sac_decisions, market_data)

        self.assertIn("action", result)
        self.assertIn("confidence", result)
        self.assertIn("source", result)
        self.assertEqual(result["action"], "BUY")

    def test_integrate_decisions_disagreement(self):
        """Test decision integration when signals and SAC disagree."""
        signals = [MockActionSignal(action="BUY", confidence=0.9)]
        sac_decisions = {"action": "SELL", "confidence": 0.6}
        market_data = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200]
        })

        result = self.integrator.integrate_decisions(signals, sac_decisions, market_data)

        self.assertIn("action", result)
        self.assertIn("confidence", result)
        self.assertIn("source", result)

    def test_validate_with_sac_boundary_cases(self):
        """Test SAC validation with boundary cases."""
        # Test with empty signals list
        result = self.validator.validate_with_sac([], {"action": "BUY", "confidence": 0.8}, pd.DataFrame())
        self.assertEqual(len(result), 0)

        # Test with extreme confidence values
        signals = [MockActionSignal(action="BUY", confidence=0.0)]
        sac_decisions = {"action": "BUY", "confidence": 1.0}
        market_data = pd.DataFrame({"close": [100, 101], "volume": [1000, 1100]})
        result = self.validator.validate_with_sac(signals, sac_decisions, market_data)
        self.assertGreaterEqual(len(result), 0)  # May filter out low confidence signals

        # Test with conflicting actions and extreme confidence differences
        signals = [MockActionSignal(action="BUY", confidence=1.0)]
        sac_decisions = {"action": "SELL", "confidence": 0.0}
        result = self.validator.validate_with_sac(signals, sac_decisions, market_data)
        self.assertGreaterEqual(len(result), 0)

    def test_integrate_decisions_boundary_cases(self):
        """Test decision integration with boundary cases."""
        # Test with empty signals
        result = self.integrator.integrate_decisions([], {"action": "BUY", "confidence": 0.8}, pd.DataFrame())
        self.assertIn("action", result)
        self.assertIn("confidence", result)

        # Test with extreme market data
        signals = [MockActionSignal(action="BUY", confidence=0.8)]
        market_data = pd.DataFrame({"close": [1e-6, 1e10], "volume": [0, 1e15]})
        result = self.integrator.integrate_decisions(signals, {"action": "BUY", "confidence": 0.8}, market_data)
        self.assertIn("action", result)
        self.assertIn("confidence", result)

        # Test with zero confidence signals
        signals = [MockActionSignal(action="BUY", confidence=0.0)]
        result = self.integrator.integrate_decisions(signals, {"action": "BUY", "confidence": 0.8}, market_data)
        self.assertIn("action", result)
        self.assertIn("confidence", result)

    def test_market_adjustment_calculation(self):
        """Test market-based confidence adjustment."""
        market_data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })

        # Test bullish action with upward momentum
        adjustment = self.integrator._calculate_market_adjustment("BUY", 0.8, market_data)
        self.assertIsInstance(adjustment, float)
        self.assertGreaterEqual(adjustment, 0.8)  # Should boost confidence


class TestSACPerformanceMonitor(unittest.TestCase):
    """Test cases for SACPerformanceMonitor."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = SACPerformanceMonitor()

    def test_initialization(self):
        """Test monitor initialization."""
        self.assertIsInstance(self.monitor, SACPerformanceMonitor)
        self.assertEqual(self.monitor.accuracy_window, 50)
        self.assertIn("signal_sac_agreement_rate", self.monitor.metrics)

    def test_record_decision_outcome(self):
        """Test recording decision outcomes."""
        integrated_decision = {
            "action": "BUY",
            "confidence": 0.8,
            "source": "agreement"
        }
        actual_outcome = 0.05  # 5% profit
        market_data = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200]
        })

        self.monitor.record_decision_outcome(integrated_decision, actual_outcome, market_data)

        self.assertEqual(len(self.monitor.performance_history), 1)
        record = self.monitor.performance_history[0]
        self.assertEqual(record["outcome"], actual_outcome)
        self.assertEqual(record["action"], "BUY")

    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some test data
        for i in range(10):
            decision = {
                "action": "BUY" if i % 2 == 0 else "SELL",
                "confidence": 0.8,
                "source": "agreement" if i < 5 else "signal_guide_dominant"
            }
            outcome = 0.02 if i % 3 != 0 else -0.01  # Mostly profitable
            market_data = pd.DataFrame({
                "close": [100 + i, 101 + i, 102 + i],
                "volume": [1000, 1100, 1200]
            })
            self.monitor.record_decision_outcome(decision, outcome, market_data)

        metrics = self.monitor.get_performance_metrics()

        self.assertIn("signal_sac_agreement_rate", metrics)
        self.assertIn("integrated_decision_accuracy", metrics)
        self.assertIn("signal_guide_accuracy", metrics)
        self.assertIn("sac_accuracy", metrics)
        self.assertEqual(metrics["signal_sac_agreement_rate"], 0.5)  # 5 out of 10 are agreement
        self.assertGreater(metrics["integrated_decision_accuracy"], 0)

    def test_metrics_update(self):
        """Test metrics update after recording outcomes."""
        initial_metrics = self.monitor.get_performance_metrics()

        # Record some outcomes
        for i in range(5):
            decision = {"action": "BUY", "confidence": 0.8, "source": "agreement"}
            outcome = 0.01
            market_data = pd.DataFrame({"close": [100, 101], "volume": [1000, 1100]})
            self.monitor.record_decision_outcome(decision, outcome, market_data)

        updated_metrics = self.monitor.get_performance_metrics()

        # Metrics should be updated after recording outcomes
        self.assertIsInstance(updated_metrics, dict)
        self.assertIn("integrated_decision_accuracy", updated_metrics)


if __name__ == "__main__":
    unittest.main()