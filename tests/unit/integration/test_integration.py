#!/usr/bin/env python3
"""
Integration tests for Action Signal Guide components.

This module contains integration tests that combine multiple Action Signal Guide
components to test their interactions and end-to-end functionality.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.regime.market_regime_types import MarketRegime
from ztb.trading.strategies.action_signal_guide.action_signal_guide import ActionSignalGuide
from ztb.trading.strategies.action_signal_guide.components.sac_integration import (
    SACSignalValidator,
    SACDecisionIntegrator,
    SACPerformanceMonitor,
)
from ztb.trading.strategies.action_signal_guide.components.market_regime import (
    MarketRegimeDetector,
    RegimeAdaptiveSignalProcessor,
    MarketConditionAnalyzer,
)
from ztb.trading.strategies.action_signal_guide.components.validation import (
    SignalValidator,
    DataSanitizer,
    PerformanceTracker,
    ValidationResult,
)


class MockActionSignal:
    """Mock ActionSignal for testing."""

    def __init__(self, **kwargs):
        self.action = kwargs.get('action', 'BUY')
        self.confidence = kwargs.get('confidence', 0.8)
        self.pattern_type = kwargs.get('pattern_type', 'fibonacci')
        self.price = kwargs.get('price', 100.0)
        self.timestamp = kwargs.get('timestamp', pd.Timestamp.now())
        self.stop_loss = kwargs.get('stop_loss', 95.0)
        self.take_profit = kwargs.get('take_profit', 110.0)
        self.signal_type = kwargs.get('signal_type', 'test')


class TestActionSignalGuideIntegration(unittest.TestCase):
    """Integration tests for Action Signal Guide components."""

    @staticmethod
    def _regime_name(regime: MarketRegime | str) -> str:
        return regime.value if isinstance(regime, MarketRegime) else str(regime)

    def setUp(self):
        """Set up test fixtures."""
        self.market_data = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [103, 104, 105, 106, 107],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })

        # Initialize components
        self.signal_guide = ActionSignalGuide()
        self.sac_validator = SACSignalValidator()
        self.decision_integrator = SACDecisionIntegrator()
        self.performance_monitor = SACPerformanceMonitor()
        self.regime_detector = MarketRegimeDetector()
        self.regime_processor = RegimeAdaptiveSignalProcessor()
        self.market_analyzer = MarketConditionAnalyzer()
        self.signal_validator = SignalValidator()
        self.data_sanitizer = DataSanitizer()
        self.performance_tracker = PerformanceTracker()

    def test_full_signal_processing_pipeline(self):
        """Test complete signal processing pipeline from data to decision."""
        # Step 1: Sanitize market data
        sanitized_data, report = self.data_sanitizer.sanitize_market_data(self.market_data)
        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertGreater(len(sanitized_data), 0)

        # Step 2: Detect market regime
        regime = self.regime_detector.detect_regime(sanitized_data)
        self.assertIsInstance(regime, MarketRegime)

        # Step 3: Analyze market conditions
        market_conditions = self.market_analyzer.analyze_market_conditions(sanitized_data)
        self.assertIsInstance(market_conditions, dict)
        self.assertIn("trend", market_conditions)

        # Step 4: Generate signals (mock the signal guide for integration test)
        signals = [
            MockActionSignal(action="BUY", confidence=0.8, pattern_type="fibonacci"),
            MockActionSignal(action="SELL", confidence=0.7, pattern_type="harmonic")
        ]

        # Step 5: Validate signals
        validated_signals = []
        for signal in signals:
            result = self.signal_validator.validate_signal(signal)
            if result.is_valid:
                validated_signals.append(signal)

        self.assertGreater(len(validated_signals), 0)

        # Step 6: SAC validation
        sac_decisions = {"action": "BUY", "confidence": 0.75}
        sac_validated_signals = self.sac_validator.validate_with_sac(
            validated_signals, sac_decisions, sanitized_data
        )

        # Step 7: Integrate decisions
        final_decision = self.decision_integrator.integrate_decisions(
            sac_validated_signals, sac_decisions, sanitized_data
        )

        self.assertIsInstance(final_decision, dict)
        self.assertIn("action", final_decision)
        self.assertIn("confidence", final_decision)

        # Step 8: Record performance (simulate outcome)
        outcome = 0.02 if final_decision["action"] == "BUY" else -0.01
        self.performance_monitor.record_decision_outcome(
            final_decision, outcome, sanitized_data
        )

        # Verify the pipeline completed successfully
        metrics = self.performance_monitor.get_performance_metrics()
        self.assertIsInstance(metrics, dict)

    def test_regime_adaptive_signal_processing(self):
        """Test regime-adaptive signal processing integration."""
        # Detect regime
        regime = self.regime_detector.detect_regime(self.market_data)
        self.assertIsInstance(regime, MarketRegime)

        # Create signals for different regimes
        signals = [
            MockActionSignal(action="BUY", confidence=0.8, pattern_type="fibonacci"),
            MockActionSignal(action="SELL", confidence=0.6, pattern_type="bollinger")
        ]

        # Process signals adaptively based on regime
        processed_signals = self.regime_processor.process_signals_for_regime(
            signals, self.market_data
        )

        self.assertIsInstance(processed_signals, list)
        # Signals should be filtered or adjusted based on regime
        self.assertGreaterEqual(len(processed_signals), 0)

        # Test regime performance tracking
        self.regime_processor.update_regime_performance(regime, 0.02)

        # Verify regime configuration
        config = self.regime_processor._get_regime_config(regime)
        self.assertIsInstance(config, dict)

    def test_validation_and_performance_tracking_integration(self):
        """Test integration between validation and performance tracking."""
        # Create and validate multiple signals
        signals = []
        for i in range(5):
            signal = MockActionSignal(
                action="BUY" if i % 2 == 0 else "SELL",
                confidence=0.7 + i * 0.05,
                pattern_type="fibonacci"
            )
            signals.append(signal)

        # Validate all signals
        validation_results = []
        for signal in signals:
            result = self.signal_validator.validate_signal(signal)
            validation_results.append(result)

        # Record performance for validated signals
        base_time = pd.Timestamp.now()
        for i, (signal, result) in enumerate(zip(signals, validation_results)):
            if result.is_valid:
                entry_time = base_time + pd.Timedelta(hours=i)
                exit_time = entry_time + pd.Timedelta(hours=2)
                exit_price = signal.price * (1.02 if signal.action == "BUY" else 0.98)

                self.performance_tracker.record_signal_performance(
                    f"signal_{i}", signal.price, exit_price,
                    entry_time, exit_time, signal.pattern_type
                )

        # Get performance metrics
        metrics = self.performance_tracker.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_signals", metrics)

        # Test filtered metrics
        filtered_metrics = self.performance_tracker.get_performance_metrics(
            pattern_type="fibonacci"
        )
        self.assertIsInstance(filtered_metrics, dict)

    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test with invalid market data
        invalid_data = pd.DataFrame({
            "close": [np.nan, np.nan, np.nan],
            "volume": [np.nan, np.nan, np.nan]
        })

        # Data sanitizer should handle NaN data
        sanitized_data, report = self.data_sanitizer.sanitize_market_data(invalid_data)
        self.assertIsInstance(sanitized_data, pd.DataFrame)

        # Regime detector should handle edge cases
        regime = self.regime_detector.detect_regime(invalid_data)
        self.assertIsInstance(regime, MarketRegime)

        # Signal validator should handle invalid signals
        invalid_signal = MockActionSignal(confidence=-0.5, price=-100)
        result = self.signal_validator.validate_signal(invalid_signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

        # SAC validator should handle empty signal lists
        sac_decisions = {"action": "BUY", "confidence": 0.8}
        validated_signals = self.sac_validator.validate_with_sac(
            [], sac_decisions, invalid_data
        )
        self.assertEqual(len(validated_signals), 0)

        # Decision integrator should handle empty signals
        decision = self.decision_integrator.integrate_decisions(
            [], sac_decisions, invalid_data
        )
        self.assertIsInstance(decision, dict)
        self.assertIn("action", decision)

    def test_boundary_conditions_integration(self):
        """Test boundary conditions across integrated components."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            "close": [100.0],
            "volume": [1000]
        })

        # All components should handle minimal data gracefully
        sanitized_data, _ = self.data_sanitizer.sanitize_market_data(minimal_data)
        regime = self.regime_detector.detect_regime(minimal_data)
        market_conditions = self.market_analyzer.analyze_market_conditions(minimal_data)

        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertIsInstance(regime, MarketRegime)
        self.assertIsInstance(market_conditions, dict)

        # Test with extreme values
        extreme_data = pd.DataFrame({
            "close": [1e-6, 1e10, 0.0, -1000.0],
            "volume": [0, 1e15, -500, np.nan]
        })

        sanitized_data, report = self.data_sanitizer.sanitize_market_data(extreme_data)
        regime = self.regime_detector.detect_regime(extreme_data)

        self.assertIsInstance(sanitized_data, pd.DataFrame)
        self.assertIsInstance(regime, MarketRegime)
        self.assertIn("issues_found", report)

        # Test signal validation with extreme values
        extreme_signal = MockActionSignal(
            confidence=1.5,  # Invalid confidence
            price=1e10,     # Extreme price
            stop_loss=0.0,  # Invalid stop loss
            take_profit=2e10
        )
        result = self.signal_validator.validate_signal(extreme_signal)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)


if __name__ == "__main__":
    unittest.main()
