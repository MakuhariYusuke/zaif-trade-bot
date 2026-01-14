#!/usr/bin/env python3
"""
End-to-end tests for Action Signal Guide.

This module contains end-to-end tests that simulate complete Action Signal Guide
workflows from market data input to final trading decisions.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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
)


class TestActionSignalGuideEndToEnd(unittest.TestCase):
    """End-to-end tests for complete Action Signal Guide workflows."""

    def setUp(self):
        """Set up test fixtures with realistic market data."""
        # Create realistic market data with trends and volatility
        dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
        np.random.seed(42)

        # Generate trending data with noise
        trend = np.linspace(100, 120, 100)  # Upward trend
        noise = np.random.normal(0, 2, 100)  # Add noise
        close_prices = trend + noise

        # Generate OHLC data
        self.market_data = pd.DataFrame({
            "timestamp": dates,
            "open": close_prices + np.random.normal(0, 1, 100),
            "high": close_prices + abs(np.random.normal(0, 1.5, 100)),
            "low": close_prices - abs(np.random.normal(0, 1.5, 100)),
            "close": close_prices,
            "volume": np.random.normal(10000, 2000, 100)
        })

        # Ensure high >= close >= low >= 0
        self.market_data["high"] = np.maximum(self.market_data["high"], self.market_data["close"])
        self.market_data["low"] = np.minimum(self.market_data["low"], self.market_data["close"])
        self.market_data["low"] = np.maximum(self.market_data["low"], 0)

        # Initialize all components
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

    def test_complete_trading_workflow_buy_scenario(self):
        """Test complete workflow resulting in BUY decision."""
        # Step 1: Data sanitization
        sanitized_data, sanitization_report = self.data_sanitizer.sanitize_market_data(self.market_data)
        self.assertGreater(len(sanitized_data), 50)
        self.assertIn("original_rows", sanitization_report)

        # Step 2: Market regime detection
        regime = self.regime_detector.detect_regime(sanitized_data)
        self.assertIn("regime", regime)
        self.assertIn("confidence", regime)

        # Step 3: Market condition analysis
        market_conditions = self.market_analyzer.analyze_market_conditions(sanitized_data)
        self.assertIn("trend", market_conditions)
        self.assertIn("volatility", market_conditions)
        self.assertIn("volume", market_conditions)

        # Step 4: Signal generation (mock realistic signals)
        signals = self._generate_realistic_signals(sanitized_data, regime["regime"])

        # Step 5: Signal validation
        validated_signals = []
        for signal in signals:
            validation_result = self.signal_validator.validate_signal(signal)
            if validation_result.is_valid:
                validated_signals.append(signal)

        self.assertGreater(len(validated_signals), 0)

        # Step 6: SAC integration
        sac_decisions = self._simulate_sac_decisions(sanitized_data, "BUY")
        sac_validated_signals = self.sac_validator.validate_with_sac(
            validated_signals, sac_decisions, sanitized_data
        )

        # Step 7: Regime-adaptive processing
        regime_processed_signals = self.regime_processor.process_signals_for_regime(
            sac_validated_signals, regime["regime"]
        )

        # Step 8: Final decision integration
        final_decision = self.decision_integrator.integrate_decisions(
            regime_processed_signals, sac_decisions, sanitized_data
        )

        self.assertIn("action", final_decision)
        self.assertIn("confidence", final_decision)
        self.assertGreater(final_decision["confidence"], 0.5)  # Should be confident BUY

        # Step 9: Performance tracking
        simulated_outcome = self._simulate_trade_outcome(final_decision, sanitized_data)
        self.performance_monitor.record_decision_outcome(
            final_decision, simulated_outcome, sanitized_data
        )

        # Verify decision was BUY
        self.assertEqual(final_decision["action"], "BUY")

    def test_complete_trading_workflow_sell_scenario(self):
        """Test complete workflow resulting in SELL decision."""
        # Create downtrending market data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
        trend = np.linspace(120, 100, 100)  # Downward trend
        noise = np.random.normal(0, 2, 100)
        close_prices = trend + noise

        bearish_data = pd.DataFrame({
            "timestamp": dates,
            "open": close_prices + np.random.normal(0, 1, 100),
            "high": close_prices + abs(np.random.normal(0, 1.5, 100)),
            "low": close_prices - abs(np.random.normal(0, 1.5, 100)),
            "close": close_prices,
            "volume": np.random.normal(10000, 2000, 100)
        })

        bearish_data["high"] = np.maximum(bearish_data["high"], bearish_data["close"])
        bearish_data["low"] = np.minimum(bearish_data["low"], bearish_data["close"])
        bearish_data["low"] = np.maximum(bearish_data["low"], 0)

        # Run complete workflow
        sanitized_data, _ = self.data_sanitizer.sanitize_market_data(bearish_data)
        regime = self.regime_detector.detect_regime(sanitized_data)
        market_conditions = self.market_analyzer.analyze_market_conditions(sanitized_data)

        signals = self._generate_realistic_signals(sanitized_data, regime["regime"])
        validated_signals = []
        for signal in signals:
            if self.signal_validator.validate_signal(signal).is_valid:
                validated_signals.append(signal)

        sac_decisions = self._simulate_sac_decisions(sanitized_data, "SELL")
        sac_validated_signals = self.sac_validator.validate_with_sac(
            validated_signals, sac_decisions, sanitized_data
        )

        regime_processed_signals = self.regime_processor.process_signals_for_regime(
            sac_validated_signals, regime["regime"]
        )

        final_decision = self.decision_integrator.integrate_decisions(
            regime_processed_signals, sac_decisions, sanitized_data
        )

        simulated_outcome = self._simulate_trade_outcome(final_decision, sanitized_data)
        self.performance_monitor.record_decision_outcome(
            final_decision, simulated_outcome, sanitized_data
        )

        # Verify decision was SELL
        self.assertEqual(final_decision["action"], "SELL")

    def test_workflow_with_market_regime_changes(self):
        """Test workflow adaptation to changing market regimes."""
        # Test multiple regime scenarios
        scenarios = [
            ("trending_bullish", "BUY"),
            ("trending_bearish", "SELL"),
            ("ranging", "HOLD"),
            ("high_volatility", "HOLD")
        ]

        for regime_type, expected_bias in scenarios:
            with self.subTest(regime=regime_type):
                # Generate data for specific regime
                test_data = self._generate_regime_specific_data(regime_type)

                # Run workflow
                sanitized_data, _ = self.data_sanitizer.sanitize_market_data(test_data)
                regime = self.regime_detector.detect_regime(sanitized_data)

                signals = self._generate_realistic_signals(sanitized_data, regime["regime"])
                validated_signals = [
                    s for s in signals
                    if self.signal_validator.validate_signal(s).is_valid
                ]

                sac_decisions = self._simulate_sac_decisions(sanitized_data, expected_bias)
                sac_validated_signals = self.sac_validator.validate_with_sac(
                    validated_signals, sac_decisions, sanitized_data
                )

                regime_processed_signals = self.regime_processor.process_signals_for_regime(
                    sac_validated_signals, regime["regime"]
                )

                final_decision = self.decision_integrator.integrate_decisions(
                    regime_processed_signals, sac_decisions, sanitized_data
                )

                # Verify workflow completed and decision was made
                self.assertIn("action", final_decision)
                self.assertIn("confidence", final_decision)
                self.assertIsInstance(final_decision["confidence"], (int, float))

    def test_performance_tracking_and_learning(self):
        """Test performance tracking and learning over multiple trades."""
        # Simulate multiple trading cycles
        num_cycles = 10
        performance_history = []

        for cycle in range(num_cycles):
            # Generate slightly different market data for each cycle
            cycle_data = self.market_data.copy()
            cycle_data["close"] += np.random.normal(0, 1, len(cycle_data))

            # Run complete workflow
            sanitized_data, _ = self.data_sanitizer.sanitize_market_data(cycle_data)
            regime = self.regime_detector.detect_regime(sanitized_data)

            signals = self._generate_realistic_signals(sanitized_data, regime["regime"])
            validated_signals = [
                s for s in signals
                if self.signal_validator.validate_signal(s).is_valid
            ]

            # Alternate between BUY and SELL decisions
            action = "BUY" if cycle % 2 == 0 else "SELL"
            sac_decisions = self._simulate_sac_decisions(sanitized_data, action)

            sac_validated_signals = self.sac_validator.validate_with_sac(
                validated_signals, sac_decisions, sanitized_data
            )

            regime_processed_signals = self.regime_processor.process_signals_for_regime(
                sac_validated_signals, regime["regime"]
            )

            final_decision = self.decision_integrator.integrate_decisions(
                regime_processed_signals, sac_decisions, sanitized_data
            )

            # Simulate realistic outcome based on market trend
            outcome = self._simulate_trade_outcome(final_decision, sanitized_data)

            # Record performance
            self.performance_monitor.record_decision_outcome(
                final_decision, outcome, sanitized_data
            )

            # Track regime performance
            self.regime_processor.update_regime_performance(regime["regime"], outcome)

            performance_history.append({
                "cycle": cycle,
                "decision": final_decision,
                "outcome": outcome,
                "regime": regime["regime"]
            })

        # Verify learning and adaptation
        metrics = self.performance_monitor.get_performance_metrics()
        self.assertIn("integrated_decision_accuracy", metrics)
        self.assertGreaterEqual(metrics["integrated_decision_accuracy"], 0.0)

        # Verify we have performance history
        self.assertEqual(len(performance_history), num_cycles)

        # Test regime-specific performance
        for regime_type in ["trending_bullish", "trending_bearish", "ranging"]:
            config = self.regime_processor.get_regime_config(regime_type)
            self.assertIsInstance(config, dict)

    def _generate_realistic_signals(self, market_data, regime):
        """Generate realistic signals based on market data and regime."""
        signals = []
        recent_data = market_data.tail(20)

        # Generate signals based on recent price action
        current_price = recent_data["close"].iloc[-1]

        # Fibonacci signal
        signals.append(MockActionSignal(
            action="BUY",
            confidence=0.75,
            pattern_type="fibonacci",
            price=current_price,
            stop_loss=current_price * 0.95,
            take_profit=current_price * 1.10,
            timestamp=pd.Timestamp.now()
        ))

        # Harmonic signal
        signals.append(MockActionSignal(
            action="SELL",
            confidence=0.70,
            pattern_type="harmonic",
            price=current_price,
            stop_loss=current_price * 1.05,
            take_profit=current_price * 0.90,
            timestamp=pd.Timestamp.now()
        ))

        return signals

    def _simulate_sac_decisions(self, market_data, bias_action):
        """Simulate SAC decisions with given bias."""
        recent_trend = market_data["close"].pct_change().tail(10).mean()

        if bias_action == "BUY":
            confidence = min(0.9, 0.5 + recent_trend * 10)
        elif bias_action == "SELL":
            confidence = min(0.9, 0.5 - recent_trend * 10)
        else:
            confidence = 0.5

        return {
            "action": bias_action,
            "confidence": max(0.1, confidence),
            "timestamp": pd.Timestamp.now()
        }

    def _simulate_trade_outcome(self, decision, market_data):
        """Simulate trade outcome based on decision and market data."""
        # Simple simulation: BUY profits if market goes up, SELL profits if market goes down
        future_returns = market_data["close"].pct_change().tail(20).mean()

        if decision["action"] == "BUY":
            outcome = future_returns * decision["confidence"]
        elif decision["action"] == "SELL":
            outcome = -future_returns * decision["confidence"]
        else:
            outcome = 0.0

        # Add some noise
        outcome += np.random.normal(0, 0.01)

        return outcome

    def _generate_regime_specific_data(self, regime_type):
        """Generate market data specific to a regime type."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="H")
        np.random.seed(42)

        if regime_type == "trending_bullish":
            trend = np.linspace(100, 120, 50)
            noise = np.random.normal(0, 1, 50)
        elif regime_type == "trending_bearish":
            trend = np.linspace(120, 100, 50)
            noise = np.random.normal(0, 1, 50)
        elif regime_type == "ranging":
            trend = 110 + np.sin(np.linspace(0, 4*np.pi, 50)) * 5
            noise = np.random.normal(0, 0.5, 50)
        else:  # high_volatility
            trend = 110 + np.random.normal(0, 10, 50)
            noise = np.random.normal(0, 2, 50)

        close_prices = trend + noise

        return pd.DataFrame({
            "timestamp": dates,
            "open": close_prices + np.random.normal(0, 0.5, 50),
            "high": close_prices + abs(np.random.normal(0, 1, 50)),
            "low": close_prices - abs(np.random.normal(0, 1, 50)),
            "close": close_prices,
            "volume": np.random.normal(10000, 1000, 50)
        })


if __name__ == "__main__":
    unittest.main()