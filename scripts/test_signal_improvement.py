"""Compatibility shim for legacy `scripts.test_signal_improvement` import path."""

from scripts.testing.smoke.test_signal_improvement import SignalGuidanceBacktestValidator

__all__ = ["SignalGuidanceBacktestValidator"]
