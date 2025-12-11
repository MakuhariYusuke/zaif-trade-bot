# Shim module to make backtest script importable by tests

from backtest.signal_guidance_backtest import (
    SignalGuidanceBacktestEnv,
    generate_synthetic_data,
)

__all__ = ["SignalGuidanceBacktestEnv", "generate_synthetic_data"]
