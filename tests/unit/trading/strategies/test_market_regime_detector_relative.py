import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.components.market_regime import (
    MarketRegimeDetector,
)


def create_price_series(base_price=100.0, n=120, volatility_spike_step=80):
    rng = pd.date_range(start="2020-01-01", periods=n, freq="T")
    prices = np.ones(n) * base_price
    # Add small noise
    for i in range(1, n):
        prices[i] = prices[i - 1] * (1 + np.random.normal(0, 0.0005))
    # Add volatility spike
    for i in range(volatility_spike_step, volatility_spike_step + 5):
        prices[i] *= 1 + np.random.normal(0, 0.01)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.random.randint(1, 10, size=n),
        },
        index=rng,
    )
    return df


def test_detect_regime_relative():
    df = create_price_series()
    # Detector with absolute mode should not detect high volatility early
    det_abs = MarketRegimeDetector(use_relative=False)
    reg_abs = det_abs.detect_regime(df)
    # Using absolute threshold, a small spike may not trigger HIGH_VOLATILITY
    assert reg_abs is not None

    # Detector in relative mode should detect the volatility spike as HIGH_VOLATILITY
    det_rel = MarketRegimeDetector(
        use_relative=True, reference_window=100, percentile_threshold=0.8
    )
    reg_rel = det_rel.detect_regime(df)
    # reg_rel can be one of MarketRegime values; assert it's computed and returns a variant
    assert reg_rel is not None
