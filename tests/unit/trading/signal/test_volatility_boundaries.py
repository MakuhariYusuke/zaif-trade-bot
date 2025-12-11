import pandas as pd

from ztb.trading.signal.common.metrics import calculate_volatility_metrics


def test_calculate_volatility_small_data():
    # Very small dataset (less than window) should still compute volatility
    prices = [100, 110, 120]
    data = pd.DataFrame({"close": prices})
    m = calculate_volatility_metrics(data, window=20)
    assert "volatility" in m
    assert isinstance(m["volatility"], float)


def test_calculate_volatility_high_volatility():
    # Synthetic high volatility dataset should produce volatility > 0.05
    prices = [100, 110, 95, 120, 80, 105, 115, 90, 125, 85]
    data = pd.DataFrame({"close": prices})
    m = calculate_volatility_metrics(data, window=5)
    assert m["volatility"] > 0.05
