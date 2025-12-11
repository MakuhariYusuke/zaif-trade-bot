import pandas as pd

from ztb.trading.signal.common.utilities import (
    calculate_volatility,
    calculate_volatility_from_prices,
    confidence_to_score_thresholds,
    get_dynamic_thresholds,
)


def test_calculate_volatility_from_prices_simple():
    prices = pd.Series([100, 101, 102, 103, 104])
    vol = calculate_volatility_from_prices(
        prices, window=3, returns_method="pct_change", annualize=False
    )
    assert isinstance(vol, float)
    assert vol >= 0.0


def test_calculate_volatility_from_prices_short_series():
    prices = pd.Series([100.0])
    vol = calculate_volatility_from_prices(prices, window=3)
    assert vol == 0.0


def test_calculate_volatility_on_returns_series():
    returns = pd.Series([0.01, -0.005, 0.002, 0.003])
    vol = calculate_volatility(returns, window=3, method="std")
    # Should return a float value
    assert isinstance(vol, float)


def test_get_dynamic_thresholds_static_mapping():
    # Without threshold_manager, fallback to base mapping
    buy, sell = get_dynamic_thresholds(
        confidence=0.7, threshold_manager=None, market_data=None, min_gap=10
    )
    buy_expected, sell_expected = confidence_to_score_thresholds(0.7, min_gap=10)
    assert buy == buy_expected and sell == sell_expected
