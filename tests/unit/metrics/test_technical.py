import numpy as np
import pandas as pd

from ztb.metrics.technical import (
    calculate_volatility,
    calculate_volatility_from_returns,
)
from ztb.trading.constants import TRADING_DAYS_PER_YEAR


def test_calculate_volatility_annualization():
    # Create a series of prices
    prices = [100.0]
    for i in range(100):
        if i % 2 == 0:
            prices.append(prices[-1] * 1.01)
        else:
            prices.append(prices[-1] / 1.01)

    # Calculate without annualization
    vol_raw = calculate_volatility(prices, window=20, annualize=False)

    # Calculate with annualization
    vol_ann = calculate_volatility(prices, window=20, annualize=True)

    # Check ratio
    assert vol_raw > 0
    assert np.isclose(vol_ann, vol_raw * np.sqrt(TRADING_DAYS_PER_YEAR))


def test_calculate_volatility_from_returns_annualization():
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 100)

    vol_raw = calculate_volatility_from_returns(returns, window=20, annualize=False)
    vol_ann = calculate_volatility_from_returns(returns, window=20, annualize=True)

    assert vol_raw > 0
    assert np.isclose(vol_ann, vol_raw * np.sqrt(TRADING_DAYS_PER_YEAR))


def test_calculate_atr():
    from ztb.metrics.technical import calculate_atr, calculate_rolling_atr

    # Create sample data
    high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 5)
    low = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18] * 5)
    close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5] * 5)

    atr = calculate_atr(high, low, close, period=14)
    assert isinstance(atr, float)
    assert atr > 0

    atr_series = calculate_rolling_atr(high, low, close, period=14)
    assert isinstance(atr_series, pd.Series)
    assert len(atr_series) == len(high)
    assert not pd.isna(atr_series.iloc[-1])


def test_calculate_adx():
    from ztb.metrics.technical import calculate_adx, calculate_rolling_adx

    # Create trending data
    high = np.linspace(10, 60, 100)
    low = np.linspace(9, 59, 100)
    close = np.linspace(9.5, 59.5, 100)

    adx = calculate_adx(high, low, close, period=14)
    assert isinstance(adx, float)
    assert 0 <= adx <= 100

    adx_series = calculate_rolling_adx(high, low, close, period=14)
    assert isinstance(adx_series, pd.Series)
    assert len(adx_series) == len(high)
    assert not pd.isna(adx_series.iloc[-1])


def test_calculate_stochastic():
    from ztb.metrics.technical import calculate_stochastic

    # Create oscillating data
    high = np.array([10, 12, 15, 14, 16] * 10)
    low = np.array([8, 9, 11, 12, 13] * 10)
    close = np.array([9, 11, 14, 13, 15] * 10)

    slowk, slowd = calculate_stochastic(
        high, low, close, fastk_period=5, slowk_period=3, slowd_period=3
    )

    assert isinstance(slowk, pd.Series)
    assert isinstance(slowd, pd.Series)
    assert len(slowk) == len(high)
    assert len(slowd) == len(high)

    # Check values are within 0-100
    valid_k = slowk.dropna()
    valid_d = slowd.dropna()

    assert not valid_k.empty
    assert not valid_d.empty

    assert (valid_k >= 0).all() and (valid_k <= 100).all()
    assert (valid_d >= 0).all() and (valid_d <= 100).all()


def test_calculate_stochastic_fast():
    from ztb.metrics.technical import calculate_stochastic_fast

    # Create oscillating data
    high = np.array([10, 12, 15, 14, 16] * 10)
    low = np.array([8, 9, 11, 12, 13] * 10)
    close = np.array([9, 11, 14, 13, 15] * 10)

    fastk, fastd = calculate_stochastic_fast(
        high, low, close, fastk_period=5, fastd_period=3
    )

    assert isinstance(fastk, pd.Series)
    assert isinstance(fastd, pd.Series)
    assert len(fastk) == len(high)
    assert len(fastd) == len(high)

    # Check values are within 0-100
    valid_k = fastk.dropna()
    valid_d = fastd.dropna()

    assert not valid_k.empty
    assert not valid_d.empty

    assert (valid_k >= 0).all() and (valid_k <= 100).all()
    assert (valid_d >= 0).all() and (valid_d <= 100).all()


def test_calculate_williams_r():
    from ztb.metrics.technical import calculate_williams_r

    # Create oscillating data
    high = np.array([10, 12, 15, 14, 16] * 10)
    low = np.array([8, 9, 11, 12, 13] * 10)
    close = np.array([9, 11, 14, 13, 15] * 10)

    willr = calculate_williams_r(high, low, close, period=5)

    assert isinstance(willr, pd.Series)
    assert len(willr) == len(high)

    # Check values are within -100 to 0
    valid_willr = willr.dropna()

    assert not valid_willr.empty
    assert (valid_willr >= -100).all() and (valid_willr <= 0).all()
