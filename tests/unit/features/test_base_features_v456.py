import numpy as np
import pandas as pd

from ztb.features.base_features_v456 import calculate_base_features


def test_calculate_base_features_basic_shape() -> None:
    df = pd.DataFrame(
        {
            "close": np.linspace(100.0, 120.0, 60),
            "high": np.linspace(101.0, 121.0, 60),
            "low": np.linspace(99.0, 119.0, 60),
            "volume": np.linspace(1000.0, 2000.0, 60),
        }
    )

    result = calculate_base_features(df)

    expected_columns = {
        "ema_5",
        "ema_20",
        "rsi_14",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "bb_upper_20",
        "macd_line",
    }
    assert expected_columns.issubset(result.columns)
    assert len(result) == len(df)
    finite_columns = [
        "ema_5",
        "ema_20",
        "rsi_14",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "macd_line",
    ]
    assert np.isfinite(result[finite_columns].to_numpy(dtype=float)).all()


def test_calculate_base_features_ema_matches_recursive_formula() -> None:
    base = np.linspace(1.0, 60.0, 60)
    df = pd.DataFrame(
        {
            "close": base,
            "high": base,
            "low": base,
            "volume": np.full(60, 10.0),
        }
    )

    result = calculate_base_features(df)

    alpha = 2.0 / 6.0
    expected = [1.0]
    for value in [2.0, 3.0, 4.0]:
        expected.append(alpha * value + (1.0 - alpha) * expected[-1])

    np.testing.assert_allclose(result["ema_5"].iloc[:4].to_numpy(), np.array(expected))


def test_calculate_base_features_adx_outputs_are_non_negative() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 107.0, 108.0] * 8,
            "high": [101.0, 103.0, 102.0, 104.0, 105.0, 106.0, 108.0, 109.0] * 8,
            "low": [99.0, 101.0, 100.0, 102.0, 103.0, 104.0, 106.0, 107.0] * 8,
            "volume": [1000.0, 1100.0, 1050.0, 1150.0, 1200.0, 1180.0, 1220.0, 1250.0]
            * 8,
        }
    )

    result = calculate_base_features(df)

    assert (result["adx_14"] >= 0).all()
    assert (result["plus_di_14"] >= 0).all()
    assert (result["minus_di_14"] >= 0).all()
