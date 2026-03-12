from ztb.trading.signal import constants


def test_constants_present_and_typed():
    assert hasattr(constants, "HIGH_SCORE_IS_BUY")
    assert isinstance(constants.HIGH_SCORE_IS_BUY, bool)

    assert hasattr(constants, "BACKTEST_HIGH_SCORE_IS_SELL")
    assert isinstance(constants.BACKTEST_HIGH_SCORE_IS_SELL, bool)

    assert hasattr(constants, "DEFAULT_FALLBACK_THRESHOLD")
    assert isinstance(constants.DEFAULT_FALLBACK_THRESHOLD, (int, float))

    assert hasattr(constants, "DEFAULT_BUY_THRESHOLD")
    assert hasattr(constants, "DEFAULT_SELL_THRESHOLD")
    assert hasattr(constants, "DEFAULT_HOLD_THRESHOLD")
    assert hasattr(constants, "CONTINUOUS_TO_SCORE_SCALE")
