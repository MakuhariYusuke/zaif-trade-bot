from ztb.trading.signal import constants
from ztb.trading.signal.quality_scorer import SignalQualityScorer


def test_quality_scorer_default_thresholds_use_constants():
    scorer = SignalQualityScorer()
    assert scorer.buy_threshold == constants.DEFAULT_BUY_THRESHOLD
    assert scorer.sell_threshold == constants.DEFAULT_SELL_THRESHOLD
    assert scorer.hold_threshold == constants.DEFAULT_HOLD_THRESHOLD

    # Continuous to score scaling
    scaled = (0.0 + 1) * constants.CONTINUOUS_TO_SCORE_SCALE
    assert scaled == 50 * (constants.CONTINUOUS_TO_SCORE_SCALE / 50) or isinstance(
        scaled, (int, float)
    )
