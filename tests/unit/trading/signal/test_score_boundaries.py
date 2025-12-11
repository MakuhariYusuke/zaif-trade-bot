from ztb.trading.signal.common.utilities import (
    calculate_confidence_score,
    score_to_discrete_action,
)


def test_score_to_discrete_action_thresholds():
    # Default thresholds buy=75, sell=25
    assert (
        score_to_discrete_action(
            75.0, buy_threshold=75, sell_threshold=25, high_score_is_buy=True
        )
        == 1
    )
    assert (
        score_to_discrete_action(
            25.0, buy_threshold=75, sell_threshold=25, high_score_is_buy=True
        )
        == -1
    )
    assert (
        score_to_discrete_action(
            74.99, buy_threshold=75, sell_threshold=25, high_score_is_buy=True
        )
        == 0
    )


def test_score_to_discrete_action_parity_inverted():
    # When HIGH_SCORE_IS_BUY False, mapping should invert
    assert (
        score_to_discrete_action(
            75.0, buy_threshold=75, sell_threshold=25, high_score_is_buy=False
        )
        == -1
    )
    assert (
        score_to_discrete_action(
            25.0, buy_threshold=75, sell_threshold=25, high_score_is_buy=False
        )
        == 1
    )


def test_calculate_confidence_score_distance():
    # Verify 'distance' method around thresholds
    thresholds = {"buy": 80, "sell": 20}
    assert (
        calculate_confidence_score(90, thresholds=thresholds, method="distance") > 0.0
    )
    assert (
        calculate_confidence_score(10, thresholds=thresholds, method="distance") > 0.0
    )
    # In hold zone
    hold_conf = calculate_confidence_score(50, thresholds=thresholds, method="distance")
    assert 0.0 <= hold_conf <= 1.0


def test_confidence_to_score_thresholds_gaps_and_clamps():
    from ztb.trading.signal.common.utilities import confidence_to_score_thresholds

    # Test large min_gap enforcement
    buy, sell = confidence_to_score_thresholds(
        0.5, min_gap=50, default_buy=75, default_sell=25
    )
    assert buy >= sell + 50

    # Negative min_gap should be allowed (treated as float) but not break invariants
    buy, sell = confidence_to_score_thresholds(
        0.5, min_gap=-10, default_buy=75, default_sell=25
    )
    assert 0 <= buy <= 100 and 0 <= sell <= 100


def test_base_signal_processor_config_merge():
    from ztb.trading.signal.common.base_classes import BaseSignalProcessor

    class DummyProcessor(BaseSignalProcessor):
        def _get_default_config(self):
            return {"alpha": 1, "beta": 2}

    dp = DummyProcessor(config={})
    # Even with empty config passed, defaults should be present
    assert dp.config.get("alpha") == 1
    assert dp.config.get("beta") == 2
