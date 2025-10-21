import pytest

from ztb.trading.environment import constants
from ztb.trading.environment.constants import (
    continuous_to_discrete_action,
    CONTINUOUS_TO_DISCRETE_THRESHOLD,
    CONTINUOUS_TO_DISCRETE_THRESHOLD_NEG,
    ACTION_BUY,
    ACTION_SELL,
    ACTION_HOLD,
)


def test_continuous_to_discrete_basic():
    # clearly above threshold -> BUY
    assert continuous_to_discrete_action(CONTINUOUS_TO_DISCRETE_THRESHOLD + 0.1) == ACTION_BUY
    # clearly below negative threshold -> SELL
    assert continuous_to_discrete_action(CONTINUOUS_TO_DISCRETE_THRESHOLD_NEG - 0.1) == ACTION_SELL
    # exactly zero -> HOLD
    assert continuous_to_discrete_action(0.0) == ACTION_HOLD


def test_continuous_to_discrete_at_thresholds():
    # exactly at positive threshold should be HOLD (threshold is exclusive for BUY)
    res_at_pos = continuous_to_discrete_action(float(CONTINUOUS_TO_DISCRETE_THRESHOLD))
    assert res_at_pos == ACTION_HOLD

    # exactly at negative threshold should be HOLD (threshold is exclusive for SELL)
    res_at_neg = continuous_to_discrete_action(float(CONTINUOUS_TO_DISCRETE_THRESHOLD_NEG))
    assert res_at_neg == ACTION_HOLD


def test_continuous_to_discrete_edge_values():
    # extreme values
    assert continuous_to_discrete_action(1.0) == ACTION_BUY
    assert continuous_to_discrete_action(-1.0) == ACTION_SELL


def test_type_coercion_and_invalid_inputs():
    # accepts floats and values that can be coerced to float
    assert continuous_to_discrete_action(0.0) == ACTION_HOLD
    assert continuous_to_discrete_action(0.3333) in (ACTION_HOLD, ACTION_BUY, ACTION_SELL)

    # invalid input type should raise (we don't coerce non-numeric types)
    with pytest.raises(TypeError):
        continuous_to_discrete_action("not-a-number")
