# mypy: disable-error-code=literal-required
import pytest
from typing import cast, List

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.behavioral_penalty_calculator import (
    BehavioralPenaltyCalculator,
)
from ztb.trading.environment.utils.config import RewardSettings


@pytest.fixture
def default_settings() -> RewardSettings:
    """Default settings with consistency penalty enabled."""
    return cast(
        RewardSettings,
        {
            "behavior": {
                "consistency_penalty": {
                    "enabled": True,
                    "value": 0.1,
                    "lookback": 3,
                }
            }
        },
    )


@pytest.fixture
def calculator(default_settings: RewardSettings) -> BehavioralPenaltyCalculator:
    """Fixture to create a BehavioralPenaltyCalculator instance."""
    return BehavioralPenaltyCalculator(default_settings)


def test_init_loads_settings_correctly(calculator: BehavioralPenaltyCalculator):
    """Test if the calculator initializes with correct settings."""
    assert calculator.consistency_penalty_enabled is True
    assert calculator.consistency_penalty_value == 0.1
    assert calculator.consistency_penalty_lookback == 3


def test_penalty_disabled(default_settings: RewardSettings):
    """Test that no penalty is applied when the feature is disabled."""
    default_settings["behavior"]["consistency_penalty"]["enabled"] = False
    calculator = BehavioralPenaltyCalculator(default_settings)

    for action in [ACTION_BUY, ACTION_HOLD, ACTION_SELL]:
        calculator.record_action(action)

    penalty = calculator.calculate_consistency_penalty(ACTION_BUY)
    assert penalty == 0.0


@pytest.mark.parametrize(
    "actions, current_action, expected_penalty",
    [
        # --- Whipsaw (Penalty Applied) ---
        # Standard BUY -> ... -> SELL
        ([ACTION_BUY, ACTION_HOLD, ACTION_HOLD], ACTION_SELL, -0.1),
        # Standard SELL -> ... -> BUY
        ([ACTION_SELL, ACTION_HOLD, ACTION_HOLD], ACTION_BUY, -0.1),
        
        # --- No Whipsaw (No Penalty) ---
        # Consistent direction
        ([ACTION_BUY, ACTION_HOLD, ACTION_BUY], ACTION_BUY, 0.0),
        # Involving HOLD
        ([ACTION_HOLD, ACTION_BUY, ACTION_SELL], ACTION_HOLD, 0.0),
        # Not enough history for lookback=3
        ([ACTION_BUY, ACTION_SELL], ACTION_BUY, 0.0),
        # Full history but no whipsaw
        ([ACTION_BUY, ACTION_HOLD, ACTION_SELL], ACTION_BUY, 0.0),
    ],
)
def test_consistency_penalty_scenarios(
    default_settings: RewardSettings,
    actions: List[int],
    current_action: int,
    expected_penalty: float,
):
    """Test various scenarios for consistency penalty."""
    calculator = BehavioralPenaltyCalculator(default_settings)
    for action in actions:
        calculator.record_action(action)

    penalty = calculator.calculate_consistency_penalty(current_action)
    assert penalty == pytest.approx(expected_penalty)


@pytest.mark.parametrize(
    "lookback, actions, current_action, expected_penalty",
    [
        # Lookback 1: immediate reversal
        (1, [ACTION_BUY], ACTION_SELL, -0.1),
        (1, [ACTION_SELL], ACTION_BUY, -0.1),
        (1, [ACTION_BUY], ACTION_BUY, 0.0),
        # Lookback 0: should be disabled
        (0, [ACTION_BUY, ACTION_SELL], ACTION_BUY, 0.0),
        # Lookback 5
        (5, [ACTION_BUY, ACTION_HOLD, ACTION_HOLD, ACTION_HOLD, ACTION_HOLD], ACTION_SELL, -0.1),
        (5, [ACTION_SELL, ACTION_BUY, ACTION_SELL, ACTION_BUY, ACTION_SELL], ACTION_BUY, -0.1), # First action is SELL, current is BUY
    ],
)
def test_boundary_and_lookback_values(
    default_settings: RewardSettings,
    lookback: int,
    actions: List[int],
    current_action: int,
    expected_penalty: float,
):
    """Test boundary conditions for the lookback parameter."""
    default_settings["behavior"]["consistency_penalty"]["lookback"] = lookback
    # The penalty value is fixed in the default settings, so we use it for assertion
    penalty_value = default_settings["behavior"]["consistency_penalty"]["value"]
    
    calculator = BehavioralPenaltyCalculator(default_settings)

    for action in actions:
        calculator.record_action(action)

    penalty = calculator.calculate_consistency_penalty(current_action)
    
    if expected_penalty != 0:
        assert penalty == pytest.approx(-penalty_value)
    else:
        assert penalty == pytest.approx(0.0)


def test_action_history_management(default_settings: RewardSettings):
    """Test if the action history deque is managed correctly."""
    lookback = 4
    default_settings["behavior"]["consistency_penalty"]["lookback"] = lookback
    calculator = BehavioralPenaltyCalculator(default_settings)

    actions_to_record = [1, 2, 0, 1, 2, 0]  # 6 actions
    for action in actions_to_record:
        calculator.record_action(action)

    # The history should only contain the last `lookback` actions
    assert len(calculator._recent_actions) == lookback
    assert calculator._recent_actions == [0, 1, 2, 0]
