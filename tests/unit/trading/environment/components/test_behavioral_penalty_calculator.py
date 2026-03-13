# mypy: disable-error-code=literal-required
from typing import List, cast

import pytest

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
            "consistency_penalty_enabled": True,
            "consistency_penalty": 0.1,
            "consistency_lookback": 3,
            # Provide legacy nested structure for tests that mutate behavior.* keys
            "behavior": {
                "consistency_penalty": {"enabled": True, "value": 0.1, "lookback": 3}
            },
        },
    )


@pytest.fixture
def calculator(default_settings: RewardSettings) -> BehavioralPenaltyCalculator:
    """Fixture to create a BehavioralPenaltyCalculator instance."""
    return BehavioralPenaltyCalculator(default_settings)


def test_init_loads_settings_correctly(calculator: BehavioralPenaltyCalculator):
    """Test if the calculator initializes with correct settings."""
    assert calculator.consistency_penalty_enabled is True
    assert calculator.penalty_value == -0.1
    assert calculator.lookback == 3


def test_penalty_disabled(default_settings: RewardSettings):
    """Test that no penalty is applied when the feature is disabled."""
    default_settings["behavior"]["consistency_penalty"]["enabled"] = False
    calculator = BehavioralPenaltyCalculator(default_settings)

    for action in [ACTION_BUY, ACTION_HOLD, ACTION_SELL]:
        calculator.record_action(action)

    penalty = calculator.calculate_consistency_penalty()
    assert penalty == 0.0


@pytest.mark.parametrize(
    "actions, current_action, expected_penalty",
    [
        # --- Whipsaw (Penalty Applied) ---
        # Make sure the prior non-HOLD is inside the lookback window
        ([ACTION_HOLD, ACTION_HOLD, ACTION_BUY], ACTION_SELL, -0.1),
        ([ACTION_HOLD, ACTION_HOLD, ACTION_SELL], ACTION_BUY, -0.1),
        # --- No Whipsaw (No Penalty) ---
        # Consistent direction
        ([ACTION_BUY, ACTION_HOLD, ACTION_BUY], ACTION_BUY, 0.0),
        # Involving HOLD
        ([ACTION_HOLD, ACTION_BUY, ACTION_SELL], ACTION_HOLD, 0.0),
        # Not enough history for lookback=3 -> now considered inside lookback window
        ([ACTION_BUY, ACTION_SELL], ACTION_BUY, -0.1),
        # Full history and reversal -> whipsaw should be detected
        ([ACTION_BUY, ACTION_HOLD, ACTION_SELL], ACTION_BUY, -0.1),
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
    # If test provides current_action param, simulate it being recorded as latest action
    calculator.record_action(current_action)

    penalty = calculator.calculate_consistency_penalty()
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
        (
            5,
            [ACTION_BUY, ACTION_HOLD, ACTION_HOLD, ACTION_HOLD, ACTION_HOLD],
            ACTION_SELL,
            -0.1,
        ),
        (
            5,
            [ACTION_SELL, ACTION_BUY, ACTION_SELL, ACTION_BUY, ACTION_SELL],
            ACTION_BUY,
            -0.1,
        ),  # First action is SELL, current is BUY
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
    # Simulate current action being applied
    calculator.record_action(current_action)

    penalty = calculator.calculate_consistency_penalty()

    if expected_penalty != 0:
        assert penalty == pytest.approx(-penalty_value)
    else:
        assert penalty == pytest.approx(0.0)


def test_action_history_management(default_settings: RewardSettings):
    """Test if the action history deque is managed correctly."""
    lookback = 4
    default_settings["behavior"]["consistency_penalty"]["lookback"] = lookback
    calculator = BehavioralPenaltyCalculator(default_settings)

    from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL

    actions_to_record = [
        ACTION_BUY,
        ACTION_SELL,
        ACTION_HOLD,
        ACTION_BUY,
        ACTION_SELL,
        ACTION_HOLD,
    ]
    for action in actions_to_record:
        calculator.record_action(action)

    # The history should contain at most the last `calculator.recent_actions.maxlen` entries
    expected_len = min(len(actions_to_record), calculator.recent_actions.maxlen)
    assert len(calculator.recent_actions) == expected_len
    assert list(calculator.recent_actions) == list(actions_to_record)[-expected_len:]


def test_forced_balance_min_actions_expand_history(default_settings: RewardSettings):
    """Forced balance thresholds should enlarge the action history window."""
    default_settings["forced_balance_min_actions"] = 120
    calculator = BehavioralPenaltyCalculator(default_settings)

    # History maxlen stores lookback+1 due to the "current" slot reservation
    assert calculator.recent_actions.maxlen == 121


def test_trend_adjustment_targets(default_settings: RewardSettings):
    """Test that trend adjustment changes buy/sell targets in the expected direction."""

    class DummyConfig:
        pass

    cfg = DummyConfig()
    # Reuse default settings but add behavior fields
    cfg.reward_settings = {
        "behavior": {
            "trend_adjustment_enabled": True,
            "trend_adjustment_strength": 0.2,
            "balance_penalty_targets": {"buy_target": 0.3, "sell_target": 0.3},
        }
    }

    # Stub TrendDetector that reports a positive trend
    class StubTrendDetector:
        def __init__(self, signal=0.5):
            self._signal = signal

        def get_trend_signal(self):
            return self._signal

        def get_statistics(self):
            return {"samples": 1, "last_signal": self._signal}

    stub = StubTrendDetector(signal=0.5)
    calc = BehavioralPenaltyCalculator(cfg, trend_detector=stub)
    adjusted = calc._adjust_targets_by_trend()
    # Positive trend should increase buy_target and decrease sell_target
    assert adjusted["buy_target"] > 0.3
    assert adjusted["sell_target"] < 0.3


def test_emergency_intervention_penalty(default_settings: RewardSettings):
    """Emergency penalty should be triggered when buy/sell imbalance exceeds threshold."""

    class DummyConfig:
        pass

    cfg = DummyConfig()
    cfg.reward_settings = {
        "behavior": {
            "emergency_intervention_enabled": True,
            "emergency_intervention_threshold": 0.1,
            "emergency_intervention_penalty": -250.0,
            "balance_penalty_min_actions": 5,
        }
    }

    calc = BehavioralPenaltyCalculator(cfg)
    # Create imbalance: 6 buys, 1 sell
    for _ in range(6):
        calc.record_action(ACTION_BUY)
    calc.record_action(ACTION_SELL)

    penalty = calc.calculate_emergency_intervention()
    assert penalty == pytest.approx(-250.0)


def test_consistency_min_actions_threshold(default_settings: RewardSettings):
    """When consistency_min_actions requires more non-HOLD actions than available, no penalty applies."""
    default_settings["behavior"]["consistency_penalty"]["lookback"] = 5
    default_settings["behavior"]["consistency_min_actions"] = 3
    calculator = BehavioralPenaltyCalculator(default_settings)
    # only 2 non-HOLD actions present (not enough)
    calculator.record_action(ACTION_BUY)
    calculator.record_action(ACTION_HOLD)
    calculator.record_action(ACTION_SELL)
    calculator.record_action(ACTION_SELL)
    # This reversal should not trigger penalty due to min_actions=3 requirement
    penalty = calculator.calculate_consistency_penalty()
    assert penalty == pytest.approx(0.0)


def test_hold_between_non_hold_actions_counts_toward_lookback(
    default_settings: RewardSettings,
):
    """Verify that HOLD entries between non-HOLD actions do not prevent whipsaw detection when lookback is 1."""
    default_settings["behavior"]["consistency_penalty"]["lookback"] = 2
    calculator = BehavioralPenaltyCalculator(default_settings)
    # previous non-HOLD is BUY with a HOLD between; current is SELL, should be a reversal
    calculator.record_action(ACTION_BUY)
    calculator.record_action(ACTION_HOLD)
    calculator.record_action(ACTION_SELL)
    penalty = calculator.calculate_consistency_penalty()
    assert penalty == pytest.approx(-0.1)


def test_action_entropy_includes_current_action(default_settings: RewardSettings):
    """Ensure action entropy shaping uses the recent actions including the current one."""
    default_settings["behavior"]["action_entropy_shaping_enabled"] = True
    default_settings["behavior"]["action_entropy_shaping_value"] = 0.05
    default_settings["behavior"]["action_entropy_lookback"] = 3
    calc = BehavioralPenaltyCalculator(default_settings)
    # Sanity-check settings were loaded properly
    assert calc.action_entropy_shaping_enabled is True
    assert calc.action_entropy_lookback == 3
    # Two buys followed by a sell (current) should be enough for lookback=3
    calc.record_action(ACTION_BUY)
    calc.record_action(ACTION_BUY)
    calc.record_action(ACTION_SELL)
    shaping = calc.calculate_action_entropy_shaping()
    assert shaping > 0


def test_get_recent_counts_lookback_counts_current(default_settings: RewardSettings):
    """_get_recent_counts(lookback) should reflect just the last N actions (including current)."""
    calculator = BehavioralPenaltyCalculator(default_settings)
    calculator.record_action(ACTION_BUY)
    calculator.record_action(ACTION_SELL)
    calculator.record_action(ACTION_HOLD)
    # Lookback 1: only most recent action = HOLD
    counts = calculator._get_recent_counts(1)
    assert counts == [1, 0, 0]
    # Lookback 2: last two actions SELL and HOLD
    counts = calculator._get_recent_counts(2)
    assert counts == [1, 0, 1]
    # Full history
    counts = calculator._get_recent_counts(None)
    assert counts == [1, 1, 1]


# ---------- 400# Tests: behavior_optimization fallback ----------


class _FakeConfig:
    """Minimal EnvironmentConfig substitute with both reward_settings and behavior_optimization."""

    def __init__(
        self,
        reward_settings: RewardSettings | None = None,
        behavior_optimization: dict | None = None,
    ):
        self.reward_settings = reward_settings
        self.behavior_optimization = behavior_optimization


def test_balance_shaping_disabled_via_behavior_optimization():
    """400# FIX: balance_shaping_enabled in behavior_optimization must override the default True."""
    config = _FakeConfig(
        reward_settings=RewardSettings(),
        behavior_optimization={"balance_shaping_enabled": False},
    )
    calc = BehavioralPenaltyCalculator(config=config)
    assert calc.balance_shaping_enabled is False


def test_action_entropy_shaping_disabled_via_behavior_optimization():
    """400# FIX: action_entropy_shaping_enabled in behavior_optimization must override the default True."""
    config = _FakeConfig(
        reward_settings=RewardSettings(),
        behavior_optimization={"action_entropy_shaping_enabled": False},
    )
    calc = BehavioralPenaltyCalculator(config=config)
    assert calc.action_entropy_shaping_enabled is False


def test_balance_shaping_default_when_no_behavior_optimization():
    """balance_shaping_enabled should default to True when behavior_optimization is absent."""
    config = _FakeConfig(reward_settings=RewardSettings())
    calc = BehavioralPenaltyCalculator(config=config)
    assert calc.balance_shaping_enabled is True


def test_behavior_optimization_keys_do_not_override_reward_settings():
    """Keys already present in RewardSettings dataclass (e.g. consistency_penalty) should be read
    from reward_settings, NOT from behavior_optimization fallback."""
    config = _FakeConfig(
        reward_settings=RewardSettings(consistency_penalty=0.42),
        behavior_optimization={"consistency_penalty": 999.0},
    )
    calc = BehavioralPenaltyCalculator(config=config)
    # penalty_value should come from RewardSettings (−0.42), not behavior_optimization (−999)
    assert abs(calc.penalty_value - (-0.42)) < 1e-6
