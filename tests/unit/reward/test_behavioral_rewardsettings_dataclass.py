import pytest

from ztb.trading.environment.components.behavioral_penalty_calculator import (
    BehavioralPenaltyCalculator,
)
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.trading.constants import ACTION_SELL, ACTION_BUY, ACTION_HOLD


def test_reward_settings_dataclass_reads_values():
    cfg = EnvironmentConfig()
    rs = RewardSettings()
    # Put experimental keys in custom_reward_params to simulate unknown keys
    rs.custom_reward_params["skewness_penalty_value"] = 1.5
    rs.custom_reward_params["skewness_penalty_enabled"] = True
    rs.custom_reward_params["skewness_penalty_tolerance"] = 0.01
    rs.custom_reward_params["skewness_lookback"] = 5

    cfg.reward_settings = rs

    calc = BehavioralPenaltyCalculator(cfg)

    assert calc.skewness_penalty_enabled is True
    assert calc.skewness_penalty_value == pytest.approx(1.5)

    # Simulate a skew via sliding window counts
    for _ in range(6):
        calc.record_action(ACTION_SELL)

    penalty = calc.calculate_skewness_penalty()
    assert penalty < 0


def test_lookback_boundary_values():
    """Test boundary values for lookback settings."""
    # lookback=0 now disables consistency penalties rather than clamping.
    # Dict-backed reward settings also keep the larger forced-balance window by default.
    cfg = EnvironmentConfig()
    cfg.reward_settings = {"consistency_lookback": 0}
    calc = BehavioralPenaltyCalculator(cfg)
    assert calc.lookback == 0
    assert calc.recent_actions.maxlen == 101

    # Test lookback = 1 (minimum valid)
    cfg.reward_settings = {"consistency_lookback": 1}
    calc = BehavioralPenaltyCalculator(cfg)
    assert calc.lookback == 1
    assert calc.recent_actions.maxlen == 101

    # Test large lookback = 1000
    cfg.reward_settings = {"consistency_lookback": 1000}
    calc = BehavioralPenaltyCalculator(cfg)
    assert calc.lookback == 1000
    assert calc.recent_actions.maxlen == 1001

    # Test default lookback = 50
    cfg = EnvironmentConfig()
    calc = BehavioralPenaltyCalculator(cfg)
    assert calc.lookback == 50
    assert calc.recent_actions.maxlen == 51


def test_sliding_window_edge_cases():
    """Test edge cases for sliding window behavior."""
    cfg = EnvironmentConfig()
    cfg.reward_settings = {
        "consistency_lookback": 3,
        "skewness_lookback": 3,
        "action_entropy_lookback": 3,
        "forced_balance_min_actions": 3,
    }
    calc = BehavioralPenaltyCalculator(cfg)

    # Empty window
    counts = calc._get_recent_counts()
    assert counts == [0, 0, 0]

    penalty = calc.calculate_skewness_penalty()
    assert penalty == 0.0  # No penalty when no actions

    # Single action
    calc.record_action(ACTION_BUY)
    counts = calc._get_recent_counts()
    assert counts == [0, 1, 0]

    penalty = calc.calculate_skewness_penalty()
    assert penalty == 0.0  # Not enough actions for penalty

    # Fill window exactly
    calc.record_action(ACTION_SELL)
    calc.record_action(ACTION_HOLD)
    counts = calc._get_recent_counts()
    assert counts == [1, 1, 1]

    # The deque keeps an extra slot so consistency logic can see previous+current actions.
    calc.record_action(ACTION_BUY)
    counts = calc._get_recent_counts()
    assert counts == [1, 2, 1]

    # Additional actions should now slide the oldest entries out.
    calc.record_action(ACTION_SELL)
    calc.record_action(ACTION_SELL)
    counts = calc._get_recent_counts()
    assert counts == [1, 1, 2]


def test_skewness_penalty_with_small_lookback():
    """Test skewness penalty with minimal lookback."""
    cfg = EnvironmentConfig()
    cfg.reward_settings = {
        "skewness_penalty_enabled": True,
        "skewness_penalty_value": 1.0,
        "skewness_penalty_tolerance": 0.05,
        "skewness_lookback": 5
    }
    calc = BehavioralPenaltyCalculator(cfg)

    # Fill with SELL actions to create skew
    for _ in range(5):
        calc.record_action(ACTION_SELL)

    penalty = calc.calculate_skewness_penalty()
    assert penalty < 0  # Should penalize SELL-heavy

    # Test with BUY actions
    calc.reset()
    for _ in range(5):
        calc.record_action(ACTION_BUY)

    penalty = calc.calculate_skewness_penalty()
    assert penalty < 0  # Should penalize BUY-heavy
