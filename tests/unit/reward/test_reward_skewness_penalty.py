
from ztb.trading.environment.components.behavioral_penalty_calculator import BehavioralPenaltyCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.constants import ACTION_BUY, ACTION_SELL


def test_skewness_penalty_sell_heavy():
    cfg = EnvironmentConfig()
    cfg.reward_settings = {"skewness_penalty_enabled": True, "skewness_penalty_value": 2.0, "skewness_penalty_tolerance": 0.05, "skewness_lookback": 5}
    calc = BehavioralPenaltyCalculator(cfg)
    # Simulate actions: many sells using the deque - ensure we hit lookback
    for _ in range(45):
        calc.record_action(ACTION_SELL)
    for _ in range(5):
        calc.record_action(ACTION_BUY)
    p = calc.calculate_skewness_penalty()
    assert p < 0.0, "Skewness penalty should be negative for SELL-heavy counts"


def test_skewness_penalty_buy_heavy():
    cfg = EnvironmentConfig()
    cfg.reward_settings = {"skewness_penalty_enabled": True, "skewness_penalty_value": 1.0, "skewness_penalty_tolerance": 0.01, "skewness_lookback": 5}
    calc = BehavioralPenaltyCalculator(cfg)
    for _ in range(40):
        calc.record_action(ACTION_BUY)
    for _ in range(4):
        calc.record_action(ACTION_SELL)
    p = calc.calculate_skewness_penalty()
    assert p < 0.0, "Skewness penalty should be negative for BUY-heavy counts"


def test_balance_shaping_reduces_deviation():
    cfg = EnvironmentConfig()
    cfg.reward_settings = {
        "balance_shaping_enabled": True,
        "balance_shaping_value": 0.5,
    }
    calc = BehavioralPenaltyCalculator(cfg)
    # Make counts skewed to SELL using deque
    # Expand lookback to include full history for this test
    cfg.reward_settings["skewness_lookback"] = 100
    cfg.reward_settings["action_entropy_lookback"] = 100
    calc = BehavioralPenaltyCalculator(cfg)
    # Make counts skewed to SELL using deque
    for _ in range(40):
        calc.record_action(ACTION_SELL)
    for _ in range(10):
        calc.record_action(ACTION_BUY)
    # BUY action should reduce sell dominance -> positive shaping
    s_buy = calc.calculate_balance_shaping(1)
    # SELL action should either increase deviation or not improve -> 0
    s_sell = calc.calculate_balance_shaping(2)
    assert s_buy > 0
    assert s_sell <= 0


def test_action_entropy_shaping():
    cfg = EnvironmentConfig()
    cfg.reward_settings = {
        "action_entropy_shaping_enabled": True,
        "action_entropy_shaping_value": 0.1,
        "action_entropy_lookback": 5,
    }
    calc = BehavioralPenaltyCalculator(cfg)
    # Fill recent_actions with mostly same action to lower entropy
    for _ in range(10):
        calc.record_action(ACTION_BUY)
    # Enough lookback
    shaping = calc.calculate_action_entropy_shaping()
    assert shaping > 0
