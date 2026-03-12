import pandas as pd

from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.trading.environment.utils.config import EnvironmentConfig


def make_env_config(**kwargs):
    cfg = EnvironmentConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_threshold_manager_non_adaptive_returns_base():
    cfg = make_env_config(
        adaptive_threshold_mode=False, continuous_to_discrete_threshold=0.01
    )
    tm = ThresholdManager(cfg)
    t = tm.get_threshold(volatility=1.0, current_price=100.0)
    assert abs(t - cfg.continuous_to_discrete_threshold) < 1e-8


def test_threshold_manager_adaptive_with_volatility_and_clamping():
    cfg = make_env_config(
        adaptive_threshold_mode=True,
        threshold_volatility_multiplier=1.0,
        min_action_threshold=0.001,
        max_action_threshold=0.05,
        continuous_to_discrete_threshold=0.01,
    )
    tm = ThresholdManager(cfg)
    # Provide volatility=1, price=100 => relative vol = 0.01 -> adjusted = base + 0.01
    t = tm.get_threshold(volatility=1.0, current_price=100.0)
    assert t >= cfg.min_action_threshold
    assert t <= cfg.max_action_threshold


def test_threshold_manager_regime_scaling():
    cfg = make_env_config(
        adaptive_threshold_mode=True,
        threshold_volatility_multiplier=1.0,
        min_action_threshold=0.001,
        max_action_threshold=0.1,
        continuous_to_discrete_threshold=0.01,
    )
    tm = ThresholdManager(cfg)
    t_ranging = tm.get_threshold(volatility=0.5, current_price=100.0, regime="ranging")
    t_trending = tm.get_threshold(
        volatility=0.5, current_price=100.0, regime="trending_bull"
    )
    # Ranging should produce higher threshold than trending (due to 1.5 multiplier applied in code)
    assert t_ranging >= t_trending


def test_threshold_manager_validate_config_errors():
    # min_threshold >= max_threshold should raise
    cfg = make_env_config(min_action_threshold=0.1, max_action_threshold=0.05)
    try:
        ThresholdManager(cfg)
        assert False, "Expected ValueError for invalid min/max"
    except ValueError:
        pass


def test_calculate_adaptive_signal_thresholds_bounds_and_cache():
    cfg = make_env_config(adaptive_threshold_mode=True)
    tm = ThresholdManager(cfg)
    data = pd.DataFrame({"close": [100, 102, 105, 101, 99, 98]})
    thr = tm.calculate_adaptive_signal_thresholds(
        data, base_confidence=0.7, base_strength=0.4
    )
    assert 0.5 <= thr["confidence_threshold"] <= 0.9
    assert 0.2 <= thr["signal_strength_threshold"] <= 0.7
    # Second call uses cache - ensure same result
    thr2 = tm.calculate_adaptive_signal_thresholds(
        data, base_confidence=0.7, base_strength=0.4
    )
    assert thr == thr2
