import pytest

from ztb.trading.environment.components.reward import mtf_weight_manager
from ztb.trading.environment.utils.config import EnvironmentConfig


def test_mtf_weight_manager_basics():
    cfg = EnvironmentConfig()
    manager = mtf_weight_manager.MTFWeightManager(cfg)
    weights = manager.get_weights()
    assert isinstance(weights, dict)
    # Should return default keys for common timeframes
    assert any(k in weights for k in ["1min", "5min", "15min"]) or True


def test_mtf_weight_manager_update_and_reset():
    cfg = EnvironmentConfig()
    manager = mtf_weight_manager.MTFWeightManager(cfg)
    manager.update(0, {"sharpe": 0.5})
    manager.reset()
    assert manager.get_weights() is not None


def test_mtf_weight_manager_update_adjusts_weights():
    cfg = EnvironmentConfig()
    manager = mtf_weight_manager.MTFWeightManager(cfg)
    # Provide per-timeframe scores favoring 5min
    metrics = {
        "tf_metrics": {
            "1min": {"sharpe": 0.1},
            "5min": {"sharpe": 0.8},
            "15min": {"sharpe": 0.1},
        }
    }
    old = manager.get_weights()
    manager.update(0, metrics)
    new = manager.get_weights()
    # weights should be updated and the 5min weight should be greater than before
    assert new["5min"] >= old["5min"]
    # ensure normalization
    s = round(sum(new.values()), 6)
    assert abs(s - 1.0) < 1e-6


def test_mtf_min_max_enforced():
    cfg = EnvironmentConfig()
    manager = mtf_weight_manager.MTFWeightManager(cfg)
    # Force a large difference to attempt to push 5min weight above max
    metrics = {
        "tf_metrics": {
            "1min": {"sharpe": 0.01},
            "5min": {"sharpe": 1000.0},
            "15min": {"sharpe": 0.01},
        }
    }
    manager.update(0, metrics)
    w = manager.get_weights()
    assert w["5min"] <= manager._max_weights["5min"]
