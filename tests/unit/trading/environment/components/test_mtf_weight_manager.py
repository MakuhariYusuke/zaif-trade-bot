from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)


def test_set_weights_respects_bounds():
    cfg = type("C", (), {})()
    mtf = MTFWeightManager(cfg)
    # try to set unrealistic weights
    mtf.set_weights({"1min": 10.0, "5min": 0.0, "15min": 0.0})
    w = mtf.get_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-12
    # each weight should be within min/max
    assert 0.1 <= w["1min"] <= 0.5
    assert 0.1 <= w["5min"] <= 0.8
    assert 0.01 <= w["15min"] <= 0.5


def test_update_with_metrics_conservative():
    cfg = type("C", (), {})()
    mtf = MTFWeightManager(cfg)
    before = mtf.get_weights()
    metrics = {
        "tf_metrics": {
            "1min": {"sharpe": 0.1},
            "5min": {"sharpe": 0.7},
            "15min": {"sharpe": 0.4},
        }
    }
    mtf.update(step=100, metrics=metrics)
    after = mtf.get_weights()
    # ensure weights updated but conservative
    assert after["5min"] >= before["5min"]
    assert abs(sum(after.values()) - 1.0) < 1e-12
