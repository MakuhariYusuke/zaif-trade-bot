import threading

from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)


def test_set_weights_clips_and_normalizes():
    mgr = MTFWeightManager(config={})
    # extreme values (some below min, some above max)
    weights = {"1min": 10.0, "5min": 0.0, "15min": -5.0}
    ok = mgr.set_weights(weights)
    assert ok is True
    w = mgr.get_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # check bounds
    assert w["1min"] <= mgr._max_weights["1min"]
    assert w["1min"] >= mgr._min_weights["1min"]
    assert w["15min"] >= mgr._min_weights["15min"]


def test_set_weights_missing_keys_uses_default_and_normalizes():
    mgr = MTFWeightManager(config={})
    before = mgr.get_weights()
    # only update 1min
    ok = mgr.set_weights({"1min": 0.5})
    assert ok is True
    w = mgr.get_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # ensure 1min changed and others adjusted
    assert w["1min"] != before["1min"]


def test_concurrent_set_weights_does_not_raise():
    mgr = MTFWeightManager(config={})

    def task_a():
        mgr.set_weights({"1min": 0.2, "5min": 0.7, "15min": 0.1})

    def task_b():
        mgr.set_weights({"1min": 0.5, "5min": 0.4, "15min": 0.1})

    t1 = threading.Thread(target=task_a)
    t2 = threading.Thread(target=task_b)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    w = mgr.get_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
