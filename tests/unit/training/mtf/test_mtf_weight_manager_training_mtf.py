import threading

from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)




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


    t2 = threading.Thread(target=task_b)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    w = mgr.get_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
