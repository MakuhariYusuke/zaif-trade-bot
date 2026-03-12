import threading

from ztb.trading.environment.components.reward.mtf_weight_manager import (
    MTFWeightManager,
)


def test_set_weights_missing_keys_uses_default_and_normalizes() -> None:
    mgr = MTFWeightManager(config={})
    before = mgr.get_weights()

    ok = mgr.set_weights({"1min": 0.5})

    assert ok is True
    weights = mgr.get_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["1min"] != before["1min"]



def test_concurrent_set_weights_keeps_normalized_state() -> None:
    mgr = MTFWeightManager(config={})

    def task_a() -> None:
        mgr.set_weights({"1min": 0.2, "5min": 0.7, "15min": 0.1})

    def task_b() -> None:
        mgr.set_weights({"1min": 0.5, "5min": 0.4, "15min": 0.1})

    t1 = threading.Thread(target=task_a)
    t2 = threading.Thread(target=task_b)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    weights = mgr.get_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9
