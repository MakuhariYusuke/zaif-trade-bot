from ztb.trading.environment.components.threshold_manager import ThresholdManager


def test_threshold_manager_clamps_base_threshold(tmp_path):
    # Supply config with base_threshold outside allowed range (less than min and more than max)
    config = type("C", (), {})()
    config.continuous_to_discrete_threshold = 10.0  # huge base
    config.min_action_threshold = 0.001
    config.max_action_threshold = 0.05
    config.adaptive_threshold_mode = False

    manager = ThresholdManager(config)
    # Should be clamped within [0.001, 0.05]
    assert manager.base_threshold <= manager.max_threshold
    assert manager.base_threshold >= manager.min_threshold

    # If base < min (simulate tiny base), should be clamped
    config2 = type("C", (), {})()
    config2.continuous_to_discrete_threshold = 0.0
    config2.min_action_threshold = 0.001
    config2.max_action_threshold = 0.05
    config2.adaptive_threshold_mode = False
    manager2 = ThresholdManager(config2)
    assert manager2.base_threshold >= manager2.min_threshold
