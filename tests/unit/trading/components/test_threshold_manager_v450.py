from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.types.common import ConfigDict


class TestThresholdManagerV450:
    @pytest.fixture
    def config(self):
        return MagicMock(spec=ConfigDict)

    def test_initialization(self, config):
        config.continuous_to_discrete_threshold = 0.01
        config.dynamic_threshold_mode = "z_score"
        config.z_score_window = 50
        config.z_score_threshold = 2.0

        manager = ThresholdManager(config)

        assert manager.dynamic_threshold_mode == "z_score"
        assert manager.z_score_window == 50
        assert manager.z_score_threshold == 2.0
        assert isinstance(manager.action_history, deque)
        assert manager.action_history.maxlen == 50

    def test_update_action_stats(self, config):
        manager = ThresholdManager(config)
        manager.update_action_stats(0.5)
        manager.update_action_stats(-0.3)

        assert len(manager.action_history) == 2
        assert manager.action_history[0] == 0.5
        assert manager.action_history[1] == 0.3  # Should be absolute value

    def test_z_score_threshold_calculation(self, config):
        config.continuous_to_discrete_threshold = 0.05
        config.dynamic_threshold_mode = "z_score"
        config.z_score_threshold = 2.0
        manager = ThresholdManager(config)

        # Fill history with small values
        for i in range(20):
            manager.update_action_stats(0.01 + (i % 2) * 0.002)

        # Mean should be 0.01, Std should be 0
        # But we have min_std protection

        # Case 1: Small action, not significant -> returns base threshold
        threshold = manager.get_threshold(base_value=0.05, raw_action_value=0.01)
        assert threshold == 0.05

        # Case 2: Large action (outlier)
        # Let's make history have some variance
        manager.action_history.clear()
        for i in range(20):
            val = 0.01 + (i % 2) * 0.002  # 0.01, 0.012, 0.01, ...
            manager.update_action_stats(val)

        # Mean approx 0.011, Std small
        # A value of 0.04 should be huge Z-score

        # We need to calculate expected Z-score to be sure
        history = np.array(manager.action_history)
        mean = np.mean(history)
        std = np.std(history)

        target_action = 0.04
        z_score = (target_action - mean) / std
        assert z_score > 2.0

        threshold = manager.get_threshold(
            base_value=0.05, raw_action_value=target_action
        )

        # Should return something close to target_action * 0.99
        assert threshold < target_action
        assert threshold == pytest.approx(target_action * 0.99)

        # Case 3: Negative base threshold (SELL)
        threshold_neg = manager.get_threshold(
            base_value=-0.05, raw_action_value=target_action
        )
        assert threshold_neg < 0
        assert abs(threshold_neg) == pytest.approx(target_action * 0.99)

    def test_fallback_when_history_empty(self, config):
        config.dynamic_threshold_mode = "z_score"
        manager = ThresholdManager(config)

        # Empty history
        threshold = manager.get_threshold(base_value=0.05, raw_action_value=0.1)
        assert threshold == 0.05

    def test_legacy_mode(self, config):
        config.dynamic_threshold_mode = "fixed"
        config.adaptive_threshold_mode = False
        manager = ThresholdManager(config)

        threshold = manager.get_threshold(base_value=0.05, raw_action_value=0.1)
        assert threshold == 0.05

    def test_z_score_threshold_calculation_mad(self, config):
        config.continuous_to_discrete_threshold = 0.05
        config.dynamic_threshold_mode = "z_score"
        config.z_score_threshold = 2.0
        config.z_score_method = "mad"
        manager = ThresholdManager(config)

        # Construct history with small variance but non-zero MAD
        history_vals = [0.01 + (i % 5) * 0.002 for i in range(20)]
        for v in history_vals:
            manager.update_action_stats(v)

        target_action = 0.05
        threshold = manager.get_threshold(
            base_value=0.05, raw_action_value=target_action
        )
        assert threshold < target_action
        assert threshold == pytest.approx(target_action * 0.99)

    def test_zscore_trigger_count(self, config):
        config.dynamic_threshold_mode = "z_score"
        config.z_score_window = 50
        config.z_score_threshold = 2.0
        manager = ThresholdManager(config)
        for i in range(20):
            manager.update_action_stats(0.01 + (i % 2) * 0.002)

        assert manager.z_score_trigger_count == 0
        # Trigger with a big action using internals (ensures calculation occurs)
        manager._calculate_z_score_threshold(0.04, 0.05)
        assert manager.z_score_trigger_count == 1
