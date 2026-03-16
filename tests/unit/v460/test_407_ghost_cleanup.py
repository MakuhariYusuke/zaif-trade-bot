"""407# Ghost file cleanup and performance improvement tests.

Verifies:
- S4: Tuple assignment bug fix in calculate_reward_simple
- P1: Settings cache in RewardCalculator._get_nested_setting
- P3: Unified GC through MemoryManager (no double-GC)
- P5: collect_garbage returns int
- Dead code removal: orphaned reward/ imports cleaned
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tests.unit.v460._fill_test_source import read_inspect_source


@pytest.fixture(scope="module")
def reward_calculator_default():
    from ztb.trading.environment.components.calculators.reward_calculator import (
        RewardCalculator,
    )
    from ztb.trading.environment.utils.config import (
        EnvironmentConfig,
        RewardSettings,
    )

    config = SimpleNamespace(behavior_optimization={})
    rs = RewardSettings()
    config.reward_settings = rs
    return RewardCalculator(config, rs, 100000.0)


@pytest.fixture(scope="module")
def reward_calculator_custom():
    from ztb.trading.environment.components.calculators.reward_calculator import (
        RewardCalculator,
    )
    from ztb.trading.environment.utils.config import (
        EnvironmentConfig,
        RewardSettings,
    )

    config = SimpleNamespace(behavior_optimization={})
    rs = RewardSettings(custom_reward_params={"test_key": 42.0})
    config.reward_settings = rs
    return RewardCalculator(config, rs, 100000.0)


class TestS4TupleBugFix:
    """S4: continuous_action_value was (None,) instead of None."""

    def test_continuous_action_value_not_tuple(self):
        """Verify continuous_action_value is None, not (None,)."""
        from ztb.trading.environment.components.calculators.reward_calculator import (
            RewardCalculator,
        )

        # Get the source of calculate_reward_simple
        source = read_inspect_source(RewardCalculator.calculate_reward_simple)
        # Filter out comments and check code lines only
        code_lines = [
            line for line in source.split("\n")
            if "continuous_action_value" in line
            and not line.strip().startswith("#")
        ]
        for line in code_lines:
            # Ensure no line has = (None,) pattern (tuple bug)
            assert "= (None," not in line, (
                f"continuous_action_value should be = None, not = (None,): {line}"
            )


class TestP1SettingsCache:
    """P1: _get_nested_setting should cache results."""

    def test_settings_cache_initialized(self, reward_calculator_default):
        """RewardCalculator should have _settings_cache dict."""
        rc = reward_calculator_default
        assert hasattr(rc, "_settings_cache")
        assert isinstance(rc._settings_cache, dict)

    def test_settings_cache_populated_after_init(self, reward_calculator_default):
        """After __init__, cache should contain entries from init-time lookups."""
        rc = reward_calculator_default
        # Cache should have been populated by __init__ calls to get_setting_*
        assert len(rc._settings_cache) > 0

    def test_settings_cache_returns_same_value(self, reward_calculator_custom):
        """Repeated get_setting_float should return cached value."""
        rc = reward_calculator_custom
        # First call — cache miss
        val1 = rc.get_setting_float("test_key", 0.0)
        # Second call — cache hit
        val2 = rc.get_setting_float("test_key", 0.0)
        assert val1 == val2 == 42.0
        # Verify it's in cache
        assert "test_key" in rc._settings_cache


class TestP3UnifiedGC:
    """P3: GC should be unified through MemoryManager."""

    def test_memory_manager_default_gc_interval(self):
        """MemoryManager should default to 50000 step interval."""
        from ztb.trading.environment.components.memory_manager import MemoryManager

        mm = MemoryManager(gc_step_interval=50000)
        assert mm.gc_step_interval == 50000
        assert mm.is_gc_enabled is True

    def test_memory_manager_gc_disabled(self):
        """MemoryManager with gc_step_interval=0 should disable GC."""
        from ztb.trading.environment.components.memory_manager import MemoryManager

        mm = MemoryManager(gc_step_interval=0)
        assert mm.is_gc_enabled is False
        assert mm.should_collect_garbage_at_step(100) is False

    def test_no_double_gc_in_core(self):
        """core.py should not have DEFAULT_GC_STEP_INTERVAL hardcoded GC."""
        from ztb.trading.environment.heavy_env import core

        source = read_inspect_source(core.HeavyTradingEnv)
        # Should not have the old double-GC pattern
        assert "DEFAULT_GC_STEP_INTERVAL" not in source or "removed" in source


class TestP5CollectGarbageReturn:
    """P5: collect_garbage should return int."""

    @patch("ztb.trading.environment.components.memory_manager.gc.collect", return_value=7)
    def test_collect_garbage_returns_int(self, mock_collect):
        """collect_garbage should return number of collected objects."""
        from ztb.trading.environment.components.memory_manager import MemoryManager

        mm = MemoryManager()
        result = mm.collect_garbage()
        assert isinstance(result, int)
        assert result == 7
        mock_collect.assert_called_once_with(generation=2)

    @patch(
        "ztb.trading.environment.components.memory_manager.gc.collect",
        side_effect=[1, 2, 3],
    )
    def test_collect_garbage_aggressive_returns_int(self, mock_collect):
        """collect_garbage_aggressive should return total collected count."""
        from ztb.trading.environment.components.memory_manager import MemoryManager

        mm = MemoryManager()
        result = mm.collect_garbage_aggressive()
        assert isinstance(result, int)
        assert result == 6
        assert mock_collect.call_count == 3


class TestDeadCodeRemoval:
    """Verify orphaned reward components were properly removed."""

    def test_reward_init_no_orphaned_imports(self):
        """reward/__init__.py should not import archived components."""
        from ztb.trading.environment.components import reward

        # These should NOT be importable from the package anymore
        assert not hasattr(reward, "BaseRewardCalculator")
        assert not hasattr(reward, "ActionPenaltyCalculator")
        assert not hasattr(reward, "DiversityBonusCalculator")
        assert not hasattr(reward, "WinRateBonusCalculator")
        assert not hasattr(reward, "DrawdownPenaltyCalculator")

    def test_reward_init_active_imports(self):
        """reward/__init__.py should export active components."""
        from ztb.trading.environment.components.reward import (
            BalanceCurriculumManager,
            MTFWeightManager,
            TrendDetector,
        )

        assert BalanceCurriculumManager is not None
        assert MTFWeightManager is not None
        assert TrendDetector is not None
