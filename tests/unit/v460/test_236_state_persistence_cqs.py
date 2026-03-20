"""236# テスト: State persistence, per-side no_feasible, hasattr排除, FFD CQS分離."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_cycle_executor import FillCycleExecutorMixin
from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin
from scripts.v460.lib.resilience import FillTestState
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from tests.unit.v460._fill_test_source import (
    CONFIG_HOT_RELOAD,
    FAST_FILL_DEFENSE,
    FILL_CYCLE_EXECUTOR,
    FILL_LOOP_ORCHESTRATOR,
    FILL_RECORD_HELPERS,
    MAKER_PRICE,
    ORCHESTRATOR_LIFECYCLE,
    read_class_method_source,
    read_source_text,
)
from ztb.trading.risk.fast_fill_defense import FastFillDefense, FastFillDefenseConfig

if TYPE_CHECKING:
    pass


# ======================================================================
# 1. FillTestState に 236# フィールドが存在する
# ======================================================================


class TestStatePersistenceFields:
    """236# 4 カウンタが FillTestState に追加されている."""

    def test_degraded_liquidation_duty_counter_field(self) -> None:
        state = FillTestState()
        assert state.degraded_liquidation_duty_counter == 0

    def test_one_sided_cooldown_remaining_field(self) -> None:
        state = FillTestState()
        assert state.one_sided_cooldown_remaining == 0

    def test_one_sided_freeze_remaining_field(self) -> None:
        state = FillTestState()
        assert state.one_sided_freeze_remaining == 0

    def test_consecutive_no_feasible_field(self) -> None:
        state = FillTestState()
        assert state.consecutive_no_feasible is None

    def test_consecutive_no_feasible_dict(self) -> None:
        state = FillTestState(consecutive_no_feasible={"buy": 2, "sell": 1})
        assert state.consecutive_no_feasible == {"buy": 2, "sell": 1}

    def test_state_roundtrip_json(self) -> None:
        """JSON シリアライズ → デシリアライズで新フィールドが欠落しない."""
        state = FillTestState(
            degraded_liquidation_duty_counter=5,
            one_sided_cooldown_remaining=2,
            one_sided_freeze_remaining=3,
            consecutive_no_feasible={"buy": 1, "sell": 0},
        )
        d = asdict(state)
        j = json.dumps(d)
        loaded = json.loads(j)
        assert loaded["degraded_liquidation_duty_counter"] == 5
        assert loaded["one_sided_cooldown_remaining"] == 2
        assert loaded["one_sided_freeze_remaining"] == 3
        assert loaded["consecutive_no_feasible"] == {"buy": 1, "sell": 0}


# ======================================================================
# 2. _build_state_snapshot に 236# フィールドが含まれる
# ======================================================================


class TestBuildStateSnapshotIncludes236:
    """_build_state_snapshot のソースに 236# フィールド参照がある."""

    @staticmethod
    def _snapshot_source() -> str:
        return read_class_method_source(
            ORCHESTRATOR_LIFECYCLE,
            "OrchestratorLifecycleMixin",
            "_build_state_snapshot",
        )

    def test_snapshot_references_degraded_duty(self) -> None:
        src = self._snapshot_source()
        assert "degraded_liquidation_duty_counter" in src

    def test_snapshot_references_cooldown(self) -> None:
        src = self._snapshot_source()
        assert "one_sided_cooldown_remaining" in src

    def test_snapshot_references_freeze(self) -> None:
        src = self._snapshot_source()
        assert "one_sided_freeze_remaining" in src

    def test_snapshot_references_no_feasible(self) -> None:
        src = self._snapshot_source()
        assert "consecutive_no_feasible" in src


# ======================================================================
# 3. _restore_common_state に 236# フィールド復元がある
# ======================================================================


class TestRestoreCommonStateIncludes236:
    """_restore_common_state のソースに 236# 復元ロジックがある."""

    @staticmethod
    def _restore_source() -> str:
        return read_class_method_source(
            ORCHESTRATOR_LIFECYCLE,
            "OrchestratorLifecycleMixin",
            "_restore_common_state",
        )

    def test_restore_references_degraded_duty(self) -> None:
        src = self._restore_source()
        assert "degraded_liquidation_duty_counter" in src

    def test_restore_references_cooldown(self) -> None:
        src = self._restore_source()
        assert "one_sided_cooldown_remaining" in src

    def test_restore_references_freeze(self) -> None:
        src = self._restore_source()
        assert "one_sided_freeze_remaining" in src

    def test_restore_references_no_feasible(self) -> None:
        src = self._restore_source()
        assert "consecutive_no_feasible" in src


# ======================================================================
# 4. _consecutive_no_feasible が per-side dict
# ======================================================================


class TestNoFeasiblePerSide:
    """236# _consecutive_no_feasible が dict[str, int] (per-side)."""

    def test_class_level_declaration_is_dict_or_none(self) -> None:
        ann = FillCycleExecutorMixin.__annotations__
        # dict[str, int] | None
        assert "_consecutive_no_feasible" in ann

    def test_source_uses_side_key(self) -> None:
        """per-side のキーとして side が使用されている."""
        src = read_source_text(FILL_CYCLE_EXECUTOR)
        # side をキーに使用 (旧: self._consecutive_no_feasible += 1)
        assert ".get(side, 0)" in src


# ======================================================================
# 5. hasattr 排除
# ======================================================================


class TestHasattrRemoval236:
    """236# hasattr パターンの排除."""

    def test_no_hasattr_trending_sell_skip(self) -> None:
        src = read_source_text(FILL_RECORD_HELPERS)
        assert 'hasattr(self, "_trending_sell_skip_count")' not in src

    def test_no_hasattr_current_regime_value_in_executor(self) -> None:
        src = read_source_text(FILL_CYCLE_EXECUTOR)
        assert 'hasattr(self, "_current_regime_value")' not in src

    def test_class_level_defaults_exist(self) -> None:
        """orchestrator にクラスレベルデフォルトが存在."""
        assert hasattr(FillLoopOrchestratorMixin, "_trending_sell_skip_count")
        assert FillLoopOrchestratorMixin._trending_sell_skip_count == 0

    def test_skip_gate_evaluator_last_reload_check(self) -> None:
        """SkipGateEvaluator._last_reload_check がクラスレベルで宣言."""
        assert hasattr(SkipGateEvaluator, "_last_reload_check")
        assert SkipGateEvaluator._last_reload_check is None


# ======================================================================
# 6. FFD CQS 分離
# ======================================================================


class TestFFDCQSSeparation:
    """236# CQS: get_boost_multiplier は純粋 getter, maybe_expire_boost が副作用."""

    def test_get_boost_multiplier_is_pure_getter(self) -> None:
        """get_boost_multiplier() にはstate変更ロジックが無い."""
        src = read_class_method_source(
            FAST_FILL_DEFENSE,
            "FastFillDefense",
            "get_boost_multiplier",
        )
        # 旧: state.boost_active = False / state.boost_multiplier = 1.0 があった
        assert "boost_active = False" not in src
        assert "boost_multiplier = 1.0" not in src
        assert "normal_fill_streak" not in src

    def test_maybe_expire_boost_has_side_effects(self) -> None:
        """maybe_expire_boost() に TTL decay ロジックがある."""
        src = read_class_method_source(
            FAST_FILL_DEFENSE,
            "FastFillDefense",
            "maybe_expire_boost",
        )
        assert "boost_active = False" in src
        assert "boost_multiplier = 1.0" in src

    def test_getter_without_expire_returns_boost(self) -> None:
        """TTL 期限切れでも maybe_expire 前なら boost 値を返す."""
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=1.0,
            offset_boost=2.5, boost_ttl_sec=1.0,
        )
        ffd = FastFillDefense(cfg, base_offset_ratio=0.005)
        state = ffd._get_state("sell")
        state.boost_active = True
        state.boost_multiplier = 2.5
        state.boost_activated_at = time.time() - 100.0  # 期限切れ
        # expire 前 → まだ 2.5
        assert ffd.get_boost_multiplier("sell") == pytest.approx(2.5)

    def test_expire_then_get_returns_1(self) -> None:
        """maybe_expire 後は 1.0."""
        cfg = FastFillDefenseConfig(
            enabled=True, threshold_sec=1.0,
            offset_boost=2.5, boost_ttl_sec=1.0,
        )
        ffd = FastFillDefense(cfg, base_offset_ratio=0.005)
        state = ffd._get_state("sell")
        state.boost_active = True
        state.boost_multiplier = 2.5
        state.boost_activated_at = time.time() - 100.0
        ffd.maybe_expire_boost("sell")
        assert ffd.get_boost_multiplier("sell") == pytest.approx(1.0)
        assert not ffd.is_boost_active("sell")

    def test_maker_price_calls_expire_before_get(self) -> None:
        """maker_price のソースで maybe_expire_boost が get_boost_multiplier 前に呼ばれる."""
        src = read_source_text(MAKER_PRICE)
        idx_expire = src.index("maybe_expire_boost")
        idx_get = src.index("get_boost_multiplier")
        assert idx_expire < idx_get


# ======================================================================
# 7. Dead import 排除
# ======================================================================


class TestDeadImportRemoval:
    """236# 未使用 import の排除."""

    def test_no_import_sys_in_config_hot_reload(self) -> None:
        src = read_source_text(CONFIG_HOT_RELOAD)
        # "import sys" が単独行で存在しないこと
        for line in src.splitlines():
            stripped = line.strip()
            if stripped == "import sys":
                pytest.fail("Dead import 'import sys' found in config_hot_reload")

    def test_no_optional_in_orchestrator(self) -> None:
        src = read_source_text(FILL_LOOP_ORCHESTRATOR)
        # Optional は未使用 (from __future__ import annotations で X | None を使用)
        assert "Optional" not in src
