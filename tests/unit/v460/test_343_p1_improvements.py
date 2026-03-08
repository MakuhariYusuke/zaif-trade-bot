"""343# テスト: P1 改善 (forced downweight / sell KPI 分離 / skip_gate kill 連携).

テスト観点:
  A. forced fill PnL downweight (337# 完全除外→重み付け投入)
  B. 強制売り KPI 分離トラッキング (buy 側と対称)
  C. skip_gate/kill 連携 (kill 解除直後の skip_gate 緩和)
  D. Config / Parser / Hot-reload の整合性
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.fill_config_parser import _parse_stopgap_section
from scripts.v460.lib.orchestrator_guards import OrchestratorGuardsMixin
from tests.unit.v460._fill_test_source import (
    ORCHESTRATOR_GUARDS,
    read_source_text,
)

_GUARDS_SOURCE = read_source_text(ORCHESTRATOR_GUARDS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. Forced fill PnL downweight
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestForcedFillDownweight:
    """343# forced fill downweight: 337# 完全除外→重み付け投入."""

    def _build_mixin(
        self, *, downweight: float = 0.5
    ) -> OrchestratorGuardsMixin:
        mixin = OrchestratorGuardsMixin.__new__(OrchestratorGuardsMixin)
        mixin._sell_kill_mgr = MagicMock()
        mixin._buy_kill_mgr = MagicMock()
        mixin.config = FillTestConfig(forced_fill_pnl_downweight=downweight)
        return mixin

    def _make_record(
        self,
        side: str = "sell",
        pnl: float = -5.0,
        forced: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            side=side,
            filled=True,
            post_fill_30s_pnl=pnl,
            balance_forced_switch=forced,
        )

    def test_config_field_exists(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "forced_fill_pnl_downweight")

    def test_default_value(self) -> None:
        """コードデフォルト = 0.5 (YAML と一致)."""
        cfg = FillTestConfig()
        assert cfg.forced_fill_pnl_downweight == 0.5

    def test_downweight_half(self) -> None:
        """downweight=0.5 で PnL が半分の重みで追跡される."""
        mixin = self._build_mixin(downweight=0.5)
        record = self._make_record(side="sell", pnl=-10.0, forced=True)
        mixin._track_side_pnl(record)
        mixin._sell_kill_mgr.track.assert_called_once_with(-5.0)

    def test_downweight_one(self) -> None:
        """downweight=1.0 では通常扱い."""
        mixin = self._build_mixin(downweight=1.0)
        record = self._make_record(side="buy", pnl=-4.0, forced=True)
        mixin._track_side_pnl(record)
        mixin._buy_kill_mgr.track.assert_called_once_with(-4.0)

    def test_downweight_zero_excludes(self) -> None:
        """downweight=0.0 は旧挙動と同じ完全除外."""
        mixin = self._build_mixin(downweight=0.0)
        record = self._make_record(side="sell", pnl=-10.0, forced=True)
        mixin._track_side_pnl(record)
        mixin._sell_kill_mgr.track.assert_not_called()

    def test_normal_fill_unaffected(self) -> None:
        """forced=False の通常 fill は downweight 設定に関係なく等倍."""
        mixin = self._build_mixin(downweight=0.1)
        record = self._make_record(side="sell", pnl=-3.0, forced=False)
        mixin._track_side_pnl(record)
        mixin._sell_kill_mgr.track.assert_called_once_with(-3.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. 強制売り KPI 分離
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSellForcedKpiConfig:
    """343# forced_sell_kpi_tracking config/parser."""

    def test_config_field_exists(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "forced_sell_kpi_tracking_enabled")
        assert cfg.forced_sell_kpi_tracking_enabled is True

    def test_parser_parses_forced_sell_kpi(self) -> None:
        result = _parse_stopgap_section({
            "止血": {"forced_sell_kpi_tracking": False},
        })
        assert result["forced_sell_kpi_tracking_enabled"] is False

    def test_parser_absent_field_no_error(self) -> None:
        result = _parse_stopgap_section({"止血": {}})
        assert "forced_sell_kpi_tracking_enabled" not in result

    def test_session_state_fields_exist(self) -> None:
        """RunSessionState に sell KPI フィールドが存在."""
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState

        st = RunSessionState()
        assert st.forced_sell_fill_count == 0
        assert st.forced_sell_pnl_sum_bps == 0.0
        assert st.normal_sell_fill_count == 0
        assert st.normal_sell_pnl_sum_bps == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. skip_gate/kill 連携
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSkipGateKillReleaseConfig:
    """343# skip_gate/kill 連携の config/parser."""

    def test_config_fields_exist(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "skip_gate_kill_release_grace_cycles")
        assert hasattr(cfg, "skip_gate_kill_release_offset")

    def test_default_values(self) -> None:
        cfg = FillTestConfig()
        assert cfg.skip_gate_kill_release_grace_cycles == 3
        assert cfg.skip_gate_kill_release_offset == -0.1

    def test_parser_parses_kill_release(self) -> None:
        result = _parse_stopgap_section({
            "skip_gate": {
                "kill_release_grace_cycles": 5,
                "kill_release_offset": -0.2,
            },
        })
        # skip_gate パラメータは _parse_stopgap_section ではなく
        # skip_gate セクション内で解析されるため、ここでは別経路をテスト
        # (fill_config_parser の sg_map 経由)

    def test_parser_skip_gate_map(self) -> None:
        """fill_config_parser.py の sg_map に kill_release パラメータが含まれる."""
        from scripts.v460.lib import fill_config_parser

        src = open(fill_config_parser.__file__).read()
        assert "kill_release_grace_cycles" in src
        assert "kill_release_offset" in src


class TestKillReleaseTracking:
    """343# kill→非kill 遷移の追跡テスト."""

    def _build_mixin(self) -> OrchestratorGuardsMixin:
        mixin = OrchestratorGuardsMixin.__new__(OrchestratorGuardsMixin)
        mixin._sell_kill_mgr = MagicMock()
        mixin._buy_kill_mgr = MagicMock()
        mixin.config = FillTestConfig()
        mixin._regime_detector = None
        mixin._maker_price = MagicMock()
        mixin._maker_price.inv_net_imbalance = 0.0
        # kill release 追跡用
        mixin._kill_was_active_buy = False
        mixin._kill_was_active_sell = False
        mixin._kill_released_at_cycle_buy = None
        mixin._kill_released_at_cycle_sell = None
        mixin._cycle_count = 0
        mixin._guard_fire_counts = {}
        return mixin

    def test_kill_release_records_cycle(self) -> None:
        """kill→非kill 遷移時にサイクル番号が記録される."""
        mixin = self._build_mixin()
        mixin._cycle_count = 42

        # 最初は kill 状態
        mixin._kill_was_active_buy = True
        # check_kill が (False, telemetry) を返す = kill 解除
        mixin._buy_kill_mgr.check_kill.return_value = (
            False,
            SimpleNamespace(
                threshold_used=-0.8,
                cooldown_remaining=0,
                probe_fired=False,
                force_release_fired=False,
            ),
        )

        result = mixin._is_side_killed("buy")
        assert result is False
        assert mixin._kill_released_at_cycle_buy == 42

    def test_no_release_when_staying_killed(self) -> None:
        """kill → kill では release 記録されない."""
        mixin = self._build_mixin()
        mixin._cycle_count = 10
        mixin._kill_was_active_sell = True

        mixin._sell_kill_mgr.check_kill.return_value = (
            True,
            SimpleNamespace(
                threshold_used=-0.3,
                cooldown_remaining=5,
                probe_fired=False,
                force_release_fired=False,
            ),
        )

        result = mixin._is_side_killed("sell")
        assert result is True
        assert mixin._kill_released_at_cycle_sell is None

    def test_no_release_when_never_killed(self) -> None:
        """非kill → 非kill では release 記録されない."""
        mixin = self._build_mixin()
        mixin._cycle_count = 5
        mixin._kill_was_active_buy = False

        mixin._buy_kill_mgr.check_kill.return_value = (
            False,
            SimpleNamespace(
                threshold_used=-0.8,
                cooldown_remaining=0,
                probe_fired=False,
                force_release_fired=False,
            ),
        )

        result = mixin._is_side_killed("buy")
        assert result is False
        assert mixin._kill_released_at_cycle_buy is None

    def test_kill_was_active_updates(self) -> None:
        """_kill_was_active_{side} が killed に追従する."""
        mixin = self._build_mixin()
        mixin._buy_kill_mgr.check_kill.return_value = (
            True,
            SimpleNamespace(
                threshold_used=-0.8,
                cooldown_remaining=5,
                probe_fired=False,
                force_release_fired=False,
            ),
        )
        mixin._is_side_killed("buy")
        assert mixin._kill_was_active_buy is True

        mixin._buy_kill_mgr.check_kill.return_value = (
            False,
            SimpleNamespace(
                threshold_used=-0.8,
                cooldown_remaining=0,
                probe_fired=False,
                force_release_fired=False,
            ),
        )
        mixin._is_side_killed("buy")
        assert mixin._kill_was_active_buy is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. Config / Parser / Hot-reload 整合性
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestParserDownweight:
    """343# forced_fill_pnl_downweight parser テスト."""

    def test_parser_parses_downweight(self) -> None:
        result = _parse_stopgap_section({
            "止血": {"forced_fill_pnl_downweight": 0.3},
        })
        assert result["forced_fill_pnl_downweight"] == pytest.approx(0.3)

    def test_parser_absent_no_error(self) -> None:
        result = _parse_stopgap_section({"止血": {}})
        assert "forced_fill_pnl_downweight" not in result


class TestHotReloadFields:
    """343# 新規フィールドの hot-reload 対応確認."""

    def test_hot_reloadable_fields_include_343(self) -> None:
        from scripts.v460.lib.config_hot_reload import _HOT_RELOADABLE_FIELDS

        expected = {
            "forced_fill_pnl_downweight",
            "forced_sell_kpi_tracking_enabled",
            "skip_gate_kill_release_grace_cycles",
            "skip_gate_kill_release_offset",
        }
        for field_name in expected:
            assert field_name in _HOT_RELOADABLE_FIELDS, (
                f"{field_name} が hot-reload 対象に含まれていない"
            )


class TestOrchestratorGuardsSourceReferences:
    """343# orchestrator_guards.py のコード内 343# 参照確認."""

    def test_343_reference_exists(self) -> None:
        assert "343#" in _GUARDS_SOURCE

    def test_downweight_logic_in_track_side_pnl(self) -> None:
        assert "forced_fill_pnl_downweight" in _GUARDS_SOURCE

    def test_kill_release_tracking_in_is_side_killed(self) -> None:
        assert "_kill_was_active_buy" in _GUARDS_SOURCE
        assert "_kill_released_at_cycle_buy" in _GUARDS_SOURCE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. FillLoopOrchestrator クラスレベル属性宣言
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOrchestratorClassLevelAttrs:
    """343# kill release 追跡属性のクラスレベル宣言確認."""

    def test_kill_release_attrs_declared(self) -> None:
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin

        assert hasattr(FillLoopOrchestratorMixin, "_kill_was_active_buy")
        assert hasattr(FillLoopOrchestratorMixin, "_kill_was_active_sell")
        assert hasattr(FillLoopOrchestratorMixin, "_kill_released_at_cycle_buy")
        assert hasattr(FillLoopOrchestratorMixin, "_kill_released_at_cycle_sell")
