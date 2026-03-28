"""648# Inventory Deadlock Detection テスト.

片側 preflight_insufficient + 反対側 no_feasible_quote の cross-cycle 膠着を
検出する仕組みの動作を検証する。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from tests.unit.v460._fill_test_source import (
    ORCHESTRATOR_BALANCE,
    ORCHESTRATOR_POST_CYCLE,
    FILL_LOOP_ORCHESTRATOR,
    read_source_text,
)
from ztb.trading.common import cancel_reasons as CR


# ═══════════════════════════════════════════════════════════════════════
# A. cancel_reasons に INVENTORY_DEADLOCK が存在すること
# ═══════════════════════════════════════════════════════════════════════


class TestInventoryDeadlockCancelReason:
    """648# INVENTORY_DEADLOCK cancel reason の存在と型安全を検証."""

    def test_constant_exists(self) -> None:
        assert hasattr(CR, "INVENTORY_DEADLOCK")
        assert CR.INVENTORY_DEADLOCK == "inventory_deadlock"

    def test_in_audit_set(self) -> None:
        assert CR.INVENTORY_DEADLOCK in CR.AUDIT_CANCEL_REASONS

    def test_in_cancel_reason_literal(self) -> None:
        """CancelReason Literal 型に inventory_deadlock が含まれること."""
        import typing
        args = typing.get_args(CR.CancelReason)
        assert "inventory_deadlock" in args


# ═══════════════════════════════════════════════════════════════════════
# B. FillTestConfig にデッドロック検出パラメータが存在すること
# ═══════════════════════════════════════════════════════════════════════


class TestInventoryDeadlockConfig:
    """648# デッドロック検出設定パラメータの存在と妥当性."""

    def test_threshold_exists(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "inventory_deadlock_threshold")
        assert cfg.inventory_deadlock_threshold > 0

    def test_alert_interval_exists(self) -> None:
        cfg = FillTestConfig()
        assert hasattr(cfg, "inventory_deadlock_alert_interval_sec")
        assert cfg.inventory_deadlock_alert_interval_sec > 0

    def test_default_threshold_is_reasonable(self) -> None:
        cfg = FillTestConfig()
        assert 5 <= cfg.inventory_deadlock_threshold <= 50


# ═══════════════════════════════════════════════════════════════════════
# C. ソース解析: 実装の構造検証
# ═══════════════════════════════════════════════════════════════════════


class TestInventoryDeadlockSourceContract:
    """648# ソースレベルで実装が正しく配置されていることの検証."""

    def test_check_method_in_balance_mixin(self) -> None:
        """_check_inventory_deadlock が orchestrator_balance.py に存在."""
        src = read_source_text(ORCHESTRATOR_BALANCE)
        assert "_check_inventory_deadlock" in src

    def test_counter_in_orchestrator(self) -> None:
        """_inventory_deadlock_counter が fill_loop_orchestrator.py に宣言."""
        src = read_source_text(FILL_LOOP_ORCHESTRATOR)
        assert "_inventory_deadlock_counter" in src

    def test_counter_reset_on_fill(self) -> None:
        """fill 成功時に _inventory_deadlock_counter = 0 にリセット."""
        src = read_source_text(ORCHESTRATOR_POST_CYCLE)
        assert "_inventory_deadlock_counter = 0" in src

    def test_counter_increment_on_unfill(self) -> None:
        """unfilled 時に _inventory_deadlock_counter += 1."""
        src = read_source_text(ORCHESTRATOR_POST_CYCLE)
        assert "_inventory_deadlock_counter += 1" in src

    def test_counter_increment_in_balance_skip(self) -> None:
        """preflight skip 時に _inventory_deadlock_counter += 1."""
        src = read_source_text(ORCHESTRATOR_BALANCE)
        assert "_inventory_deadlock_counter += 1" in src

    def test_guard_fire_recorded(self) -> None:
        """デッドロック検出時に _inc_guard_fire("inventory_deadlock") が呼ばれる."""
        src = read_source_text(ORCHESTRATOR_BALANCE)
        assert '"inventory_deadlock"' in src

    def test_consecutive_no_feasible_checked(self) -> None:
        """反対側の _consecutive_no_feasible を参照して検出判定."""
        src = read_source_text(ORCHESTRATOR_BALANCE)
        assert "_consecutive_no_feasible" in src

    def test_alert_interval_throttling(self) -> None:
        """連続 alert 防止のための interval チェックが存在."""
        src = read_source_text(ORCHESTRATOR_BALANCE)
        assert "inventory_deadlock_alert_interval_sec" in src


# ═══════════════════════════════════════════════════════════════════════
# D. 648# σ stale fix 関連: spread guard 前の σ refresh
# ═══════════════════════════════════════════════════════════════════════


class TestSigmaRefreshOrder:
    """648# _estimate_sigma が _enforce_spread_guards の前に呼ばれることの検証."""

    def test_sigma_refresh_before_spread_guard(self) -> None:
        """compute() 内で _estimate_sigma が _enforce_spread_guards より前."""
        from tests.unit.v460._fill_test_source import MAKER_PRICE, read_source_text

        src = read_source_text(MAKER_PRICE)
        idx_sigma = src.index("self._estimate_sigma(spread, mid_price)")
        idx_guard = src.index("self._enforce_spread_guards(")
        assert idx_sigma < idx_guard, (
            "_estimate_sigma must be called before _enforce_spread_guards"
        )
