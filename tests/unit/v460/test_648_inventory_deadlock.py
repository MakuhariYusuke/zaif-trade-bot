"""648# Inventory Deadlock Detection テスト.

片側 preflight_insufficient + 反対側 no_feasible_quote の cross-cycle 膠着を
検出する仕組みの動作を検証する。
648# Part 3: 低優先度改善のテストも含む。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from tests.unit.v460._fill_test_source import (
    CONFIG_HOT_RELOAD,
    FILL_LOOP_ORCHESTRATOR,
    MAKER_MICROSTRUCTURE,
    ORCHESTRATOR_BALANCE,
    ORCHESTRATOR_POST_CYCLE,
    ORCHESTRATOR_PRE_CYCLE,
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


# ═══════════════════════════════════════════════════════════════════════
# E. 648# Part 3: preflight_pause_count 日替わりリセット
# ═══════════════════════════════════════════════════════════════════════


class TestPreflightPauseCountDailyReset:
    """648# _preflight_pause_count が日替わりでリセットされることの検証."""

    def test_reset_in_daily_reset(self) -> None:
        """_process_daily_reset 内で preflight_pause_count がリセットされる."""
        src = read_source_text(ORCHESTRATOR_PRE_CYCLE)
        assert "_preflight_pause_count = 0" in src
        # リセットが _process_daily_reset メソッド内あること
        idx_method = src.index("def _process_daily_reset")
        idx_reset = src.index("_preflight_pause_count = 0")
        # 次のメソッド定義より前にあること
        next_def = src.find("\n    def ", idx_method + 1)
        assert idx_method < idx_reset < next_def

    def test_class_level_attribute_declared(self) -> None:
        """fill_loop_orchestrator にクラスレベル属性が宣言されている."""
        src = read_source_text(FILL_LOOP_ORCHESTRATOR)
        assert "_preflight_pause_count: int = 0" in src


# ═══════════════════════════════════════════════════════════════════════
# F. 648# Part 3: Parkinson σ 窓境界フォールバック改善
# ═══════════════════════════════════════════════════════════════════════


class TestParkinsonSigmaWindowBoundary:
    """648# 窓リセット直後の Roll proxy フォールバックではなく前窓 σ を使用."""

    def test_prev_sigma_fallback_in_source(self) -> None:
        """H==L (窓リセット直後) で _prev_sigma を使用するロジックが存在."""
        src = read_source_text(MAKER_MICROSTRUCTURE)
        assert "_prev_sigma" in src
        assert "_prev_sigma = self._last_sigma" in src

    def test_prev_sigma_divides_by_vol_ratio(self) -> None:
        """前窓 σ は vol_ratio で除算して raw σ に戻してから再適用."""
        src = read_source_text(MAKER_MICROSTRUCTURE)
        assert "_prev_sigma / vol_ratio" in src

    def test_roll_proxy_only_on_initial(self) -> None:
        """初回 (_prev_sigma == 0) のみ Roll proxy にフォールバック."""
        src = read_source_text(MAKER_MICROSTRUCTURE)
        # _prev_sigma > 0 のチェックが存在
        assert "_prev_sigma > 0" in src
        # else 句で Roll proxy が使われる
        idx_check = src.index("_prev_sigma > 0")
        remainder = src[idx_check:idx_check + 300]
        assert "spread / (2.0 * mid_price)" in remainder


# ═══════════════════════════════════════════════════════════════════════
# G. 648# Part 3: deadlock config の Hot-Reload 登録
# ═══════════════════════════════════════════════════════════════════════


class TestDeadlockConfigHotReload:
    """648# inventory_deadlock config が hot-reload 対象に登録されている."""

    def test_threshold_in_hot_reloadable(self) -> None:
        src = read_source_text(CONFIG_HOT_RELOAD)
        assert '"inventory_deadlock_threshold"' in src

    def test_alert_interval_in_hot_reloadable(self) -> None:
        src = read_source_text(CONFIG_HOT_RELOAD)
        assert '"inventory_deadlock_alert_interval_sec"' in src
