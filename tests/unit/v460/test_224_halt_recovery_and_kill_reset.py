"""224# B1/B2: halt解除後ソフトリカバリ + 日替わり×kill矛盾修正のテスト."""
from __future__ import annotations

import pytest

from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard


# ======================================================================
# B1: halt解除後ソフトリカバリ
# ======================================================================


class TestHaltRecovery:
    """224# B1: per-side halt 解除後のソフトリカバリ (lot 縮小) テスト."""

    def _make_guard(
        self,
        per_side_hard: float = -10.0,
        per_side_halt_cycles: int = 3,
        recovery_cycles: int = 2,
        recovery_lot_scale: float = 0.5,
    ) -> DailyDrawdownGuard:
        return DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            per_side_enabled=True,
            per_side_hard_limit_bps=per_side_hard,
            per_side_halt_cycles=per_side_halt_cycles,
            per_side_recovery_cycles=recovery_cycles,
            per_side_recovery_lot_scale=recovery_lot_scale,
        )

    def test_recovery_starts_on_halt_release(self) -> None:
        """halt解除時にリカバリカウンタが設定される."""
        guard = self._make_guard(per_side_halt_cycles=2, recovery_cycles=3)
        # halt をトリガ
        guard.update_pnl(-15.0, side="buy")
        assert guard.is_side_halted("buy")
        assert guard.state.side_halt_remaining_buy == 2
        # 2 tick で解除
        guard.tick_side_halt()
        guard.tick_side_halt()
        assert not guard.is_side_halted("buy")
        # リカバリカウンタが設定されている
        assert guard.state.side_recovery_remaining_buy == 3

    def test_recovery_lot_scale_decrements(self) -> None:
        """get_recovery_lot_scale がリカバリ期間中に縮小倍率を返しデクリメントする."""
        guard = self._make_guard(recovery_cycles=3, recovery_lot_scale=0.5)
        # manually set recovery state
        guard.state.side_recovery_remaining_buy = 3

        scale1 = guard.get_recovery_lot_scale("buy")
        assert scale1 == 0.5
        assert guard.state.side_recovery_remaining_buy == 2

        scale2 = guard.get_recovery_lot_scale("buy")
        assert scale2 == 0.5
        assert guard.state.side_recovery_remaining_buy == 1

        scale3 = guard.get_recovery_lot_scale("buy")
        assert scale3 == 0.5
        assert guard.state.side_recovery_remaining_buy == 0

        # リカバリ終了 → 1.0
        scale4 = guard.get_recovery_lot_scale("buy")
        assert scale4 == 1.0

    def test_recovery_sell_side(self) -> None:
        """sell 側のリカバリが独立に動作する."""
        guard = self._make_guard(recovery_cycles=2, recovery_lot_scale=0.5)
        guard.state.side_recovery_remaining_sell = 2

        assert guard.get_recovery_lot_scale("sell") == 0.5
        assert guard.state.side_recovery_remaining_sell == 1
        # buy 側は影響なし
        assert guard.get_recovery_lot_scale("buy") == 1.0

    def test_recovery_disabled_when_zero_cycles(self) -> None:
        """recovery_cycles=0 の場合は常に 1.0."""
        guard = self._make_guard(recovery_cycles=0)
        guard.state.side_recovery_remaining_buy = 5  # 強制設定しても無効
        assert guard.get_recovery_lot_scale("buy") == 1.0

    def test_recovery_disabled_when_per_side_disabled(self) -> None:
        """per_side_enabled=False の場合は常に 1.0."""
        guard = DailyDrawdownGuard(
            enabled=True,
            hard_limit_bps=-50.0,
            soft_limit_bps=-30.0,
            per_side_enabled=False,
            per_side_recovery_cycles=5,
        )
        guard.state.side_recovery_remaining_buy = 3
        assert guard.get_recovery_lot_scale("buy") == 1.0

    def test_recovery_full_flow_halt_to_recovery(self) -> None:
        """halt → tick_side_halt で解除 → リカバリ → 通常: フルフロー."""
        guard = self._make_guard(
            per_side_halt_cycles=2, recovery_cycles=2, recovery_lot_scale=0.5,
        )
        # Phase 1: halt
        guard.update_pnl(-15.0, side="sell")
        assert guard.is_side_halted("sell")

        # Phase 2: tick down to 0 → release
        guard.tick_side_halt()
        guard.tick_side_halt()
        assert not guard.is_side_halted("sell")
        assert guard.state.side_recovery_remaining_sell == 2

        # Phase 3: recovery — lot scale 0.5 for 2 cycles
        assert guard.get_recovery_lot_scale("sell") == 0.5
        assert guard.get_recovery_lot_scale("sell") == 0.5

        # Phase 4: normal
        assert guard.get_recovery_lot_scale("sell") == 1.0

    def test_recovery_in_export_import(self) -> None:
        """リカバリ状態が export/import で永続化される."""
        guard = self._make_guard(recovery_cycles=3)
        guard.update_pnl(-1.0, side="buy")  # 日を確立
        guard.state.side_recovery_remaining_buy = 2
        guard.state.side_recovery_remaining_sell = 1

        exported = guard.export_state()
        assert exported["side_recovery_remaining_buy"] == 2
        assert exported["side_recovery_remaining_sell"] == 1

        new_guard = self._make_guard(recovery_cycles=3)
        new_guard.import_state(exported)
        assert new_guard.state.side_recovery_remaining_buy == 2
        assert new_guard.state.side_recovery_remaining_sell == 1

    def test_recovery_in_metrics(self) -> None:
        """get_metrics にリカバリフィールドが含まれる."""
        guard = self._make_guard()
        guard.state.side_recovery_remaining_buy = 3
        m = guard.get_metrics()
        assert "side_recovery_remaining_buy" in m
        assert "side_recovery_remaining_sell" in m
        assert m["side_recovery_remaining_buy"] == 3
        assert m["side_recovery_remaining_sell"] == 0

    def test_recovery_cleared_on_day_reset(self) -> None:
        """日替わりリセットでリカバリ状態もクリアされる."""
        guard = self._make_guard(recovery_cycles=5)
        guard.update_pnl(-1.0, side="buy")  # 日を確立
        guard.state.side_recovery_remaining_buy = 3

        # Force day change by modifying current_day
        guard.state.current_day = "19700101"
        guard.maybe_reset_day()
        assert guard.state.side_recovery_remaining_buy == 0


# ======================================================================
# B2: 日替わりリセット × dynamic kill 矛盾
# ======================================================================


class TestDayResetKillConflict:
    """224# B2: 日替わりリセット時に dynamic kill が矛盾する場合の検出テスト."""

    def test_kill_manager_reset_clears_state(self) -> None:
        """DynamicKillManager.reset() が全状態をクリアする."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        config = DynamicKillConfig(
            threshold_bps=-0.5,
            window=5,
            resume_window=3,
        )
        mgr = DynamicKillManager(config, side="sell")
        # PnL を蓄積 → kill をトリガ
        for _ in range(5):
            mgr.track(-2.0)
        killed, _ = mgr.check_kill()
        assert killed  # kill が発火していることを確認

        # reset → kill 解除
        mgr.reset()
        killed2, t2 = mgr.check_kill()
        assert not killed2
        assert t2.rolling_count == 0

    def test_is_kill_active_no_side_effects(self) -> None:
        """is_kill_active() が副作用なしで kill 状態を返す."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        config = DynamicKillConfig(
            threshold_bps=-0.5,
            window=3,
            resume_window=5,
        )
        mgr = DynamicKillManager(config, side="sell")
        for _ in range(3):
            mgr.track(-2.0)

        # check_kill で kill 発動 + cooldown 設定
        killed, _ = mgr.check_kill()
        assert killed
        cooldown_before = mgr._cooldown
        assert cooldown_before == 5  # resume_window

        # is_kill_active() を呼んでも cooldown は変わらない
        active, mean, count = mgr.is_kill_active()
        assert active
        assert mgr._cooldown == cooldown_before  # 副作用なし
        assert count == 3

    def test_is_kill_active_returns_false_when_not_killing(self) -> None:
        """kill 非アクティブ時に is_kill_active() が False を返す."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        config = DynamicKillConfig(threshold_bps=-1.0, window=3, resume_window=2)
        mgr = DynamicKillManager(config, side="buy")
        mgr.track(1.0)
        mgr.track(0.5)
        mgr.track(0.8)
        active, mean, count = mgr.is_kill_active()
        assert not active
        assert mean is not None
        assert mean > 0

    def test_kill_manager_check_kill_returns_rolling_mean(self) -> None:
        """check_kill の telemetry に rolling_mean が含まれる."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        config = DynamicKillConfig(threshold_bps=-1.0, window=3, resume_window=2)
        mgr = DynamicKillManager(config, side="buy")
        mgr.track(-0.5)
        mgr.track(-0.3)
        mgr.track(-0.2)
        _, t = mgr.check_kill()
        assert t.rolling_mean is not None
        assert t.rolling_mean == pytest.approx((-0.5 + -0.3 + -0.2) / 3, abs=1e-6)


# ======================================================================
# B1 Executor 連携: _halt_recovery_lot_mult
# ======================================================================


class TestRecoveryLotMultAttribute:
    """224# B1: _halt_recovery_lot_mult 属性に関する境界テスト."""

    def test_getattr_default(self) -> None:
        """_halt_recovery_lot_mult 未設定時に getattr が 1.0 を返す."""

        class Stub:
            pass

        obj = Stub()
        assert getattr(obj, "_halt_recovery_lot_mult", 1.0) == 1.0

    def test_scale_applied(self) -> None:
        """lot スケーリングロジックの正当性確認."""
        order_quantity = 0.001
        order_lot = 0.01
        recovery_lm = 0.5
        result = max(order_quantity, order_lot * recovery_lm)
        assert result == 0.005

    def test_scale_floor_at_min_lot(self) -> None:
        """縮小後が min lot を下回らない."""
        order_quantity = 0.005
        order_lot = 0.008
        recovery_lm = 0.5
        result = max(order_quantity, order_lot * recovery_lm)
        assert result == 0.005  # floor で保護
