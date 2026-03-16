"""273# テスト: 268# I3/I5/I6 対策.

- I5: DynamicKillManager の max_kill_duration_sec 時間上限
- I3: DailyDrawdownGuard.untick_side_halt() 空サイクル halt カウント除外
- I6: CycleGateAggregator halt_recovery_active ソフトゲート grace period
- Pattern B: kill↔halt 相互ロック防止 (I5 + I3 の組み合わせ)
"""

from __future__ import annotations

import time

import pytest

from ztb.risk.sell_dynamic_kill import (
    DynamicKillConfig,
    DynamicKillManager,
    DynamicKillTelemetry,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I5: Kill Time Limit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKillTimeLimit:
    """273# I5: max_kill_duration_sec による kill 自動解除."""

    def _make_mgr(self, duration_sec: float = 60.0) -> DynamicKillManager:
        return DynamicKillManager(
            DynamicKillConfig(
                window=3,
                threshold_bps=-0.5,
                resume_window=100,  # long cooldown — time limit が先に来るはず
                max_kill_duration_sec=duration_sec,
                max_stale_kill_cycles=0,  # probe 無効
            ),
            side="sell",
        )

    def test_kill_expires_after_duration(self) -> None:
        """kill 発動後、max_kill_duration_sec 経過で自動解除."""
        mgr = self._make_mgr(duration_sec=60.0)
        # kill を発動させる
        for _ in range(3):
            mgr.track(-1.0)
        killed, _ = mgr.check_kill()
        assert killed is True

        # kill 起動時刻だけを直接古くして、module-level time patch を避ける
        mgr._kill_activated_at = time.time() - 61
        killed, telem = mgr.check_kill()
        assert killed is False, "kill should auto-expire after duration"

    def test_kill_persists_before_duration(self) -> None:
        """max_kill_duration_sec 未満では kill 持続."""
        mgr = self._make_mgr(duration_sec=60.0)
        for _ in range(3):
            mgr.track(-1.0)
        killed, _ = mgr.check_kill()
        assert killed is True

        # 発動時刻を現在に寄せたまま再評価 → まだ kill
        mgr._kill_activated_at = time.time()
        killed, _ = mgr.check_kill()
        assert killed is True, "kill should persist before duration limit"

    def test_kill_disabled_when_zero(self) -> None:
        """max_kill_duration_sec=0 → 時間上限無効 (従来互換)."""
        mgr = DynamicKillManager(
            DynamicKillConfig(
                window=3,
                threshold_bps=-0.5,
                resume_window=100,
                max_kill_duration_sec=0.0,
                max_stale_kill_cycles=0,
            ),
            side="sell",
        )
        for _ in range(3):
            mgr.track(-1.0)
        killed, _ = mgr.check_kill()
        assert killed is True

        # 無制限モードでは古い起動時刻でも kill 持続
        mgr._kill_activated_at = 0.0
        killed, _ = mgr.check_kill()
        assert killed is True

    def test_track_resets_kill_timestamp(self) -> None:
        """新データ (track) が来ると kill timestamp がリセット."""
        mgr = self._make_mgr(duration_sec=60.0)
        for _ in range(3):
            mgr.track(-1.0)
        killed, _ = mgr.check_kill()
        assert killed is True
        assert mgr._kill_activated_at is not None

        # track で新データ投入 → timestamp リセット
        mgr.track(0.1)
        assert mgr._kill_activated_at is None

    def test_reset_clears_kill_timestamp(self) -> None:
        """reset() が kill timestamp をクリア."""
        mgr = self._make_mgr(duration_sec=60.0)
        for _ in range(3):
            mgr.track(-1.0)
        mgr.check_kill()
        assert mgr._kill_activated_at is not None

        mgr.reset()
        assert mgr._kill_activated_at is None

    def test_export_import_preserves_kill_timestamp(self) -> None:
        """export/import で kill timestamp が保持される."""
        mgr = self._make_mgr(duration_sec=60.0)
        for _ in range(3):
            mgr.track(-1.0)
        mgr.check_kill()

        state = mgr.export_state()
        assert "kill_activated_at" in state
        assert state["kill_activated_at"] is not None

        mgr2 = self._make_mgr(duration_sec=60.0)
        mgr2.import_state(state)
        assert mgr2._kill_activated_at == state["kill_activated_at"]

    def test_import_state_none_kill_timestamp(self) -> None:
        """kill_activated_at=None の state を import."""
        mgr = self._make_mgr(duration_sec=60.0)
        state = {"pnl_history": [], "kill_activated_at": None}
        mgr.import_state(state)
        assert mgr._kill_activated_at is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I3: untick_side_halt (空サイクル halt カウント除外)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestUntickSideHalt:
    """273# I3: 空サイクルの halt カウント除外."""

    def _make_guard(self, halt_cycles: int = 10):
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
        return DailyDrawdownGuard(
            enabled=True,
            per_side_enabled=True,
            per_side_hard_limit_bps=-5.0,
            per_side_halt_cycles=halt_cycles,
        )

    def test_untick_compensates_tick(self) -> None:
        """tick + untick → カウンタが元に戻る."""
        guard = self._make_guard(halt_cycles=10)
        # sell halt を発動
        guard.update_pnl(-6.0, side="sell")
        assert guard.is_side_halted("sell")
        remaining_before = guard._state.side_halt_remaining_sell

        # tick → decrement
        guard.tick_side_halt()
        remaining_after_tick = guard._state.side_halt_remaining_sell
        assert remaining_after_tick == remaining_before - 1

        # untick → compensate
        guard.untick_side_halt()
        remaining_after_untick = guard._state.side_halt_remaining_sell
        assert remaining_after_untick == remaining_before

    def test_untick_does_not_exceed_max(self) -> None:
        """untick は halt_cycles 以上にならない."""
        guard = self._make_guard(halt_cycles=10)
        guard.update_pnl(-6.0, side="sell")
        assert guard._state.side_halt_remaining_sell == 10

        # tick なしで untick → 10 のまま
        guard.untick_side_halt()
        assert guard._state.side_halt_remaining_sell == 10

    def test_untick_noop_when_not_halted(self) -> None:
        """halt されていない場合は untick 無効."""
        guard = self._make_guard(halt_cycles=10)
        assert not guard.is_side_halted("sell")
        # 例外なし
        guard.untick_side_halt()

    def test_untick_buy_side(self) -> None:
        """buy 側でも untick が正しく動作."""
        guard = self._make_guard(halt_cycles=5)
        guard.update_pnl(-6.0, side="buy")
        assert guard.is_side_halted("buy")

        guard.tick_side_halt()
        remaining = guard._state.side_halt_remaining_buy
        guard.untick_side_halt()
        assert guard._state.side_halt_remaining_buy == remaining + 1

    def test_deadlock_cycles_preserved_with_untick(self) -> None:
        """デッドロック中の N 空サイクルで untick すると halt 期間が保持される."""
        guard = self._make_guard(halt_cycles=5)
        guard.update_pnl(-6.0, side="sell")
        assert guard._state.side_halt_remaining_sell == 5

        # 5 回の空サイクル: tick + untick で補償
        for _ in range(5):
            guard.tick_side_halt()
            guard.untick_side_halt()

        # halt は解除されていない (カウンタが保持)
        assert guard.is_side_halted("sell")
        assert guard._state.side_halt_remaining_sell == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I6: is_in_recovery & halt_recovery_active grace period
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestIsInRecovery:
    """273# DailyDrawdownGuard.is_in_recovery()."""

    def _make_guard(self, recovery_cycles: int = 3):
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
        return DailyDrawdownGuard(
            enabled=True,
            per_side_enabled=True,
            per_side_hard_limit_bps=-5.0,
            per_side_halt_cycles=2,
            per_side_recovery_cycles=recovery_cycles,
            per_side_recovery_lot_scale=0.5,
        )

    def test_not_in_recovery_before_halt(self) -> None:
        """halt 前は recovery でない."""
        guard = self._make_guard()
        assert guard.is_in_recovery("sell") is False

    def test_in_recovery_after_halt_release(self) -> None:
        """halt 解除後は recovery 中."""
        guard = self._make_guard(recovery_cycles=3)
        guard.update_pnl(-6.0, side="sell")
        assert guard.is_side_halted("sell")

        # halt を消化 (2 cycles)
        guard.tick_side_halt()
        guard.tick_side_halt()
        assert not guard.is_side_halted("sell")
        assert guard.is_in_recovery("sell") is True

    def test_recovery_ends_after_consume(self) -> None:
        """recovery cycles を consume すると recovery 終了."""
        guard = self._make_guard(recovery_cycles=2)
        guard.update_pnl(-6.0, side="sell")
        guard.tick_side_halt()
        guard.tick_side_halt()

        assert guard.is_in_recovery("sell") is True
        guard.consume_recovery_cycle("sell")
        guard.consume_recovery_cycle("sell")
        assert guard.is_in_recovery("sell") is False


class TestHaltRecoveryGraceInGate:
    """273# I6: CycleGateAggregator で halt_recovery_active grace period."""

    def _make_gate(self):
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator
        config = FillTestConfig(
            skip_sell_trending=True,
        )
        return CycleGateAggregator(config)

    def test_soft_gate_blocks_normally(self) -> None:
        """通常時はソフトゲートがブロックする."""
        gate = self._make_gate()
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=None,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            halt_recovery_active=False,
        )
        # trending_sell_skip がブロックするはず
        assert result.blocked is True

    def test_soft_gate_bypassed_during_recovery(self) -> None:
        """halt_recovery_active=True でソフトゲートがバイパスされる."""
        gate = self._make_gate()
        result = gate.evaluate(
            side="sell",
            regime="trending_up",
            vol_ratio=None,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=False,
            halt_recovery_active=True,
        )
        # trending_sell_skip がバイパスされる
        assert result.blocked is False

    def test_hard_gate_still_blocks_during_recovery(self) -> None:
        """halt_recovery_active=True でもハードゲート (kill) はブロックする."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator
        config = FillTestConfig(
            sell_dynamic_kill_enabled=True,  # kill gate を有効化
        )
        gate = CycleGateAggregator(config)
        result = gate.evaluate(
            side="sell",
            regime="ranging",
            vol_ratio=None,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=True,  # sell kill active
            halt_recovery_active=True,
        )
        # kill gate はバイパスされない
        assert result.blocked is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pattern B: kill ↔ halt 相互ロック防止 統合テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPatternBMitigation:
    """268# Pattern B: kill↔halt 相互ロック防止の統合テスト."""

    def test_kill_time_limit_breaks_pattern_b(self) -> None:
        """kill 時間上限により、halt 中でも kill が自動解除される."""
        mgr = DynamicKillManager(
            DynamicKillConfig(
                window=3,
                threshold_bps=-0.5,
                resume_window=50,
                max_kill_duration_sec=120.0,  # 2分
                max_stale_kill_cycles=0,
            ),
            side="sell",
        )
        # kill 発動
        for _ in range(3):
            mgr.track(-1.0)
        killed, _ = mgr.check_kill()
        assert killed is True

        # halt 中: track は来ない (新データなし)
        # 2分経過 → kill 自動解除
        mgr._kill_activated_at = time.time() - 121
        killed, _ = mgr.check_kill()
        assert killed is False, "Pattern B: kill should auto-expire after time limit"

    def test_untick_preserves_halt_during_deadlock(self) -> None:
        """空サイクルの untick で halt が無駄に消費されない."""
        from scripts.v460.lib.daily_drawdown_guard import DailyDrawdownGuard
        guard = DailyDrawdownGuard(
            enabled=True,
            per_side_enabled=True,
            per_side_hard_limit_bps=-5.0,
            per_side_halt_cycles=10,
        )
        guard.update_pnl(-6.0, side="sell")
        assert guard.is_side_halted("sell")

        # 10 空サイクル: tick + untick
        for _ in range(10):
            guard.tick_side_halt()
            guard.untick_side_halt()

        # halt は保持 (10 サイクル分のカウンタが温存)
        assert guard.is_side_halted("sell")
        assert guard._state.side_halt_remaining_sell > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config wiring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestConfigWiring:
    """273# YAML → FillTestConfig → DynamicKillConfig の配線テスト."""

    def test_fill_config_has_max_duration_fields(self) -> None:
        """FillTestConfig に max_duration_sec フィールドがある."""
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "sell_dynamic_kill_max_duration_sec")
        assert hasattr(cfg, "buy_dynamic_kill_max_duration_sec")
        assert cfg.sell_dynamic_kill_max_duration_sec == 1800.0  # 336# drift fix
        assert cfg.buy_dynamic_kill_max_duration_sec == 1800.0  # 336# drift fix

    def test_dynamic_kill_config_has_max_duration(self) -> None:
        """DynamicKillConfig に max_kill_duration_sec フィールドがある."""
        cfg = DynamicKillConfig()
        assert hasattr(cfg, "max_kill_duration_sec")
        assert cfg.max_kill_duration_sec == 0.0

    def test_yaml_parsing_max_kill_duration(self) -> None:
        """YAML から max_kill_duration_sec が読み込まれる."""
        from scripts.v460.lib.fill_config import FillTestConfig
        yaml_data = {
            "止血": {
                "sell_dynamic_kill": {
                    "enabled": True,
                    "max_kill_duration_sec": 1800,
                },
                "buy_dynamic_kill": {
                    "enabled": True,
                    "max_kill_duration_sec": 900,
                },
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.sell_dynamic_kill_max_duration_sec == 1800.0
        assert cfg.buy_dynamic_kill_max_duration_sec == 900.0
