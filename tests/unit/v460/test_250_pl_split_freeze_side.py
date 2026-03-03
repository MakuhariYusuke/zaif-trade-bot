"""250# P/L 3分離, freeze side 紐付け, quiescence deadlock 防御テスト.

P0 items:
1. P/L 3分離追跡基盤 — adverse selection 累積カウント/bps (248# P1-5)
2. freeze/cooldown side 紐付け — 反対 side 通過 (247# P1-4)
3. quiescence + balance_forced deadlock 防御 (code sweep finding)
4. probe 廃止コメント補強 (247# P1-6) — テスト不要
5. rearm halt_triggered_at 防御コメント — テスト不要
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.cycle_gate_aggregator import CycleGateAggregator
from scripts.v460.lib.fill_config import FillTestConfig


# =============================================================
# 1. P/L 3分離: FillRecord の adverse_selected 累積ロジック
# =============================================================


class TestPLSplit250:
    """250# adverse selection 累積カウントのレジューム復元ロジック.

    fill_loop_orchestrator の resume loop 内ロジックを直接模倣してテスト。
    実機では clean_records をループして累積値を計算。
    """

    @staticmethod
    def _make_record(
        *,
        filled: bool = True,
        adverse_selected: bool | None = None,
        post_fill_30s_pnl: float | None = None,
        side: str = "buy",
        order_quantity: float = 0.001,
    ):
        """最小 FillRecord ダミー."""
        from ztb.metrics.fill_quality import FillRecord

        return FillRecord(
            cycle_id="test",
            timestamp=1000.0,
            side=side,
            order_price=14_000_000.0,
            order_quantity=order_quantity,
            filled=filled,
            adverse_selected=adverse_selected,
            post_fill_30s_pnl=post_fill_30s_pnl,
        )

    @staticmethod
    def _compute_adverse_stats(records):
        """fill_loop_orchestrator の resume ループと同一ロジック."""
        cumulative_adverse_count = 0
        cumulative_adverse_bps = 0.0
        for r in records:
            if r.filled and r.adverse_selected is True:
                cumulative_adverse_count += 1
                if r.post_fill_30s_pnl is not None:
                    cumulative_adverse_bps += r.post_fill_30s_pnl
        return cumulative_adverse_count, cumulative_adverse_bps

    def test_no_adverse_fills(self) -> None:
        """adverse_selected=False のみ → カウント 0."""
        records = [
            self._make_record(adverse_selected=False, post_fill_30s_pnl=0.5),
            self._make_record(adverse_selected=False, post_fill_30s_pnl=0.3),
        ]
        count, bps = self._compute_adverse_stats(records)
        assert count == 0
        assert bps == pytest.approx(0.0)

    def test_all_adverse_fills(self) -> None:
        """全て adverse → 全カウント + bps 合計."""
        records = [
            self._make_record(adverse_selected=True, post_fill_30s_pnl=-1.5),
            self._make_record(adverse_selected=True, post_fill_30s_pnl=-0.8),
        ]
        count, bps = self._compute_adverse_stats(records)
        assert count == 2
        assert bps == pytest.approx(-2.3)

    def test_mixed_fills(self) -> None:
        """混在: adverse のみカウント."""
        records = [
            self._make_record(adverse_selected=True, post_fill_30s_pnl=-1.0),
            self._make_record(adverse_selected=False, post_fill_30s_pnl=0.5),
            self._make_record(adverse_selected=True, post_fill_30s_pnl=-0.3),
            self._make_record(adverse_selected=None, post_fill_30s_pnl=-0.2),
        ]
        count, bps = self._compute_adverse_stats(records)
        assert count == 2
        assert bps == pytest.approx(-1.3)

    def test_unfilled_record_excluded(self) -> None:
        """filled=False は adverse_selected=True でもカウントしない."""
        records = [
            self._make_record(filled=False, adverse_selected=True, post_fill_30s_pnl=-1.0),
            self._make_record(filled=True, adverse_selected=True, post_fill_30s_pnl=-0.5),
        ]
        count, bps = self._compute_adverse_stats(records)
        assert count == 1
        assert bps == pytest.approx(-0.5)

    def test_adverse_with_none_pnl(self) -> None:
        """adverse=True だが pnl=None → カウントされるが bps には加算しない."""
        records = [
            self._make_record(adverse_selected=True, post_fill_30s_pnl=None),
            self._make_record(adverse_selected=True, post_fill_30s_pnl=-1.0),
        ]
        count, bps = self._compute_adverse_stats(records)
        assert count == 2
        assert bps == pytest.approx(-1.0)

    def test_incremental_tracking(self) -> None:
        """250# インクリメンタル追跡: 新規 fill を 1 件ずつ追加."""
        # fill_loop_orchestrator の run_loop 内ロジック模倣
        cumulative_adverse_count = 0
        cumulative_adverse_bps = 0.0

        # 1st fill: adverse
        r1 = self._make_record(adverse_selected=True, post_fill_30s_pnl=-2.0)
        if r1.adverse_selected is True:
            cumulative_adverse_count += 1
            if r1.post_fill_30s_pnl is not None:
                cumulative_adverse_bps += r1.post_fill_30s_pnl
        assert cumulative_adverse_count == 1
        assert cumulative_adverse_bps == pytest.approx(-2.0)

        # 2nd fill: not adverse
        r2 = self._make_record(adverse_selected=False, post_fill_30s_pnl=0.3)
        if r2.adverse_selected is True:
            cumulative_adverse_count += 1
        assert cumulative_adverse_count == 1  # no change

        # 3rd fill: adverse
        r3 = self._make_record(adverse_selected=True, post_fill_30s_pnl=-0.7)
        if r3.adverse_selected is True:
            cumulative_adverse_count += 1
            if r3.post_fill_30s_pnl is not None:
                cumulative_adverse_bps += r3.post_fill_30s_pnl
        assert cumulative_adverse_count == 2
        assert cumulative_adverse_bps == pytest.approx(-2.7)


# =============================================================
# 2. freeze/cooldown side 紐付け
# =============================================================


class TestFreezeSideTracking250:
    """250# P1-4: freeze/cooldown は紐付け side のみ block.

    fill_loop_orchestrator のロジックを直接テストする:
    - _one_sided_frozen_side が設定されていれば、その side のみ block
    - 反対 side は pass through
    - None (未設定) は全 side block (従来互換)
    """

    @staticmethod
    def _should_skip_freeze(
        frozen_side: str | None,
        current_side: str,
        freeze_remaining: int,
    ) -> bool:
        """fill_loop_orchestrator の freeze skip 判定ロジックを再現."""
        if freeze_remaining <= 0:
            return False
        if frozen_side is None or frozen_side == current_side:
            return True  # blocked
        return False  # opposite side passes through

    @staticmethod
    def _should_skip_cooldown(
        frozen_side: str | None,
        current_side: str,
        cooldown_remaining: int,
    ) -> bool:
        """fill_loop_orchestrator の cooldown skip 判定ロジックを再現."""
        if cooldown_remaining <= 0:
            return False
        if frozen_side is None or frozen_side == current_side:
            return True
        return False

    # --- Freeze tests ---

    def test_freeze_blocks_frozen_side(self) -> None:
        """frozen_side=buy → buy は block."""
        assert self._should_skip_freeze("buy", "buy", 5)

    def test_freeze_passes_opposite_side(self) -> None:
        """frozen_side=buy → sell は pass through."""
        assert not self._should_skip_freeze("buy", "sell", 5)

    def test_freeze_blocks_sell_frozen(self) -> None:
        """frozen_side=sell → sell は block."""
        assert self._should_skip_freeze("sell", "sell", 5)

    def test_freeze_passes_buy_when_sell_frozen(self) -> None:
        """frozen_side=sell → buy は pass through."""
        assert not self._should_skip_freeze("sell", "buy", 5)

    def test_freeze_none_blocks_all(self) -> None:
        """frozen_side=None (従来互換) → 全 side block."""
        assert self._should_skip_freeze(None, "buy", 5)
        assert self._should_skip_freeze(None, "sell", 5)

    def test_freeze_zero_remaining_no_block(self) -> None:
        """remaining=0 → block しない (freeze 消化済み)."""
        assert not self._should_skip_freeze("buy", "buy", 0)
        assert not self._should_skip_freeze(None, "buy", 0)

    # --- Cooldown tests ---

    def test_cooldown_blocks_frozen_side(self) -> None:
        """frozen_side=buy → buy は cooldown block."""
        assert self._should_skip_cooldown("buy", "buy", 3)

    def test_cooldown_passes_opposite_side(self) -> None:
        """frozen_side=buy → sell は pass through."""
        assert not self._should_skip_cooldown("buy", "sell", 3)

    def test_cooldown_none_blocks_all(self) -> None:
        """frozen_side=None → 全 side cooldown block."""
        assert self._should_skip_cooldown(None, "buy", 3)
        assert self._should_skip_cooldown(None, "sell", 3)


# =============================================================
# 3. quiescence + balance_forced deadlock 防御
# =============================================================


class TestQuiescenceDeadlockDefense250:
    """250# quiescence + balance_forced 時に degraded liquidation を許容.

    cycle_gate_aggregator の evaluate():
    - quiescence=True + dual_kill + balance_forced=True + degraded=True
      → quiescence を緩和 (degraded 清算で rescue)
    - quiescence=True + dual_kill + balance_forced=False
      → 従来通り pure quiescence (resting)
    """

    @staticmethod
    def _make_gate(
        *,
        quiescence: bool = True,
        degraded: bool = True,
    ) -> CycleGateAggregator:
        cfg = FillTestConfig(
            buy_dynamic_kill_enabled=True,
            sell_dynamic_kill_enabled=True,
            dual_kill_quiescence_enabled=quiescence,
            degraded_liquidation_enabled=degraded,
            # 先行ゲートを無効化して dual_kill 到達を保証
            skip_buy_unknown_regime=False,
            skip_sell_unknown_regime=False,
            skip_ranging_buy_low_vol=False,
            skip_sell_trending=False,
            sell_velocity_skip_enabled=False,
            buy_velocity_skip_enabled=False,
        )
        return CycleGateAggregator(cfg)

    def test_quiescence_pure_resting_no_balance_forced(self) -> None:
        """balance_forced=False → pure quiescence → blocked."""
        gate = self._make_gate(quiescence=True, degraded=True)
        result = gate.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=0.5,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=True,
            is_sell_killed=True,
        )
        assert result.blocked
        assert not result.dual_kill_bypassed

    def test_quiescence_balance_forced_degraded_allows_cycle(self) -> None:
        """balance_forced=True + degraded=True → quiescence 緩和.

        kill gate が block するが、degraded_liquidation=True で
        result.blocked=False (縮退清算パスで通過)。
        """
        gate = self._make_gate(quiescence=True, degraded=True)
        result = gate.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=0.5,
            balance_forced=True,
            inv_net_imbalance=0.0,
            is_buy_killed=True,
            is_sell_killed=True,
        )
        # degraded_liquidation が有効なので blocked=False (縮退清算パス)
        assert not result.blocked
        assert not result.dual_kill_bypassed
        # degraded_liquidation が True になることを確認
        assert result.degraded_liquidation

    def test_quiescence_balance_forced_no_degraded(self) -> None:
        """balance_forced=True + degraded=False → pure quiescence (no degraded available)."""
        gate = self._make_gate(quiescence=True, degraded=False)
        result = gate.evaluate(
            side="buy",
            regime="ranging",
            vol_ratio=0.5,
            balance_forced=True,
            inv_net_imbalance=0.0,
            is_buy_killed=True,
            is_sell_killed=True,
        )
        assert result.blocked
        assert not result.dual_kill_bypassed

    def test_legacy_bypass_still_works(self) -> None:
        """quiescence=False → 旧挙動で dual_kill_bypass 発動."""
        gate = self._make_gate(quiescence=False)
        result = gate.evaluate(
            side="sell",
            regime="ranging",
            vol_ratio=0.5,
            balance_forced=True,
            inv_net_imbalance=0.0,
            is_buy_killed=True,
            is_sell_killed=True,
        )
        assert not result.blocked
        assert result.dual_kill_bypassed

    def test_single_kill_unaffected(self) -> None:
        """sell killed + balance_forced → degraded 縮退清算で通過."""
        gate = self._make_gate(quiescence=True, degraded=True)
        result = gate.evaluate(
            side="sell",
            regime="ranging",
            vol_ratio=0.5,
            balance_forced=True,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=True,
        )
        # sell killed + balance_forced + degraded → degraded で blocked=False
        assert not result.blocked
        assert not result.dual_kill_bypassed
        assert result.degraded_liquidation

    def test_single_kill_no_balance_forced(self) -> None:
        """sell killed + balance_forced=False → blocked (no degraded rescue)."""
        gate = self._make_gate(quiescence=True, degraded=True)
        result = gate.evaluate(
            side="sell",
            regime="ranging",
            vol_ratio=0.5,
            balance_forced=False,
            inv_net_imbalance=0.0,
            is_buy_killed=False,
            is_sell_killed=True,
        )
        assert result.blocked
        assert not result.dual_kill_bypassed
        assert not result.degraded_liquidation


# =============================================================
# 4. probe 廃止コメント: max_stale_kill_cycles=0 で probe 無効化
# =============================================================


class TestProbeDisable250:
    """250# probe 廃止基盤: max_stale_kill_cycles=0 で probe 完全無効化."""

    def test_probe_disabled_zero(self) -> None:
        """max_stale_kill_cycles=0 → probe 発火しない."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        cfg = DynamicKillConfig(
            enabled=True,
            window=3,
            threshold_bps=-1.0,
            resume_window=2,
            max_stale_kill_cycles=0,  # probe 無効
        )
        mgr = DynamicKillManager(cfg, side="sell")

        # Kill 発動
        mgr.track(-2.0)
        mgr.track(-2.0)
        mgr.track(-2.0)
        killed, _ = mgr.check_kill()
        assert killed

        # 100 cycles stale — probe 発火なし (max_stale=0)
        for _ in range(100):
            killed, telem = mgr.check_kill()
            # cooldown 消化後は kill 維持 (probe なし)
            assert not telem.probe_fired

    def test_probe_enabled_default(self) -> None:
        """max_stale_kill_cycles=10 (default) → probe 発火あり."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        cfg = DynamicKillConfig(
            enabled=True,
            window=3,
            threshold_bps=-1.0,
            resume_window=2,
            max_stale_kill_cycles=10,
        )
        mgr = DynamicKillManager(cfg, side="sell")

        mgr.track(-2.0)
        mgr.track(-2.0)
        mgr.track(-2.0)

        # Kill + cooldown 消化
        probe_found = False
        for _ in range(50):
            killed, telem = mgr.check_kill()
            if telem.probe_fired:
                probe_found = True
                break
        assert probe_found, "probe should fire when max_stale_kill_cycles > 0"
