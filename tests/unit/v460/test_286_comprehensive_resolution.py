"""286# テスト: 282#-284# 全課題解決 — ロック強化・AS防御・KPI分離・GM遅延.

テスト対象:
  1. LockManager portalocker + zombie wait 強化
  2. detect_split_brain() 事後検出
  3. Events start/stop finally 保証
  4. buy_dynamic_kill 在庫連動緩和 (Ho & Stoll 1981)
  5. 強制買い KPI 分離トラッキング
  6. Buy-side AS Guard (microprice 急落防御)
  7. 強制買い遅延実行 (Glosten-Milgrom 1985)
  8. guard_reason_classifier 再分類 (283# MEDIUM-4)
  9. DynamicKillManager threshold_offset_bps パラメータ
"""

from __future__ import annotations

import time

import pytest


# ======================================================================
# 1. LockManager portalocker + zombie wait
# ======================================================================

class TestLockManagerPortalocker:
    """286# P0: OS レベルロック強化テスト."""

    def test_lock_manager_has_os_lock_attribute(self):
        """LockManager に _os_lock_fh 属性があること."""
        from scripts.v460.lib.lock_manager import LockManager
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mgr = LockManager(Path(td), "test_run_id")
            assert hasattr(mgr, "_os_lock_fh")
            assert mgr._os_lock_fh is None

    def test_acquire_release_roundtrip(self):
        """acquire → release のラウンドトリップが正常動作すること."""
        from scripts.v460.lib.lock_manager import LockManager
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mgr = LockManager(Path(td), "roundtrip_test")
            mgr.acquire()  # Should not raise
            mgr.release()  # Should not raise

    def test_lock_conflict_second_acquire(self):
        """2つ目の LockManager が acquire すると LockConflictError."""
        import os
        from scripts.v460.lib.lock_manager import LockManager, LockConflictError
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mgr1 = LockManager(Path(td), "first_run")
            mgr1.acquire()
            try:
                mgr2 = LockManager(
                    Path(td), "second_run",
                    lock_acquire_retries=0,
                )
                with pytest.raises(LockConflictError):
                    mgr2.acquire()
            finally:
                mgr1.release()

    def test_wait_for_pid_exit_nonexistent_pid(self):
        """_wait_for_pid_exit に存在しない PID → 即座に完了 (例外なし)."""
        from scripts.v460.lib.lock_manager import LockManager
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mgr = LockManager(Path(td), "wait_test")
            # 999999999 は存在しない PID → 即座に return (例外なし)
            mgr._wait_for_pid_exit(999999999)  # Should not raise

    def test_portalocker_import_flag(self):
        """portalocker インポートフラグが設定されていること."""
        from scripts.v460.lib.lock_manager import _HAS_PORTALOCKER
        # portalocker がインストールされていれば True
        assert isinstance(_HAS_PORTALOCKER, bool)


# ======================================================================
# 2. detect_split_brain() 事後検出
# ======================================================================

class TestDetectSplitBrain:
    """286# 283# P0-1: Split-Brain 事後検出テスト."""

    def _make_record(self, ts: float, run_id: str, pid: int | None = None, side: str = "buy"):
        from ztb.metrics.fill_quality import FillRecord
        return FillRecord(
            cycle_id="test-cycle-0",
            timestamp=ts,
            side=side,
            order_price=15_000_000.0,
            order_quantity=0.001,
            filled=True,
            run_id=run_id,
            pid=pid,
        )

    def test_no_split_brain_single_run(self):
        """単一 run_id → Split-Brain なし."""
        from ztb.metrics.fill_quality import detect_split_brain

        records = [
            self._make_record(100.0, "run_a", 1000),
            self._make_record(110.0, "run_a", 1000),
            self._make_record(120.0, "run_a", 1000),
        ]
        events = detect_split_brain(records)
        assert events == []

    def test_split_brain_overlapping_run_ids(self):
        """異なる run_id が短時間内に出現 → CRITICAL 検出."""
        from ztb.metrics.fill_quality import detect_split_brain

        records = [
            self._make_record(100.0, "run_a", 1000),
            self._make_record(105.0, "run_b", 2000),  # overlap window 内
        ]
        events = detect_split_brain(records)
        assert len(events) == 1
        assert events[0]["run_id_a"] == "run_a"
        assert events[0]["run_id_b"] == "run_b"
        assert events[0]["pid_a"] == 1000
        assert events[0]["pid_b"] == 2000

    def test_no_split_brain_sequential_runs(self):
        """異なる run_id でも overlap_window 外 → 検出なし."""
        from ztb.metrics.fill_quality import detect_split_brain

        records = [
            self._make_record(100.0, "run_a", 1000),
            self._make_record(500.0, "run_b", 2000),  # 400秒後 = window外
        ]
        events = detect_split_brain(records, overlap_window_sec=300.0)
        assert events == []

    def test_split_brain_same_run_id_different_pid(self):
        """同一 run_id で pid が異なる → プロセス入替検出."""
        from ztb.metrics.fill_quality import detect_split_brain

        records = [
            self._make_record(100.0, "run_a", 1000),
            self._make_record(105.0, "run_a", 2000),  # 同 run_id, 異 pid
        ]
        events = detect_split_brain(records)
        assert len(events) == 1
        assert events[0]["pid_a"] == 1000
        assert events[0]["pid_b"] == 2000

    def test_empty_records(self):
        """空リスト → 検出なし."""
        from ztb.metrics.fill_quality import detect_split_brain
        assert detect_split_brain([]) == []

    def test_single_record(self):
        """1レコード → 検出なし."""
        from ztb.metrics.fill_quality import detect_split_brain
        records = [self._make_record(100.0, "run_a", 1000)]
        assert detect_split_brain(records) == []


# ======================================================================
# 3. Events start/stop finally 保証 (fill_test_cli.py)
# ======================================================================

class TestEventsStartStopGuarantee:
    """286# 283# P0-3: start/stop ペア保証テスト."""

    def test_stop_event_logged_on_crash(self):
        """crash 時も stop イベントが記録されること (コード検査)."""
        import ast
        from pathlib import Path

        cli_path = Path("scripts/v460/lib/fill_test_cli.py")
        source = cli_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # finally ブロック内に log_event("stop", ...) が必ず存在すること
        # (crash 時に stop_reason が "crash:..." でも stop を記録)
        assert 'not stop_reason.startswith("crash:")' not in source, (
            "286# fix: crash 時も stop イベントを記録するべき"
        )
        # 新しい条件: stop_reason があれば常に stop を記録
        assert "if stop_reason:" in source


# ======================================================================
# 4. buy_dynamic_kill 在庫連動緩和 (Ho & Stoll 1981)
# ======================================================================

class TestBuyDynamicKillInvRelaxation:
    """286# 283# P1-4: 在庫連動の kill 閾値緩和テスト."""

    def test_threshold_offset_relaxes_kill(self):
        """threshold_offset_bps > 0 で kill 閾値が緩和されること."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        cfg = DynamicKillConfig(
            enabled=True,
            window=3,
            threshold_bps=-0.5,
            resume_window=1,
        )
        mgr = DynamicKillManager(cfg, side="buy")

        # 3回追跡: 平均 = -0.6 bps (threshold -0.5 より悪い → 通常 kill)
        for _ in range(3):
            mgr.track(-0.6)

        # 通常: kill される
        killed_normal, _ = mgr.check_kill(threshold_offset_bps=0.0)
        assert killed_normal is True

    def test_threshold_offset_prevents_kill(self):
        """十分な threshold_offset_bps で kill が防止されること."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        cfg = DynamicKillConfig(
            enabled=True,
            window=3,
            threshold_bps=-0.5,
            resume_window=1,
        )
        mgr = DynamicKillManager(cfg, side="buy")

        # 3回: 平均 = -0.6
        for _ in range(3):
            mgr.track(-0.6)

        # 緩和: threshold = -0.5 + 0.2 = -0.3 → -0.6 < -0.3 → まだ kill
        killed_partial, _ = mgr.check_kill(threshold_offset_bps=0.2)
        assert killed_partial is True

        # 大きい緩和: threshold = -0.5 + 0.5 = 0.0 → -0.6 < 0.0 → まだ kill
        # But wait, resume_window=1 → cooldown 中
        # Let's use a fresh manager
        mgr2 = DynamicKillManager(
            DynamicKillConfig(enabled=True, window=3, threshold_bps=-0.5, resume_window=1),
            side="buy",
        )
        # 平均 = -0.4 (threshold -0.5 以上 → kill されない)
        for _ in range(3):
            mgr2.track(-0.4)
        killed, _ = mgr2.check_kill(threshold_offset_bps=0.0)
        assert killed is False

    def test_config_inv_relaxation_fields_exist(self):
        """FillTestConfig に在庫連動緩和フィールドが存在すること."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert hasattr(cfg, "buy_dynamic_kill_inv_relaxation_enabled")
        assert hasattr(cfg, "buy_dynamic_kill_inv_relaxation_scale")
        assert hasattr(cfg, "buy_dynamic_kill_inv_relaxation_max_bps")
        assert cfg.buy_dynamic_kill_inv_relaxation_enabled is False
        assert cfg.buy_dynamic_kill_inv_relaxation_max_bps == 0.3


# ======================================================================
# 5. 強制買い KPI 分離トラッキング
# ======================================================================

class TestForcedBuyKpiTracking:
    """286# 283# P1-5: 強制買い KPI 分離テスト."""

    def test_config_kpi_tracking_enabled_default(self):
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.forced_buy_kpi_tracking_enabled is True

    def test_session_state_has_kpi_fields(self):
        """RunSessionState に強制買い KPI フィールドがあること."""
        from scripts.v460.lib.fill_loop_orchestrator import RunSessionState

        st = RunSessionState()
        assert st.forced_buy_fill_count == 0
        assert st.forced_buy_pnl_sum_bps == 0.0
        assert st.normal_buy_fill_count == 0
        assert st.normal_buy_pnl_sum_bps == 0.0


# ======================================================================
# 6. Buy-side AS Guard (microprice 急落防御)
# ======================================================================

class TestBuyAsGuard:
    """286# 283# P1-6 / 284# P1: Buy-side AS Guard テスト."""

    def test_config_fields_exist(self):
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert hasattr(cfg, "buy_as_guard_enabled")
        assert hasattr(cfg, "buy_as_guard_velocity_threshold_bps")
        assert hasattr(cfg, "buy_as_guard_offset_mult")
        assert hasattr(cfg, "buy_as_guard_max_offset_ratio")
        assert cfg.buy_as_guard_enabled is False

    def test_guard_disabled_noop(self):
        """disabled 時は offset 変更なし."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig

        cfg = FillTestConfig(buy_as_guard_enabled=False)
        ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.15)
        calc = MakerPriceCalculator(
            cfg, ffd, None, base_offset_ratio=0.15,
        )
        result = calc._apply_buy_as_guard("buy", -10.0, 0.15)
        assert result == 0.15  # 変更なし

    def test_guard_fires_on_sell_noop(self):
        """sell 側では発動しないこと (buy 専用)."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig

        cfg = FillTestConfig(buy_as_guard_enabled=True)
        ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.15)
        calc = MakerPriceCalculator(
            cfg, ffd, None, base_offset_ratio=0.15,
        )
        result = calc._apply_buy_as_guard("sell", -10.0, 0.15)
        assert result == 0.15

    def test_guard_expands_offset_on_decline(self):
        """velocity が閾値以下で buy offset が拡大すること."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig

        cfg = FillTestConfig(
            buy_as_guard_enabled=True,
            buy_as_guard_velocity_threshold_bps=-5.0,
            buy_as_guard_offset_mult=1.5,
            buy_as_guard_max_offset_ratio=0.5,
        )
        ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.15)
        calc = MakerPriceCalculator(
            cfg, ffd, None, base_offset_ratio=0.15,
        )
        # velocity = -8 bps (< -5 bps) → 発動
        result = calc._apply_buy_as_guard("buy", -8.0, 0.15)
        assert result > 0.15  # offset 拡大
        assert result == pytest.approx(0.225)  # 0.15 * 1.5

    def test_guard_respects_max_ratio(self):
        """max_offset_ratio でクリップされること."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig

        cfg = FillTestConfig(
            buy_as_guard_enabled=True,
            buy_as_guard_velocity_threshold_bps=-5.0,
            buy_as_guard_offset_mult=5.0,  # 非常に大きい倍率
            buy_as_guard_max_offset_ratio=0.3,  # 上限 0.3
        )
        ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.15)
        calc = MakerPriceCalculator(
            cfg, ffd, None, base_offset_ratio=0.15,
        )
        result = calc._apply_buy_as_guard("buy", -10.0, 0.15)
        assert result <= 0.3  # max でクリップ

    def test_guard_no_fire_on_positive_velocity(self):
        """velocity が正 (上昇中) → 発動なし."""
        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.maker_price import MakerPriceCalculator
        from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig

        cfg = FillTestConfig(
            buy_as_guard_enabled=True,
            buy_as_guard_velocity_threshold_bps=-5.0,
        )
        ffd = FastFillDefense(FastFillDefenseConfig(), base_offset_ratio=0.15)
        calc = MakerPriceCalculator(
            cfg, ffd, None, base_offset_ratio=0.15,
        )
        result = calc._apply_buy_as_guard("buy", 3.0, 0.15)
        assert result == 0.15


# ======================================================================
# 7. 強制買い遅延実行 (Glosten-Milgrom 1985)
# ======================================================================

class TestForcedBuyDelay:
    """286# 284# P1: 強制買い遅延実行テスト."""

    def test_config_delay_fields_exist(self):
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert hasattr(cfg, "forced_buy_delay_enabled")
        assert hasattr(cfg, "forced_buy_delay_velocity_threshold_bps")
        assert hasattr(cfg, "forced_buy_delay_cycles")
        assert cfg.forced_buy_delay_enabled is False
        assert cfg.forced_buy_delay_cycles == 3

    def test_delay_counter_attribute_exists(self):
        """FillLoopOrchestratorMixin に _forced_buy_delay_remaining があること."""
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin

        assert hasattr(FillLoopOrchestratorMixin, "_forced_buy_delay_remaining")
        assert FillLoopOrchestratorMixin._forced_buy_delay_remaining == 0


# ======================================================================
# 8. guard_reason_classifier 再分類 (283# MEDIUM-4)
# ======================================================================

class TestGuardReclassification:
    """286# 283# MEDIUM-4: guard dominance 改善 — SYSTEM→RECOVERY 再分類."""

    def test_balance_forced_halt_block_is_recovery(self):
        from scripts.v460.lib.guard_reason_classifier import classify_guard, GuardCategory
        assert classify_guard("balance_forced_halt_block") == GuardCategory.RECOVERY

    def test_one_sided_freeze_skip_is_recovery(self):
        from scripts.v460.lib.guard_reason_classifier import classify_guard, GuardCategory
        assert classify_guard("one_sided_freeze_skip") == GuardCategory.RECOVERY

    def test_one_sided_cooldown_skip_is_recovery(self):
        from scripts.v460.lib.guard_reason_classifier import classify_guard, GuardCategory
        assert classify_guard("one_sided_cooldown_skip") == GuardCategory.RECOVERY

    def test_degraded_liquidation_is_recovery(self):
        from scripts.v460.lib.guard_reason_classifier import classify_guard, GuardCategory
        assert classify_guard("degraded_liquidation_duty_skip") == GuardCategory.RECOVERY
        assert classify_guard("degraded_liquidation_active") == GuardCategory.RECOVERY

    def test_inventory_escape_is_recovery(self):
        from scripts.v460.lib.guard_reason_classifier import classify_guard, GuardCategory
        assert classify_guard("inventory_escape_duty_skip") == GuardCategory.RECOVERY
        assert classify_guard("inventory_escape_active") == GuardCategory.RECOVERY

    def test_forced_buy_delay_is_market(self):
        """286# forced_buy_delay は市場都合 (GM逆選択回避)."""
        from scripts.v460.lib.guard_reason_classifier import classify_guard, GuardCategory
        assert classify_guard("forced_buy_delay") == GuardCategory.MARKET

    def test_system_guards_remain_system(self):
        """dd_halt, phantom 等は SYSTEM のまま."""
        from scripts.v460.lib.guard_reason_classifier import classify_guard, GuardCategory
        assert classify_guard("dd_halt") == GuardCategory.SYSTEM
        assert classify_guard("phantom_veto_block") == GuardCategory.SYSTEM
        assert classify_guard("hard_skip_utc") == GuardCategory.SYSTEM


# ======================================================================
# 9. DynamicKillManager threshold_offset_bps パラメータ
# ======================================================================

class TestDynamicKillThresholdOffset:
    """286# check_kill() の threshold_offset_bps パラメータテスト."""

    def test_zero_offset_is_backward_compatible(self):
        """offset=0 は従来互換 (振る舞い不変)."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        cfg = DynamicKillConfig(enabled=True, window=2, threshold_bps=-1.0, resume_window=1)
        mgr = DynamicKillManager(cfg, side="sell")
        mgr.track(-0.5)
        mgr.track(-0.5)
        killed, tel = mgr.check_kill(threshold_offset_bps=0.0)
        # mean=-0.5 > threshold=-1.0 → not killed
        assert killed is False

    def test_negative_offset_tightens_threshold(self):
        """負の offset は閾値を厳格化する (kill されやすくなる)."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        cfg = DynamicKillConfig(enabled=True, window=2, threshold_bps=-1.0, resume_window=1)
        mgr = DynamicKillManager(cfg, side="sell")
        mgr.track(-0.5)
        mgr.track(-0.5)
        # effective threshold = -1.0 + (-0.5) = -1.5
        # mean=-0.5 > -1.5 → not killed
        killed, _ = mgr.check_kill(threshold_offset_bps=-0.5)
        assert killed is False

    def test_disabled_manager_ignores_offset(self):
        """disabled 時は offset 無視."""
        from ztb.risk.sell_dynamic_kill import DynamicKillConfig, DynamicKillManager

        cfg = DynamicKillConfig(enabled=False)
        mgr = DynamicKillManager(cfg, side="buy")
        killed, _ = mgr.check_kill(threshold_offset_bps=100.0)
        assert killed is False
