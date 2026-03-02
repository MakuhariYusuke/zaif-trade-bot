"""223# テスト — 222# レビュー対応の検証.

対象:
  A. CycleGateResult.dual_kill_bypassed フラグ
  B. DynamicKillTelemetry.probe_fired / force_release_fired フラグ
  C. balance_forced + per-side halt 再チェック (統合テスト相当)
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.cycle_gate_aggregator import (
    CycleGateAggregator,
    CycleGateResult,
)
from ztb.risk.sell_dynamic_kill import (
    DynamicKillConfig,
    DynamicKillManager,
    DynamicKillTelemetry,
)


# ─── ヘルパー ──────────────────────────────────────────────────────────


def _make_config(**overrides: object) -> FillTestConfig:
    """テスト用の最小 FillTestConfig."""
    defaults: dict[str, object] = {
        "skip_buy_unknown_regime": True,
        "skip_ranging_buy_low_vol": True,
        "low_vol_threshold": 0.75,
        "skip_sell_trending": True,
        "skip_sell_trending_up_only": False,
        "max_consecutive_trending_sell_skip": 30,
        "sell_guard_inv_bypass_threshold": 0.3,
        "buy_dynamic_kill_enabled": True,
        "sell_dynamic_kill_enabled": True,
        "buy_dynamic_kill_threshold_bps": -5.0,
        "sell_dynamic_kill_threshold_bps": -5.0,
        "sell_velocity_skip_enabled": True,
        "sell_velocity_skip_threshold_bps": 8.0,
        "buy_velocity_skip_enabled": True,
        "buy_velocity_skip_threshold_bps": -8.0,
        "skip_sell_unknown_regime": True,
    }
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def _make_gate(**overrides: object) -> CycleGateAggregator:
    return CycleGateAggregator(_make_config(**overrides))


def _default_ctx(**overrides: object) -> dict:
    ctx: dict = {
        "side": "buy",
        "regime": "ranging",
        "vol_ratio": 1.0,
        "balance_forced": False,
        "inv_net_imbalance": 0.0,
        "is_buy_killed": False,
        "is_sell_killed": False,
    }
    ctx.update(overrides)
    return ctx


# ═══════════════════════════════════════════════════════════════════════
# A. CycleGateResult.dual_kill_bypassed
# ═══════════════════════════════════════════════════════════════════════


class TestDualKillBypassedFlag:
    """223# dual_kill_bypassed フラグの検証."""

    def test_dual_kill_sets_flag(self) -> None:
        """buy+sell 両方 kill → dual_kill_bypassed=True."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=True,
        ))
        assert not r.blocked
        assert r.dual_kill_bypassed is True

    def test_single_kill_no_flag(self) -> None:
        """片方だけ kill → dual_kill_bypassed=False."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="sell", is_buy_killed=False, is_sell_killed=True,
        ))
        assert r.blocked
        assert r.dual_kill_bypassed is False

    def test_no_kill_no_flag(self) -> None:
        """kill なし → dual_kill_bypassed=False (デフォルト)."""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx())
        assert not r.blocked
        assert r.dual_kill_bypassed is False

    def test_dual_kill_with_balance_forced_flag_true(self) -> None:
        """234#: balance_forced=True + dual kill → dual_kill_bypassed=True.

        234# で `not balance_forced` を _dual_kill 条件から削除したため、
        balance_forced=True でも dual_kill が正しく検出される。"""
        gate = _make_gate()
        r = gate.evaluate(**_default_ctx(
            side="buy", is_buy_killed=True, is_sell_killed=True,
            balance_forced=True,
        ))
        assert not r.blocked
        assert r.dual_kill_bypassed is True  # 234# dual kill detected

    def test_default_result_has_false(self) -> None:
        """CycleGateResult デフォルトは dual_kill_bypassed=False."""
        r = CycleGateResult()
        assert r.dual_kill_bypassed is False


# ═══════════════════════════════════════════════════════════════════════
# B. DynamicKillTelemetry.probe_fired / force_release_fired
# ═══════════════════════════════════════════════════════════════════════


class TestTelemetryFlags:
    """223# probe_fired / force_release_fired フラグの検証."""

    def _make_mgr(
        self,
        *,
        window: int = 5,
        threshold_bps: float = -0.5,
        resume_window: int = 1,
        max_stale: int = 4,
        min_probe_interval: int = 2,
        max_force_release_probes: int = 3,
    ) -> DynamicKillManager:
        return DynamicKillManager(
            DynamicKillConfig(
                enabled=True,
                window=window,
                threshold_bps=threshold_bps,
                resume_window=resume_window,
                max_stale_kill_cycles=max_stale,
                min_probe_interval=min_probe_interval,
                max_force_release_probes=max_force_release_probes,
            ),
            side="sell",
        )

    def _fill_bad_data(self, mgr: DynamicKillManager, n: int = 5) -> None:
        """window 分のネガティブ PnL を投入."""
        for _ in range(n):
            mgr.track(-1.0)

    def test_normal_kill_no_flags(self) -> None:
        """通常の kill → probe_fired=False, force_release_fired=False."""
        mgr = self._make_mgr()
        self._fill_bad_data(mgr)
        killed, telem = mgr.check_kill()
        assert killed
        assert telem.probe_fired is False
        assert telem.force_release_fired is False

    def test_probe_sets_probe_fired(self) -> None:
        """stale probe 発動 → probe_fired=True."""
        mgr = self._make_mgr(max_stale=3, resume_window=1)
        self._fill_bad_data(mgr)

        probe_seen = False
        for _ in range(20):
            killed, telem = mgr.check_kill()
            if not killed and telem.probe_fired:
                probe_seen = True
                break

        assert probe_seen, "probe_fired should be True on probe cycle"

    def test_probe_fired_only_on_probe_cycle(self) -> None:
        """probe でないサイクルでは probe_fired=False."""
        mgr = self._make_mgr(max_stale=5, resume_window=1)
        self._fill_bad_data(mgr)

        # 最初のサイクルは kill (probe ではない)
        killed, telem = mgr.check_kill()
        assert killed
        assert telem.probe_fired is False

    def test_force_release_sets_flag(self) -> None:
        """force release 発動 → force_release_fired=True."""
        mgr = self._make_mgr(
            max_stale=3, resume_window=1,
            max_force_release_probes=2, min_probe_interval=2,
        )
        self._fill_bad_data(mgr)

        force_release_seen = False
        for _ in range(50):
            killed, telem = mgr.check_kill()
            if not killed and telem.force_release_fired:
                force_release_seen = True
                break

        assert force_release_seen, "force_release_fired should be True"
        assert mgr._force_released

    def test_force_release_flag_only_once(self) -> None:
        """force release 後の通常チェックでは flag=False."""
        mgr = self._make_mgr(
            max_stale=3, resume_window=1,
            max_force_release_probes=2, min_probe_interval=2,
        )
        self._fill_bad_data(mgr)

        # force release まで進める
        for _ in range(50):
            mgr.check_kill()
            if mgr._force_released:
                break

        # force release 中の後続チェック
        killed, telem = mgr.check_kill()
        assert not killed
        # force_release が進行中だが、fired は最初の 1 回だけ
        assert telem.force_release_fired is False
        assert telem.probe_fired is False

    def test_default_telemetry_flags(self) -> None:
        """DynamicKillTelemetry デフォルト値."""
        t = DynamicKillTelemetry(
            killed=False,
            cooldown_remaining=0,
            rolling_mean=None,
            rolling_count=0,
            threshold_used=-0.5,
            regime=None,
            total_kills=0,
            total_cooldown_cycles=0,
        )
        assert t.probe_fired is False
        assert t.force_release_fired is False
