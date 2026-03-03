"""242# Liveness Constraint Relaxation (233# P1) tests.

テスト対象:
  A. DynamicKillManager: toxic_kill_stale_multiplier — toxicity KILL 時の probe interval 延長
  B. FillConfig: quiescence_gate_blocks_threshold / quiescence_sleep_sec
  C. FillLoopOrchestrator: _effective_sleep max_override / quiescence ログ正常化
"""
from __future__ import annotations

import asyncio

import pytest

from ztb.risk.sell_dynamic_kill import (
    DynamicKillConfig,
    DynamicKillManager,
    ToxicityLevel,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. DynamicKillManager: toxic_kill_stale_multiplier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestToxicKillProbeRelaxation242:
    """242# toxicity KILL 時の probe interval 延長."""

    def _make_mgr(
        self,
        *,
        window: int = 5,
        threshold_bps: float = -0.5,
        resume_window: int = 2,
        max_stale: int = 10,
        toxic_mult: int = 10,
        toxicity_budget_enabled: bool = True,
        min_probe_interval: int = 2,
        max_force_release_probes: int = 5,
    ) -> DynamicKillManager:
        return DynamicKillManager(
            DynamicKillConfig(
                enabled=True,
                window=window,
                threshold_bps=threshold_bps,
                resume_window=resume_window,
                max_stale_kill_cycles=max_stale,
                toxic_kill_stale_multiplier=toxic_mult,
                toxicity_budget_enabled=toxicity_budget_enabled,
                min_probe_interval=min_probe_interval,
                max_force_release_probes=max_force_release_probes,
            ),
            side="sell",
        )

    def _fill_bad_data(self, mgr: DynamicKillManager, n: int = 5) -> None:
        """window 分のネガティブ PnL を投入."""
        for _ in range(n):
            mgr.track(-1.0)

    def test_probe_delayed_when_toxicity_kill(self) -> None:
        """toxicity KILL + データ裏付けあり → probe interval が延長される."""
        mgr = self._make_mgr(
            max_stale=10, toxic_mult=10, resume_window=1,
        )
        self._fill_bad_data(mgr)

        # toxicity assessment should be KILL
        tox = mgr.assess_toxicity()
        assert tox.level == ToxicityLevel.KILL

        # Kill fires immediately
        killed, _ = mgr.check_kill()
        assert killed

        # Without multiplier, probe at ~10 cycles. With ×10, should take ~100.
        # Run 50 cycles — should NOT see probe
        kills = 0
        for _ in range(50):
            killed, _ = mgr.check_kill()
            if killed:
                kills += 1
            else:
                break  # probe fired unexpectedly

        assert kills == 50, (
            f"Expected 50 consecutive kills (probe at ~100), got probe at {kills}"
        )

    def test_probe_fires_at_extended_interval(self) -> None:
        """toxic_kill_stale_multiplier 適用後も最終的には probe が発火する."""
        mgr = self._make_mgr(
            max_stale=5, toxic_mult=3, resume_window=1,
            min_probe_interval=2, max_force_release_probes=0,
        )
        self._fill_bad_data(mgr)

        killed, _ = mgr.check_kill()
        assert killed

        # effective interval = 5 * 3 = 15. Run up to 20 cycles to find probe.
        probe_at = None
        kills = 0
        for i in range(25):
            killed, tel = mgr.check_kill()
            if not killed:
                probe_at = i
                break
            kills += 1

        assert probe_at is not None, "Probe should eventually fire"
        # probe should be at ~15 cycles (max_stale=5 × toxic_mult=3)
        # accounting for cooldown (resume_window=1), it's stale reaches 5*3=15
        assert kills >= 10, (
            f"Probe too early: {kills} kills before probe "
            f"(expected ~15 with toxic_mult=3)"
        )

    def test_no_delay_when_toxicity_budget_disabled(self) -> None:
        """toxicity_budget_enabled=False → 従来互換 (延長なし)."""
        mgr = self._make_mgr(
            max_stale=10, toxic_mult=10,
            toxicity_budget_enabled=False, resume_window=1,
        )
        self._fill_bad_data(mgr)

        killed, _ = mgr.check_kill()
        assert killed

        # 通常の probe interval (10 cycles)
        kills = 0
        for _ in range(20):
            killed, _ = mgr.check_kill()
            if killed:
                kills += 1
            else:
                break

        assert kills < 15, (
            f"Without toxicity budget, probe should fire at ~10 cycles, got {kills}"
        )

    def test_no_delay_when_multiplier_is_one(self) -> None:
        """toxic_kill_stale_multiplier=1 → 従来互換."""
        mgr = self._make_mgr(
            max_stale=10, toxic_mult=1, resume_window=1,
        )
        self._fill_bad_data(mgr)

        killed, _ = mgr.check_kill()
        assert killed

        kills = 0
        for _ in range(20):
            killed, _ = mgr.check_kill()
            if killed:
                kills += 1
            else:
                break

        assert kills < 15, (
            f"With toxic_mult=1, probe should fire at ~10, got {kills}"
        )

    def test_no_delay_when_data_insufficient(self) -> None:
        """データ不足 (rolling_mean=None) → 延長なし (probe fires normally)."""
        mgr = self._make_mgr(
            window=10, max_stale=5, toxic_mult=10, resume_window=1,
        )
        # Only 3 data points (window=10 requires 10)
        for _ in range(3):
            mgr.track(-1.0)

        # assess_toxicity should return GREEN (insufficient data)
        tox = mgr.assess_toxicity()
        assert tox.level == ToxicityLevel.GREEN
        assert tox.rolling_mean is None

        # With only 3 data points, check_kill won't kill (insufficient data)
        killed, _ = mgr.check_kill()
        assert not killed, "Should not kill with insufficient data"

    def test_no_delay_when_toxicity_green(self) -> None:
        """toxicity GREEN (良好な PnL) → 延長なし."""
        mgr = self._make_mgr(
            max_stale=5, toxic_mult=10, resume_window=1,
        )
        # Good PnL data - all positive
        for _ in range(5):
            mgr.track(1.0)

        tox = mgr.assess_toxicity()
        assert tox.level == ToxicityLevel.GREEN

        # PnL is good → no kill at all
        killed, _ = mgr.check_kill()
        assert not killed

    def test_toxic_multiplier_method_directly(self) -> None:
        """_toxic_kill_multiplier の直接テスト."""
        mgr = self._make_mgr(
            max_stale=10, toxic_mult=5, resume_window=1,
        )
        # No data → multiplier=1 (no toxicity assessment possible)
        assert mgr._toxic_kill_multiplier(None) == 1

        # Sufficient bad data → KILL → multiplier=5
        self._fill_bad_data(mgr)
        assert mgr._toxic_kill_multiplier(None) == 5

        # Good data → GREEN → multiplier=1
        mgr.reset()
        for _ in range(5):
            mgr.track(1.0)
        assert mgr._toxic_kill_multiplier(None) == 1

    def test_toxic_multiplier_disabled_when_budget_off(self) -> None:
        """toxicity_budget_enabled=False → multiplier=1."""
        mgr = self._make_mgr(
            max_stale=10, toxic_mult=5,
            toxicity_budget_enabled=False,
        )
        self._fill_bad_data(mgr)
        assert mgr._toxic_kill_multiplier(None) == 1

    def test_force_release_also_delayed(self) -> None:
        """toxic_kill_stale_multiplier は force_release にも影響する.

        probe 間隔が延長されるため、force_release までの総サイクル数も増加。
        これにより kill gate が長期間維持される。
        """
        mgr = self._make_mgr(
            max_stale=3, toxic_mult=3, resume_window=1,
            min_probe_interval=2, max_force_release_probes=2,
        )
        self._fill_bad_data(mgr)

        # Without multiplier: force_release at ~3+2=5 probes cycle count
        # With multiplier ×3: each probe gap is 3×3=9, 2 probes → ~18+ cycles
        force_released_at = None
        for i in range(60):
            killed, _ = mgr.check_kill()
            if mgr._force_released:
                force_released_at = i
                break

        assert force_released_at is not None, "Force release should eventually fire"
        # With toxic_mult=3, force_release should take significantly longer
        assert force_released_at > 10, (
            f"Force release too early: cycle {force_released_at} "
            f"(expected >10 with toxic_mult=3)"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. FillConfig: quiescence settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQuiescenceConfig242:
    """242# FillConfig quiescence 設定のデフォルト値テスト."""

    def test_default_values(self) -> None:
        """デフォルト値が期待通り."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.quiescence_gate_blocks_threshold == 20
        assert cfg.quiescence_sleep_sec == 1800.0

    def test_max_cycle_sleep_unchanged(self) -> None:
        """既存の max_cycle_sleep_sec は変更なし."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.max_cycle_sleep_sec == 600.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. _effective_sleep: max_override パラメータ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEffectiveSleepOverride242:
    """242# _effective_sleep の max_override パラメータ."""

    @pytest.fixture
    def orchestrator_stub(self):
        """_effective_sleep テスト用の最小スタブ.

        _effective_sleep の sleep 値計算ロジックだけをテストする。
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from scripts.v460.lib.fill_config import FillTestConfig
        from scripts.v460.lib.fill_loop_orchestrator import FillLoopOrchestratorMixin

        class Stub(FillLoopOrchestratorMixin):
            pass

        stub = object.__new__(Stub)
        stub.config = FillTestConfig(
            max_cycle_sleep_sec=600.0,
            quiescence_sleep_sec=1800.0,
        )
        stub._regime_detector = None
        stub._soft_drawdown_interval_multiplier = 1.0
        stub._alert_interval_mult = 1.0
        stub._cycle_strategy = MagicMock()
        stub._cycle_strategy.effective_interval.return_value = 120.0
        # _current_regime_value() は _regime_detector=None → "unknown" を返す
        stub._current_regime_value = lambda: "unknown"
        return stub

    @pytest.mark.asyncio
    async def test_default_uses_max_cycle_sleep(self, orchestrator_stub) -> None:
        """max_override=0 → max_cycle_sleep_sec を使用."""
        import unittest.mock

        with unittest.mock.patch("asyncio.sleep", new=unittest.mock.AsyncMock()) as mock_sleep:
            await orchestrator_stub._effective_sleep(multiplier=100.0)
            # 120 * 100 = 12000, capped by 600
            mock_sleep.assert_awaited_once_with(600.0)

    @pytest.mark.asyncio
    async def test_override_uses_quiescence_cap(self, orchestrator_stub) -> None:
        """max_override=1800 → quiescence sleep cap を使用."""
        import unittest.mock

        with unittest.mock.patch("asyncio.sleep", new=unittest.mock.AsyncMock()) as mock_sleep:
            await orchestrator_stub._effective_sleep(
                multiplier=100.0, max_override=1800.0,
            )
            # 120 * 100 = 12000, capped by 1800
            mock_sleep.assert_awaited_once_with(1800.0)

    @pytest.mark.asyncio
    async def test_override_no_effect_when_raw_smaller(self, orchestrator_stub) -> None:
        """raw sleep < max_override → raw sleep を使用."""
        import unittest.mock

        with unittest.mock.patch("asyncio.sleep", new=unittest.mock.AsyncMock()) as mock_sleep:
            await orchestrator_stub._effective_sleep(
                multiplier=1.0, max_override=1800.0,
            )
            # 120 * 1 = 120, < 1800 → use 120
            mock_sleep.assert_awaited_once_with(120.0)
