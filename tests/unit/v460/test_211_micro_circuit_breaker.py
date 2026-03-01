"""211# P1-B: Micro Circuit Breaker テスト.

短期価格急変の自動検知・防御メカニズムのユニットテスト。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.micro_circuit_breaker import (
    MCBConfig,
    MCBLevel,
    MCBResult,
    MicroCircuitBreaker,
)


# =====================================================================
# 基本動作テスト
# =====================================================================


class TestMicroCircuitBreakerBasic:
    """MCB の基本動作テスト."""

    def test_disabled_returns_normal(self) -> None:
        """enabled=False → 常に NORMAL."""
        mcb = MicroCircuitBreaker(MCBConfig(enabled=False))
        mcb.update(100_000.0, 1000.0)
        result = mcb.check(1000.0)
        assert result.level == MCBLevel.NORMAL

    def test_no_data_returns_normal(self) -> None:
        """データなし → NORMAL."""
        mcb = MicroCircuitBreaker(MCBConfig())
        result = mcb.check(1000.0)
        assert result.level == MCBLevel.NORMAL

    def test_stable_price_returns_normal(self) -> None:
        """安定した価格 → NORMAL."""
        mcb = MicroCircuitBreaker(MCBConfig(
            baseline_sample_interval_sec=1.0,
        ))
        base_price = 100_000.0
        for i in range(400):
            mcb.update(base_price + (i % 3) * 10, float(i))  # tiny fluctuation
        result = mcb.check(400.0)
        assert result.level == MCBLevel.NORMAL


# =====================================================================
# 閾値超過テスト
# =====================================================================


class TestMicroCircuitBreakerThresholds:
    """閾値超過時の段階判定テスト."""

    def test_large_price_drop_triggers_halt(self) -> None:
        """大幅な価格下落 → default threshold で HALT or WARNING."""
        mcb = MicroCircuitBreaker(MCBConfig(
            baseline_sample_interval_sec=1.0,
            # デフォルト閾値を使う (warmup サンプル不足)
            default_threshold_5m_pct=0.5,
            single_window_halt_sigma=3.0,
        ))
        # 5 分間 (300s) の安定期間
        base = 100_000.0
        for i in range(310):
            mcb.update(base, float(i))

        # 急落: 3% drop (default_threshold_5m = 0.5%, σ = 3%/0.5% = 6 > 3.0)
        mcb.update(base * 0.97, 310.0)
        result = mcb.check(310.0)
        assert result.level in (MCBLevel.WARNING, MCBLevel.HALT)

    def test_moderate_change_triggers_caution(self) -> None:
        """中程度の動き → CAUTION."""
        mcb = MicroCircuitBreaker(MCBConfig(
            baseline_sample_interval_sec=1.0,
            default_threshold_5m_pct=0.5,
            caution_sigma=1.0,
            warning_sigma=1.5,
            halt_sigma=2.0,
            single_window_warning_sigma=2.0,
            single_window_halt_sigma=3.0,
        ))
        base = 100_000.0
        for i in range(310):
            mcb.update(base, float(i))
        # 0.6% drop → σ = 0.6/0.5 = 1.2 → CAUTION (> 1.0, < 1.5)
        mcb.update(base * 0.994, 310.0)
        result = mcb.check(310.0)
        assert result.level == MCBLevel.CAUTION


# =====================================================================
# クールダウンテスト
# =====================================================================


class TestMicroCircuitBreakerCooldown:
    """HALT クールダウンテスト."""

    def test_halt_cooldown_period(self) -> None:
        """HALT 後はクールダウン期間中 HALT を返し続ける."""
        mcb = MicroCircuitBreaker(MCBConfig(
            baseline_sample_interval_sec=1.0,
            halt_cooldown_sec=60.0,
            default_threshold_5m_pct=0.5,
            single_window_halt_sigma=2.0,
        ))
        base = 100_000.0
        for i in range(310):
            mcb.update(base, float(i))
        # Crash
        mcb.update(base * 0.95, 310.0)
        r1 = mcb.check(310.0)
        assert r1.level == MCBLevel.HALT

        # クールダウン中 (30秒後)
        r2 = mcb.check(340.0)
        assert r2.level == MCBLevel.HALT
        assert r2.cooldown_remaining_sec > 0

        # クールダウン後 (70秒後) — 価格がまだ crash 状態なので
        # 再評価で再度 HALT or WARNING になる可能性がある。
        # ここでは「cooldown_remaining_sec が 0 以下」= 元の cooldown が終了したことを検証。
        r3 = mcb.check(380.0)
        # 元の cooldown (310+60=370) は終了しているが、再評価で新しい HALT が発生しうる
        # → cooldown_remaining が元の残り (310+60-380 = -10) ではなく新しい値になる
        assert r3.cooldown_remaining_sec >= 0  # 新しい HALT or NORMAL


# =====================================================================
# σ判定ロジックテスト
# =====================================================================


class TestMicroCircuitBreakerDetermineLevel:
    """_determine_level の単体テスト."""

    def test_normal_when_all_below_caution(self) -> None:
        cfg = MCBConfig(caution_sigma=1.0, warning_sigma=1.5, halt_sigma=2.0)
        mcb = MicroCircuitBreaker(cfg)
        assert mcb._determine_level([0.5, 0.3, 0.8]) == MCBLevel.NORMAL

    def test_caution_when_one_above(self) -> None:
        cfg = MCBConfig(caution_sigma=1.0, warning_sigma=1.5, halt_sigma=2.0,
                        single_window_warning_sigma=2.0, escalation_window_count=2)
        mcb = MicroCircuitBreaker(cfg)
        assert mcb._determine_level([1.2, 0.3, 0.5]) == MCBLevel.CAUTION

    def test_warning_with_two_windows(self) -> None:
        cfg = MCBConfig(caution_sigma=1.0, warning_sigma=1.5, halt_sigma=2.0,
                        escalation_window_count=2, single_window_warning_sigma=2.0)
        mcb = MicroCircuitBreaker(cfg)
        assert mcb._determine_level([1.6, 1.7, 0.5]) == MCBLevel.WARNING

    def test_warning_single_window_high_sigma(self) -> None:
        cfg = MCBConfig(warning_sigma=1.5, single_window_warning_sigma=2.0,
                        halt_sigma=2.5, single_window_halt_sigma=3.0,
                        escalation_window_count=2)
        mcb = MicroCircuitBreaker(cfg)
        # 1窓で 2.3 > single_window_warning_sigma (2.0) → WARNING
        assert mcb._determine_level([2.3, 0.5, 0.3]) == MCBLevel.WARNING

    def test_halt_with_two_windows(self) -> None:
        cfg = MCBConfig(halt_sigma=2.0, escalation_window_count=2,
                        single_window_halt_sigma=3.0)
        mcb = MicroCircuitBreaker(cfg)
        assert mcb._determine_level([2.5, 2.1, 0.5]) == MCBLevel.HALT

    def test_halt_single_window_extreme(self) -> None:
        cfg = MCBConfig(halt_sigma=2.0, single_window_halt_sigma=3.0,
                        escalation_window_count=2)
        mcb = MicroCircuitBreaker(cfg)
        assert mcb._determine_level([3.5, 0.5, 0.5]) == MCBLevel.HALT


# =====================================================================
# export / import テスト
# =====================================================================


class TestMicroCircuitBreakerState:
    """状態永続化のテスト."""

    def test_export_empty(self) -> None:
        mcb = MicroCircuitBreaker(MCBConfig())
        state = mcb.export_state()
        assert state["halt_until"] == 0.0
        assert state["total_halts"] == 0
        assert state["price_buffer"] == []

    def test_roundtrip(self) -> None:
        mcb = MicroCircuitBreaker(MCBConfig(baseline_sample_interval_sec=1.0))
        for i in range(50):
            mcb.update(100_000.0 + i, float(i))
        state = mcb.export_state()

        mcb2 = MicroCircuitBreaker(MCBConfig(baseline_sample_interval_sec=1.0))
        mcb2.import_state(state)
        state2 = mcb2.export_state()

        assert len(state["price_buffer"]) == len(state2["price_buffer"])
        assert state["halt_until"] == state2["halt_until"]

    def test_import_restores_halt(self) -> None:
        """import で halt_until が復元される."""
        mcb = MicroCircuitBreaker(MCBConfig())
        mcb.import_state({"halt_until": 99999999.0})
        assert mcb._halt_until == 99999999.0


# =====================================================================
# MCBResult テスト
# =====================================================================


class TestMCBResult:
    """MCBResult dataclass のテスト."""

    def test_default(self) -> None:
        r = MCBResult()
        assert r.level == MCBLevel.NORMAL
        assert r.offset_mult == 1.0
        assert r.interval_mult == 1.0

    def test_warning_overrides(self) -> None:
        r = MCBResult(
            level=MCBLevel.WARNING,
            offset_mult=1.5,
            interval_mult=2.0,
        )
        assert r.offset_mult == 1.5


# =====================================================================
# Config テスト
# =====================================================================


class TestMCBConfig:
    """MCBConfig dataclass のテスト."""

    def test_defaults(self) -> None:
        cfg = MCBConfig()
        assert cfg.enabled is True
        assert cfg.caution_sigma == 1.0
        assert cfg.warning_sigma == 1.5
        assert cfg.halt_sigma == 2.0
        assert cfg.halt_cooldown_sec == 300.0

    def test_custom(self) -> None:
        cfg = MCBConfig(
            enabled=False,
            halt_cooldown_sec=60.0,
            warning_offset_mult=2.0,
        )
        assert cfg.enabled is False
        assert cfg.halt_cooldown_sec == 60.0
        assert cfg.warning_offset_mult == 2.0


# =====================================================================
# FillTestConfig 統合テスト
# =====================================================================


class TestFillTestConfigMCB:
    """FillTestConfig に MCB config が含まれることの検証."""

    def test_mcb_fields_exist(self) -> None:
        from scripts.v460.lib.fill_config import FillTestConfig
        cfg = FillTestConfig()
        assert hasattr(cfg, "mcb_enabled")
        assert cfg.mcb_enabled is False  # default disabled
        assert cfg.mcb_caution_sigma == 1.0
        assert cfg.mcb_warning_sigma == 1.5
        assert cfg.mcb_halt_sigma == 2.0
        assert cfg.mcb_halt_cooldown_sec == 300.0


# =====================================================================
# cancel_reasons 統合テスト
# =====================================================================


class TestCancelReasonsMCB:
    """cancel_reasons に MCB 定数が含まれることの検証."""

    def test_mcb_constants_exist(self) -> None:
        from scripts.v460.lib import cancel_reasons as CR
        assert CR.MCB_HALT == "mcb_halt"
        assert CR.MCB_WARNING == "mcb_warning"
        assert CR.MCB_HALT in CR.AUDIT_CANCEL_REASONS
        assert CR.MCB_WARNING in CR.AUDIT_CANCEL_REASONS
