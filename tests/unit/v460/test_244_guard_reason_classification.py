"""244# Guard Reason Classification テスト.

guard_fire_counts の reason を market/system/recovery に分類する
guard_reason_classifier モジュールの単体テスト + CycleGateResult 統合テスト。
"""

from __future__ import annotations

import pytest

from scripts.v460.lib.guard_reason_classifier import (
    GuardCategory,
    categorize_guard_fire_counts,
    classify_guard,
    guard_category_totals,
)


# ============================================================
# A. classify_guard — 個別分類
# ============================================================
class TestClassifyGuard244:
    """classify_guard の分類精度テスト."""

    @pytest.mark.parametrize(
        "guard_name, expected",
        [
            # Market guards
            ("gate_unknown_regime_buy_skip", GuardCategory.MARKET),
            ("gate_trending_sell_skip", GuardCategory.MARKET),
            ("gate_buy_dynamic_kill", GuardCategory.MARKET),
            ("gate_sell_dynamic_kill", GuardCategory.MARKET),
            ("gate_narrow_spread_pause", GuardCategory.MARKET),
            ("gate_sell_guard_reject", GuardCategory.MARKET),
            ("mcb_halt", GuardCategory.MARKET),
            ("sad_frozen", GuardCategory.MARKET),
            ("sad_dry", GuardCategory.MARKET),
            ("sad_wide", GuardCategory.MARKET),
            ("toxic_veto_block", GuardCategory.MARKET),
            ("quiescence", GuardCategory.MARKET),
            # System guards
            ("dd_halt", GuardCategory.SYSTEM),
            ("per_side_dd_both_halt", GuardCategory.SYSTEM),
            # 286# 283# MEDIUM-4: SYSTEM → RECOVERY 再分類
            ("preflight_insufficient", GuardCategory.SYSTEM),
            ("one_sided_freeze_skip", GuardCategory.RECOVERY),
            ("hard_skip_utc", GuardCategory.SYSTEM),
            ("phantom_position_detected", GuardCategory.SYSTEM),
            ("phantom_veto_block", GuardCategory.SYSTEM),
            ("degraded_liquidation_active", GuardCategory.RECOVERY),
            # 286# 新規 guard reasons
            # 244# SR-1: toxic_veto_set は市場都合
            ("toxic_veto_set", GuardCategory.MARKET),
            # Recovery guards
            ("dynamic_kill_probe_sell", GuardCategory.RECOVERY),
            ("dynamic_kill_probe_buy", GuardCategory.RECOVERY),
            ("dynamic_kill_force_release_sell", GuardCategory.RECOVERY),
            ("dual_kill_bypass", GuardCategory.RECOVERY),
            ("per_side_halt_recovery_active", GuardCategory.RECOVERY),
        ],
    )
    def test_known_guards(self, guard_name: str, expected: GuardCategory) -> None:
        assert classify_guard(guard_name) == expected

    def test_unknown_guard_defaults_to_system(self) -> None:
        """未知の guard は保守的に SYSTEM 扱い."""
        assert classify_guard("totally_unknown_guard") == GuardCategory.SYSTEM


# ============================================================
# B. guard_category_totals — カテゴリ集計
# ============================================================
class TestGuardCategoryTotals244:
    """guard_category_totals の集計精度テスト."""

    def test_empty_returns_zeros(self) -> None:
        result = guard_category_totals(None)
        assert result == {"market": 0, "system": 0, "recovery": 0}

    def test_empty_dict_returns_zeros(self) -> None:
        result = guard_category_totals({})
        assert result == {"market": 0, "system": 0, "recovery": 0}

    def test_mixed_counts(self) -> None:
        counts = {
            "mcb_halt": 3,
            "sad_frozen": 2,
            "dd_halt": 5,
            "preflight_insufficient": 1,
            "dynamic_kill_probe_sell": 4,
        }
        result = guard_category_totals(counts)
        assert result["market"] == 5  # mcb_halt(3) + sad_frozen(2)
        assert result["system"] == 6  # dd_halt(5) + preflight(1)
        assert result["recovery"] == 4  # probe_sell(4)


# ============================================================
# C. categorize_guard_fire_counts — カテゴリ別内訳
# ============================================================
class TestCategorizeGuardFireCounts244:
    """categorize_guard_fire_counts の内訳分解テスト."""

    def test_empty(self) -> None:
        result = categorize_guard_fire_counts(None)
        assert result == {"market": {}, "system": {}, "recovery": {}}

    def test_categorization(self) -> None:
        counts = {
            "gate_trending_sell_skip": 10,
            "dd_halt": 2,
            "dual_kill_bypass": 1,
        }
        result = categorize_guard_fire_counts(counts)
        assert result["market"] == {"gate_trending_sell_skip": 10}
        assert result["system"] == {"dd_halt": 2}
        assert result["recovery"] == {"dual_kill_bypass": 1}


# ============================================================
# D. CycleGateResult.blocking_category 統合テスト
# ============================================================
class TestCycleGateResultBlockingCategory244:
    """CycleGateResult.blocking_category プロパティ統合テスト."""

    def test_empty_blocking_reason(self) -> None:
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateResult
        r = CycleGateResult()
        assert r.blocking_category == ""

    def test_market_category(self) -> None:
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateResult
        r = CycleGateResult(blocked=True, blocking_reason="trending_sell_skip")
        assert r.blocking_category == "market"

    def test_system_category_for_unknown(self) -> None:
        """未知の gate reason は system 扱い."""
        from scripts.v460.lib.cycle_gate_aggregator import CycleGateResult
        r = CycleGateResult(blocked=True, blocking_reason="some_new_gate")
        assert r.blocking_category == "system"


# ============================================================
# E. GuardCategory enum values
# ============================================================
class TestGuardCategoryEnum244:
    """GuardCategory の値の安定性."""

    def test_values(self) -> None:
        assert GuardCategory.MARKET.value == "market"
        assert GuardCategory.SYSTEM.value == "system"
        assert GuardCategory.RECOVERY.value == "recovery"

    def test_all_categories(self) -> None:
        """3 カテゴリのみ存在."""
        assert len(GuardCategory) == 3
