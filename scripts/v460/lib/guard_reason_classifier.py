"""244# Guard Reason Classification (232# P2-2).

guard_fire_counts の各 reason を「市場都合 (MARKET)」と「システム都合 (SYSTEM)」に
分類し、運用判断を支援する。

設計思想:
  - Market guards: 外部市場環境（ボラ、スプレッド、逆選択リスク）に基づく判定。
    発火は「市場が危険」を意味し、正常な防御動作。
  - System guards: 内部システム状態（DD上限、残高不足、時間フィルター等）に基づく判定。
    発火は「システム制約で動けない」を意味し、設定見直しの示唆。
  - Recovery guards: kill/halt からの回復動作。発火は「システムが自己修復中」。
"""

from __future__ import annotations

import enum
from typing import Final


class GuardCategory(enum.Enum):
    """Guard 発火カテゴリ."""

    MARKET = "market"      # 市場都合: 外部環境に起因する防御
    SYSTEM = "system"      # システム都合: 内部制約に起因する停止
    RECOVERY = "recovery"  # 回復動作: kill/halt からの復帰


# ────────────────────────────────────────────────────────────────
# 分類マッピング (guard_name → GuardCategory)
# ────────────────────────────────────────────────────────────────
#  gate_* prefix は CycleGateAggregator 経由の発火 (orchestrator L1651)
# ────────────────────────────────────────────────────────────────

_CLASSIFICATION: Final[dict[str, GuardCategory]] = {
    # ── CycleGate 経由: 市場都合 ──
    "gate_unknown_regime_buy_skip": GuardCategory.MARKET,
    "gate_ranging_low_vol_skip": GuardCategory.MARKET,
    "gate_trending_sell_skip": GuardCategory.MARKET,
    "gate_buy_dynamic_kill": GuardCategory.MARKET,
    "gate_sell_dynamic_kill": GuardCategory.MARKET,
    "gate_rule_velocity_sell_skip": GuardCategory.MARKET,
    "gate_rule_velocity_buy_skip": GuardCategory.MARKET,
    "gate_rule_skip_unknown_sell": GuardCategory.MARKET,
    "gate_narrow_spread_pause": GuardCategory.MARKET,
    "gate_spread_too_narrow": GuardCategory.MARKET,
    "gate_sell_guard_reject": GuardCategory.MARKET,
    "gate_toxicity_participation_skip": GuardCategory.MARKET,
    # ── Orchestrator 直接: 市場都合 ──
    "mcb_halt": GuardCategory.MARKET,
    "mcb_warning": GuardCategory.MARKET,
    "mcb_sad_escalation": GuardCategory.MARKET,
    "sad_frozen": GuardCategory.MARKET,
    "sad_dry": GuardCategory.MARKET,
    "sad_wide": GuardCategory.MARKET,
    "toxic_veto_block": GuardCategory.MARKET,
    "toxicity_participation_skip": GuardCategory.MARKET,
    "quiescence": GuardCategory.MARKET,
    # ── システム都合 ──
    "dd_halt": GuardCategory.SYSTEM,
    "per_side_dd_both_halt": GuardCategory.SYSTEM,
    "per_side_halt_switch": GuardCategory.SYSTEM,
    "balance_forced_halt_block": GuardCategory.SYSTEM,
    "preflight_insufficient": GuardCategory.SYSTEM,
    "one_sided_freeze_skip": GuardCategory.SYSTEM,
    "one_sided_cooldown_skip": GuardCategory.SYSTEM,
    "hard_skip_utc": GuardCategory.SYSTEM,
    "time_filter_both_sides": GuardCategory.SYSTEM,
    "phantom_position_detected": GuardCategory.SYSTEM,
    "phantom_veto_block": GuardCategory.SYSTEM,
    "day_reset_kill_conflict": GuardCategory.SYSTEM,
    "degraded_liquidation_duty_skip": GuardCategory.SYSTEM,
    "degraded_liquidation_active": GuardCategory.SYSTEM,
    # ── 回復動作 ──
    "dynamic_kill_probe_sell": GuardCategory.RECOVERY,
    "dynamic_kill_probe_buy": GuardCategory.RECOVERY,
    "dynamic_kill_force_release_sell": GuardCategory.RECOVERY,
    "dynamic_kill_force_release_buy": GuardCategory.RECOVERY,
    "dual_kill_bypass": GuardCategory.RECOVERY,
    "per_side_halt_recovery_active": GuardCategory.RECOVERY,
}


def classify_guard(guard_name: str) -> GuardCategory:
    """guard_name を GuardCategory に分類.

    未知の guard_name は SYSTEM (保守的) として扱う。
    """
    return _CLASSIFICATION.get(guard_name, GuardCategory.SYSTEM)


def categorize_guard_fire_counts(
    counts: dict[str, int] | None,
) -> dict[str, dict[str, int]]:
    """guard_fire_counts を category 別に集約.

    Returns:
        {"market": {"mcb_halt": 3, ...}, "system": {...}, "recovery": {...}}
    """
    result: dict[str, dict[str, int]] = {
        cat.value: {} for cat in GuardCategory
    }
    if not counts:
        return result
    for guard_name, count in counts.items():
        cat = classify_guard(guard_name)
        result[cat.value][guard_name] = count
    return result


def guard_category_totals(
    counts: dict[str, int] | None,
) -> dict[str, int]:
    """guard_fire_counts のカテゴリ別合計.

    Returns:
        {"market": 45, "system": 12, "recovery": 3}
    """
    totals: dict[str, int] = {cat.value: 0 for cat in GuardCategory}
    if not counts:
        return totals
    for guard_name, count in counts.items():
        cat = classify_guard(guard_name)
        totals[cat.value] += count
    return totals
