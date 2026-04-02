from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ztb.trading.common import cancel_reasons as CR


class SkipCategory(str, Enum):
    ENV_HALT = "env_halt"
    SIDE_BLOCK = "side_block"
    GATE_BLOCK = "gate_block"
    RESOURCE = "resource"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True)
class CancelReasonMeta:
    reason: str
    category: SkipCategory
    expected_update_last_side: bool
    description: str


def _meta(
    reason: str,
    category: SkipCategory,
    expected_update_last_side: bool,
    description: str,
) -> CancelReasonMeta:
    return CancelReasonMeta(
        reason=reason,
        category=category,
        expected_update_last_side=expected_update_last_side,
        description=description,
    )


CANCEL_REASON_REGISTRY: dict[str, CancelReasonMeta] = {
    CR.MCB_HALT: _meta(CR.MCB_HALT, SkipCategory.ENV_HALT, False, "Market circuit breaker halt"),
    CR.SAD_FROZEN: _meta(CR.SAD_FROZEN, SkipCategory.ENV_HALT, False, "Spread anomaly frozen"),
    CR.MCB_SAD_ESCALATION: _meta(
        CR.MCB_SAD_ESCALATION, SkipCategory.ENV_HALT, False, "MCB and SAD escalation"
    ),
    CR.PER_SIDE_DD_HALT: _meta(
        CR.PER_SIDE_DD_HALT, SkipCategory.ENV_HALT, False, "Per-side drawdown halt"
    ),
    CR.TOXIC_FILL_SIDE_VETO: _meta(
        CR.TOXIC_FILL_SIDE_VETO, SkipCategory.ENV_HALT, False, "Toxic fill side veto"
    ),
    CR.PHANTOM_SIDE_VETO: _meta(
        CR.PHANTOM_SIDE_VETO, SkipCategory.ENV_HALT, False, "Phantom side veto"
    ),
    CR.OPERATOR_HALT: _meta(CR.OPERATOR_HALT, SkipCategory.ENV_HALT, False, "Operator halt"),
    CR.PREFLIGHT_INSUFFICIENT: _meta(
        CR.PREFLIGHT_INSUFFICIENT, SkipCategory.RESOURCE, True, "Preflight insufficient funds"
    ),
    CR.PREFLIGHT_PAUSE: _meta(
        CR.PREFLIGHT_PAUSE, SkipCategory.RESOURCE, True, "Preflight pause before safe stop"
    ),
    CR.ONE_SIDED_FREEZE_SKIP: _meta(
        CR.ONE_SIDED_FREEZE_SKIP, SkipCategory.SIDE_BLOCK, True, "One-sided freeze"
    ),
    CR.ONE_SIDED_COOLDOWN_SKIP: _meta(
        CR.ONE_SIDED_COOLDOWN_SKIP, SkipCategory.SIDE_BLOCK, True, "One-sided cooldown"
    ),
    CR.TOXICITY_PARTICIPATION_SKIP: _meta(
        CR.TOXICITY_PARTICIPATION_SKIP, SkipCategory.GATE_BLOCK, True, "Toxicity participation skip"
    ),
    CR.DEGRADED_LIQUIDATION_DUTY_SKIP: _meta(
        CR.DEGRADED_LIQUIDATION_DUTY_SKIP,
        SkipCategory.GATE_BLOCK,
        True,
        "Degraded liquidation duty skip",
    ),
    CR.TIME_FILTER_BOTH_SIDES: _meta(
        CR.TIME_FILTER_BOTH_SIDES, SkipCategory.MAINTENANCE, False, "Time filter all sides"
    ),
    CR.TIME_FILTER_086_DEADLOCK: _meta(
        CR.TIME_FILTER_086_DEADLOCK,
        SkipCategory.MAINTENANCE,
        False,
        "Time filter deadlock protection",
    ),
    CR.HARD_SKIP_UTC_HOUR: _meta(
        CR.HARD_SKIP_UTC_HOUR, SkipCategory.MAINTENANCE, False, "Hard skip UTC hour"
    ),
    CR.SKIP_GATE: _meta(CR.SKIP_GATE, SkipCategory.GATE_BLOCK, True, "SkipGate block"),
    CR.CROSS_VENUE_LEAD_LAG_VETO: _meta(
        CR.CROSS_VENUE_LEAD_LAG_VETO, SkipCategory.GATE_BLOCK, True, "Cross-venue veto"
    ),
    CR.CROSS_VENUE_BUY_VETO: _meta(
        CR.CROSS_VENUE_BUY_VETO,
        SkipCategory.GATE_BLOCK,
        True,
        "694# Cross-venue buy-side protection veto",
    ),
    CR.AS_TRAILING_GATE_VETO: _meta(
        CR.AS_TRAILING_GATE_VETO,
        SkipCategory.GATE_BLOCK,
        True,
        "694# AS trailing rate gate veto",
    ),
    "entry_gate_ev_negative": _meta(
        "entry_gate_ev_negative", SkipCategory.GATE_BLOCK, True, "Entry gate negative EV"
    ),
    "composite_risk_exceeded": _meta(
        "composite_risk_exceeded", SkipCategory.GATE_BLOCK, True, "Composite risk gate"
    ),
    "ppo_sidecar_skip": _meta(
        "ppo_sidecar_skip", SkipCategory.GATE_BLOCK, True, "PPO sidecar skip"
    ),
    "ppo_sidecar_side_conflict": _meta(
        "ppo_sidecar_side_conflict", SkipCategory.GATE_BLOCK, True, "PPO side conflict"
    ),
}


def validate_skip_consistency(cancel_reason: str, update_last_side: bool) -> bool:
    meta = CANCEL_REASON_REGISTRY.get(cancel_reason)
    if meta is None:
        return False
    return meta.expected_update_last_side == update_last_side
