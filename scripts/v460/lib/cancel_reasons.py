"""145# §9-#6: cancel_reason 定数集約.

cancel_reason 文字列が run_fill_test, fill_quality, skip_gate_evaluator,
order_monitor 等に散在し、命名変更時のドリフトリスクが高い。
全レイヤで同一参照に統一するため定数化。

カテゴリ:
  AUDIT  — 監査系・スキップ系 (quarantine bypass 対象)
  EXEC   — 実行系 (発注結果)
  GUARD  — 保護系 (circuit_breaker, timeout 等)

173#: CancelReason Literal 型エイリアスを追加。型安全性を向上。
"""

from __future__ import annotations

from typing import Literal

# ======================================================================
# AUDIT: quarantine bypass 対象
# ======================================================================
CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
PREFLIGHT_PAUSE = "preflight_pause"
PREFLIGHT_INSUFFICIENT = "preflight_insufficient"
TIME_FILTER_BOTH_SIDES = "time_filter_both_sides"
TIME_FILTER_086_DEADLOCK = "time_filter_086_deadlock"
NARROW_SPREAD_PAUSE = "narrow_spread_pause"
# 348# balance_forced 撤廃: BALANCE_FORCED_SKIP を削除
UNKNOWN_REGIME_BUY_SKIP = "unknown_regime_buy_skip"
UNKNOWN_REGIME_SELL_SKIP = "unknown_regime_sell_skip"  # 156# §16: buy側と対称化
SELL_DYNAMIC_KILL = "sell_dynamic_kill"
BUY_DYNAMIC_KILL = "buy_dynamic_kill"  # 157# §19: buy 側動的 kill
TRENDING_SELL_SKIP = "trending_sell_skip"  # 155# §9: trending regime sell 抑制
RANGING_LOW_VOL_SKIP = "ranging_low_vol_skip"  # 169# B1': ranging_buy at low_vol hard skip
SKIP_GATE = "skip_gate"
# 165# AS-R1: velocity-based skip
SKIP_GATE_RULE_VELOCITY_SELL = "skip_gate_rule_velocity_sell"
SKIP_GATE_RULE_VELOCITY_BUY = "skip_gate_rule_velocity_buy"
# 168# §4.1 #3: daily drawdown halt
DAILY_DRAWDOWN_HALT = "daily_drawdown_halt"
# 200# B/I: postonly_guard crossing → skip (offset pipeline 無効化 防止)
POSTONLY_CROSSING_SKIP = "postonly_crossing_skip"
# 205# §9.4: 時間帯 Hard Skip (Kyle proxy — 最悪時間帯は取引完全停止)
HARD_SKIP_UTC_HOUR = "hard_skip_utc_hour"
# 205# §9.2: Toxic Fill 同一サイド拒否 (大損後に同一方向を N サイクル封鎖)
TOXIC_FILL_SIDE_VETO = "toxic_fill_side_veto"
# 238# S-2: Phantom 検出後の同 side 一時拒否 (逆選択防御)
PHANTOM_SIDE_VETO = "phantom_side_veto"
# 240# Toxicity Budget: 確率的不参加 (232# §2.2 Glosten-Milgrom)
TOXICITY_PARTICIPATION_SKIP = "toxicity_participation_skip"
# 205# §9.5: 片側 DD Halt (サイド別累積損失超過で片側封鎖)
PER_SIDE_DD_HALT = "per_side_dd_halt"
# 215# P0-C: alert_mode.json によるオペレータ緊急停止
OPERATOR_HALT = "operator_halt"
# 211# P1-B: Micro Circuit Breaker (短期価格急変の自動防御)
MCB_HALT = "mcb_halt"
MCB_WARNING = "mcb_warning"
# 211# P1-C: Spread Anomaly Detector (流動性枯渇検知)
SAD_FROZEN = "sad_frozen"
SAD_DRY = "sad_dry"
# 211# P1-D: MCB×SAD AND Escalation (両方 WARNING 以上で即 HALT)
MCB_SAD_ESCALATION = "mcb_sad_escalation"
# 234# 縮退清算 + one-sided エスカレーション
DEGRADED_LIQUIDATION_DUTY_SKIP = "degraded_liquidation_duty_skip"
ONE_SIDED_COOLDOWN_SKIP = "one_sided_cooldown_skip"
ONE_SIDED_FREEZE_SKIP = "one_sided_freeze_skip"
# 294# forced_buy_delay (regime-aware 購入遅延)
# 348# balance_forced 撤廃: FORCED_BUY_DELAY を削除
# 296# skip_gate_rule: unknown regime sell skip
SKIP_GATE_RULE_UNKNOWN_SELL = "skip_gate_rule_unknown_sell"
# 373# F9: order_monitor poll エラー連続限度超過
POLL_ERROR_LIMIT = "poll_error_limit"
# 418# P0: Execution Final Clamp hard skip (全 multiplier 適用後に offset が極端)
FINAL_CLAMP_HARD_SKIP = "final_clamp_hard_skip"
# 418# P0: Route-to-Kill Deadlock (buy残高不足×sell kill-gated ⇒ 両側封鎖)
ROUTE_TO_KILL_DEADLOCK = "route_to_kill_deadlock"

AUDIT_CANCEL_REASONS: frozenset[str] = frozenset({
    CIRCUIT_BREAKER_OPEN,
    PREFLIGHT_PAUSE,
    PREFLIGHT_INSUFFICIENT,
    TIME_FILTER_BOTH_SIDES,
    TIME_FILTER_086_DEADLOCK,
    NARROW_SPREAD_PAUSE,
    UNKNOWN_REGIME_BUY_SKIP,
    UNKNOWN_REGIME_SELL_SKIP,
    SELL_DYNAMIC_KILL,
    BUY_DYNAMIC_KILL,
    TRENDING_SELL_SKIP,
    RANGING_LOW_VOL_SKIP,
    SKIP_GATE,
    SKIP_GATE_RULE_VELOCITY_SELL,
    SKIP_GATE_RULE_VELOCITY_BUY,
    DAILY_DRAWDOWN_HALT,
    POSTONLY_CROSSING_SKIP,
    HARD_SKIP_UTC_HOUR,
    TOXIC_FILL_SIDE_VETO,
    PHANTOM_SIDE_VETO,
    TOXICITY_PARTICIPATION_SKIP,
    PER_SIDE_DD_HALT,
    OPERATOR_HALT,
    MCB_HALT,
    MCB_WARNING,
    SAD_FROZEN,
    SAD_DRY,
    MCB_SAD_ESCALATION,
    DEGRADED_LIQUIDATION_DUTY_SKIP,
    ONE_SIDED_COOLDOWN_SKIP,
    ONE_SIDED_FREEZE_SKIP,
    SKIP_GATE_RULE_UNKNOWN_SELL,
    POLL_ERROR_LIMIT,
    FINAL_CLAMP_HARD_SKIP,
    ROUTE_TO_KILL_DEADLOCK,
})

# ======================================================================
# EXEC: 発注実行結果
# ======================================================================
POST_ONLY_REJECT = "post_only_reject"
INSUFFICIENT_FUNDS = "insufficient_funds"
MINIMUM_SIZE = "minimum_size"
API_ERROR = "api_error"
TIMEOUT = "timeout"
UNKNOWN = "unknown"

# ======================================================================
# GUARD: 保護系 (order_monitor)
# ======================================================================
STALE_SKIP_GATE_BLOCKED = "stale_skip_gate_blocked"
STALE_REPRICE_FAILED = "stale_reprice_failed"
STALE_ADVERSE_DRIFT = "stale_adverse_drift"  # 200# P0-1: 不利方向 drift cancel-only

# ======================================================================
# ORDERBOOK: 板取得エラー細分化
# ======================================================================
ORDERBOOK_ERROR = "orderbook_error"
ORDERBOOK_TIMEOUT = "orderbook_timeout"
ORDERBOOK_RATE_LIMIT = "orderbook_rate_limit"
ORDERBOOK_EMPTY = "orderbook_empty"
SELL_GUARD_REJECT = "sell_guard_reject"
SPREAD_TOO_NARROW = "spread_too_narrow"  # 158# §20-D
NO_FEASIBLE_QUOTE = "no_feasible_quote"  # 234# 制約集合崩壊: 有効価格域なし

# ======================================================================
# 173# 型安全: CancelReason Literal 型エイリアス
# すべての有効な cancel_reason 値を列挙。新規追加時はここにも追記。
# ======================================================================
CancelReason = Literal[
    # AUDIT
    "circuit_breaker_open",
    "preflight_pause",
    "preflight_insufficient",
    "time_filter_both_sides",
    "time_filter_086_deadlock",
    "narrow_spread_pause",
    "unknown_regime_buy_skip",
    "unknown_regime_sell_skip",
    "sell_dynamic_kill",
    "buy_dynamic_kill",
    "trending_sell_skip",
    "ranging_low_vol_skip",
    "skip_gate",
    "skip_gate_rule_velocity_sell",
    "skip_gate_rule_velocity_buy",
    "daily_drawdown_halt",
    "postonly_crossing_skip",
    "hard_skip_utc_hour",
    "toxic_fill_side_veto",
    "phantom_side_veto",
    "toxicity_participation_skip",
    "per_side_dd_halt",
    "operator_halt",
    "mcb_halt",
    "mcb_warning",
    "sad_frozen",
    "sad_dry",
    "mcb_sad_escalation",
    # EXEC
    "post_only_reject",
    "insufficient_funds",
    "minimum_size",
    "api_error",
    "timeout",
    "unknown",
    # GUARD
    "stale_skip_gate_blocked",
    "stale_reprice_failed",
    "stale_adverse_drift",
    # ORDERBOOK
    "orderbook_error",
    "orderbook_timeout",
    "orderbook_rate_limit",
    "orderbook_empty",
    "sell_guard_reject",
    "spread_too_narrow",
    "no_feasible_quote",
    # 234# 縮退清算 + one-sided エスカレーション
    "degraded_liquidation_duty_skip",
    "one_sided_cooldown_skip",
    "one_sided_freeze_skip",
    # 296# skip_gate_rule: unknown regime sell
    "skip_gate_rule_unknown_sell",
    # 373# F9: poll error limit
    "poll_error_limit",
    # 418# P0: Final Clamp + Route-to-Kill
    "final_clamp_hard_skip",
    "route_to_kill_deadlock",
]
