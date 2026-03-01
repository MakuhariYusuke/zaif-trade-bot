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
BALANCE_FORCED_SKIP = "balance_forced_skip"
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
# 205# §9.5: 片側 DD Halt (サイド別累積損失超過で片側封鎖)
PER_SIDE_DD_HALT = "per_side_dd_halt"

AUDIT_CANCEL_REASONS: frozenset[str] = frozenset({
    CIRCUIT_BREAKER_OPEN,
    PREFLIGHT_PAUSE,
    PREFLIGHT_INSUFFICIENT,
    TIME_FILTER_BOTH_SIDES,
    TIME_FILTER_086_DEADLOCK,
    NARROW_SPREAD_PAUSE,
    BALANCE_FORCED_SKIP,
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
    PER_SIDE_DD_HALT,
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
    "balance_forced_skip",
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
    "per_side_dd_halt",
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
]
