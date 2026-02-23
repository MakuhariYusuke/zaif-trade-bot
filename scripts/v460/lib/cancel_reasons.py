"""145# §9-#6: cancel_reason 定数集約.

cancel_reason 文字列が run_fill_test, fill_quality, skip_gate_evaluator,
order_monitor 等に散在し、命名変更時のドリフトリスクが高い。
全レイヤで同一参照に統一するため定数化。

カテゴリ:
  AUDIT  — 監査系・スキップ系 (quarantine bypass 対象)
  EXEC   — 実行系 (発注結果)
  GUARD  — 保護系 (circuit_breaker, timeout 等)
"""

from __future__ import annotations

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
SELL_DYNAMIC_KILL = "sell_dynamic_kill"
TRENDING_SELL_SKIP = "trending_sell_skip"  # 155# §9: trending regime sell 抑制
SKIP_GATE = "skip_gate"

AUDIT_CANCEL_REASONS: frozenset[str] = frozenset({
    CIRCUIT_BREAKER_OPEN,
    PREFLIGHT_PAUSE,
    PREFLIGHT_INSUFFICIENT,
    TIME_FILTER_BOTH_SIDES,
    TIME_FILTER_086_DEADLOCK,
    NARROW_SPREAD_PAUSE,
    BALANCE_FORCED_SKIP,
    UNKNOWN_REGIME_BUY_SKIP,
    SELL_DYNAMIC_KILL,
    TRENDING_SELL_SKIP,
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

# ======================================================================
# ORDERBOOK: 板取得エラー細分化
# ======================================================================
ORDERBOOK_ERROR = "orderbook_error"
ORDERBOOK_TIMEOUT = "orderbook_timeout"
ORDERBOOK_RATE_LIMIT = "orderbook_rate_limit"
ORDERBOOK_EMPTY = "orderbook_empty"
SELL_GUARD_REJECT = "sell_guard_reject"
