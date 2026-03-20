from __future__ import annotations

from collections.abc import Iterable


def get_trade_field(
    trade: object,
    *,
    key: str,
    fallback_key: str | None = None,
    default: object = None,
) -> object:
    """dict / object の両方から trade field を取得する."""
    if isinstance(trade, dict):
        if key in trade:
            return trade[key]
        if fallback_key is not None and fallback_key in trade:
            return trade[fallback_key]
        return default

    value = getattr(trade, key, default)
    if value is default and fallback_key is not None:
        value = getattr(trade, fallback_key, default)
    return value


def normalize_recent_trades(
    trades: object,
    *,
    fallback_timestamp: float,
) -> list[dict[str, object]] | None:
    """adapter の recent_trades を skip-gate 共通形式へ正規化する."""
    if not trades:
        return None

    try:
        iterator = iter(trades if isinstance(trades, Iterable) else [])
    except TypeError:
        return None

    normalized: list[dict[str, object]] = []
    for trade in iterator:
        if trade is None:
            continue
        normalized.append(
            {
                "ts": get_trade_field(
                    trade,
                    key="timestamp",
                    fallback_key="ts",
                    default=fallback_timestamp,
                ),
                "price": get_trade_field(
                    trade,
                    key="price",
                    default=0.0,
                ),
                "amount": get_trade_field(
                    trade,
                    key="amount",
                    fallback_key="quantity",
                    default=0.0,
                ),
                "side": get_trade_field(
                    trade,
                    key="side",
                    default="buy",
                ),
            }
        )
    return normalized or None


__all__ = [
    "get_trade_field",
    "normalize_recent_trades",
]
