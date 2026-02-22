"""145# §9-#3 / §10.2-#1: OB (orderbook) 正規化ユーティリティ.

OrderBookSnapshot.bids/asks は list[tuple[float, float]] だが、
一部モジュール (skip_gate_evaluator) では .price/.quantity アクセスをしていた。
tuple/object 両対応の安全な抽出関数を提供し、散在する dual-format ロジックを一元化する。
"""

from __future__ import annotations

from typing import Any, Sequence


def extract_price(level: Any) -> float:
    """板レベルから price を抽出 (tuple / object 両対応)."""
    if isinstance(level, (list, tuple)):
        return float(level[0])
    return float(getattr(level, "price", 0.0))


def extract_size(level: Any) -> float:
    """板レベルから size (quantity) を抽出 (tuple / object 両対応)."""
    if isinstance(level, (list, tuple)):
        return float(level[1])
    return float(getattr(level, "quantity", getattr(level, "size", 0.0)))


def best_bid_ask(
    ob: Any,
) -> tuple[float | None, float | None]:
    """OrderBookSnapshot から best bid/ask を安全に抽出.

    Returns:
        (best_bid, best_ask) — データ不足時は None.
    """
    bid = extract_price(ob.bids[0]) if ob and ob.bids else None
    ask = extract_price(ob.asks[0]) if ob and ob.asks else None
    return bid, ask


def depth_volume(levels: Sequence[Any], depth: int = 5) -> float:
    """板の指定深さまでの合計出来高を計算."""
    return sum(extract_size(lv) for lv in levels[:depth])
