from __future__ import annotations

from typing import Final


ORDER_STATE_FILLED: Final[str] = "filled"
ORDER_STATE_CANCELLED: Final[str] = "cancelled"
ORDER_STATE_REJECTED: Final[str] = "rejected"
ORDER_STATE_PENDING: Final[str] = "pending"
ORDER_STATE_CONFIRMED: Final[str] = "confirmed"
ORDER_STATE_PARTIAL: Final[str] = "partial"

_STATUS_TO_STATE: dict[str, str] = {
    "filled": ORDER_STATE_FILLED,
    "cancelled": ORDER_STATE_CANCELLED,
    "rejected": ORDER_STATE_REJECTED,
    "pending": ORDER_STATE_PENDING,
    "confirmed": ORDER_STATE_CONFIRMED,
    "partial": ORDER_STATE_PARTIAL,
}


def parse_order_state(status_str: str) -> str:
    """exchange status 文字列を内部比較用の正規化文字列へ変換."""
    return _STATUS_TO_STATE.get(status_str, ORDER_STATE_PENDING)


class CancelFillCheck:
    """cancel → fill recheck 結果."""

    __slots__ = ("was_filled", "fill_price", "t_fill", "cancel_succeeded")

    def __init__(
        self,
        was_filled: bool = False,
        fill_price: float | None = None,
        t_fill: float | None = None,
        cancel_succeeded: bool = True,
    ) -> None:
        self.was_filled = was_filled
        self.fill_price = fill_price
        self.t_fill = t_fill
        self.cancel_succeeded = cancel_succeeded
