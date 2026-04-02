from __future__ import annotations

from typing import Protocol

import numpy as np


class SupportsFillRecordPnl(Protocol):
    filled: bool
    post_fill_30s_pnl: float | None
    fill_price: float | None
    order_quantity: float


def compute_record_pnl_jpy(record: SupportsFillRecordPnl) -> float | None:
    """FillRecord 互換オブジェクトの 30s PnL を JPY 概算へ変換する."""
    if not record.filled:
        return None
    if record.post_fill_30s_pnl is None or record.fill_price is None:
        return None
    pnl_bps = float(record.post_fill_30s_pnl)
    fill_price = float(record.fill_price)
    order_qty = float(record.order_quantity)
    if not (np.isfinite(pnl_bps) and np.isfinite(fill_price) and np.isfinite(order_qty)):
        return None
    return pnl_bps * 1e-4 * fill_price * order_qty


__all__ = ["SupportsFillRecordPnl", "compute_record_pnl_jpy"]
