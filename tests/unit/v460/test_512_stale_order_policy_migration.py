from __future__ import annotations

from scripts.v460.lib.order_monitor import _CancelFillCheck, _parse_order_state
from ztb.trading.execution.stale_order_policy import (
    CancelFillCheck,
    ORDER_STATE_FILLED,
    ORDER_STATE_PENDING,
    parse_order_state,
)


class TestStaleOrderPolicyMigration:
    def test_parse_order_state_shim_points_to_canonical(self) -> None:
        assert _parse_order_state is parse_order_state
        assert _parse_order_state("filled") == ORDER_STATE_FILLED
        assert _parse_order_state("mystery") == ORDER_STATE_PENDING

    def test_cancel_fill_check_shim_points_to_canonical(self) -> None:
        assert _CancelFillCheck is CancelFillCheck
        result = _CancelFillCheck(was_filled=True, fill_price=1.0)
        assert result.was_filled is True
        assert result.fill_price == 1.0
