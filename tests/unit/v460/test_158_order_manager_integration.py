"""P2-2: OrderManager.execute_trade 統合テスト.

158# backlog: execute_trade フルフローのモックベース統合テスト。
OrderManager → exchange_adapter → place_order の全パスをカバー。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from ztb.trading.live_trader.components.order_manager import OrderManager
from ztb.utils.errors import ValidationError as QuantityValidationError
from ztb.utils.exceptions.custom_exceptions import ValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_order_manager(
    *,
    demo_mode: bool = False,
    has_adapter: bool = True,
    last_valid_price: Optional[float] = 15_000_000.0,
) -> tuple[OrderManager, MagicMock]:
    """Create OrderManager with mocked LiveTrader."""
    live_trader = MagicMock()
    live_trader.demo_mode = demo_mode
    live_trader._last_valid_price = last_valid_price
    live_trader._current_prices = {"btc_jpy": last_valid_price or 0.0}
    live_trader.position = "flat"
    live_trader._send_notification = MagicMock()

    if has_adapter:
        adapter = AsyncMock()
        adapter.place_order = AsyncMock(
            return_value=SimpleNamespace(order_id="ORD-001")
        )
        live_trader.exchange_adapter = adapter
    else:
        live_trader.exchange_adapter = None

    om = OrderManager(live_trader)
    return om, live_trader


@pytest.fixture
def demo_om() -> tuple[OrderManager, MagicMock]:
    return _make_order_manager(demo_mode=True)


@pytest.fixture
def live_om() -> tuple[OrderManager, MagicMock]:
    return _make_order_manager(demo_mode=False)


# ---------------------------------------------------------------------------
# Demo mode tests
# ---------------------------------------------------------------------------

class TestDemoMode:
    """Demo mode bypasses exchange and returns True."""

    def test_demo_buy_returns_true(self, demo_om: tuple) -> None:
        om, lt = demo_om
        assert om.execute_trade("buy", 0.001) is True

    def test_demo_sell_returns_true(self, demo_om: tuple) -> None:
        om, lt = demo_om
        assert om.execute_trade("sell", 0.001) is True

    def test_demo_does_not_call_exchange(self, demo_om: tuple) -> None:
        om, lt = demo_om
        om.execute_trade("buy", 0.001)
        # exchange_adapter should NOT be accessed for place_order
        adapter = getattr(lt, "exchange_adapter", None)
        if adapter is not None:
            adapter.place_order.assert_not_called()

    def test_demo_sends_notification(self, demo_om: tuple) -> None:
        om, lt = demo_om
        om.execute_trade("buy", 0.001)
        lt._send_notification.assert_called_once()
        args = lt._send_notification.call_args
        assert "Demo" in args[0][0] or "demo" in args[0][0].lower() or "DEMO" in str(args)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Input validation before execution."""

    def test_invalid_side_raises(self, live_om: tuple) -> None:
        om, _ = live_om
        with pytest.raises(ValidationError, match="side"):
            om.execute_trade("hold", 0.001)

    def test_zero_amount_raises(self, live_om: tuple) -> None:
        om, _ = live_om
        with pytest.raises(QuantityValidationError, match="positive"):
            om.execute_trade("buy", 0.0)

    def test_negative_amount_raises(self, live_om: tuple) -> None:
        om, _ = live_om
        with pytest.raises(QuantityValidationError, match="positive"):
            om.execute_trade("buy", -0.001)

    def test_validate_trade_parameters_valid(self, live_om: tuple) -> None:
        om, _ = live_om
        om.validate_trade_parameters("buy", 0.001)  # should not raise

    def test_validate_trade_parameters_invalid_side(self, live_om: tuple) -> None:
        om, _ = live_om
        with pytest.raises(ValidationError, match="Invalid trade side"):
            om.validate_trade_parameters("hold", 0.001)


# ---------------------------------------------------------------------------
# Live mode — success path
# ---------------------------------------------------------------------------

class TestLiveModeSuccess:
    """Live mode success: exchange_adapter.place_order returns Order."""

    def test_buy_returns_true(self, live_om: tuple) -> None:
        om, lt = live_om
        assert om.execute_trade("buy", 0.001) is True

    def test_sell_returns_true(self, live_om: tuple) -> None:
        om, lt = live_om
        assert om.execute_trade("sell", 0.001) is True

    def test_place_order_called_with_correct_args(self, live_om: tuple) -> None:
        om, lt = live_om
        om.execute_trade("buy", 0.001)
        adapter = lt.exchange_adapter
        adapter.place_order.assert_called_once_with(
            symbol="btc_jpy",
            side="buy",
            quantity=0.001,
            price=15_000_000.0,
            order_type="limit",
        )

    def test_success_notification_sent(self, live_om: tuple) -> None:
        om, lt = live_om
        om.execute_trade("sell", 0.001)
        # At least one notification should be sent on success
        assert lt._send_notification.call_count >= 1
        # Check for success-related notification
        any_success = any(
            "Order" in str(c) or "order" in str(c).lower()
            for c in lt._send_notification.call_args_list
        )
        assert any_success, "Expected order-placed notification"


# ---------------------------------------------------------------------------
# Live mode — no exchange adapter
# ---------------------------------------------------------------------------

class TestLiveModeNoAdapter:
    """No exchange_adapter → returns False."""

    def test_returns_false(self) -> None:
        om, lt = _make_order_manager(demo_mode=False, has_adapter=False)
        assert om.execute_trade("buy", 0.001) is False

    def test_sends_error_notification(self) -> None:
        om, lt = _make_order_manager(demo_mode=False, has_adapter=False)
        om.execute_trade("buy", 0.001)
        lt._send_notification.assert_called_once()
        err_msg = str(lt._send_notification.call_args)
        assert "adapter" in err_msg.lower() or "Adapter" in err_msg


# ---------------------------------------------------------------------------
# Live mode — no valid price
# ---------------------------------------------------------------------------

class TestLiveModeNoPrice:
    """No valid price → returns False."""

    def test_returns_false_when_price_none(self) -> None:
        om, lt = _make_order_manager(
            demo_mode=False, has_adapter=True, last_valid_price=None
        )
        lt._current_prices = {"btc_jpy": 0.0}
        assert om.execute_trade("buy", 0.001) is False

    def test_returns_false_when_price_zero(self) -> None:
        om, lt = _make_order_manager(
            demo_mode=False, has_adapter=True, last_valid_price=0.0
        )
        lt._current_prices = {"btc_jpy": 0.0}
        assert om.execute_trade("buy", 0.001) is False

    def test_sends_price_error_notification(self) -> None:
        om, lt = _make_order_manager(
            demo_mode=False, has_adapter=True, last_valid_price=None
        )
        lt._current_prices = {"btc_jpy": 0.0}
        om.execute_trade("buy", 0.001)
        assert lt._send_notification.call_count >= 1
        err_msg = str(lt._send_notification.call_args)
        assert "price" in err_msg.lower() or "Price" in err_msg


# ---------------------------------------------------------------------------
# Live mode — exchange raises exception
# ---------------------------------------------------------------------------

class TestLiveModeExchangeError:
    """exchange_adapter.place_order raises → returns False."""

    def test_returns_false_on_exchange_exception(self, live_om: tuple) -> None:
        om, lt = live_om
        lt.exchange_adapter.place_order = AsyncMock(
            side_effect=ConnectionError("API timeout")
        )
        assert om.execute_trade("buy", 0.001) is False

    def test_sends_error_notification_on_exception(self, live_om: tuple) -> None:
        om, lt = live_om
        lt.exchange_adapter.place_order = AsyncMock(
            side_effect=ConnectionError("API timeout")
        )
        om.execute_trade("buy", 0.001)
        assert lt._send_notification.call_count >= 1
        err_msg = str(lt._send_notification.call_args)
        assert "Error" in err_msg or "error" in err_msg.lower()


# ---------------------------------------------------------------------------
# Live mode — place_order returns None
# ---------------------------------------------------------------------------

class TestLiveModeOrderNone:
    """place_order returns None → returns False."""

    def test_returns_false_when_order_none(self, live_om: tuple) -> None:
        om, lt = live_om
        lt.exchange_adapter.place_order = AsyncMock(return_value=None)
        assert om.execute_trade("buy", 0.001) is False

    def test_no_success_notification_when_order_none(self, live_om: tuple) -> None:
        om, lt = live_om
        lt.exchange_adapter.place_order = AsyncMock(return_value=None)
        om.execute_trade("buy", 0.001)
        # Should not have success notification, but may have error notification
        for c in lt._send_notification.call_args_list:
            assert "Live Order Placed" not in str(c)


# ---------------------------------------------------------------------------
# _execute_order_async bridge
# ---------------------------------------------------------------------------

class TestExecuteOrderAsync:
    """Test sync→async bridge in _execute_order_async."""

    def test_returns_order_on_success(self, live_om: tuple) -> None:
        om, lt = live_om
        order = om._execute_order_async(
            exchange_adapter=lt.exchange_adapter,
            symbol="btc_jpy",
            side="buy",
            quantity=0.001,
            price=15_000_000.0,
            order_type="limit",
        )
        assert order is not None
        assert order.order_id == "ORD-001"

    def test_raises_on_adapter_failure(self, live_om: tuple) -> None:
        om, lt = live_om
        lt.exchange_adapter.place_order = AsyncMock(
            side_effect=RuntimeError("network error")
        )
        with pytest.raises(RuntimeError, match="network error"):
            om._execute_order_async(
                exchange_adapter=lt.exchange_adapter,
                symbol="btc_jpy",
                side="sell",
                quantity=0.001,
                price=15_000_000.0,
            )


# ---------------------------------------------------------------------------
# get_trade_info
# ---------------------------------------------------------------------------

class TestGetTradeInfo:
    """get_trade_info returns correct state dict."""

    def test_returns_expected_keys(self, live_om: tuple) -> None:
        om, lt = live_om
        lt.position = "long"
        lt.entry_price = 15_000_000.0
        lt.total_pnl = 1234.5
        lt.trades_count = 7
        lt.demo_mode = False

        info = om.get_trade_info()
        assert info == {
            "position": "long",
            "entry_price": 15_000_000.0,
            "total_pnl": 1234.5,
            "trades_count": 7,
            "demo_mode": False,
        }
