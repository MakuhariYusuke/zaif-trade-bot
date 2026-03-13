"""013# Appendix D fixes — unit tests.

Validates C-3/C-4/C-7/C-9/D-1/D-3/D-5 fixes applied to
CoincheckAdapter and OrderManager.
"""

import asyncio
import json
import urllib.parse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.unit.v460._fill_test_source import read_inspect_source
from ztb.trading.live.exchanges.bitflyer.adapter import BitFlyerAdapter
from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter
from ztb.trading.live_trader.components.order_manager import OrderManager


# ---------------------------------------------------------------------------
# C-3: Signature body consistency
# ---------------------------------------------------------------------------

class TestC3SignatureConsistency:
    """C-3: Signature must match actual request body."""

    def test_make_api_request_signs_urlencode_body(self):
        """Signature = nonce + url + urlencode(data), NOT json.dumps(data)."""
        adapter = CoincheckAdapter(
            api_key="test_key",
            api_secret="test_secret",
            dry_run=False,
        )

        data = {"pair": "btc_jpy", "order_type": "buy", "amount": "0.01", "rate": "5000000"}
        url = "https://coincheck.com/api/exchange/orders"

        # 146# §13: Mock the session returned by _get_session()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "id": 12345}
        mock_response.raise_for_status = MagicMock()
        mock_response.text = '{"success": true, "id": 12345}'

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response

        with patch.object(adapter, "_get_session", return_value=mock_session):
            adapter._make_api_request("POST", url, data)

            # Verify POST was called
            mock_session.post.assert_called_once()
            call_kwargs = mock_session.post.call_args

            # The data sent must be urlencode format
            sent_data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
            expected_body = urllib.parse.urlencode(data)
            assert sent_data == expected_body, (
                f"Sent body={sent_data!r} != expected urlencode={expected_body!r}"
            )

            # Verify the signature was computed over the SAME urlencode body
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            nonce = headers["ACCESS-NONCE"]
            signature = headers["ACCESS-SIGNATURE"]

            # Recompute expected signature
            expected_message = nonce + url + expected_body
            expected_signature = adapter._create_signature(expected_message)
            assert signature == expected_signature, (
                "Signature was not computed over urlencode body"
            )

    def test_signature_no_body_for_get(self):
        """GET requests: signature = nonce + url (no body)."""
        adapter = CoincheckAdapter(
            api_key="test_key",
            api_secret="test_secret",
            dry_run=False,
        )

        url = "https://coincheck.com/api/accounts/balance"

        # 146# §13: Mock the session returned by _get_session()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "jpy": "1000"}
        mock_response.raise_for_status = MagicMock()
        mock_response.text = '{"success": true}'

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch.object(adapter, "_get_session", return_value=mock_session):
            adapter._make_api_request("GET", url)

            headers = mock_session.get.call_args.kwargs.get("headers") or mock_session.get.call_args[1].get("headers")
            nonce = headers["ACCESS-NONCE"]
            signature = headers["ACCESS-SIGNATURE"]

            expected_message = nonce + url
            expected_signature = adapter._create_signature(expected_message)
            assert signature == expected_signature

    def test_old_json_dumps_not_used_in_signature(self):
        """Ensure json.dumps is NOT used for signature computation."""
        source = read_inspect_source(CoincheckAdapter._make_api_request)
        # json.dumps should not appear in the signature computation section
        # It may still be imported but should not be used for message construction
        assert "json.dumps(data" not in source, (
            "json.dumps(data) should not be used in _make_api_request for signature"
        )


# ---------------------------------------------------------------------------
# C-4: async/sync unification
# ---------------------------------------------------------------------------

class TestC4AsyncUnification:
    """C-4: All private API methods must use asyncio.to_thread.

    145# §13: CoincheckAdapter → BaseExchangeAdapter migration.
    public methods (place_order etc.) are now inherited from base;
    asyncio.to_thread は _xxx_real() に移動済み.
    """

    def test_place_order_real_uses_asyncio_to_thread(self):
        """_place_order_real must use asyncio.to_thread."""
        source = read_inspect_source(CoincheckAdapter._place_order_real)
        assert "asyncio.to_thread" in source, (
            "_place_order_real must use asyncio.to_thread for sync requests"
        )

    def test_cancel_order_real_uses_asyncio_to_thread(self):
        """_cancel_order_real must use asyncio.to_thread."""
        source = read_inspect_source(CoincheckAdapter._cancel_order_real)
        assert "asyncio.to_thread" in source, (
            "_cancel_order_real must use asyncio.to_thread for sync requests"
        )

    def test_get_balance_real_uses_asyncio_to_thread(self):
        """_get_balance_real must use asyncio.to_thread."""
        source = read_inspect_source(CoincheckAdapter._get_balance_real)
        assert "asyncio.to_thread" in source, (
            "_get_balance_real must use asyncio.to_thread for sync requests"
        )


# ---------------------------------------------------------------------------
# C-7: order_type mapping
# ---------------------------------------------------------------------------

class TestC7OrderTypeMapping:
    """C-7: Coincheck order_type must be buy/sell/market_buy/market_sell."""

    def test_limit_buy_sends_correct_order_type(self):
        """Limit buy: order_type='buy', rate set, time_in_force='post_only'."""
        adapter = CoincheckAdapter(
            api_key="k", api_secret="s", dry_run=False,
        )
        adapter._current_prices["btc_jpy"] = 5000000.0

        captured_data = {}
        def mock_make_request(method, url, data=None):
            captured_data.update(data or {})
            return {"success": True, "id": 123}

        async def _test():
            with patch.object(adapter, "_make_api_request", side_effect=mock_make_request):
                with patch.object(adapter, "_check_rate_limit", new_callable=AsyncMock):
                    with patch.object(adapter, "_simulate_delay", new_callable=AsyncMock):
                        await adapter.place_order(
                            symbol="btc_jpy", side="buy", quantity=0.01,
                            price=5000000, order_type="limit",
                        )

        asyncio.run(_test())

        assert captured_data["order_type"] == "buy"
        assert captured_data["rate"] == "5000000"
        assert captured_data["amount"] == "0.01"
        assert captured_data["time_in_force"] == "post_only"

    def test_limit_sell_sends_correct_order_type(self):
        """Limit sell: order_type='sell', rate set, time_in_force='post_only'."""
        adapter = CoincheckAdapter(
            api_key="k", api_secret="s", dry_run=False,
        )

        captured_data = {}
        def mock_make_request(method, url, data=None):
            captured_data.update(data or {})
            return {"success": True, "id": 124}

        async def _test():
            with patch.object(adapter, "_make_api_request", side_effect=mock_make_request):
                with patch.object(adapter, "_check_rate_limit", new_callable=AsyncMock):
                    with patch.object(adapter, "_simulate_delay", new_callable=AsyncMock):
                        await adapter.place_order(
                            symbol="btc_jpy", side="sell", quantity=0.01,
                            price=5100000, order_type="limit",
                        )

        asyncio.run(_test())

        assert captured_data["order_type"] == "sell"
        assert captured_data["time_in_force"] == "post_only"

    def test_market_buy_sends_market_buy_amount(self):
        """Market buy: order_type='market_buy', market_buy_amount in JPY."""
        adapter = CoincheckAdapter(
            api_key="k", api_secret="s", dry_run=False,
        )
        adapter._current_prices["btc_jpy"] = 5000000.0

        captured_data = {}
        def mock_make_request(method, url, data=None):
            captured_data.update(data or {})
            return {"success": True, "id": 125}

        async def _test():
            with patch.object(adapter, "_make_api_request", side_effect=mock_make_request):
                with patch.object(adapter, "_check_rate_limit", new_callable=AsyncMock):
                    with patch.object(adapter, "_simulate_delay", new_callable=AsyncMock):
                        await adapter.place_order(
                            symbol="btc_jpy", side="buy", quantity=0.01,
                            order_type="market",
                        )

        asyncio.run(_test())

        assert captured_data["order_type"] == "market_buy"
        assert "market_buy_amount" in captured_data
        assert "amount" not in captured_data  # market_buy does NOT use amount
        jpy = int(captured_data["market_buy_amount"])
        assert jpy == 50000  # 0.01 * 5,000,000

    def test_market_sell_sends_correct_order_type(self):
        """Market sell: order_type='market_sell', amount in BTC."""
        adapter = CoincheckAdapter(
            api_key="k", api_secret="s", dry_run=False,
        )

        captured_data = {}
        def mock_make_request(method, url, data=None):
            captured_data.update(data or {})
            return {"success": True, "id": 126}

        async def _test():
            with patch.object(adapter, "_make_api_request", side_effect=mock_make_request):
                with patch.object(adapter, "_check_rate_limit", new_callable=AsyncMock):
                    with patch.object(adapter, "_simulate_delay", new_callable=AsyncMock):
                        await adapter.place_order(
                            symbol="btc_jpy", side="sell", quantity=0.01,
                            order_type="market",
                        )

        asyncio.run(_test())

        assert captured_data["order_type"] == "market_sell"
        assert captured_data["amount"] == "0.01"
        assert "market_buy_amount" not in captured_data

    def test_no_limit_buy_limit_sell_values(self):
        """'limit_buy'/'limit_sell' must NOT appear in source (145# _place_order_real)."""
        source = read_inspect_source(CoincheckAdapter._place_order_real)
        assert "limit_buy" not in source
        assert "limit_sell" not in source


# ---------------------------------------------------------------------------
# D-1: OrderManager exchange connection
# ---------------------------------------------------------------------------

class TestD1OrderManagerConnection:
    """D-1: OrderManager must call exchange_adapter.place_order()."""

    def test_no_todo_in_execute_trade(self):
        """The TODO comment should be removed."""
        source = read_inspect_source(OrderManager.execute_trade)
        assert "TODO" not in source, (
            "TODO should be removed — live trading is now implemented"
        )

    def test_live_mode_calls_exchange_adapter(self):
        """Live mode must use exchange_adapter.place_order()."""
        # Create mock live_trader
        mock_trader = MagicMock()
        mock_trader.demo_mode = False
        mock_trader.position = 0
        mock_trader.entry_price = 0.0
        mock_trader._last_valid_price = 5000000.0
        mock_trader._current_prices = {"btc_jpy": 5000000.0}
        mock_trader._send_notification = MagicMock()

        # Create mock exchange adapter
        mock_adapter = MagicMock()
        mock_order = MagicMock()
        mock_order.order_id = "test-order-123"

        # Setup async place_order
        async def mock_place_order(**kwargs):
            return mock_order

        mock_adapter.place_order = mock_place_order
        mock_trader.exchange_adapter = mock_adapter

        om = OrderManager(mock_trader)
        result = om.execute_trade("buy", 0.01)
        assert result is True

    def test_live_mode_no_adapter_returns_false(self):
        """Live mode without exchange_adapter returns False."""
        mock_trader = MagicMock()
        mock_trader.demo_mode = False
        mock_trader.exchange_adapter = None
        mock_trader._send_notification = MagicMock()

        om = OrderManager(mock_trader)
        result = om.execute_trade("buy", 0.01)
        assert result is False


# ---------------------------------------------------------------------------
# D-3: post_only support
# ---------------------------------------------------------------------------

class TestD3PostOnly:
    """D-3: Limit orders must include time_in_force='post_only'."""

    def test_place_order_source_contains_post_only(self):
        """_place_order_real source must reference post_only (145# §13)."""
        source = read_inspect_source(CoincheckAdapter._place_order_real)
        assert "post_only" in source, "post_only must be in _place_order_real"


# ---------------------------------------------------------------------------
# D-5: Rate limiter
# ---------------------------------------------------------------------------

class TestD5RateLimiter:
    """D-5: Default rate limiter must be 4 req/s (Coincheck limit)."""

    def test_default_rate_limit_is_4(self):
        """Default CoincheckAdapter rate limit should be 4 req/s."""
        adapter = CoincheckAdapter(dry_run=True)
        config = adapter.rate_limiter.config
        assert config.requests_per_second == 4.0, (
            f"Expected 4.0 req/s, got {config.requests_per_second}"
        )


# ---------------------------------------------------------------------------
# C-9: Docstring accuracy
# ---------------------------------------------------------------------------

class TestC9Docstring:
    """C-9: CoincheckAdapter docstring must reflect real trading support."""

    def test_docstring_not_stub_only(self):
        """Docstring should not claim 'real trading not implemented'."""
        docstring = CoincheckAdapter.__doc__ or ""
        assert "not implemented" not in docstring.lower(), (
            "Docstring still claims real trading is not implemented"
        )
        assert "simulation only" not in docstring.lower()


# ---------------------------------------------------------------------------
# C-5: bitFlyer product_code normalization
# ---------------------------------------------------------------------------

class TestC5BitFlyerNormalization:
    """C-5: bitFlyer private API must normalize product_code to uppercase."""

    def test_place_order_normalizes_product_code(self):
        """_place_order_real must uppercase symbol for product_code."""
        source = read_inspect_source(BitFlyerAdapter._place_order_real)
        assert "symbol.upper()" in source or "product_code = symbol.upper()" in source

    def test_get_current_price_normalizes_product_code(self):
        """_get_current_price_real must uppercase symbol."""
        source = read_inspect_source(BitFlyerAdapter._get_current_price_real)
        assert "symbol.upper()" in source or "product_code = symbol.upper()" in source
