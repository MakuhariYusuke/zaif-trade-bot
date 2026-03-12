"""
WebSocket クライアントのユニットテスト.

014# T4: CoincheckPublicWS / CoincheckPrivateWS のパース・ライフサイクルテスト.
WebSocket サーバーをモックし、外部接続なしでテスト可能。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ztb.trading.live.exchanges.base.broker_interfaces import (
    OrderBookSnapshot,
    TradeRecord,
)
from ztb.trading.live.exchanges.coincheck.websocket_client import (
    Channel,
    CoincheckPrivateWS,
    CoincheckPublicWS,
    WebSocketConfig,
    _parse_orderbook,
    _parse_trades,
    _StreamStats,
)


# ---------------------------------------------------------------------------
# Parser tests (pure functions, no network)
# ---------------------------------------------------------------------------

class TestParseTrades:
    """_parse_trades のテスト."""

    def test_valid_single_trade(self) -> None:
        """正常な 1 件の約定をパースできる."""
        data = [
            [1700000000.0, 12345, "btc_jpy", 10300000.0, 0.001, "buy", "t1", "m1", None],
        ]
        result = _parse_trades(data)
        assert len(result) == 1
        assert result[0].timestamp == 1700000000.0
        assert result[0].price == 10300000.0
        assert result[0].amount == 0.001
        assert result[0].side == "buy"

    def test_valid_multiple_trades(self) -> None:
        """複数約定を一度にパースできる."""
        data = [
            [1700000000.0, 1, "btc_jpy", 10300000.0, 0.001, "buy", "t1", "m1", None],
            [1700000001.0, 2, "btc_jpy", 10300500.0, 0.002, "sell", "t2", "m2", None],
        ]
        result = _parse_trades(data)
        assert len(result) == 2
        assert result[1].side == "sell"

    def test_empty_list(self) -> None:
        """空リストは空結果."""
        assert _parse_trades([]) == []

    def test_malformed_row_skipped(self) -> None:
        """不正データはスキップされる."""
        data = [
            [1700000000.0, 1, "btc_jpy"],  # too short
            [1700000001.0, 2, "btc_jpy", 10300000.0, 0.001, "buy"],  # valid
        ]
        result = _parse_trades(data)
        assert len(result) == 1

    def test_non_list_row_skipped(self) -> None:
        """非リスト行はスキップ."""
        data = ["not a list", [1700000001.0, 2, "btc_jpy", 10300000.0, 0.001, "buy"]]
        result = _parse_trades(data)
        assert len(result) == 1


class TestParseOrderbook:
    """_parse_orderbook のテスト."""

    def test_valid_orderbook(self) -> None:
        """正常な板データをパースできる."""
        data = [
            "btc_jpy",
            {
                "bids": [["10300000", "0.5"], ["10299000", "1.0"]],
                "asks": [["10301000", "0.3"], ["10302000", "0.8"]],
                "last_update_at": 1700000000,
            },
        ]
        result = _parse_orderbook(data)
        assert result is not None
        assert result.exchange == "coincheck"
        assert result.timestamp == 1700000000
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.bids[0] == (10300000.0, 0.5)
        assert result.asks[0] == (10301000.0, 0.3)

    def test_empty_orderbook(self) -> None:
        """空の板データをパースできる."""
        data = ["btc_jpy", {"bids": [], "asks": [], "last_update_at": 1700000000}]
        result = _parse_orderbook(data)
        assert result is not None
        assert len(result.bids) == 0
        assert len(result.asks) == 0

    def test_invalid_format_returns_none(self) -> None:
        """不正形式は None を返す."""
        assert _parse_orderbook([]) is None
        assert _parse_orderbook(["btc_jpy"]) is None
        assert _parse_orderbook(["btc_jpy", "not a dict"]) is None


class TestStreamStats:
    """_StreamStats のテスト."""

    def test_defaults(self) -> None:
        s = _StreamStats()
        assert s.connections == 0
        assert s.messages_received == 0

    def test_str(self) -> None:
        s = _StreamStats(connections=2, messages_received=100)
        text = str(s)
        assert "conn=2" in text
        assert "msgs=100" in text


# ---------------------------------------------------------------------------
# Channel enum tests
# ---------------------------------------------------------------------------

class TestChannel:
    def test_channel_values(self) -> None:
        assert Channel.TRADES.value == "btc_jpy-trades"
        assert Channel.ORDERBOOK.value == "btc_jpy-orderbook"
        assert Channel.ORDER_EVENTS.value == "order-events"
        assert Channel.EXECUTION_EVENTS.value == "execution-events"


# ---------------------------------------------------------------------------
# WebSocketConfig tests
# ---------------------------------------------------------------------------

class TestWebSocketConfig:
    def test_defaults(self) -> None:
        cfg = WebSocketConfig()
        assert cfg.public_url == "wss://ws-api.coincheck.com"
        assert cfg.private_url == "wss://stream.coincheck.com"
        assert cfg.reconnect_delay_initial == 1.0
        assert cfg.max_reconnects == 0  # unlimited

    def test_frozen(self) -> None:
        cfg = WebSocketConfig()
        with pytest.raises(AttributeError):
            cfg.public_url = "wss://other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CoincheckPublicWS unit tests (mocked WebSocket)
# ---------------------------------------------------------------------------

class TestPublicWSDispatch:
    """Dispatch logic tests without real connections."""

    @pytest.fixture
    def ws_client(self) -> CoincheckPublicWS:
        return CoincheckPublicWS(config=WebSocketConfig(max_reconnects=1))

    @pytest.mark.asyncio
    async def test_dispatch_trades(self, ws_client: CoincheckPublicWS) -> None:
        """約定メッセージが on_trade コールバックに渡される."""
        received: list[list[TradeRecord]] = []

        async def on_trade(trades: list[TradeRecord]) -> None:
            received.append(trades)

        ws_client.on_trade = on_trade

        trade_msg = [
            [1700000000.0, 1, "btc_jpy", 10300000.0, 0.001, "buy", "t1", "m1", None],
        ]
        await ws_client._dispatch(trade_msg)

        assert len(received) == 1
        assert len(received[0]) == 1
        assert received[0][0].price == 10300000.0

    @pytest.mark.asyncio
    async def test_dispatch_orderbook(self, ws_client: CoincheckPublicWS) -> None:
        """板メッセージが on_orderbook コールバックに渡される."""
        received: list[OrderBookSnapshot] = []

        async def on_ob(ob: OrderBookSnapshot) -> None:
            received.append(ob)

        ws_client.on_orderbook = on_ob

        ob_msg = [
            "btc_jpy",
            {
                "bids": [["10300000", "0.5"]],
                "asks": [["10301000", "0.3"]],
                "last_update_at": 1700000000,
            },
        ]
        await ws_client._dispatch(ob_msg)

        assert len(received) == 1
        assert received[0].bids[0] == (10300000.0, 0.5)

    @pytest.mark.asyncio
    async def test_dispatch_no_callback(self, ws_client: CoincheckPublicWS) -> None:
        """コールバック未設定でもエラーにならない."""
        trade_msg = [
            [1700000000.0, 1, "btc_jpy", 10300000.0, 0.001, "buy", "t1", "m1", None],
        ]
        # Should not raise
        await ws_client._dispatch(trade_msg)
        assert ws_client.stats.trades_received == 1

    @pytest.mark.asyncio
    async def test_dispatch_non_list_ignored(self, ws_client: CoincheckPublicWS) -> None:
        """非リストメッセージは無視される."""
        await ws_client._dispatch({"type": "info"})
        assert ws_client.stats.trades_received == 0

    @pytest.mark.asyncio
    async def test_stats_update(self, ws_client: CoincheckPublicWS) -> None:
        """Stats が正しく更新される."""
        trade_msg = [
            [1700000000.0, 1, "btc_jpy", 10300000.0, 0.001, "buy", "t1", "m1", None],
            [1700000001.0, 2, "btc_jpy", 10300500.0, 0.002, "sell", "t2", "m2", None],
        ]
        await ws_client._dispatch(trade_msg)
        assert ws_client.stats.trades_received == 2

        ob_msg = [
            "btc_jpy",
            {"bids": [], "asks": [], "last_update_at": 1700000000},
        ]
        await ws_client._dispatch(ob_msg)
        assert ws_client.stats.orderbooks_received == 1


# ---------------------------------------------------------------------------
# CoincheckPrivateWS unit tests
# ---------------------------------------------------------------------------

class TestPrivateWSSignature:
    """Private WS signature generation test."""

    def test_signature_deterministic(self) -> None:
        """同じ入力で同じ署名が生成される."""
        ws = CoincheckPrivateWS(api_key="test_key", api_secret="test_secret")
        sig1 = ws._create_signature("12345")
        sig2 = ws._create_signature("12345")
        assert sig1 == sig2
        assert len(sig1) == 64  # SHA-256 hex digest length

    def test_signature_changes_with_nonce(self) -> None:
        """nonce が異なれば署名も変わる."""
        ws = CoincheckPrivateWS(api_key="test_key", api_secret="test_secret")
        sig1 = ws._create_signature("12345")
        sig2 = ws._create_signature("12346")
        assert sig1 != sig2


class TestPrivateWSDispatch:
    """Private dispatch logic tests."""

    @pytest.fixture
    def private_ws(self) -> CoincheckPrivateWS:
        return CoincheckPrivateWS(
            api_key="k", api_secret="s",
            config=WebSocketConfig(max_reconnects=1),
        )

    @pytest.mark.asyncio
    async def test_dispatch_order_event(self, private_ws: CoincheckPrivateWS) -> None:
        """Order event が正しくディスパッチされる."""
        events: list[dict[str, Any]] = []

        async def on_order(ev: dict[str, Any]) -> None:
            events.append(ev)

        private_ws.on_order_event = on_order

        msg = [
            "order-events",
            {"event": "FILL", "order_id": 123, "rate": "10300000"},
        ]
        await private_ws._dispatch_private(msg)

        assert len(events) == 1
        assert events[0]["event"] == "FILL"
        assert private_ws.stats.order_events == 1

    @pytest.mark.asyncio
    async def test_dispatch_execution_event(self, private_ws: CoincheckPrivateWS) -> None:
        """Execution event が正しくディスパッチされる."""
        events: list[dict[str, Any]] = []

        async def on_exec(ev: dict[str, Any]) -> None:
            events.append(ev)

        private_ws.on_execution_event = on_exec

        msg = [
            "execution-events",
            {"id": 1, "order_id": 123, "rate": "10300000", "fee": "0", "liquidity": "M"},
        ]
        await private_ws._dispatch_private(msg)

        assert len(events) == 1
        assert events[0]["liquidity"] == "M"
        assert private_ws.stats.execution_events == 1

    @pytest.mark.asyncio
    async def test_dispatch_unknown_channel(self, private_ws: CoincheckPrivateWS) -> None:
        """不明チャネルはスキップされる."""
        msg = ["unknown-channel", {"data": 1}]
        await private_ws._dispatch_private(msg)
        assert private_ws.stats.order_events == 0
        assert private_ws.stats.execution_events == 0
