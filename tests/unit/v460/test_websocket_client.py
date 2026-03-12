"""
WebSocket Client テスト — CoincheckPublicWS / CoincheckPrivateWS.

websockets を完全モック化し、接続なしでメッセージパース・コールバック・
再接続ロジック・統計カウンタをテストする。
"""

from __future__ import annotations

import asyncio
import json
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
    _StreamStats,
    _parse_orderbook,
    _parse_trades,
)


class _AwaitRecorder:
    def __init__(self) -> None:
        self.await_count = 0

    async def __call__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.await_count += 1


# ======================================================================
# Parsers
# ======================================================================


class TestParseTrades:
    """_parse_trades — WS 形式 → TradeRecord 変換."""

    def test_single_trade(self) -> None:
        data = [[1700000000.0, 12345, "btc_jpy", 10000000.0, 0.01, "buy"]]
        result = _parse_trades(data)
        assert len(result) == 1
        t = result[0]
        assert isinstance(t, TradeRecord)
        assert t.timestamp == 1700000000.0
        assert t.price == 10000000.0
        assert t.amount == 0.01
        assert t.side == "buy"

    def test_multiple_trades(self) -> None:
        data = [
            [1700000000.0, 1, "btc_jpy", 10000000.0, 0.01, "buy"],
            [1700000001.0, 2, "btc_jpy", 10000100.0, 0.02, "sell"],
            [1700000002.0, 3, "btc_jpy", 9999900.0, 0.005, "buy"],
        ]
        result = _parse_trades(data)
        assert len(result) == 3
        assert result[0].side == "buy"
        assert result[1].side == "sell"
        assert result[2].amount == 0.005

    def test_empty_list(self) -> None:
        assert _parse_trades([]) == []

    def test_malformed_row_skipped(self) -> None:
        data = [
            [1700000000.0, 1, "btc_jpy", 10000000.0, 0.01, "buy"],
            [1700000001.0],  # too short
            "not a list",  # wrong type
        ]
        result = _parse_trades(data)
        assert len(result) == 1

    def test_string_values_coerced(self) -> None:
        data = [["1700000000", 1, "btc_jpy", "10000000", "0.01", "BUY"]]
        result = _parse_trades(data)
        assert len(result) == 1
        assert result[0].price == 10000000.0
        assert result[0].side == "buy"  # lowered


class TestParseOrderbook:
    """_parse_orderbook — WS 形式 → OrderBookSnapshot 変換."""

    def test_normal_orderbook(self) -> None:
        data = [
            "btc_jpy",
            {
                "bids": [["10000000", "0.5"], ["9999000", "1.0"]],
                "asks": [["10001000", "0.3"], ["10002000", "0.7"]],
                "last_update_at": 1700000000.0,
            },
        ]
        result = _parse_orderbook(data)
        assert result is not None
        assert isinstance(result, OrderBookSnapshot)
        assert result.timestamp == 1700000000.0
        assert len(result.bids) == 2
        assert len(result.asks) == 2
        assert result.bids[0] == (10000000.0, 0.5)
        assert result.asks[0] == (10001000.0, 0.3)
        assert result.exchange == "coincheck"

    def test_empty_orderbook(self) -> None:
        data = ["btc_jpy", {"bids": [], "asks": [], "last_update_at": 0}]
        result = _parse_orderbook(data)
        assert result is not None
        assert result.bids == []
        assert result.asks == []

    def test_invalid_too_short(self) -> None:
        assert _parse_orderbook(["btc_jpy"]) is None

    def test_invalid_second_not_dict(self) -> None:
        assert _parse_orderbook(["btc_jpy", [1, 2, 3]]) is None


# ======================================================================
# WebSocketConfig
# ======================================================================


class TestWebSocketConfig:
    def test_defaults(self) -> None:
        cfg = WebSocketConfig()
        assert cfg.public_url == "wss://ws-api.coincheck.com"
        assert cfg.private_url == "wss://stream.coincheck.com"
        assert cfg.reconnect_delay_initial == 1.0
        assert cfg.reconnect_delay_max == 60.0
        assert cfg.max_reconnects == 0  # unlimited

    def test_custom_config(self) -> None:
        cfg = WebSocketConfig(
            public_url="wss://test.example.com",
            max_reconnects=5,
            ping_interval=15.0,
        )
        assert cfg.public_url == "wss://test.example.com"
        assert cfg.max_reconnects == 5
        assert cfg.ping_interval == 15.0


# ======================================================================
# Channel Enum
# ======================================================================


class TestChannel:
    def test_values(self) -> None:
        assert Channel.TRADES.value == "btc_jpy-trades"
        assert Channel.ORDERBOOK.value == "btc_jpy-orderbook"
        assert Channel.ORDER_EVENTS.value == "order-events"
        assert Channel.EXECUTION_EVENTS.value == "execution-events"

    def test_is_str(self) -> None:
        # Channel は str Enum なので文字列比較可能
        assert Channel.TRADES == "btc_jpy-trades"


# ======================================================================
# _StreamStats
# ======================================================================


class TestStreamStats:
    def test_defaults(self) -> None:
        s = _StreamStats()
        assert s.connections == 0
        assert s.messages_received == 0
        assert s.errors == 0

    def test_str_repr(self) -> None:
        s = _StreamStats(connections=3, messages_received=100, errors=2)
        text = str(s)
        assert "conn=3" in text
        assert "msgs=100" in text
        assert "errors=2" in text


# ======================================================================
# CoincheckPublicWS
# ======================================================================


class TestCoincheckPublicWS:
    """PublicWS — ライフサイクル・ディスパッチのユニットテスト."""

    def test_init_defaults(self) -> None:
        ws = CoincheckPublicWS()
        assert ws.on_trade is None
        assert ws.on_orderbook is None
        assert ws.on_error is None
        assert ws._running is False
        assert ws.is_connected is False

    def test_init_custom_config(self) -> None:
        cfg = WebSocketConfig(max_reconnects=3)
        ws = CoincheckPublicWS(config=cfg)
        assert ws.config.max_reconnects == 3

    @pytest.mark.asyncio
    async def test_start_sets_running(self) -> None:
        ws = CoincheckPublicWS(
            config=WebSocketConfig(
                max_reconnects=1,
                reconnect_delay_initial=0.01,
                reconnect_delay_max=0.01,
            )
        )
        with patch(
            "ztb.trading.live.exchanges.coincheck.websocket_client.websockets.connect",
            side_effect=ConnectionRefusedError("test"),
        ):
            await ws.start([Channel.TRADES])
            assert ws._task is not None
            await asyncio.wait_for(ws._task, timeout=0.2)
            assert ws._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        ws = CoincheckPublicWS()
        ws._running = True
        ws._task = asyncio.create_task(asyncio.sleep(100))
        await ws.stop()
        assert ws._running is False
        assert ws._task is None

    @pytest.mark.asyncio
    async def test_dispatch_trades(self) -> None:
        """_dispatch が trade callback を呼ぶこと."""
        ws = CoincheckPublicWS()
        received: list[list[TradeRecord]] = []

        async def on_trade(trades: list[TradeRecord]) -> None:
            received.append(trades)

        ws.on_trade = on_trade

        trade_data = [[1700000000.0, 1, "btc_jpy", 10000000.0, 0.01, "buy"]]
        await ws._dispatch(trade_data)
        assert len(received) == 1
        assert len(received[0]) == 1
        assert received[0][0].price == 10000000.0

    @pytest.mark.asyncio
    async def test_dispatch_orderbook(self) -> None:
        ws = CoincheckPublicWS()
        received: list[OrderBookSnapshot] = []

        async def on_ob(ob: OrderBookSnapshot) -> None:
            received.append(ob)

        ws.on_orderbook = on_ob

        ob_data = [
            "btc_jpy",
            {
                "bids": [["10000000", "1.0"]],
                "asks": [["10001000", "0.5"]],
                "last_update_at": 1700000000.0,
            },
        ]
        await ws._dispatch(ob_data)
        assert len(received) == 1
        assert received[0].bids[0] == (10000000.0, 1.0)

    @pytest.mark.asyncio
    async def test_dispatch_non_list_ignored(self) -> None:
        ws = CoincheckPublicWS()
        ws.on_trade = _AwaitRecorder()
        await ws._dispatch({"type": "heartbeat"})
        assert ws.on_trade.await_count == 0

    @pytest.mark.asyncio
    async def test_stats_increment(self) -> None:
        ws = CoincheckPublicWS()
        data = [[1700000000.0, 1, "btc_jpy", 10000000.0, 0.01, "buy"]]
        await ws._dispatch(data)
        assert ws.stats.trades_received == 1

    @pytest.mark.asyncio
    async def test_double_start_warning(self) -> None:
        ws = CoincheckPublicWS()
        ws._running = True
        # Should return immediately without error
        await ws.start([Channel.TRADES])

    def test_is_connected_false_when_no_ws(self) -> None:
        ws = CoincheckPublicWS()
        assert ws.is_connected is False

    def test_is_connected_with_mock_ws(self) -> None:
        ws = CoincheckPublicWS()
        mock_ws = MagicMock()
        mock_ws.open = True
        ws._ws = mock_ws
        assert ws.is_connected is True


# ======================================================================
# CoincheckPrivateWS
# ======================================================================


class TestCoincheckPrivateWS:
    """PrivateWS — 認証・ディスパッチのユニットテスト."""

    def test_init(self) -> None:
        ws = CoincheckPrivateWS(api_key="test_key", api_secret="test_secret")
        assert ws.api_key == "test_key"
        assert ws.api_secret == "test_secret"
        assert ws._authenticated is False
        assert ws.is_connected is False

    def test_create_signature(self) -> None:
        ws = CoincheckPrivateWS(api_key="key", api_secret="secret")
        sig = ws._create_signature("1700000000000")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex = 64 chars

    def test_create_signature_deterministic(self) -> None:
        ws = CoincheckPrivateWS(api_key="key", api_secret="secret")
        sig1 = ws._create_signature("12345")
        sig2 = ws._create_signature("12345")
        assert sig1 == sig2

    def test_create_signature_different_nonce(self) -> None:
        ws = CoincheckPrivateWS(api_key="key", api_secret="secret")
        sig1 = ws._create_signature("11111")
        sig2 = ws._create_signature("22222")
        assert sig1 != sig2

    @pytest.mark.asyncio
    async def test_dispatch_order_event(self) -> None:
        ws = CoincheckPrivateWS(api_key="k", api_secret="s")
        received: list[dict] = []

        async def on_order(event: dict[str, Any]) -> None:
            received.append(event)

        ws.on_order_event = on_order

        event = [Channel.ORDER_EVENTS.value, {"id": 123, "status": "FILL"}]
        await ws._dispatch_private(event)
        assert len(received) == 1
        assert received[0]["id"] == 123
        assert ws.stats.order_events == 1

    @pytest.mark.asyncio
    async def test_dispatch_execution_event(self) -> None:
        ws = CoincheckPrivateWS(api_key="k", api_secret="s")
        received: list[dict] = []

        async def on_exec(event: dict[str, Any]) -> None:
            received.append(event)

        ws.on_execution_event = on_exec

        event = [Channel.EXECUTION_EVENTS.value, {"id": 456, "price": 10000000}]
        await ws._dispatch_private(event)
        assert len(received) == 1
        assert received[0]["id"] == 456
        assert ws.stats.execution_events == 1

    @pytest.mark.asyncio
    async def test_dispatch_unknown_channel_ignored(self) -> None:
        ws = CoincheckPrivateWS(api_key="k", api_secret="s")
        ws.on_order_event = _AwaitRecorder()
        ws.on_execution_event = _AwaitRecorder()

        await ws._dispatch_private(["unknown-channel", {"data": True}])
        assert ws.on_order_event.await_count == 0
        assert ws.on_execution_event.await_count == 0

    @pytest.mark.asyncio
    async def test_dispatch_non_list_ignored(self) -> None:
        ws = CoincheckPrivateWS(api_key="k", api_secret="s")
        ws.on_order_event = _AwaitRecorder()
        await ws._dispatch_private({"type": "something"})
        assert ws.on_order_event.await_count == 0

    @pytest.mark.asyncio
    async def test_dispatch_short_list_ignored(self) -> None:
        ws = CoincheckPrivateWS(api_key="k", api_secret="s")
        ws.on_order_event = _AwaitRecorder()
        await ws._dispatch_private(["only_one_element"])
        assert ws.on_order_event.await_count == 0

    @pytest.mark.asyncio
    async def test_stop_idempotent(self) -> None:
        ws = CoincheckPrivateWS(api_key="k", api_secret="s")
        await ws.stop()  # should not raise
        assert ws._running is False

    def test_is_connected_requires_auth(self) -> None:
        ws = CoincheckPrivateWS(api_key="k", api_secret="s")
        mock_ws = MagicMock()
        mock_ws.open = True
        ws._ws = mock_ws
        ws._authenticated = False
        assert ws.is_connected is False

        ws._authenticated = True
        assert ws.is_connected is True

    @pytest.mark.asyncio
    async def test_authenticate_success(self) -> None:
        ws = CoincheckPrivateWS(api_key="key", api_secret="secret")
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"status": "ok"}))

        result = await ws._authenticate(mock_ws)
        assert result is True
        assert ws._authenticated is True
        mock_ws.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_authenticate_error_response(self) -> None:
        ws = CoincheckPrivateWS(api_key="key", api_secret="secret")
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"error": "invalid_key"})
        )

        result = await ws._authenticate(mock_ws)
        assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_timeout(self) -> None:
        ws = CoincheckPrivateWS(api_key="key", api_secret="secret")
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=asyncio.TimeoutError)

        result = await ws._authenticate(mock_ws)
        assert result is False


# ======================================================================
# Monitor スクリプト — §3.9 ルール評価
# ======================================================================


class TestMonitorStopRules:
    """monitor_fill_test.py の §3.9 ルール評価テスト."""

    def _make_record(
        self,
        filled: bool = True,
        pnl_bps: float = 0.0,
        fill_price: float = 10_000_000.0,
        quantity: float = 0.001,
        adverse: bool = False,
    ) -> Any:
        """テスト用 FillRecord を作成."""
        from ztb.metrics.fill_quality import FillRecord
        return FillRecord(
            cycle_id="test",
            timestamp=1700000000.0,
            side="buy",
            order_price=fill_price,
            order_quantity=quantity,
            fill_price=fill_price if filled else None,
            filled=filled,
            cancelled=not filled,
            queue_wait_sec=10.0,
            mid_at_fill=fill_price if filled else None,
            mid_30s_after=fill_price if filled else None,
            post_fill_30s_pnl=pnl_bps if filled else None,
            adverse_selected=adverse if filled else None,
        )

    def test_cumulative_loss_below_threshold(self) -> None:
        from scripts.v460.monitor_fill_test import _check_cumulative_loss
        # 小さな損失: -1 bps × 10M × 0.001 = -1 JPY per record
        records = [self._make_record(pnl_bps=-1.0) for _ in range(50)]
        assert _check_cumulative_loss(records) is False  # 50 JPY < 10,000

    def test_cumulative_loss_above_threshold(self) -> None:
        from scripts.v460.monitor_fill_test import _check_cumulative_loss
        # 大きな損失: -20 bps × 10M × 0.001 = -20 JPY per record
        records = [self._make_record(pnl_bps=-20.0) for _ in range(600)]
        assert _check_cumulative_loss(records) is True  # 12,000 > 10,000

    def test_evaluate_stop_rules_skip_low_n(self) -> None:
        from scripts.v460.monitor_fill_test import evaluate_stop_rules
        from ztb.metrics.fill_quality import FillMetrics
        metrics = FillMetrics(total_orders=50)
        records = [self._make_record() for _ in range(50)]
        results = evaluate_stop_rules(metrics, records)
        # R1 (min_n=200) と R2 (min_n=500) は SKIP
        r1 = next(r for r in results if r["rule"] == "R1_fill_rate")
        r2 = next(r for r in results if r["rule"] == "R2_as_ratio")
        assert r1["status"] == "SKIP"
        assert r2["status"] == "SKIP"

    def test_evaluate_r5_ok(self) -> None:
        from scripts.v460.monitor_fill_test import evaluate_stop_rules
        from ztb.metrics.fill_quality import FillMetrics
        metrics = FillMetrics(total_orders=10)
        records = [self._make_record(pnl_bps=1.0) for _ in range(10)]
        results = evaluate_stop_rules(metrics, records)
        r5 = next(r for r in results if r["rule"] == "R5_cumulative_loss")
        assert r5["status"] == "OK"
        assert r5["triggered"] is False
