"""
Coincheck WebSocket クライアント — Public / Private チャネル対応.

014# T4: REST 5秒ポーリングを補完・代替する低遅延データ取得.

Public  (wss://ws-api.coincheck.com):
  - btc_jpy-trades    : 約定ストリーム (0.1s 間隔配信)
  - btc_jpy-orderbook : 板差分ストリーム (0.1s 間隔配信)

Private (wss://stream.coincheck.com):
  - order-events      : TRIGGERED/PARTIALLY_FILL/FILL/EXPIRY/CANCEL
  - execution-events  : 約定詳細 (fee, liquidity T/M, side)

NOTE:
  - メッセージ再送機能なし — 切断中の欠損は REST で補完すること.
  - Private 利用には API キーの「WebSocket」権限チェックが必要.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, Sequence

import websockets
from websockets.asyncio.client import ClientConnection

from ztb.trading.live.exchanges.base.broker_interfaces import (
    OrderBookSnapshot,
    TradeRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Channel(str, Enum):
    """Coincheck WebSocket チャネル名."""
    TRADES = "btc_jpy-trades"
    ORDERBOOK = "btc_jpy-orderbook"
    ORDER_EVENTS = "order-events"
    EXECUTION_EVENTS = "execution-events"


@dataclass(frozen=True)
class WebSocketConfig:
    """WebSocket 接続設定."""
    public_url: str = "wss://ws-api.coincheck.com"
    private_url: str = "wss://stream.coincheck.com"
    reconnect_delay_initial: float = 1.0
    reconnect_delay_max: float = 60.0
    reconnect_delay_factor: float = 2.0
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    message_timeout: float = 60.0  # no message → reconnect
    max_reconnects: int = 0  # 0 = unlimited


# Callback type aliases
OnTrade = Callable[[list[TradeRecord]], Coroutine[Any, Any, None]]
OnOrderBook = Callable[[OrderBookSnapshot], Coroutine[Any, Any, None]]
OnOrderEvent = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]
OnExecutionEvent = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]
OnError = Callable[[Exception], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Public WebSocket Client
# ---------------------------------------------------------------------------

class CoincheckPublicWS:
    """Coincheck Public WebSocket — trades + orderbook ストリーミング.

    使用例::

        async def on_trade(trades: list[TradeRecord]) -> None:
            for t in trades:
                print(t.price, t.amount, t.side)

        ws = CoincheckPublicWS()
        ws.on_trade = on_trade
        await ws.start([Channel.TRADES, Channel.ORDERBOOK])
        # ... later ...
        await ws.stop()
    """

    def __init__(self, config: Optional[WebSocketConfig] = None) -> None:
        self.config = config or WebSocketConfig()
        self.on_trade: Optional[OnTrade] = None
        self.on_orderbook: Optional[OnOrderBook] = None
        self.on_error: Optional[OnError] = None

        self._ws: Optional[ClientConnection] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._channels: list[Channel] = []
        self._reconnect_count = 0

        # Stats
        self.stats = _StreamStats()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        channels: Optional[Sequence[Channel]] = None,
    ) -> None:
        """Start streaming. Default: TRADES + ORDERBOOK."""
        if self._running:
            logger.warning("PublicWS already running")
            return
        self._channels = list(channels or [Channel.TRADES, Channel.ORDERBOOK])
        self._running = True
        self._reconnect_count = 0
        self._task = asyncio.create_task(self._run_loop(), name="coincheck-public-ws")
        logger.info(f"PublicWS started: channels={[c.value for c in self._channels]}")

    async def stop(self) -> None:
        """Gracefully stop streaming."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info(f"PublicWS stopped. {self.stats}")

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Connection loop with exponential backoff reconnect."""
        delay = self.config.reconnect_delay_initial

        while self._running:
            try:
                async with websockets.connect(
                    self.config.public_url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                ) as ws:
                    self._ws = ws
                    delay = self.config.reconnect_delay_initial  # reset on success
                    self._reconnect_count = 0

                    # Subscribe
                    for ch in self._channels:
                        await ws.send(json.dumps({
                            "type": "subscribe",
                            "channel": ch.value,
                        }))
                        logger.debug(f"Subscribed: {ch.value}")

                    self.stats.connections += 1
                    logger.info(
                        f"PublicWS connected (#{self.stats.connections})"
                    )

                    # Message loop
                    await self._recv_loop(ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.stats.errors += 1
                if self.on_error:
                    try:
                        await self.on_error(e)
                    except Exception:
                        pass
                logger.warning(f"PublicWS error: {e}")

            # Reconnect logic
            if not self._running:
                break
            self._reconnect_count += 1
            max_r = self.config.max_reconnects
            if max_r > 0 and self._reconnect_count >= max_r:
                logger.error(f"PublicWS max reconnects ({max_r}) exceeded")
                self._running = False
                break

            logger.info(f"PublicWS reconnecting in {delay:.1f}s ...")
            await asyncio.sleep(delay)
            delay = min(delay * self.config.reconnect_delay_factor,
                        self.config.reconnect_delay_max)

    async def _recv_loop(self, ws: ClientConnection) -> None:
        """Receive and dispatch messages."""
        while self._running:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=self.config.message_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("PublicWS no message timeout — reconnecting")
                return
            except websockets.exceptions.ConnectionClosed:
                logger.info("PublicWS connection closed by server")
                return

            try:
                data = json.loads(raw)
                self.stats.messages_received += 1
                await self._dispatch(data)
            except Exception as e:
                self.stats.parse_errors += 1
                logger.error(f"PublicWS message parse/dispatch error: {e}")

    async def _dispatch(self, data: Any) -> None:
        """Parse and dispatch to callbacks."""
        if not isinstance(data, list):
            logger.debug(f"PublicWS non-list message: {data}")
            return

        # Trades: [[ts, id, pair, rate, amount, order_type, ...], ...]
        if len(data) >= 1 and isinstance(data[0], list):
            trades = _parse_trades(data)
            self.stats.trades_received += len(trades)
            if self.on_trade and trades:
                await self.on_trade(trades)
            return

        # Orderbook: [pair, {bids: [...], asks: [...], last_update_at: ...}]
        if len(data) == 2 and isinstance(data[1], dict):
            ob = _parse_orderbook(data)
            self.stats.orderbooks_received += 1
            if self.on_orderbook and ob:
                await self.on_orderbook(ob)
            return

        logger.debug(f"PublicWS unknown format: {str(data)[:200]}")


# ---------------------------------------------------------------------------
# Private WebSocket Client
# ---------------------------------------------------------------------------

class CoincheckPrivateWS:
    """Coincheck Private WebSocket — order / execution イベント.

    使用例::

        ws = CoincheckPrivateWS(api_key="...", api_secret="...")
        ws.on_fill = my_fill_handler
        await ws.start([Channel.ORDER_EVENTS, Channel.EXECUTION_EVENTS])
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        config: Optional[WebSocketConfig] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.config = config or WebSocketConfig()
        self.on_order_event: Optional[OnOrderEvent] = None
        self.on_execution_event: Optional[OnExecutionEvent] = None
        self.on_error: Optional[OnError] = None

        self._ws: Optional[ClientConnection] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._channels: list[Channel] = []
        self._reconnect_count = 0
        self._authenticated = False
        self.stats = _StreamStats()

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _create_signature(self, nonce: str) -> str:
        """HMAC-SHA256 signature: sign(nonce + private_uri)."""
        uri = f"{self.config.private_url}/private"
        message = nonce + uri
        return hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def _authenticate(self, ws: ClientConnection) -> bool:
        """Send login command and wait for success response."""
        nonce = str(int(time.time() * 1000))
        signature = self._create_signature(nonce)

        login_msg = json.dumps({
            "type": "login",
            "access_key": self.api_key,
            "access_nonce": nonce,
            "access_signature": signature,
        })
        await ws.send(login_msg)
        logger.debug("PrivateWS login sent")

        # Wait for auth response (timeout 10s)
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            resp = json.loads(raw)
            # Coincheck doesn't document success response format clearly;
            # treat any non-error as success
            if isinstance(resp, dict) and resp.get("error"):
                logger.error(f"PrivateWS auth failed: {resp}")
                return False
            logger.info("PrivateWS authenticated")
            self._authenticated = True
            return True
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"PrivateWS auth timeout/error: {e}")
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        channels: Optional[Sequence[Channel]] = None,
    ) -> None:
        """Start private streaming."""
        if self._running:
            logger.warning("PrivateWS already running")
            return
        self._channels = list(
            channels or [Channel.ORDER_EVENTS, Channel.EXECUTION_EVENTS]
        )
        self._running = True
        self._reconnect_count = 0
        self._task = asyncio.create_task(self._run_loop(), name="coincheck-private-ws")
        logger.info(
            f"PrivateWS started: channels={[c.value for c in self._channels]}"
        )

    async def stop(self) -> None:
        """Gracefully stop."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info(f"PrivateWS stopped. {self.stats}")

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.open and self._authenticated

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Connection loop with auto-reconnect."""
        delay = self.config.reconnect_delay_initial

        while self._running:
            try:
                async with websockets.connect(
                    self.config.private_url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.ping_timeout,
                ) as ws:
                    self._ws = ws
                    self._authenticated = False
                    delay = self.config.reconnect_delay_initial

                    # Authenticate
                    if not await self._authenticate(ws):
                        logger.error("PrivateWS auth failed — will retry")
                        continue  # triggers reconnect

                    # Subscribe
                    for ch in self._channels:
                        await ws.send(json.dumps({
                            "type": "subscribe",
                            "channel": ch.value,
                        }))
                        logger.debug(f"PrivateWS subscribed: {ch.value}")

                    self.stats.connections += 1
                    logger.info(
                        f"PrivateWS connected (#{self.stats.connections})"
                    )

                    # Message loop
                    await self._recv_loop(ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.stats.errors += 1
                if self.on_error:
                    try:
                        await self.on_error(e)
                    except Exception:
                        pass
                logger.warning(f"PrivateWS error: {e}")

            if not self._running:
                break
            self._reconnect_count += 1
            max_r = self.config.max_reconnects
            if max_r > 0 and self._reconnect_count >= max_r:
                logger.error(f"PrivateWS max reconnects ({max_r}) exceeded")
                self._running = False
                break

            logger.info(f"PrivateWS reconnecting in {delay:.1f}s ...")
            await asyncio.sleep(delay)
            delay = min(delay * self.config.reconnect_delay_factor,
                        self.config.reconnect_delay_max)

    async def _recv_loop(self, ws: ClientConnection) -> None:
        """Receive and dispatch private messages."""
        while self._running:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=self.config.message_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("PrivateWS no message timeout — reconnecting")
                return
            except websockets.exceptions.ConnectionClosed:
                logger.info("PrivateWS connection closed")
                return

            try:
                data = json.loads(raw)
                self.stats.messages_received += 1
                await self._dispatch_private(data)
            except Exception as e:
                self.stats.parse_errors += 1
                logger.error(f"PrivateWS parse/dispatch error: {e}")

    async def _dispatch_private(self, data: Any) -> None:
        """Dispatch private channel messages."""
        if not isinstance(data, list) or len(data) < 2:
            logger.debug(f"PrivateWS non-list/short: {data}")
            return

        channel = data[0] if isinstance(data[0], str) else None
        payload = data[1] if len(data) >= 2 else None

        if channel == Channel.ORDER_EVENTS.value:
            if isinstance(payload, dict):
                self.stats.order_events += 1
                if self.on_order_event:
                    await self.on_order_event(payload)
        elif channel == Channel.EXECUTION_EVENTS.value:
            if isinstance(payload, dict):
                self.stats.execution_events += 1
                if self.on_execution_event:
                    await self.on_execution_event(payload)
        else:
            logger.debug(f"PrivateWS unknown channel: {channel}")


# ---------------------------------------------------------------------------
# Parsers — WS format → broker_interfaces types
# ---------------------------------------------------------------------------

def _parse_trades(data: list[list[Any]]) -> list[TradeRecord]:
    """Parse WS trade messages to TradeRecord list.

    WS format: [timestamp, trade_id, pair, rate, amount, order_type, ...]
    """
    result: list[TradeRecord] = []
    for row in data:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            ts = float(row[0])
            price = float(row[3])
            amount = float(row[4])
            side = str(row[5]).lower()  # "buy" or "sell"
            result.append(TradeRecord(
                timestamp=ts,
                price=price,
                amount=amount,
                side=side,
            ))
        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Trade parse skip: {e}")
    return result


def _parse_orderbook(data: list[Any]) -> Optional[OrderBookSnapshot]:
    """Parse WS orderbook message to OrderBookSnapshot.

    WS format: [pair, {bids: [[rate, amount], ...], asks: [...], last_update_at: ts}]
    """
    if len(data) < 2 or not isinstance(data[1], dict):
        return None

    ob_data = data[1]
    try:
        ts = float(ob_data.get("last_update_at", time.time()))
        bids_raw = ob_data.get("bids", [])
        asks_raw = ob_data.get("asks", [])

        bids = [(float(r), float(a)) for r, a in bids_raw]
        asks = [(float(r), float(a)) for r, a in asks_raw]

        return OrderBookSnapshot(
            timestamp=ts,
            bids=bids,
            asks=asks,
            exchange="coincheck",
        )
    except (ValueError, TypeError) as e:
        logger.debug(f"Orderbook parse error: {e}")
        return None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class _StreamStats:
    """Streaming statistics for monitoring."""
    connections: int = 0
    messages_received: int = 0
    trades_received: int = 0
    orderbooks_received: int = 0
    order_events: int = 0
    execution_events: int = 0
    errors: int = 0
    parse_errors: int = 0

    def __str__(self) -> str:
        return (
            f"Stats(conn={self.connections}, msgs={self.messages_received}, "
            f"trades={self.trades_received}, ob={self.orderbooks_received}, "
            f"order_ev={self.order_events}, exec_ev={self.execution_events}, "
            f"errors={self.errors}, parse_err={self.parse_errors})"
        )
