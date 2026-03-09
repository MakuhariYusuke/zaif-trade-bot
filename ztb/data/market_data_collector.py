"""
Market data collector — tick raw 収集 + 1分集約の二層保存.

v460 §1.5 準拠.

一次保存: JSONL (gzip) per day → data/v460/raw/{orderbook,trades}/
二次生成: 1分集約 Parquet    → data/v460/features/

NOTE: DataAcquisitionScheduler (ztb/data/scheduler.py) は Binance 専用のため
      コードは流用せず、スケジュール思想 (APScheduler cron) のみ参考。
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ztb.data.raw_paths import resolve_raw_dir
from ztb.io.jsonl_gz import append_jsonl_gz, read_jsonl_gz
from ztb.trading.live.exchanges.base.broker_interfaces import (
    IBroker,
    MarketDataNotSupported,
    OrderBookSnapshot,
    TradeRecord,
)

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AGG_DIR = _PROJECT_ROOT / "data" / "v460" / "features"


def _extract_orderbook_top_features(
    ob_df: pd.DataFrame,
) -> pd.DataFrame:
    """板の top-of-book / top-5 depth を単一パスで抽出する."""
    bids_list = ob_df["bids"].tolist()
    asks_list = ob_df["asks"].tolist()
    n_rows = len(ob_df)

    best_bid = np.empty(n_rows, dtype=np.float64)
    best_ask = np.empty(n_rows, dtype=np.float64)
    bid_vol_5 = np.empty(n_rows, dtype=np.float64)
    ask_vol_5 = np.empty(n_rows, dtype=np.float64)

    for idx, (bids, asks) in enumerate(zip(bids_list, asks_list)):
        if bids:
            best_bid[idx] = float(bids[0][0])
            bid_vol_5[idx] = float(sum(size for _, size in bids[:5]))
        else:
            best_bid[idx] = np.nan
            bid_vol_5[idx] = 0.0

        if asks:
            best_ask[idx] = float(asks[0][0])
            ask_vol_5[idx] = float(sum(size for _, size in asks[:5]))
        else:
            best_ask[idx] = np.nan
            ask_vol_5[idx] = 0.0

    top = pd.DataFrame(index=ob_df.index)
    top["best_bid"] = best_bid
    top["best_ask"] = best_ask
    top["bid_vol_5"] = bid_vol_5
    top["ask_vol_5"] = ask_vol_5
    top["mid_price"] = (best_bid + best_ask) / 2.0
    top["spread"] = (best_ask - best_bid) / top["mid_price"]
    denom = bid_vol_5 + ask_vol_5
    top["depth_imbalance"] = np.divide(
        bid_vol_5 - ask_vol_5,
        denom,
        out=np.full(n_rows, np.nan, dtype=np.float64),
        where=denom != 0.0,
    )
    return top


def _aggregate_orderbook_1min(
    ob_df: pd.DataFrame,
) -> pd.DataFrame:
    """板特徴量を 1 分集約する."""
    top = _extract_orderbook_top_features(ob_df)
    ob_1m = top.resample("1min").agg(
        best_bid=("best_bid", "last"),
        best_ask=("best_ask", "last"),
        mid_price=("mid_price", "last"),
        spread=("spread", "last"),
        bid_vol_5=("bid_vol_5", "last"),
        ask_vol_5=("ask_vol_5", "last"),
        depth_imbalance=("depth_imbalance", "last"),
        spread_min=("spread", "min"),
        spread_max=("spread", "max"),
    )
    ob_1m["spread_range"] = ob_1m.pop("spread_max") - ob_1m.pop("spread_min")
    return ob_1m.dropna(how="all")


def _aggregate_trades_1min(
    tr_df: pd.DataFrame,
) -> pd.DataFrame:
    """約定特徴量を 1 分集約する."""
    side_lower = tr_df["side"].astype("string").str.lower()
    amount = tr_df["amount"].astype(np.float64, copy=False)
    price = tr_df["price"].astype(np.float64, copy=False)
    buy_mask = side_lower.eq("buy").to_numpy(dtype=bool, na_value=False)
    amount_arr = amount.to_numpy(copy=False)
    price_arr = price.to_numpy(copy=False)

    tr_features = pd.DataFrame(index=tr_df.index)
    tr_features["buy_vol"] = np.where(buy_mask, amount_arr, 0.0)
    tr_features["sell_vol"] = np.where(buy_mask, 0.0, amount_arr)
    tr_features["_pv"] = price_arr * amount_arr
    tr_features["_amount"] = amount_arr

    tr_1m = tr_features.resample("1min").agg(
        buy_volume=("buy_vol", "sum"),
        sell_volume=("sell_vol", "sum"),
        trade_count=("buy_vol", "size"),
        _sum_pv=("_pv", "sum"),
        _sum_amount=("_amount", "sum"),
    )
    tr_1m["vwap"] = tr_1m["_sum_pv"] / tr_1m["_sum_amount"].replace(0, np.nan)
    tr_1m = tr_1m.drop(columns=["_sum_pv", "_sum_amount"])
    denom = (tr_1m["buy_volume"] + tr_1m["sell_volume"]).replace(0, np.nan)
    tr_1m["trade_flow_imbalance"] = (
        (tr_1m["buy_volume"] - tr_1m["sell_volume"]) / denom
    )
    return tr_1m.dropna(how="all")

class MarketDataCollector:
    """Tick raw 収集 + 1分集約の二層保存を行うデータ収集サービス.

    Phase 0 で使用。板情報・約定フローを定期ポーリングし、
    raw を JSONL (gzip)、1 分集約を Parquet で保存する。
    """

    def __init__(
        self,
        adapter: IBroker,
        symbol: str = "btc_jpy",
        raw_dir: str | Path | None = None,
        agg_dir: str | Path | None = None,
        poll_interval_sec: float = 5.0,
    ) -> None:
        self.adapter = adapter
        self.symbol = symbol
        self.raw_dir = resolve_raw_dir(raw_dir)
        self.agg_dir = Path(agg_dir) if agg_dir else DEFAULT_AGG_DIR
        self.poll_interval_sec = poll_interval_sec

        # Ensure dirs exist
        (self.raw_dir / "orderbook").mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "trades").mkdir(parents=True, exist_ok=True)
        self.agg_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer for current day (cap でメモリ保護)
        self._ob_buffer: list[dict[str, Any]] = []
        self._tr_buffer: list[dict[str, Any]] = []
        self._buffer_cap: int = 50_000  # 自動 flush 閾値
        self._last_trade_id: tuple[float, float, float, str] | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Tick collection
    # ------------------------------------------------------------------

    async def collect_tick(
        self,
    ) -> tuple[OrderBookSnapshot | None, list[TradeRecord]]:
        """Fetch one tick of orderbook + trades from the adapter."""
        ob: OrderBookSnapshot | None = None
        trades: list[TradeRecord] = []

        try:
            ob = await self.adapter.get_orderbook(self.symbol, depth=10)
        except MarketDataNotSupported:
            logger.warning("Adapter does not support orderbook")
        except Exception as e:
            logger.error(f"Orderbook fetch error: {e}")

        try:
            trades = await self.adapter.get_recent_trades(self.symbol, limit=100)
        except MarketDataNotSupported:
            logger.warning("Adapter does not support trades")
        except Exception as e:
            logger.error(f"Trades fetch error: {e}")

        return ob, trades

    # ------------------------------------------------------------------
    # Raw JSONL persistence (gzip, daily rotation)
    # ------------------------------------------------------------------

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _append_raw_ob(self, ob: OrderBookSnapshot) -> None:
        record = {
            "ts": ob.timestamp,
            "bids": ob.bids,
            "asks": ob.asks,
            "exchange": ob.exchange,
        }
        self._ob_buffer.append(record)

    def _append_raw_trades(self, trades: list[TradeRecord]) -> None:
        """Append trades with dedup via _last_trade_id.

        003# #5: _last_trade_id was declared but never used.
        Using timestamp+price+amount as composite trade ID for dedup.
        007# F8 / 009# P2-2.1: 文字列比較 → タプル数値比較に修正.
        文字列だと "10.5:..." < "9.5:..." (辞書順) になるバグがあった.
        """
        new_trades: list[TradeRecord] = []
        for t in trades:
            trade_key = (t.timestamp, t.price, t.amount, t.side)
            if self._last_trade_id is not None:
                last_key = self._last_trade_id
                # タプル比較: (timestamp, price, amount) で時系列順を判定
                if trade_key[:3] <= last_key[:3]:
                    continue  # Already seen or older
            new_trades.append(t)

        if new_trades:
            # Update _last_trade_id to latest (tuple)
            latest = new_trades[-1]
            self._last_trade_id = (latest.timestamp, latest.price, latest.amount, latest.side)

        for t in new_trades:
            self._tr_buffer.append(
                {
                    "ts": t.timestamp,
                    "price": t.price,
                    "amount": t.amount,
                    "side": t.side,
                }
            )

    def flush_raw(self, day_str: str | None = None) -> tuple[Path, Path]:
        """Flush in-memory buffers to JSONL gzip files and return paths."""
        day = day_str or self._today_str()

        ob_path = self.raw_dir / "orderbook" / f"{day}.jsonl.gz"
        tr_path = self.raw_dir / "trades" / f"{day}.jsonl.gz"

        # Append mode — open existing gz and add lines
        self._write_jsonl_gz(ob_path, self._ob_buffer)
        self._write_jsonl_gz(tr_path, self._tr_buffer)

        n_ob, n_tr = len(self._ob_buffer), len(self._tr_buffer)
        self._ob_buffer.clear()
        self._tr_buffer.clear()
        logger.info(f"Flushed raw: {n_ob} ob snapshots, {n_tr} trades → {day}")
        return ob_path, tr_path

    @staticmethod
    def _write_jsonl_gz(path: Path, records: list[dict[str, Any]]) -> None:
        append_jsonl_gz(path, records)

    # ------------------------------------------------------------------
    # 1-min aggregation (raw → Parquet)
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_to_1min(
        ob_path: Path,
        tr_path: Path,
        output_path: Path | None = None,
    ) -> pd.DataFrame:
        """Read raw JSONL gzips and produce 1-min aggregated Parquet.

        Returns the aggregated DataFrame.
        """
        # --- Orderbook ---
        ob_records = _read_jsonl_gz(ob_path)
        if ob_records:
            ob_df = pd.DataFrame(ob_records)
            ob_df.index = pd.to_datetime(ob_df["ts"], unit="s", utc=True)
            ob_1m = _aggregate_orderbook_1min(ob_df)
        else:
            ob_1m = pd.DataFrame()

        # --- Trades ---
        tr_records = _read_jsonl_gz(tr_path)
        if tr_records:
            tr_df = pd.DataFrame(tr_records)
            tr_df.index = pd.to_datetime(tr_df["ts"], unit="s", utc=True)
            tr_1m = _aggregate_trades_1min(tr_df)
        else:
            tr_1m = pd.DataFrame()

        # Merge
        if not ob_1m.empty and not tr_1m.empty:
            merged = ob_1m.join(tr_1m, how="outer")
        elif not ob_1m.empty:
            merged = ob_1m
        elif not tr_1m.empty:
            merged = tr_1m
        else:
            merged = pd.DataFrame()

        if not merged.empty and output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(output_path, engine="pyarrow")
            logger.info(f"Aggregated {len(merged)} 1-min rows → {output_path}")

        return merged

    def _flush_and_aggregate(self, auto_aggregate: bool) -> None:
        """バッファ flush + オプション集約. DRY ヘルパー."""
        ob_path, tr_path = self.flush_raw()
        if auto_aggregate:
            try:
                day = self._today_str()
                agg_out = self.agg_dir / f"{day}.parquet"
                self.aggregate_to_1min(ob_path, tr_path, agg_out)
            except Exception as e:
                logger.warning(f"Auto-aggregate failed: {e}")

    def _check_buffer_cap(self) -> bool:
        """バッファが上限超過時に緊急 flush. メモリ保護."""
        if len(self._ob_buffer) + len(self._tr_buffer) > self._buffer_cap:
            logger.warning(
                f"Buffer cap ({self._buffer_cap}) exceeded: "
                f"ob={len(self._ob_buffer)}, tr={len(self._tr_buffer)} — emergency flush"
            )
            self.flush_raw()
            return True
        return False

    # ------------------------------------------------------------------
    # Continuous collection loop
    # ------------------------------------------------------------------

    async def run_continuous(
        self,
        duration_hours: float = 24.0,
        auto_aggregate: bool = False,
    ) -> None:
        """Run collection loop for *duration_hours*.

        Flushes raw buffers every 10 minutes and at end.
        003# #11: auto_aggregate=True triggers aggregate_to_1min after each flush.
        """
        self._running = True
        end_time = time.time() + duration_hours * 3600
        flush_interval = 600  # 10 min
        last_flush = time.time()
        ticks_collected = 0

        logger.info(
            f"Starting continuous collection: symbol={self.symbol}, "
            f"interval={self.poll_interval_sec}s, duration={duration_hours}h"
        )

        try:
            while self._running and time.time() < end_time:
                ob, trades = await self.collect_tick()
                if ob:
                    self._append_raw_ob(ob)
                if trades:
                    self._append_raw_trades(trades)
                ticks_collected += 1

                # Periodic flush
                if time.time() - last_flush > flush_interval:
                    self._flush_and_aggregate(auto_aggregate)
                    last_flush = time.time()

                await asyncio.sleep(self.poll_interval_sec)
        except asyncio.CancelledError:
            logger.info("Collection cancelled")
        finally:
            self._flush_and_aggregate(auto_aggregate)
            logger.info(f"Collection finished. Total ticks: {ticks_collected}")

    # ------------------------------------------------------------------
    # WebSocket-based continuous collection
    # ------------------------------------------------------------------

    async def run_continuous_ws(
        self,
        duration_hours: float = 24.0,
        auto_aggregate: bool = False,
    ) -> None:
        """WebSocket ストリーミングによるデータ収集.

        REST ポーリング (run_continuous) の代替。
        0.1s 間隔のリアルタイムデータを受信し、同じ raw 形式で保存する。
        014# T4: REST 5秒 → WS 0.1秒 で情報密度 50x 向上。
        """
        from ztb.trading.live.exchanges.coincheck.websocket_client import (
            Channel,
            CoincheckPublicWS,
        )

        self._running = True
        end_time = time.time() + duration_hours * 3600
        flush_interval = 600  # 10 min
        last_flush = time.time()
        ticks_collected = 0

        # Callbacks — buffer data using existing methods
        async def _on_trade(trades: list["TradeRecord"]) -> None:
            nonlocal ticks_collected
            self._append_raw_trades(trades)
            ticks_collected += len(trades)
            self._check_buffer_cap()

        async def _on_orderbook(ob: "OrderBookSnapshot") -> None:
            nonlocal ticks_collected
            self._append_raw_ob(ob)
            ticks_collected += 1
            self._check_buffer_cap()

        ws = CoincheckPublicWS()
        ws.on_trade = _on_trade
        ws.on_orderbook = _on_orderbook

        logger.info(
            f"Starting WS collection: symbol={self.symbol}, "
            f"duration={duration_hours}h"
        )

        await ws.start([Channel.TRADES, Channel.ORDERBOOK])

        try:
            while self._running and time.time() < end_time:
                await asyncio.sleep(min(10.0, flush_interval))

                # Periodic flush
                if time.time() - last_flush > flush_interval:
                    self._flush_and_aggregate(auto_aggregate)
                    last_flush = time.time()
                    logger.info(
                        f"WS collection: {ticks_collected} items, "
                        f"ws_stats={ws.stats}"
                    )
        except asyncio.CancelledError:
            logger.info("WS collection cancelled")
        finally:
            await ws.stop()
            self._flush_and_aggregate(auto_aggregate)
            logger.info(
                f"WS collection finished. Total items: {ticks_collected}"
            )

    def stop(self) -> None:
        """Signal the collection loop to stop gracefully."""
        self._running = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around ztb.io.jsonl_gz.read_jsonl_gz."""
    return cast(list[dict[str, Any]], read_jsonl_gz(path))
