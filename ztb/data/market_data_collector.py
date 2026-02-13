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
import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ztb.trading.live.exchanges.base.broker_interfaces import (
    IBroker,
    MarketDataNotSupported,
    OrderBookSnapshot,
    TradeRecord,
)

logger = logging.getLogger(__name__)

# Default paths (project root relative)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "v460" / "raw"
DEFAULT_AGG_DIR = _PROJECT_ROOT / "data" / "v460" / "features"


class MarketDataCollector:
    """Tick raw 収集 + 1分集約の二層保存を行うデータ収集サービス.

    Phase 0 で使用。板情報・約定フローを定期ポーリングし、
    raw を JSONL (gzip)、1 分集約を Parquet で保存する。
    """

    def __init__(
        self,
        adapter: IBroker,
        symbol: str = "btc_jpy",
        raw_dir: Optional[str | Path] = None,
        agg_dir: Optional[str | Path] = None,
        poll_interval_sec: float = 5.0,
    ) -> None:
        self.adapter = adapter
        self.symbol = symbol
        self.raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
        self.agg_dir = Path(agg_dir) if agg_dir else DEFAULT_AGG_DIR
        self.poll_interval_sec = poll_interval_sec

        # Ensure dirs exist
        (self.raw_dir / "orderbook").mkdir(parents=True, exist_ok=True)
        (self.raw_dir / "trades").mkdir(parents=True, exist_ok=True)
        self.agg_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer for current day
        self._ob_buffer: list[dict[str, Any]] = []
        self._tr_buffer: list[dict[str, Any]] = []
        self._last_trade_id: Optional[tuple[float, float, float, str]] = None
        self._running = False

    # ------------------------------------------------------------------
    # Tick collection
    # ------------------------------------------------------------------

    async def collect_tick(
        self,
    ) -> tuple[Optional[OrderBookSnapshot], list[TradeRecord]]:
        """Fetch one tick of orderbook + trades from the adapter."""
        ob: Optional[OrderBookSnapshot] = None
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

    def flush_raw(self, day_str: Optional[str] = None) -> tuple[Path, Path]:
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
        if not records:
            return
        with gzip.open(path, "at", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 1-min aggregation (raw → Parquet)
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_to_1min(
        ob_path: Path, tr_path: Path, output_path: Path
    ) -> pd.DataFrame:
        """Read raw JSONL gzips and produce 1-min aggregated Parquet.

        Returns the aggregated DataFrame.
        """
        # --- Orderbook ---
        ob_records = _read_jsonl_gz(ob_path)
        if ob_records:
            ob_df = pd.DataFrame(ob_records)
            ob_df["dt"] = pd.to_datetime(ob_df["ts"], unit="s", utc=True)
            ob_df = ob_df.set_index("dt")

            # Extract top-of-book features
            ob_df["best_bid"] = ob_df["bids"].apply(
                lambda b: b[0][0] if b else np.nan
            )
            ob_df["best_ask"] = ob_df["asks"].apply(
                lambda a: a[0][0] if a else np.nan
            )
            ob_df["mid_price"] = (ob_df["best_bid"] + ob_df["best_ask"]) / 2
            ob_df["spread"] = (
                (ob_df["best_ask"] - ob_df["best_bid"]) / ob_df["mid_price"]
            )

            # Depth top-5
            ob_df["bid_vol_5"] = ob_df["bids"].apply(
                lambda b: sum(s for _, s in b[:5])
            )
            ob_df["ask_vol_5"] = ob_df["asks"].apply(
                lambda a: sum(s for _, s in a[:5])
            )
            ob_df["depth_imbalance"] = (ob_df["bid_vol_5"] - ob_df["ask_vol_5"]) / (
                ob_df["bid_vol_5"] + ob_df["ask_vol_5"]
            ).replace(0, np.nan)

            # Resample to 1min — last snapshot of each minute
            ob_1m = (
                ob_df[["best_bid", "best_ask", "mid_price", "spread",
                       "bid_vol_5", "ask_vol_5", "depth_imbalance"]]
                .resample("1min")
                .last()
                .dropna(how="all")
            )
            # Also add spread range within the minute
            ob_1m["spread_range"] = (
                ob_df["spread"].resample("1min").max()
                - ob_df["spread"].resample("1min").min()
            )
        else:
            ob_1m = pd.DataFrame()

        # --- Trades ---
        tr_records = _read_jsonl_gz(tr_path)
        if tr_records:
            tr_df = pd.DataFrame(tr_records)
            tr_df["dt"] = pd.to_datetime(tr_df["ts"], unit="s", utc=True)
            tr_df = tr_df.set_index("dt")

            buy_mask = tr_df["side"].str.lower() == "buy"
            tr_df["buy_vol"] = tr_df["amount"].where(buy_mask, 0)
            tr_df["sell_vol"] = tr_df["amount"].where(~buy_mask, 0)

            tr_1m = tr_df.resample("1min").agg(
                buy_volume=("buy_vol", "sum"),
                sell_volume=("sell_vol", "sum"),
                trade_count=("price", "count"),
                vwap=("price", lambda x: np.average(x, weights=tr_df.loc[x.index, "amount"]) if len(x) > 0 else np.nan),
            )
            tr_1m["trade_flow_imbalance"] = (
                (tr_1m["buy_volume"] - tr_1m["sell_volume"])
                / (tr_1m["buy_volume"] + tr_1m["sell_volume"]).replace(0, np.nan)
            )
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

        if not merged.empty:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(output_path, engine="pyarrow")
            logger.info(f"Aggregated {len(merged)} 1-min rows → {output_path}")

        return merged

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
                    ob_path, tr_path = self.flush_raw()
                    if auto_aggregate:
                        try:
                            # 007# F3: pass correct paths to static method
                            day = self._today_str()
                            agg_out = self.agg_dir / f"{day}.parquet"
                            self.aggregate_to_1min(ob_path, tr_path, agg_out)
                        except Exception as e:
                            logger.warning(f"Auto-aggregate failed: {e}")
                    last_flush = time.time()

                await asyncio.sleep(self.poll_interval_sec)
        except asyncio.CancelledError:
            logger.info("Collection cancelled")
        finally:
            ob_path, tr_path = self.flush_raw()
            if auto_aggregate:
                try:
                    # 007# F3: pass correct paths to static method
                    day = self._today_str()
                    agg_out = self.agg_dir / f"{day}.parquet"
                    self.aggregate_to_1min(ob_path, tr_path, agg_out)
                except Exception as e:
                    logger.warning(f"Final auto-aggregate failed: {e}")
            logger.info(f"Collection finished. Total ticks: {ticks_collected}")

    def stop(self) -> None:
        """Signal the collection loop to stop gracefully."""
        self._running = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL gzip file into a list of dicts."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
