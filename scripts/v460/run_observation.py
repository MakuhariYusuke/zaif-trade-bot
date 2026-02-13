#!/usr/bin/env python3
"""
観測トラック — MarketDataCollector による板・約定データ収集.

012# §5.2「改訂前でも着手可能な作業」として、
既存 MarketDataCollector を使い orderbook/trades の raw データを蓄積する。
注文は出さない (observation only)。コスト 0 JPY。

Usage:
  # 24時間収集 (デフォルト)
  python scripts/v460/run_observation.py

  # 指定時間収集
  python scripts/v460/run_observation.py --hours 72

  # ポーリング間隔変更 (デフォルト 5秒)
  python scripts/v460/run_observation.py --interval 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.data.market_data_collector import MarketDataCollector
from ztb.trading.live.exchanges.coincheck.adapter import CoincheckAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main(hours: float, interval: float, auto_agg: bool) -> None:
    """Observation-only data collection."""
    # dry_run=True でも get_orderbook / get_recent_trades は public API を叩く
    adapter = CoincheckAdapter(dry_run=True)

    collector = MarketDataCollector(
        adapter=adapter,
        symbol="btc_jpy",
        poll_interval_sec=interval,
    )

    logger.info(
        f"=== Observation Track Start ===\n"
        f"  Duration: {hours}h\n"
        f"  Interval: {interval}s\n"
        f"  Raw dir:  {collector.raw_dir}\n"
        f"  Agg dir:  {collector.agg_dir}\n"
        f"  Auto-aggregate: {auto_agg}\n"
        f"  Mode: observation only (no orders, 0 JPY)\n"
    )

    try:
        await collector.run_continuous(
            duration_hours=hours,
            auto_aggregate=auto_agg,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        collector.stop()


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Observation-only data collection (012# §5)"
    )
    parser.add_argument(
        "--hours", type=float, default=24.0,
        help="Collection duration in hours (default: 24)",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Polling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--auto-aggregate", action="store_true",
        help="Auto-aggregate raw data to 1-min Parquet on each flush",
    )
    args = parser.parse_args()
    asyncio.run(main(args.hours, args.interval, args.auto_aggregate))


if __name__ == "__main__":
    cli()
