#!/usr/bin/env python3
"""
Bridge replay system for backtesting trading strategies with realistic slippage.
取引戦略のバックテストのための現実的なスリッページを考慮したブリッジリプレイシステム
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BridgeReplay:
    """
    Replay trading logs against order book data to simulate realistic execution.
    注文ログを板情報に対してリプレイし、現実的な約定をシミュレート
    """

    def __init__(
        self,
        orderbook_dir: str = "data/orderbooks",
        trade_log_dir: str = "logs/trade_logs",
    ):
        """
        Initialize bridge replay system

        Args:
            orderbook_dir: Directory containing order book snapshots
            trade_log_dir: Directory containing trade log files
        """
        self.orderbook_dir = Path(orderbook_dir)
        self.trade_log_dir = Path(trade_log_dir)
        self.orderbook_cache: Dict[str, pd.DataFrame] = {}

    def load_orderbook_snapshot(
        self, timestamp: datetime, pair: str = "btc_jpy"
    ) -> Optional[pd.DataFrame]:
        """
        Load order book snapshot closest to the given timestamp

        Args:
            timestamp: Target timestamp
            pair: Trading pair

        Returns:
            Order book DataFrame or None if not found
        """
        # Create date-based directory path
        date_str = timestamp.strftime("%Y-%m-%d")
        pair_dir = self.orderbook_dir / pair / date_str

        if not pair_dir.exists():
            return None

        # Find closest orderbook file
        orderbook_files = list(pair_dir.glob("*.json"))
        if not orderbook_files:
            return None

        # Find file with timestamp closest to target
        closest_file = None
        min_diff = float("inf")

        for file_path in orderbook_files:
            try:
                # Extract timestamp from filename (assuming format: orderbook_YYYYMMDD_HHMMSS.json)
                filename = file_path.stem
                if "_" in filename:
                    time_part = filename.split("_")[-1]
                    file_timestamp = datetime.strptime(
                        f"{date_str} {time_part}", "%Y-%m-%d %H%M%S"
                    )
                    diff = abs((file_timestamp - timestamp).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_file = file_path
            except ValueError:
                continue

        if closest_file is None:
            return None

        # Load from cache or file
        cache_key = f"{pair}_{date_str}_{closest_file.stem}"
        if cache_key in self.orderbook_cache:
            return self.orderbook_cache[cache_key]

        try:
            with open(closest_file, "r") as f:
                data = json.load(f)

            # Convert to DataFrame
            bids = pd.DataFrame(data.get("bids", []), columns=["price", "size"])
            asks = pd.DataFrame(data.get("asks", []), columns=["price", "size"])

            orderbook = {
                "timestamp": data.get("timestamp", timestamp.isoformat()),
                "bids": bids,
                "asks": asks,
            }

            df = pd.DataFrame([orderbook])
            self.orderbook_cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"Failed to load orderbook {closest_file}: {e}")
            return None

    def load_trade_logs(
        self, start_date: datetime, end_date: datetime, pair: str = "btc_jpy"
    ) -> pd.DataFrame:
        """
        Load trade logs for the specified date range

        Args:
            start_date: Start date for logs
            end_date: End date for logs
            pair: Trading pair

        Returns:
            DataFrame with trade log entries
        """
        all_logs = []

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            log_file = self.trade_log_dir / f"trade_log_{date_str}.jsonl"

            if log_file.exists():
                try:
                    logs = []
                    with open(log_file, "r") as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                # Filter by pair if specified
                                if entry.get("pair", "").lower() == pair.lower():
                                    logs.append(entry)

                    if logs:
                        df = pd.DataFrame(logs)
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        all_logs.append(df)

                except Exception as e:
                    logger.error(f"Failed to load trade log {log_file}: {e}")

            current_date += timedelta(days=1)

        if not all_logs:
            return pd.DataFrame()

        return pd.concat(all_logs, ignore_index=True)

    def simulate_order_execution(
        self, order: Dict[str, Any], orderbook: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Simulate order execution against order book

        Args:
            order: Order details (price, size, side)
            orderbook: Current order book snapshot

        Returns:
            Execution result with slippage analysis
        """
        if orderbook.empty:
            return {
                "executed": False,
                "reason": "no_orderbook",
                "slippage": 0.0,
                "executed_size": 0.0,
                "executed_price": order.get("price", 0.0),
            }

        side = order.get("side", "").lower()
        order_price = order.get("price", 0.0)
        order_size = order.get("size", 0.0)

        if side not in ["buy", "sell"]:
            return {
                "executed": False,
                "reason": "invalid_side",
                "slippage": 0.0,
                "executed_size": 0.0,
                "executed_price": order_price,
            }

        # Get relevant side of order book
        book_side = (
            orderbook["bids"].iloc[0] if side == "sell" else orderbook["asks"].iloc[0]
        )
        if book_side.empty:
            return {
                "executed": False,
                "reason": "empty_book_side",
                "slippage": 0.0,
                "executed_size": 0.0,
                "executed_price": order_price,
            }

        # Sort by price (best bids/asks first)
        if side == "sell":  # Selling to bids (highest prices first)
            book_side = book_side.sort_values("price", ascending=False)
        else:  # Buying from asks (lowest prices first)
            book_side = book_side.sort_values("price", ascending=True)

        # Simulate execution
        remaining_size = order_size
        executed_size = 0.0
        executed_value = 0.0

        for _, level in book_side.iterrows():
            level_price = level["price"]
            level_size = level["size"]

            # Check if order can execute at this level
            if side == "sell" and level_price < order_price:
                # Would sell at lower price than requested
                continue
            elif side == "buy" and level_price > order_price:
                # Would buy at higher price than requested
                continue

            # Execute at this level
            execute_size = min(remaining_size, level_size)
            executed_size += execute_size
            executed_value += execute_size * level_price
            remaining_size -= execute_size

            if remaining_size <= 0:
                break

        if executed_size == 0:
            return {
                "executed": False,
                "reason": "no_liquidity",
                "slippage": 0.0,
                "executed_size": 0.0,
                "executed_price": order_price,
            }

        # Calculate slippage
        avg_executed_price = executed_value / executed_size
        slippage = (
            abs(avg_executed_price - order_price) / order_price
            if order_price > 0
            else 0.0
        )

        return {
            "executed": True,
            "reason": "success",
            "slippage": slippage,
            "executed_size": executed_size,
            "executed_price": avg_executed_price,
            "requested_size": order_size,
            "fill_ratio": executed_size / order_size if order_size > 0 else 0.0,
        }

    def replay_trades(
        self, start_date: datetime, end_date: datetime, pair: str = "btc_jpy"
    ) -> Iterator[Dict[str, Any]]:
        """
        Replay trades against historical order books

        Args:
            start_date: Start date for replay
            end_date: End date for replay
            pair: Trading pair

        Yields:
            Dictionary with replay results for each trade
        """
        logger.info(
            f"Starting trade replay for {pair} from {start_date.date()} to {end_date.date()}"
        )

        # Load trade logs
        trade_logs = self.load_trade_logs(start_date, end_date, pair)

        if trade_logs.empty:
            logger.warning(f"No trade logs found for {pair} in date range")
            return

        logger.info(f"Loaded {len(trade_logs)} trade entries")

        # Sort by timestamp
        trade_logs = trade_logs.sort_values("timestamp")

        replayed_count = 0
        executed_count = 0

        for _, trade in trade_logs.iterrows():
            timestamp = trade["timestamp"]

            # Load corresponding order book
            orderbook = self.load_orderbook_snapshot(timestamp, pair)

            # Simulate execution
            order = {
                "price": trade.get("price", 0.0),
                "size": trade.get("size", 0.0),
                "side": trade.get("side", ""),
            }

            if orderbook is not None:
                execution_result = self.simulate_order_execution(order, orderbook)
            else:
                execution_result = {
                    "executed": False,
                    "reason": "no_orderbook",
                    "slippage": 0.0,
                    "executed_size": 0.0,
                    "executed_price": order.get("price", 0.0),
                }

            # Combine results
            result = {
                "timestamp": timestamp.isoformat(),
                "pair": pair,
                "original_order": order,
                "orderbook_available": orderbook is not None,
                "execution": execution_result,
            }

            replayed_count += 1
            if execution_result["executed"]:
                executed_count += 1

            # Progress logging
            if replayed_count % 100 == 0:
                logger.info(
                    f"Replayed {replayed_count} trades, executed {executed_count}"
                )

            yield result

        logger.info(
            f"Replay completed: {executed_count}/{replayed_count} trades executed"
        )

    def get_replay_summary(
        self, replay_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from replay results

        Args:
            replay_results: List of replay result dictionaries

        Returns:
            Summary statistics
        """
        if not replay_results:
            return {"message": "No replay results to summarize"}

        executed_trades = [r for r in replay_results if r["execution"]["executed"]]
        slippage_values = [r["execution"]["slippage"] for r in executed_trades]

        summary = {
            "total_trades": len(replay_results),
            "executed_trades": len(executed_trades),
            "execution_rate": (
                len(executed_trades) / len(replay_results) if replay_results else 0.0
            ),
            "avg_slippage": np.mean(slippage_values) if slippage_values else 0.0,
            "max_slippage": np.max(slippage_values) if slippage_values else 0.0,
            "median_slippage": np.median(slippage_values) if slippage_values else 0.0,
            "slippage_std": np.std(slippage_values) if slippage_values else 0.0,
            "orderbook_coverage": sum(
                1 for r in replay_results if r["orderbook_available"]
            )
            / len(replay_results),
        }

        return summary
