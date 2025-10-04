#!/usr/bin/env python3
"""
Slippage analysis for trading execution quality.
取引約定品質のためのスリッページ分析
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SlippageAnalysis:
    """
    Analyze trading slippage based on order book liquidity and execution data.
    注文板の流動性と約定データに基づく取引スリッページの分析
    """

    def __init__(
        self,
        orderbook_dir: str = "data/orderbooks",
        analysis_output_dir: str = "reports/slippage",
    ):
        """
        Initialize slippage analysis

        Args:
            orderbook_dir: Directory containing order book data
            analysis_output_dir: Directory to save analysis results
        """
        self.orderbook_dir = Path(orderbook_dir)
        self.analysis_output_dir = Path(analysis_output_dir)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_orderbook_liquidity(
        self, timestamp: datetime, pair: str = "btc_jpy", depth_levels: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze order book liquidity at a specific timestamp

        Args:
            timestamp: Analysis timestamp
            pair: Trading pair
            depth_levels: Number of price levels to analyze

        Returns:
            Liquidity analysis results
        """
        # Load order book snapshot
        from .bridge_replay import BridgeReplay

        replay = BridgeReplay(str(self.orderbook_dir))
        orderbook = replay.load_orderbook_snapshot(timestamp, pair)

        if orderbook is None or orderbook.empty:
            return {
                "timestamp": timestamp.isoformat(),
                "liquidity_score": 0.0,
                "spread_bps": 0.0,
                "bid_depth": 0.0,
                "ask_depth": 0.0,
                "mid_price": 0.0,
            }

        bids_df = orderbook["bids"].iloc[0]
        asks_df = orderbook["asks"].iloc[0]

        if bids_df.empty or asks_df.empty:
            return {
                "timestamp": timestamp.isoformat(),
                "liquidity_score": 0.0,
                "spread_bps": 0.0,
                "bid_depth": 0.0,
                "ask_depth": 0.0,
                "mid_price": 0.0,
            }

        # Calculate metrics
        best_bid = bids_df["price"].max()  # Highest bid
        best_ask = asks_df["price"].min()  # Lowest ask
        mid_price = (best_bid + best_ask) / 2

        # Spread in basis points
        spread_bps = ((best_ask - best_bid) / mid_price) * 10000

        # Depth analysis (top N levels)
        bid_depth = bids_df.nlargest(depth_levels, "price")["size"].sum()
        ask_depth = asks_df.nsmallest(depth_levels, "price")["size"].sum()

        # Liquidity score (combination of spread and depth)
        # Lower spread and higher depth = higher liquidity
        spread_score = max(
            0, 1 - (spread_bps / 100)
        )  # Normalize spread (100bps = 0 score)
        depth_score = min(
            1.0, (bid_depth + ask_depth) / 10
        )  # Normalize depth (10 units = 1 score)
        liquidity_score = (spread_score + depth_score) / 2

        return {
            "timestamp": timestamp.isoformat(),
            "liquidity_score": liquidity_score,
            "spread_bps": spread_bps,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "mid_price": mid_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
        }

    def analyze_execution_slippage(
        self, trades_df: pd.DataFrame, orderbooks_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze slippage for executed trades

        Args:
            trades_df: DataFrame with trade execution data
            orderbooks_dir: Directory with order book data

        Returns:
            DataFrame with slippage analysis
        """
        if orderbooks_dir:
            self.orderbook_dir = Path(orderbooks_dir)

        results = []

        for _, trade in trades_df.iterrows():
            timestamp = pd.to_datetime(trade["timestamp"])

            # Get order book liquidity at execution time
            liquidity = self.analyze_orderbook_liquidity(
                timestamp, trade.get("pair", "btc_jpy")
            )

            # Calculate slippage metrics
            requested_price = trade.get("requested_price", 0.0)
            executed_price = trade.get("executed_price", 0.0)
            order_size = trade.get("size", 0.0)
            executed_size = trade.get("executed_size", 0.0)

            # Price slippage
            if requested_price > 0:
                price_slippage = abs(executed_price - requested_price) / requested_price
                price_slippage_bps = price_slippage * 10000
            else:
                price_slippage = 0.0
                price_slippage_bps = 0.0

            # Size slippage (partial fills)
            size_slippage = (
                (order_size - executed_size) / order_size if order_size > 0 else 0.0
            )

            # Market impact (if available)
            market_impact = trade.get("market_impact", 0.0)

            result = {
                "timestamp": timestamp,
                "pair": trade.get("pair", "btc_jpy"),
                "side": trade.get("side", ""),
                "requested_price": requested_price,
                "executed_price": executed_price,
                "order_size": order_size,
                "executed_size": executed_size,
                "price_slippage": price_slippage,
                "price_slippage_bps": price_slippage_bps,
                "size_slippage": size_slippage,
                "market_impact": market_impact,
                "liquidity_score": liquidity["liquidity_score"],
                "spread_bps": liquidity["spread_bps"],
                "execution_success": executed_size > 0,
            }

            results.append(result)

        return pd.DataFrame(results)

    def generate_slippage_report(
        self, slippage_df: pd.DataFrame, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive slippage analysis report

        Args:
            slippage_df: DataFrame with slippage analysis
            output_file: Output file path (optional)

        Returns:
            Report summary
        """
        if slippage_df.empty:
            return {"message": "No slippage data to analyze"}

        # Overall statistics
        successful_trades = slippage_df[slippage_df["execution_success"]]

        report = {
            "analysis_period": {
                "start": (
                    slippage_df["timestamp"].min().isoformat()
                    if not slippage_df.empty
                    else None
                ),
                "end": (
                    slippage_df["timestamp"].max().isoformat()
                    if not slippage_df.empty
                    else None
                ),
                "total_trades": len(slippage_df),
                "successful_trades": len(successful_trades),
                "success_rate": (
                    len(successful_trades) / len(slippage_df)
                    if len(slippage_df) > 0
                    else 0.0
                ),
            },
            "slippage_statistics": {},
            "liquidity_analysis": {},
            "recommendations": [],
        }

        if not successful_trades.empty:
            # Price slippage statistics
            report["slippage_statistics"] = {
                "avg_price_slippage_bps": successful_trades[
                    "price_slippage_bps"
                ].mean(),
                "median_price_slippage_bps": successful_trades[
                    "price_slippage_bps"
                ].median(),
                "max_price_slippage_bps": successful_trades["price_slippage_bps"].max(),
                "price_slippage_std_bps": successful_trades["price_slippage_bps"].std(),
                "avg_size_slippage": successful_trades["size_slippage"].mean(),
                "size_slippage_std": successful_trades["size_slippage"].std(),
            }

            # Liquidity analysis
            report["liquidity_analysis"] = {
                "avg_liquidity_score": successful_trades["liquidity_score"].mean(),
                "avg_spread_bps": successful_trades["spread_bps"].mean(),
                "liquidity_correlation": successful_trades["price_slippage_bps"].corr(
                    successful_trades["liquidity_score"]
                ),
            }

            # Side-specific analysis
            for side in ["buy", "sell"]:
                side_trades = successful_trades[
                    successful_trades["side"].str.lower() == side
                ]
                if not side_trades.empty:
                    report[f"{side}_analysis"] = {
                        "count": len(side_trades),
                        "avg_slippage_bps": side_trades["price_slippage_bps"].mean(),
                        "max_slippage_bps": side_trades["price_slippage_bps"].max(),
                    }

            # Generate recommendations
            avg_slippage = report["slippage_statistics"].get("avg_price_slippage_bps")
            if avg_slippage is not None and avg_slippage > 50:  # > 0.5%
                report["recommendations"].append(
                    "High average slippage detected. Consider using limit orders or reducing order size."
                )
            elif avg_slippage is not None and avg_slippage > 20:  # > 0.2%
                report["recommendations"].append(
                    "Moderate slippage detected. Monitor market conditions closely."
                )

            liquidity_corr = report["liquidity_analysis"].get("liquidity_correlation")
            if (
                liquidity_corr is not None and liquidity_corr < -0.3
            ):  # Negative correlation between liquidity and slippage
                report["recommendations"].append(
                    "Slippage increases during low liquidity periods. Consider trading during high liquidity hours."
                )

        # Save report
        if output_file:
            output_path = self.analysis_output_dir / output_file
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Slippage report saved to {output_path}")

        return report

    def monitor_real_time_slippage(
        self, execution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Monitor slippage for real-time trading execution

        Args:
            execution_data: Real-time execution data

        Returns:
            Slippage analysis for the execution
        """
        timestamp = datetime.now()

        # Analyze current liquidity
        pair = execution_data.get("pair", "btc_jpy")
        liquidity = self.analyze_orderbook_liquidity(timestamp, pair)

        # Calculate slippage
        requested_price = execution_data.get("requested_price", 0.0)
        executed_price = execution_data.get("executed_price", 0.0)

        if requested_price > 0:
            price_slippage = abs(executed_price - requested_price) / requested_price
            price_slippage_bps = price_slippage * 10000
        else:
            price_slippage = 0.0
            price_slippage_bps = 0.0

        result = {
            "timestamp": timestamp.isoformat(),
            "pair": pair,
            "price_slippage_bps": price_slippage_bps,
            "liquidity_score": liquidity["liquidity_score"],
            "spread_bps": liquidity["spread_bps"],
            "alert_level": self._determine_alert_level(
                price_slippage_bps, liquidity["liquidity_score"]
            ),
        }

        return result

    def _determine_alert_level(
        self, slippage_bps: float, liquidity_score: float
    ) -> str:
        """
        Determine alert level based on slippage and liquidity

        Args:
            slippage_bps: Slippage in basis points
            liquidity_score: Liquidity score (0-1)

        Returns:
            Alert level string
        """
        if (
            slippage_bps > 100 and liquidity_score < 0.3
        ):  # High slippage + low liquidity
            return "critical"
        elif slippage_bps > 50 or (slippage_bps > 25 and liquidity_score < 0.5):
            return "warning"
        else:
            return "normal"
