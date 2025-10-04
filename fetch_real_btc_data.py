#!/usr/bin/env python3
"""
Fetch real BTC/JPY historical data from Coincheck API for validation.
"""

import csv
from datetime import datetime
from typing import Any, Dict, List, cast

import requests

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoincheckDataFetcher:
    """Fetch historical BTC/JPY data from Coincheck API."""

    BASE_URL = "https://coincheck.com"

    def __init__(self) -> None:
        self.session = requests.Session()

    def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades from Coincheck API."""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/api/trades",
                params=cast(Dict[str, Any], {"pair": "btc_jpy", "limit": limit}),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            logger.info(f"API Response: {data}")  # Debug: show actual response

            if not data.get("success", False):
                logger.error("Coincheck API returned success=False")
                return []

            return cast(List[Dict[str, Any]], data.get("data", []))
        except Exception as e:
            logger.error(f"Failed to fetch recent trades: {e}")
            return []

    def get_ohlc_data(
        self, period: str = "1min", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get OHLC data from Coincheck API."""
        try:
            # Coincheck OHLC endpoint
            response = self.session.get(
                f"{self.BASE_URL}/api/exchange/rate/ohlc",
                params=cast(Dict[str, Any], {"pair": "btc_jpy", "period": period, "limit": limit}),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success", False):
                logger.error("Coincheck API returned success=False")
                return []

            return cast(List[Dict[str, Any]], data.get("data", []))
        except Exception as e:
            logger.error(f"Failed to fetch OHLC data: {e}")
            return []

    def create_dataset_from_trades(
        self, trades: List[Dict[str, Any]], output_file: str
    ) -> None:
        """Create a dataset similar to ml-dataset-enhanced.csv from trades data."""
        if not trades:
            logger.error("No trades data to process")
            return

        # Sort trades by timestamp (oldest first)
        trades_sorted = sorted(trades, key=lambda x: x.get("created_at", 0))

        # Prepare CSV data
        csv_data = []

        for i, trade in enumerate(trades_sorted):
            try:
                # Basic trade data
                timestamp_str = trade["created_at"]
                # Parse ISO 8601 format (e.g., "2025-10-03T06:41:18.000Z")
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                price = float(trade["rate"])
                qty = float(trade["amount"])

                # Create basic OHLCV data (simplified)
                open_price = price
                high_price = price
                low_price = price
                close_price = price
                volume = qty

                # Create row similar to ml-dataset-enhanced.csv format
                row = {
                    "ts": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "pair": "BTC/JPY",
                    "side": trade.get("order_type", "buy"),
                    "rsi": 50.0,  # Placeholder
                    "sma_short": price,  # Placeholder
                    "sma_long": price,  # Placeholder
                    "price": price,
                    "qty": qty,
                    "pnl": 0.0,  # Placeholder
                    "win": 1 if trade.get("order_type") == "buy" else 0,  # Placeholder
                    "source": "coincheck_real",
                    # Add basic technical indicators (placeholders)
                    "ADX": 0.0,
                    "ATR": 0.0,
                    "ATR_simplified": 0.0,
                    "BB_Lower": price * 0.98,
                    "BB_Middle": price,
                    "BB_Position": 0.5,
                    "BB_Upper": price * 1.02,
                    "BB_Width": 0.04,
                    "CCI": 0.0,
                    "DOW": 0.0,
                    "Donchian_Pos_20": 0.5,
                    "Donchian_Slope_20": 0.0,
                    "Donchian_Width_Rel_20": 0.0,
                    "EMACross_Diff": 0.0,
                    "EMACross_Signal": 0,
                    "HV": 0.0,
                    "HeikinAshi_Close": price,
                    "HeikinAshi_High": price,
                    "HeikinAshi_Low": price,
                    "HeikinAshi_Open": price,
                    "HourOfDay": timestamp.hour,
                    "Ichimoku_Chikou": price,
                    "Ichimoku_Cloud_Thickness": 0.0,
                    "Ichimoku_Composite_Signal": 0,
                    "Ichimoku_Cross": 0,
                    "Ichimoku_Diff_Norm": 0.0,
                    "Ichimoku_Kijun": price,
                    "Ichimoku_Price_Cloud_Distance": 0.0,
                    "Ichimoku_Senkou_A": price,
                    "Ichimoku_Senkou_B": price,
                    "Ichimoku_Tenkan": price,
                    "Ichimoku_Trend": 0,
                    "KAMA": price,
                    "Kalman_Estimate": price,
                    "Kalman_Residual": 0.0,
                    "Kalman_Residual_Norm": 0.0,
                    "MACD": 0.0,
                    "MFI": 50.0,
                    "MinusDI": 0.0,
                    "OBV": 0.0,
                    "PlusDI": 0.0,
                    "PriceVolumeCorr": 0.0,
                    "ROC": 0.0,
                    "RSI": 50.0,
                    "ReturnMA_Medium": 0.0,
                    "ReturnMA_Short": 0.0,
                    "ReturnStdDev": 0.0,
                    "Stochastic": 50.0,
                    "Supertrend": price,
                    "Supertrend_Direction": 0,
                    "TEMA": price,
                    "VWAP": price,
                    "ZScore": 0.0,
                    "atr_10": 0.0,
                    "close": close_price,
                    "ema_5": price,
                    "high": high_price,
                    "low": low_price,
                    "open": open_price,
                    "rolling_mean_20": price,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "volume": volume,
                }

                csv_data.append(row)

            except Exception as e:
                logger.warning(f"Failed to process trade {i}: {e}")
                continue

        # Write to CSV
        if csv_data:
            fieldnames = csv_data[0].keys()
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

            logger.info(f"Created dataset with {len(csv_data)} rows: {output_file}")
        else:
            logger.error("No valid data to write")


def main() -> None:
    """Main function to fetch and create real BTC/JPY dataset."""
    fetcher = CoincheckDataFetcher()

    # Get recent trades
    logger.info("Fetching recent trades from Coincheck...")
    trades = fetcher.get_recent_trades(limit=500)  # Get more data

    if not trades:
        logger.error("Failed to fetch trades data")
        return

    logger.info(f"Fetched {len(trades)} trades")

    # Create dataset
    output_file = "btc_jpy_real_dataset.csv"
    fetcher.create_dataset_from_trades(trades, output_file)

    # Also try OHLC data if available
    logger.info("Fetching OHLC data...")
    ohlc_data = fetcher.get_ohlc_data(limit=100)
    if ohlc_data:
        logger.info(f"Fetched {len(ohlc_data)} OHLC records")
    else:
        logger.info("OHLC data not available or failed to fetch")


if __name__ == "__main__":
    main()
