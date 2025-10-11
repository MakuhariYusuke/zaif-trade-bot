#!/usr/bin/env python3
"""
Fetch real BTC/JPY historical data from Yahoo Finance for validation.
"""

from datetime import datetime, timedelta
from typing import cast

import pandas as pd
import ta  # type: ignore[import-untyped]  # Technical Analysis library
import yfinance as yf  # type: ignore[import-untyped]

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class YahooFinanceDataFetcher:
    """Fetch historical BTC/JPY data from Yahoo Finance."""

    def __init__(self) -> None:
        pass

    def fetch_btc_jpy_data(
        self, start_date: str, end_date: str | None = None, interval: str = "1m"
    ) -> pd.DataFrame:
        """Fetch BTC/JPY data from Yahoo Finance."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            f"Fetching BTC/JPY data from {start_date} to {end_date} with {interval} interval"
        )

        # BTC-JPY ticker on Yahoo Finance
        ticker = yf.Ticker("BTC-JPY")

        # Fetch data
        df = cast(pd.DataFrame, ticker.history(start=start_date, end=end_date, interval=interval))

        if df.empty:
            logger.error("No data fetched from Yahoo Finance")
            return pd.DataFrame()

        logger.info(f"Fetched {len(df)} rows of data")
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        if df.empty:
            return df

        # Basic price data
        df = df.copy()
        df["close"] = df["Close"]
        df["high"] = df["High"]
        df["low"] = df["Low"]
        df["open"] = df["Open"]
        df["volume"] = df["Volume"]

        # Add technical indicators using ta library
        try:
            # RSI
            df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

            # SMA
            df["sma_short"] = ta.trend.SMAIndicator(
                df["close"], window=5
            ).sma_indicator()
            df["sma_long"] = ta.trend.SMAIndicator(
                df["close"], window=20
            ).sma_indicator()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
            df["BB_Lower"] = bb.bollinger_lband()
            df["BB_Middle"] = bb.bollinger_mavg()
            df["BB_Upper"] = bb.bollinger_hband()
            df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
            df["BB_Position"] = (df["close"] - df["BB_Lower"]) / (
                df["BB_Upper"] - df["BB_Lower"]
            )

            # MACD
            macd = ta.trend.MACD(df["close"])
            df["MACD"] = macd.macd()

            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
            df["Stochastic"] = stoch.stoch()

            # ATR
            df["ATR"] = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"]
            ).average_true_range()

            # Fill NaN values
            df = df.fillna(50.0)  # Default neutral values

        except Exception as e:
            logger.warning(f"Failed to add technical indicators: {e}")
            # Fill with default values
            df["rsi"] = 50.0
            df["sma_short"] = df["close"]
            df["sma_long"] = df["close"]
            df["BB_Lower"] = df["close"] * 0.98
            df["BB_Middle"] = df["close"]
            df["BB_Upper"] = df["close"] * 1.02
            df["BB_Width"] = 0.04
            df["BB_Position"] = 0.5
            df["MACD"] = 0.0
            df["Stochastic"] = 50.0
            df["ATR"] = 0.0

        return df

    def create_ml_dataset(self, df: pd.DataFrame, output_file: str) -> None:
        """Create ML dataset in the required format."""
        if df.empty:
            logger.error("No data to process")
            return

        ml_data = []

        for i, (timestamp, row) in enumerate(df.iterrows()):
            timestamp = cast(datetime, timestamp)
            try:
                # Create row in ml-dataset-enhanced.csv format
                ml_row = {
                    "ts": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "pair": "BTC/JPY",
                    "side": "buy" if i % 2 == 0 else "sell",  # Alternate for simulation
                    "rsi": float(row.get("rsi", 50.0)),
                    "sma_short": float(row.get("sma_short", row["close"])),
                    "sma_long": float(row.get("sma_long", row["close"])),
                    "price": float(row["close"]),
                    "qty": 0.001,  # Fixed small amount
                    "pnl": 0.0,
                    "win": 1,
                    "source": "yahoo_finance_real",
                    # Add all required columns with default/placeholder values
                    "ADX": 0.0,
                    "ATR": float(row.get("ATR", 0.0)),
                    "ATR_simplified": float(row.get("ATR", 0.0)),
                    "BB_Lower": float(row.get("BB_Lower", row["close"] * 0.98)),
                    "BB_Middle": float(row.get("BB_Middle", row["close"])),
                    "BB_Position": float(row.get("BB_Position", 0.5)),
                    "BB_Upper": float(row.get("BB_Upper", row["close"] * 1.02)),
                    "BB_Width": float(row.get("BB_Width", 0.04)),
                    "CCI": 0.0,
                    "DOW": 0.0,
                    "Donchian_Pos_20": 0.5,
                    "Donchian_Slope_20": 0.0,
                    "Donchian_Width_Rel_20": 0.0,
                    "EMACross_Diff": 0.0,
                    "EMACross_Signal": 0,
                    "HV": 0.0,
                    "HeikinAshi_Close": float(row["close"]),
                    "HeikinAshi_High": float(row["high"]),
                    "HeikinAshi_Low": float(row["low"]),
                    "HeikinAshi_Open": float(row["open"]),
                    "HourOfDay": timestamp.hour,
                    "Ichimoku_Chikou": float(row["close"]),
                    "Ichimoku_Cloud_Thickness": 0.0,
                    "Ichimoku_Composite_Signal": 0,
                    "Ichimoku_Cross": 0,
                    "Ichimoku_Diff_Norm": 0.0,
                    "Ichimoku_Kijun": float(row["close"]),
                    "Ichimoku_Price_Cloud_Distance": 0.0,
                    "Ichimoku_Senkou_A": float(row["close"]),
                    "Ichimoku_Senkou_B": float(row["close"]),
                    "Ichimoku_Tenkan": float(row["close"]),
                    "Ichimoku_Trend": 0,
                    "KAMA": float(row["close"]),
                    "Kalman_Estimate": float(row["close"]),
                    "Kalman_Residual": 0.0,
                    "Kalman_Residual_Norm": 0.0,
                    "MACD": float(row.get("MACD", 0.0)),  # Added MACD
                    "MFI": 50.0,
                    "MinusDI": 0.0,
                    "OBV": 0.0,
                    "PlusDI": 0.0,
                    "PriceVolumeCorr": 0.0,
                    "ROC": 0.0,
                    "RSI": float(row.get("rsi", 50.0)),
                    "ReturnMA_Medium": 0.0,
                    "ReturnMA_Short": 0.0,
                    "ReturnStdDev": 0.0,
                    "Stochastic": float(
                        row.get("Stochastic", 50.0)
                    ),  # Added Stochastic
                    "Supertrend": float(row["close"]),
                    "Supertrend_Direction": 0,
                    "TEMA": float(row["close"]),
                    "VWAP": float(row["close"]),
                    "ZScore": 0.0,
                    "atr_10": float(row.get("ATR", 0.0)),
                    "close": float(row["close"]),  # Added close
                    "ema_5": float(row.get("sma_short", row["close"])),
                    "high": float(row["high"]),  # Added high
                    "low": float(row["low"]),  # Added low
                    "open": float(row["open"]),  # Added open
                    "rolling_mean_20": float(row.get("sma_long", row["close"])),
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "volume": float(row["volume"]),
                }

                ml_data.append(ml_row)

            except Exception as e:
                logger.warning(f"Failed to process row {i}: {e}")
                continue

        # Save to CSV
        if ml_data:
            ml_df = pd.DataFrame(ml_data)
            ml_df.to_csv(output_file, index=False)
            logger.info(f"Created ML dataset with {len(ml_data)} rows: {output_file}")
        else:
            logger.error("No valid data to write")


def main() -> None:
    """Main function to fetch and create real BTC/JPY dataset."""
    fetcher = YahooFinanceDataFetcher()

    # Fetch recent data (last 7 days, 1-minute intervals)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    df = fetcher.fetch_btc_jpy_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1m",
    )

    if df.empty:
        logger.error("Failed to fetch data")
        return

    # Add technical indicators
    df_with_indicators = fetcher.add_technical_indicators(df)

    # Create ML dataset
    output_file = "btc_jpy_yahoo_real_dataset.csv"
    fetcher.create_ml_dataset(df_with_indicators, output_file)


if __name__ == "__main__":
    main()
