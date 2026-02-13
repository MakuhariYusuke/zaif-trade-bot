"""
マイクロストラクチャ特徴量算出モジュール.

001# §2.2 の 10 候補特徴量を 1 分集約 Parquet から生成する.
既存 hft_proxies.py (OHLCV proxy) とは別系統 — リアル板/約定データベース.

Usage:
    from ztb.features.microstructure import add_microstructure_features
    df = add_microstructure_features(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_microstructure_features(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Add microstructure features from 1-min aggregated data.

    Expected input columns (from MarketDataCollector aggregate_to_1min):
        - best_bid, best_ask, mid_price, spread
        - bid_vol_5, ask_vol_5, depth_imbalance
        - buy_volume, sell_volume, trade_count, vwap, trade_flow_imbalance
        - close (from OHLCV)

    Generated features (§2.2):
        1. bid_ask_spread (pass-through or computed)
        2. depth_imbalance (pass-through)
        3. trade_flow_imbalance (pass-through)
        4. vwap_deviation
        5. trade_intensity
        6. order_flow_toxicity (VPIN approximation)
        7. price_impact
        8. micro_return_vol
        9. bid_depth_slope
        10. ask_depth_slope

    Args:
        df: DataFrame with required columns.
        window: Rolling window for derived features (default 20 = 20 min).

    Returns:
        DataFrame with added microstructure feature columns.
    """
    df = df.copy()
    eps = 1e-10

    # ---- 1. bid_ask_spread ----
    # If already present from aggregation, keep it; otherwise compute
    if "bid_ask_spread" not in df.columns:
        if "best_ask" in df.columns and "best_bid" in df.columns:
            mid = (df["best_ask"] + df["best_bid"]) / 2
            df["bid_ask_spread"] = (df["best_ask"] - df["best_bid"]) / (mid + eps)
        elif "spread" in df.columns:
            df["bid_ask_spread"] = df["spread"]

    # ---- 2. depth_imbalance  (pass-through) ----
    # Already in DataFrame from aggregation

    # ---- 3. trade_flow_imbalance  (pass-through) ----
    # Already in DataFrame from aggregation

    # ---- 4. vwap_deviation ----
    if "vwap" in df.columns and "close" in df.columns:
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / (df["close"] + eps)

    # ---- 5. trade_intensity ----
    if "trade_count" in df.columns:
        tc_mean = df["trade_count"].rolling(window, min_periods=1).mean()
        df["trade_intensity"] = df["trade_count"] / (tc_mean + eps)

    # ---- 6. order_flow_toxicity (VPIN approximation) ----
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        total_vol = df["buy_volume"] + df["sell_volume"]
        abs_imbalance = (df["buy_volume"] - df["sell_volume"]).abs()
        # VPIN = rolling mean of |buy - sell| / total_volume
        df["order_flow_toxicity"] = (
            abs_imbalance.rolling(window, min_periods=1).sum()
            / (total_vol.rolling(window, min_periods=1).sum() + eps)
        )

    # ---- 7. price_impact ----
    if "close" in df.columns:
        total_vol = (
            (df["buy_volume"] + df["sell_volume"])
            if "buy_volume" in df.columns
            else df.get("volume", pd.Series(1.0, index=df.index))
        )
        delta_price = df["close"].diff().abs()
        raw_impact = delta_price / (total_vol + eps)
        df["price_impact"] = raw_impact.rolling(window, min_periods=1).mean()

    # ---- 8. micro_return_vol ----
    if "close" in df.columns:
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["micro_return_vol"] = log_ret.rolling(window, min_periods=1).std()

    # ---- 9 & 10. bid/ask depth slope ----
    # These require raw orderbook data (bid_price_1..N, bid_size_1..N).
    # For 1-min aggregated data, approximate using bid_vol_5 / mid_price range.
    # Full implementation will be added in P0-10+ when raw depth-level data is available.
    if "bid_vol_5" in df.columns and "best_bid" in df.columns and "mid_price" in df.columns:
        bid_range = (df["mid_price"] - df["best_bid"]).clip(lower=eps)
        df["bid_depth_slope"] = df["bid_vol_5"] / bid_range
    if "ask_vol_5" in df.columns and "best_ask" in df.columns and "mid_price" in df.columns:
        ask_range = (df["best_ask"] - df["mid_price"]).clip(lower=eps)
        df["ask_depth_slope"] = df["ask_vol_5"] / ask_range

    # Fill NaN from rolling ops
    df = df.ffill().bfill().fillna(0)

    return df


# Canonical list of v460 microstructure feature columns
MICROSTRUCTURE_FEATURES: list[str] = [
    "bid_ask_spread",
    "depth_imbalance",
    "trade_flow_imbalance",
    "vwap_deviation",
    "trade_intensity",
    "order_flow_toxicity",
    "price_impact",
    "micro_return_vol",
    "bid_depth_slope",
    "ask_depth_slope",
]
