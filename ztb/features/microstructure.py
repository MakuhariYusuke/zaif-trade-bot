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
    eps = 1e-10
    out = pd.DataFrame(index=df.index)
    close = df["close"] if "close" in df.columns else None
    buy_volume = df["buy_volume"] if "buy_volume" in df.columns else None
    sell_volume = df["sell_volume"] if "sell_volume" in df.columns else None
    total_vol = (
        buy_volume + sell_volume
        if buy_volume is not None and sell_volume is not None
        else None
    )

    # ---- 1. bid_ask_spread ----
    # If already present from aggregation, keep it; otherwise compute
    if "bid_ask_spread" not in df.columns:
        if "best_ask" in df.columns and "best_bid" in df.columns:
            mid = (df["best_ask"] + df["best_bid"]) / 2
            out["bid_ask_spread"] = (df["best_ask"] - df["best_bid"]) / (mid + eps)
        elif "spread" in df.columns:
            out["bid_ask_spread"] = df["spread"]

    # ---- 2. depth_imbalance  (pass-through) ----
    # Already in DataFrame from aggregation

    # ---- 3. trade_flow_imbalance  (pass-through) ----
    # Already in DataFrame from aggregation

    # ---- 4. vwap_deviation ----
    if "vwap" in df.columns and "close" in df.columns:
        out["vwap_deviation"] = (df["close"] - df["vwap"]) / (df["close"] + eps)

    # ---- 5. trade_intensity ----
    if "trade_count" in df.columns:
        tc_mean = df["trade_count"].rolling(window, min_periods=1).mean()
        out["trade_intensity"] = df["trade_count"] / (tc_mean + eps)

    # ---- 6. order_flow_toxicity (VPIN approximation) ----
    if total_vol is not None and buy_volume is not None and sell_volume is not None:
        if bool(total_vol.eq(0).all()):
            out["order_flow_toxicity"] = 0.0
        else:
            abs_imbalance = (buy_volume - sell_volume).abs()
            # VPIN = rolling mean of |buy - sell| / total_volume
            out["order_flow_toxicity"] = (
                abs_imbalance.rolling(window, min_periods=1).sum()
                / (total_vol.rolling(window, min_periods=1).sum() + eps)
            )

    # ---- 7. price_impact ----
    if close is not None:
        volume_for_impact = total_vol
        if volume_for_impact is None:
            volume_for_impact = df.get("volume", pd.Series(1.0, index=df.index))
        delta_price = close.diff().abs()
        raw_impact = delta_price / (volume_for_impact + eps)
        out["price_impact"] = raw_impact.rolling(window, min_periods=1).mean()

    # ---- 8. micro_return_vol ----
    if close is not None:
        log_ret = np.log(close / close.shift(1))
        out["micro_return_vol"] = log_ret.rolling(window, min_periods=1).std()

    # ---- 9 & 10. bid/ask depth slope ----
    # These require raw orderbook data (bid_price_1..N, bid_size_1..N).
    # For 1-min aggregated data, approximate using bid_vol_5 / mid_price range.
    # Full implementation will be added in P0-10+ when raw depth-level data is available.
    if "bid_vol_5" in df.columns and "best_bid" in df.columns and "mid_price" in df.columns:
        bid_range = (df["mid_price"] - df["best_bid"]).clip(lower=eps)
        out["bid_depth_slope"] = df["bid_vol_5"] / bid_range
    if "ask_vol_5" in df.columns and "best_ask" in df.columns and "mid_price" in df.columns:
        ask_range = (df["best_ask"] - df["mid_price"]).clip(lower=eps)
        out["ask_depth_slope"] = df["ask_vol_5"] / ask_range

    result = df.join(out, how="left")
    feature_cols = [col for col in MICROSTRUCTURE_FEATURES if col in result.columns]
    if feature_cols:
        # Fill NaN from rolling ops on feature columns only.
        # 003# #8: bfill() removed — causes look-ahead leakage in training
        result.loc[:, feature_cols] = result.loc[:, feature_cols].ffill().fillna(0)

    return result

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
