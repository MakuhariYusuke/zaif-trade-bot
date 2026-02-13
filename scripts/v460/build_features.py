#!/usr/bin/env python3
"""
v460 特徴量 Parquet 生成スクリプト.

OHLCV → マイクロストラクチャ proxy 特徴量を生成し、
data/v460/features/btc_jpy_1m_v460_features.parquet に保存する。

Phase 0 G0-data の前提データを構築する。
リアル板/約定データが収集できた段階で proxy → real への差替が可能な設計。

§2.2 特徴量候補 10 種:
  1. bid_ask_spread      — OHLCV proxy: (high-low)/mid
  2. depth_imbalance      — OHLCV proxy: CLV ベース
  3. trade_flow_imbalance — OHLCV proxy: signed volume
  4. vwap_deviation       — 近似 VWAP vs close
  5. trade_intensity      — volume / rolling mean volume
  6. order_flow_toxicity  — VPIN 近似 (|buy-sell|/total)
  7. price_impact         — |Δclose| / volume
  8. micro_return_vol     — log return rolling std
  9. bid_depth_slope      — vol proxied via (volume * CLV+) / range
  10. ask_depth_slope     — vol proxied via (volume * CLV-) / range

Usage:
  python scripts/v460/build_features.py
  python scripts/v460/build_features.py --source data/btc_jpy_1m_v451_optimized_features.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SOURCE = "data/btc_jpy_1m_v451_optimized_features.parquet"
DEFAULT_OUTPUT = "data/v460/features/btc_jpy_1m_v460_features.parquet"

# v460 10 microstructure feature names
V460_FEATURES = [
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


def build_proxy_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """OHLCV からマイクロストラクチャ proxy 特徴量を生成.

    リアル板データがない段階で、OHLCV パターンから合理的に導出できる
    proxy 値を用いてパイプラインを構築する。

    Args:
        df: OHLCV DataFrame (open, high, low, close, volume).
        window: Rolling window for derived features.

    Returns:
        DataFrame with close + 10 microstructure features.
    """
    eps = 1e-10
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    open_ = df["open"].astype(np.float64)
    volume = df["volume"].astype(np.float64).clip(lower=eps)

    out = pd.DataFrame(index=df.index)
    out["close"] = close

    # ---- 1. bid_ask_spread ----
    # Proxy: (high - low) / mid — intra-bar range as spread proxy
    mid = (high + low) / 2
    out["bid_ask_spread"] = (high - low) / (mid + eps)

    # ---- 2. depth_imbalance ----
    # Proxy: CLV (Close Location Value) — where close falls in [low, high]
    # Range [-1, +1]. Positive = buy-side depth dominance proxy
    clv = ((close - low) / (high - low + eps)) * 2 - 1
    out["depth_imbalance"] = clv

    # ---- 3. trade_flow_imbalance ----
    # Proxy: signed volume — CLV × volume, normalized
    signed_vol = clv * volume
    sv_std = signed_vol.rolling(window, min_periods=1).std() + eps
    out["trade_flow_imbalance"] = signed_vol / sv_std

    # ---- 4. vwap_deviation ----
    # Proxy VWAP: typical price = (H+L+C)/3 cumulated with volume
    typical = (high + low + close) / 3
    cum_vol = volume.rolling(window, min_periods=1).sum()
    cum_tp_vol = (typical * volume).rolling(window, min_periods=1).sum()
    vwap_proxy = cum_tp_vol / (cum_vol + eps)
    out["vwap_deviation"] = (close - vwap_proxy) / (close + eps)

    # ---- 5. trade_intensity ----
    # Proxy: volume / rolling mean volume
    vol_mean = volume.rolling(window, min_periods=1).mean()
    out["trade_intensity"] = volume / (vol_mean + eps)

    # ---- 6. order_flow_toxicity (VPIN approximation) ----
    # Proxy: rolling |CLV × volume| / total volume
    abs_signed = (clv * volume).abs()
    out["order_flow_toxicity"] = (
        abs_signed.rolling(window, min_periods=1).sum()
        / (volume.rolling(window, min_periods=1).sum() + eps)
    )

    # ---- 7. price_impact ----
    # |Δclose| / volume, smoothed
    delta_price = close.diff().abs()
    raw_impact = delta_price / (volume + eps)
    out["price_impact"] = raw_impact.rolling(window, min_periods=1).mean()

    # ---- 8. micro_return_vol ----
    log_ret = np.log(close / close.shift(1))
    out["micro_return_vol"] = log_ret.rolling(window, min_periods=1).std()

    # ---- 9. bid_depth_slope ----
    # Proxy: buy-side volume / bid-side range
    # buy fraction of volume approximated by (CLV+1)/2
    buy_frac = ((clv + 1) / 2).clip(0, 1)
    buy_vol = volume * buy_frac
    bid_range = (mid - low).clip(lower=eps)
    out["bid_depth_slope"] = buy_vol / bid_range

    # ---- 10. ask_depth_slope ----
    sell_frac = 1 - buy_frac
    sell_vol = volume * sell_frac
    ask_range = (high - mid).clip(lower=eps)
    out["ask_depth_slope"] = sell_vol / ask_range

    # Fill NaN from rolling (no bfill — 003# #8)
    out = out.ffill().fillna(0)

    return out


def compute_sha256(path: Path) -> str:
    """File SHA-256."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_and_save(
    source_path: str | Path,
    output_path: str | Path,
    window: int = 20,
) -> dict:
    """Load source → build features → save → return metadata."""
    src = Path(source_path)
    if not src.is_absolute():
        src = _PROJECT_ROOT / src
    out = Path(output_path)
    if not out.is_absolute():
        out = _PROJECT_ROOT / out

    logger.info(f"Reading source: {src}")
    df = pd.read_parquet(src)
    logger.info(f"Source shape: {df.shape}")

    # Ensure required OHLCV columns
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing OHLCV columns: {missing}")

    logger.info(f"Building v460 proxy features (window={window})...")
    result = build_proxy_features(df, window=window)
    logger.info(f"Output shape: {result.shape}")

    # Validate: all 10 features present
    for feat in V460_FEATURES:
        assert feat in result.columns, f"Missing feature: {feat}"

    # NaN check
    nan_count = int(result[V460_FEATURES].isna().sum().sum())
    total_cells = len(result) * len(V460_FEATURES)
    nan_ratio = nan_count / max(total_cells, 1)
    logger.info(f"NaN count: {nan_count}/{total_cells} ({nan_ratio:.6f})")
    assert nan_ratio <= 0.01, f"NaN ratio {nan_ratio:.4f} exceeds 1% threshold"

    # Save
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, engine="pyarrow", index=False)
    logger.info(f"Saved: {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    # Hash
    data_hash = compute_sha256(out)
    logger.info(f"SHA-256: {data_hash[:16]}...")

    return {
        "source_path": str(src),
        "output_path": str(out),
        "rows": len(result),
        "features": V460_FEATURES,
        "n_features": len(V460_FEATURES),
        "nan_ratio": round(nan_ratio, 8),
        "sha256": data_hash,
        "window": window,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v460 Feature Builder")
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"Source OHLCV parquet (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--window", type=int, default=20, help="Rolling window")
    args = parser.parse_args()

    meta = build_and_save(args.source, args.output, args.window)

    print("\n" + "=" * 60)
    print("  v460 Feature Build Complete")
    print("=" * 60)
    print(f"  Rows:       {meta['rows']:,}")
    print(f"  Features:   {meta['n_features']}")
    print(f"  NaN ratio:  {meta['nan_ratio']}")
    print(f"  SHA-256:    {meta['sha256'][:16]}...")
    print(f"  Output:     {meta['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
