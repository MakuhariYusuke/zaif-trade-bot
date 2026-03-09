#!/usr/bin/env python3
"""
v460 特徴量 Parquet 生成スクリプト.

2 モード対応:
  proxy: OHLCV → マイクロストラクチャ proxy 特徴量を生成 (G0-data 用)
  real:  raw orderbook/trades JSONL.gz → 1分集約 → real 特徴量を生成 (G1 再検証用)

Phase 0 G0-data の前提データを構築する。
リアル板/約定データが収集できた段階で proxy → real への差替が可能な設計。

§2.2 特徴量候補 10 種:
  1. bid_ask_spread      — real: (best_ask-best_bid)/mid / proxy: (high-low)/mid
  2. depth_imbalance      — real: (bid_vol_5-ask_vol_5) / proxy: CLV ベース
  3. trade_flow_imbalance — real: (buy_vol-sell_vol) / proxy: signed volume
  4. vwap_deviation       — 近似 VWAP vs close
  5. trade_intensity      — volume / rolling mean volume
  6. order_flow_toxicity  — VPIN 近似 (|buy-sell|/total)
  7. price_impact         — |Δclose| / volume
  8. micro_return_vol     — log return rolling std
  9. bid_depth_slope      — real: bid_vol_5/bid_range / proxy: (volume*CLV+)/range
  10. ask_depth_slope     — real: ask_vol_5/ask_range / proxy: (volume*CLV-)/range

Usage:
  # proxy モード (デフォルト — 従来互換)
  python scripts/v460/build_features.py
  python scripts/v460/build_features.py --mode proxy --source data/btc_jpy_1m_v451_optimized_features.parquet

  # real モード — raw data から生成
  python scripts/v460/build_features.py --mode real
  python scripts/v460/build_features.py --mode real --raw-dir data/v460/raw --date 20260213

  # real モード — 全日付一括
  python scripts/v460/build_features.py --mode real --all-dates
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from ztb.data.market_data_collector import MarketDataCollector
from ztb.features.microstructure import add_microstructure_features, MICROSTRUCTURE_FEATURES
from ztb.utils.run_manifest import compute_file_hash as _compute_shared_file_hash

# Default paths
DEFAULT_SOURCE = "data/btc_jpy_1m_v451_optimized_features.parquet"
DEFAULT_OUTPUT = "data/v460/features/btc_jpy_1m_v460_features.parquet"
DEFAULT_RAW_DIR = "data/v460/raw"
DEFAULT_REAL_OUTPUT = "data/v460/features/btc_jpy_1m_v460_real_features.parquet"

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
    return _compute_shared_file_hash(path)


# ---------------------------------------------------------------------------
# Real data pipeline: raw JSONL.gz → 1min agg → microstructure features
# ---------------------------------------------------------------------------

def _discover_dates(raw_dir: Path) -> list[str]:
    """raw_dir 内の日付ファイルを検出して日付文字列リストを返す."""
    return sorted(_discover_daily_inputs(raw_dir))


def _discover_daily_inputs(raw_dir: Path) -> dict[str, tuple[Path | None, Path | None]]:
    """利用可能な日次 raw 入力を date -> (orderbook, trades) で返す."""
    ob_dir = raw_dir / "orderbook"
    daily_inputs: dict[str, tuple[Path | None, Path | None]] = {}
    if ob_dir.is_dir():
        for f in ob_dir.glob("*.jsonl.gz"):
            date_str = f.name.replace(".jsonl.gz", "")
            _, tr_path = daily_inputs.get(date_str, (None, None))
            daily_inputs[date_str] = (f, tr_path)
    tr_dir = raw_dir / "trades"
    if tr_dir.is_dir():
        for f in tr_dir.glob("*.jsonl.gz"):
            date_str = f.name.replace(".jsonl.gz", "")
            ob_path, _ = daily_inputs.get(date_str, (None, None))
            daily_inputs[date_str] = (ob_path, f)
    return daily_inputs


def build_real_features(
    raw_dir: str | Path,
    output_path: str | Path,
    dates: list[str] | None = None,
    window: int = 20,
) -> dict:
    """raw JSONL.gz → aggregate_to_1min → microstructure features → Parquet.

    Args:
        raw_dir: raw data ディレクトリ (orderbook/, trades/ サブディレクトリ含む)
        output_path: 出力 Parquet パス
        dates: 処理対象の日付リスト (None = 全日付)
        window: 特徴量の rolling window

    Returns:
        メタデータ dict
    """
    raw = Path(raw_dir)
    if not raw.is_absolute():
        raw = _PROJECT_ROOT / raw
    out = Path(output_path)
    if not out.is_absolute():
        out = _PROJECT_ROOT / out

    # Discover dates and reuse resolved daily inputs to avoid repeated exists/stat checks.
    daily_inputs = _discover_daily_inputs(raw)
    all_dates = sorted(daily_inputs)
    if not all_dates:
        raise FileNotFoundError(f"No raw data found in {raw}")

    target_dates = dates if dates else all_dates
    logger.info(f"Target dates: {target_dates} (available: {all_dates})")

    # Aggregate each date
    dfs: list[pd.DataFrame] = []
    for d in target_dates:
        ob_path, tr_path = daily_inputs.get(d, (None, None))
        if ob_path is None and tr_path is None:
            logger.warning(f"No data for date {d}, skipping")
            continue

        agg_df = MarketDataCollector.aggregate_to_1min(ob_path, tr_path, output_path=None)
        if not agg_df.empty:
            dfs.append(agg_df)
            logger.info(f"  {d}: {len(agg_df)} rows aggregated")
        else:
            logger.warning(f"  {d}: empty aggregation result")

    if not dfs:
        raise ValueError("No data aggregated from any date")

    # Merge all dates
    merged = pd.concat(dfs, axis=0).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    logger.info(f"Merged 1-min data: {len(merged)} rows, cols: {list(merged.columns)}")

    # Generate close from mid_price if not present (real data doesn't have OHLCV close)
    if "close" not in merged.columns:
        if "mid_price" in merged.columns:
            merged["close"] = merged["mid_price"]
            logger.info("Using mid_price as close surrogate")
        else:
            raise KeyError("Neither 'close' nor 'mid_price' found in aggregated data")

    # Apply microstructure features
    logger.info(f"Adding microstructure features (window={window})...")
    result = add_microstructure_features(merged, window=window)
    logger.info(f"Result shape: {result.shape}")

    # Validate: all 10 features present
    missing_feats = [f for f in MICROSTRUCTURE_FEATURES if f not in result.columns]
    if missing_feats:
        logger.warning(f"Missing features (will be filled with 0): {missing_feats}")

    # NaN check
    feat_cols = [f for f in MICROSTRUCTURE_FEATURES if f in result.columns]
    nan_count = int(result[feat_cols].isna().sum().sum())
    total_cells = len(result) * len(feat_cols)
    nan_ratio = nan_count / max(total_cells, 1)
    logger.info(f"NaN count: {nan_count}/{total_cells} ({nan_ratio:.6f})")

    # Save
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, engine="pyarrow")
    logger.info(f"Saved: {out} ({out.stat().st_size / 1024:.1f} KB)")

    data_hash = compute_sha256(out)
    logger.info(f"SHA-256: {data_hash[:16]}...")

    return {
        "mode": "real",
        "raw_dir": str(raw),
        "output_path": str(out),
        "dates": target_dates,
        "rows": len(result),
        "columns": list(result.columns),
        "features": feat_cols,
        "n_features": len(feat_cols),
        "nan_ratio": round(nan_ratio, 8),
        "sha256": data_hash,
        "window": window,
    }


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
        "--mode", choices=["proxy", "real"], default="proxy",
        help="proxy: OHLCV proxy features / real: raw orderbook+trades features",
    )
    # Proxy mode options
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"Source OHLCV parquet for proxy mode (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (auto-determined by mode if omitted)",
    )
    # Real mode options
    parser.add_argument(
        "--raw-dir", default=DEFAULT_RAW_DIR,
        help=f"Raw data directory for real mode (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Specific date to process in real mode (e.g. 20260213)",
    )
    parser.add_argument(
        "--all-dates", action="store_true",
        help="Process all available dates in real mode",
    )
    parser.add_argument("--window", type=int, default=20, help="Rolling window")
    args = parser.parse_args()

    if args.mode == "real":
        output = args.output or DEFAULT_REAL_OUTPUT
        dates = None
        if args.date:
            dates = [args.date]
        elif not args.all_dates:
            dates = None  # default: all dates

        meta = build_real_features(
            raw_dir=args.raw_dir,
            output_path=output,
            dates=dates,
            window=args.window,
        )
        print("\n" + "=" * 60)
        print("  v460 Real Feature Build Complete")
        print("=" * 60)
        print(f"  Mode:       real")
        print(f"  Dates:      {meta['dates']}")
        print(f"  Rows:       {meta['rows']:,}")
        print(f"  Features:   {meta['n_features']}")
        print(f"  NaN ratio:  {meta['nan_ratio']}")
        print(f"  SHA-256:    {meta['sha256'][:16]}...")
        print(f"  Output:     {meta['output_path']}")
        print("=" * 60)
    else:
        output = args.output or DEFAULT_OUTPUT
        meta = build_and_save(args.source, output, args.window)
        print("\n" + "=" * 60)
        print("  v460 Feature Build Complete")
        print("=" * 60)
        print(f"  Mode:       proxy")
        print(f"  Rows:       {meta['rows']:,}")
        print(f"  Features:   {meta['n_features']}")
        print(f"  NaN ratio:  {meta['nan_ratio']}")
        print(f"  SHA-256:    {meta['sha256'][:16]}...")
        print(f"  Output:     {meta['output_path']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
