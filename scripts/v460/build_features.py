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
from ztb.data.raw_paths import resolve_available_raw_dates, resolve_raw_dir
from ztb.features.microstructure import add_microstructure_features, MICROSTRUCTURE_FEATURES
from ztb.utils.run_manifest import compute_file_hash as _compute_shared_file_hash
from scripts.v460.ml.feature_enricher import discover_raw_daily_inputs

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

# 377# M2-M5 市場理論 proxy 特徴量 (Phase 3.2 準備)
M2_M5_FEATURES = [
    "posterior_trending_up",    # M2: BayesianRegimeFilter posterior
    "posterior_trending_down",  # M2
    "posterior_ranging",        # M2
    "posterior_volatile",       # M2
    "vol_cluster",             # M3: σ-Clustering (0=LOW,1=MID,2=HIGH,3=EXTREME)
    "fill_prob",               # M4: GLFT fill probability proxy
    "vpin_vol_sync",           # M5: VPIN (= order_flow_toxicity のエイリアス)
]

# 379# 035#-306# pre-366# 市場理論 proxy 特徴量 (SAC 接続用)
PRE366_FEATURES = [
    "parkinson_sigma",        # 305# Parkinson (1980) H/L σ (intra-bar volatility)
    "ema_velocity_bps",       # 227#/200# EMA smoothed velocity (bps)
    "kyle_lambda_proxy",      # 266# Kyle (1985) 価格インパクト係数
    "amihud_illiq_proxy",     # 266# Amihud (2002) 非流動性比率
    "vpin_toxicity",          # 107# VPIN order flow toxicity
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

    # ---- 377# M2-M5 市場理論 proxy 特徴量 ----
    out = _add_m2_m5_proxy(out, close, high, low, volume, window)

    # ---- 379# 035#-306# pre-366# 市場理論 proxy 特徴量 ----
    out = _add_pre366_proxy(out, close, high, low, volume, window)

    return out


def _add_m2_m5_proxy(
    out: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.DataFrame:
    """377# M2-M5 市場理論 proxy 特徴量を OHLCV から計算.

    Phase 3.2 準備: build_features.py の M2-M5 ゼロ行を解消する。
    オンラインモジュール (bayesian_regime_filter, sigma_clustering,
    fill_probability_model) と同等のロジックを batch 適用する。

    Args:
        out: 既存の特徴量 DataFrame (close + 10 microstructure)
        close, high, low, volume: OHLCV Series
        window: rolling window

    Returns:
        out に M2-M5 の 7 列を追加した DataFrame
    """
    eps = 1e-10
    n = len(out)

    # ---- M2: Bayesian Regime posterior (Hamilton filter proxy) ----
    # シーケンシャル処理が必要 (path-dependent)
    from scripts.v460.lib.bayesian_regime_filter import BayesianRegimeFilter

    brf = BayesianRegimeFilter()
    log_ret = np.log(close / close.shift(1)).fillna(0.0).values

    posteriors = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        result = brf.update(float(log_ret[i]))
        posteriors[i] = result.posterior

    out["posterior_trending_up"] = posteriors[:, 0]
    out["posterior_trending_down"] = posteriors[:, 1]
    out["posterior_ranging"] = posteriors[:, 2]
    out["posterior_volatile"] = posteriors[:, 3]
    logger.info(f"  M2 BayesianRegime: {n} updates, final MAP={np.argmax(posteriors[-1])}")

    # ---- M3: σ-Clustering (vol_ratio → cluster int) ----
    from scripts.v460.lib.sigma_clustering import (
        VolatilityCluster,
        VolatilityRegimeClassifier,
    )

    classifier = VolatilityRegimeClassifier()
    _cluster_to_int = {
        VolatilityCluster.LOW: 0,
        VolatilityCluster.MID: 1,
        VolatilityCluster.HIGH: 2,
        VolatilityCluster.EXTREME: 3,
    }

    # vol_ratio proxy: rolling std / rolling mean of |return|
    abs_ret = np.abs(log_ret)
    abs_ret_series = pd.Series(abs_ret, index=out.index)
    rolling_std = abs_ret_series.rolling(window, min_periods=1).std().fillna(eps)
    baseline_std = abs_ret_series.expanding(min_periods=max(window, 1)).std().fillna(eps)
    vol_ratio = (rolling_std / (baseline_std + eps)).clip(lower=0.01, upper=10.0)

    vol_cluster = np.ones(n, dtype=np.int32)  # default MID=1
    for i in range(n):
        cluster = classifier.classify(float(vol_ratio.iloc[i]))
        vol_cluster[i] = _cluster_to_int[cluster]

    out["vol_cluster"] = vol_cluster
    logger.info(f"  M3 σ-Clustering: distribution={np.bincount(vol_cluster, minlength=4).tolist()}")

    # ---- M4: GLFT fill probability proxy ----
    # offset_ratio proxy: bid_ask_spread / 2 (half-spread as offset)
    from scripts.v460.lib.fill_probability_model import FillProbabilityModel

    fpm = FillProbabilityModel()
    spread = out["bid_ask_spread"].values if "bid_ask_spread" in out.columns else np.full(n, 0.005)
    offset_proxy = np.clip(spread / 2.0, 0.0, 1.0)
    fill_prob = np.array([fpm.predict_fill_prob(float(o)) for o in offset_proxy], dtype=np.float64)
    out["fill_prob"] = fill_prob
    logger.info(f"  M4 GLFT fill_prob: mean={fill_prob.mean():.4f}, std={fill_prob.std():.4f}")

    # ---- M5: VPIN volume sync ----
    # order_flow_toxicity は既に VPIN proxy として計算済み → エイリアス
    if "order_flow_toxicity" in out.columns:
        out["vpin_vol_sync"] = out["order_flow_toxicity"]
    else:
        out["vpin_vol_sync"] = 0.5  # neutral fallback
    logger.info("  M5 VPIN: aliased from order_flow_toxicity")

    return out


def _add_pre366_proxy(
    out: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.DataFrame:
    """379# 035#-306# pre-366# 市場理論 proxy 特徴量を OHLCV から計算.

    10 個の pre-366# 市場理論システムのうち、OHLCV から合理的に
    proxy 可能な 5 特徴量を batch 計算する。

    FeatureRegistry (ztb/features/market_theory.py) と同等のロジック。
    build_features.py Parquet 向けのバッチ生成版。

    Args:
        out: 既存の特徴量 DataFrame
        close, high, low, volume: OHLCV Series
        window: rolling window

    Returns:
        out に PRE366_FEATURES の 5 列を追加した DataFrame
    """
    import math as _math

    eps = 1e-10
    n = len(out)

    # ---- 305# Parkinson σ: intra-bar H/L volatility ----
    parkinson_denom = 2.0 * _math.sqrt(_math.log(2.0))
    hl_valid = (high > 0) & (low > 0) & (high > low)
    log_hl = pd.Series(0.0, index=out.index, dtype=np.float64)
    log_hl[hl_valid] = np.log(high[hl_valid] / low[hl_valid])
    parkinson_raw = log_hl / parkinson_denom
    out["parkinson_sigma"] = parkinson_raw.rolling(window, min_periods=1).mean()
    logger.info(
        f"  379# Parkinson σ: mean={out['parkinson_sigma'].mean():.6f}, "
        f"std={out['parkinson_sigma'].std():.6f}"
    )

    # ---- 227#/200# EMA smoothed velocity (bps) ----
    prev_close = close.shift(1).fillna(close.iloc[0] if n > 0 else 0)
    valid_prev = prev_close > eps
    velocity_bps = pd.Series(0.0, index=out.index, dtype=np.float64)
    velocity_bps[valid_prev] = (
        (close[valid_prev] - prev_close[valid_prev])
        / prev_close[valid_prev]
        * 10000.0
    )
    out["ema_velocity_bps"] = velocity_bps.ewm(span=5, min_periods=1, adjust=False).mean()
    logger.info(
        f"  379# EMA velocity: mean={out['ema_velocity_bps'].mean():.4f} bps, "
        f"std={out['ema_velocity_bps'].std():.4f}"
    )

    # ---- 266# Kyle λ proxy: range / (2·volume) → z-score ----
    bar_range = high - low
    safe_vol = volume.clip(lower=eps)
    raw_kyle = bar_range / (2.0 * safe_vol)
    kyle_mean = raw_kyle.rolling(window, min_periods=1).mean()
    kyle_std = raw_kyle.rolling(window, min_periods=1).std().fillna(eps)
    out["kyle_lambda_proxy"] = (raw_kyle - kyle_mean) / (kyle_std + eps)
    logger.info(
        f"  379# Kyle λ: raw_mean={raw_kyle.mean():.6e}, "
        f"z_mean={out['kyle_lambda_proxy'].mean():.4f}"
    )

    # ---- 266# Amihud ILLIQ: |return|/volume → z-score ----
    abs_return = (close / prev_close - 1).abs().fillna(0)
    raw_amihud = abs_return / safe_vol
    amihud_mean = raw_amihud.rolling(window, min_periods=1).mean()
    amihud_std = raw_amihud.rolling(window, min_periods=1).std().fillna(eps)
    out["amihud_illiq_proxy"] = (raw_amihud - amihud_mean) / (amihud_std + eps)
    logger.info(
        f"  379# Amihud ILLIQ: raw_mean={raw_amihud.mean():.6e}, "
        f"z_mean={out['amihud_illiq_proxy'].mean():.4f}"
    )

    # ---- 107# VPIN toxicity (order_flow_toxicity の明示的別名) ----
    if "order_flow_toxicity" in out.columns:
        out["vpin_toxicity"] = out["order_flow_toxicity"]
    else:
        # フォールバック: CLV-based VPIN を再計算
        hl_range = high - low
        safe_range = hl_range.where(hl_range > eps, 1.0)
        clv = ((close - low) / safe_range) * 2 - 1
        abs_signed = (clv * volume).abs()
        out["vpin_toxicity"] = (
            abs_signed.rolling(window, min_periods=1).sum()
            / (volume.rolling(window, min_periods=1).sum() + eps)
        )
    logger.info(
        f"  379# VPIN toxicity: mean={out['vpin_toxicity'].mean():.4f}"
    )

    # NaN fill (no bfill — 003# #8)
    for col in PRE366_FEATURES:
        if col in out.columns:
            out[col] = out[col].ffill().fillna(0)

    return out


def compute_sha256(path: Path) -> str:
    """File SHA-256."""
    return _compute_shared_file_hash(path)


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
    raw = resolve_raw_dir(raw_dir)
    out = Path(output_path)
    if not out.is_absolute():
        out = _PROJECT_ROOT / out

    # Discover dates and reuse resolved daily inputs to avoid repeated exists/stat checks.
    daily_inputs = discover_raw_daily_inputs(raw)
    all_dates = sorted(daily_inputs)
    if not all_dates:
        raise FileNotFoundError(f"No raw data found in {raw}")

    target_dates = resolve_available_raw_dates(daily_inputs, dates)
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

    # Validate: all 10 + 7 + 5 features present
    all_expected = V460_FEATURES + M2_M5_FEATURES + PRE366_FEATURES
    for feat in all_expected:
        assert feat in result.columns, f"Missing feature: {feat}"

    # NaN check (V460 base features のみ — M2-M5/PRE366 は path-dependent で warmup NaN が許容)
    nan_count = int(result[V460_FEATURES].isna().sum().sum())
    total_cells = len(result) * len(V460_FEATURES)
    nan_ratio = nan_count / max(total_cells, 1)
    logger.info(f"NaN count: {nan_count}/{total_cells} ({nan_ratio:.6f})")
    assert nan_ratio <= 0.01, f"NaN ratio {nan_ratio:.4f} exceeds 1% threshold"

    logger.info(
        f"Total features: {len(all_expected)} "
        f"({len(V460_FEATURES)} base + {len(M2_M5_FEATURES)} M2-M5 + "
        f"{len(PRE366_FEATURES)} pre-366#)"
    )

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
        "features": all_expected,
        "n_features": len(all_expected),
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
