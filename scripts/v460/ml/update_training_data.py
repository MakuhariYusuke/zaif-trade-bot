"""552# SAC 訓練データ自動更新モジュール.

yfinance から BTC-JPY 1分足を取得し、FeatureRegistry で特徴量を計算して
full_registry_features.parquet に追記する。

retrain_scheduler から呼べるライブラリ関数 + CLI の二面対応。

Usage:
  # CLI (手動実行)
  python scripts/v460/ml/update_training_data.py

  # ライブラリ (retrain_scheduler から呼出)
  from scripts.v460.ml.update_training_data import ensure_data_fresh
  updated = ensure_data_fresh(parquet_path, max_stale_hours=48)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PARQUET_PATH = _PROJECT_ROOT / "data" / "btc_jpy_1m_full_registry_features.parquet"
_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

# SAC が使用する 17 特徴量 (g2_sac_train.yaml features.selected)
_SAC_FEATURES = [
    "price_velocity",
    "micro_trend",
    "price_acceleration",
    "volume_surge",
    "momentum_divergence",
    "tick_volume_ratio",
    "order_flow_imbalance",
    "micro_volatility",
    "spread_pressure",
    "momentum_burst",
    "liquidity_surge",
    "realized_volatility",
    "parkinson_sigma",
    "vpin_proxy",
    "kyle_lambda_proxy",
    "amihud_illiq",
    "ema_velocity_bps",
]

# FeatureRegistry が存在しない列 (parquet にあるが計算不要)
# → 既存列をそのまま保持し、新規行には NaN or 0 を入れる
_WARMUP_ROWS = 500  # RSI 等のウォームアップに必要な行数


def _get_parquet_last_timestamp(parquet_path: Path) -> datetime | None:
    """parquet の最終タイムスタンプを取得."""
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path, columns=["timestamp"])
    if df.empty:
        return None
    last_ts = df["timestamp"].iloc[-1]
    if hasattr(last_ts, "to_pydatetime"):
        dt = last_ts.to_pydatetime()
    else:
        dt = pd.Timestamp(last_ts).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _hours_since_last_update(parquet_path: Path) -> float:
    """parquet 最終タイムスタンプからの経過時間 (hours)."""
    last_ts = _get_parquet_last_timestamp(parquet_path)
    if last_ts is None:
        return float("inf")
    now = datetime.now(timezone.utc)
    return (now - last_ts).total_seconds() / 3600.0


def _download_ohlcv(period: str = "7d") -> pd.DataFrame:
    """yfinance から BTC-JPY 1分足をダウンロード."""
    import yfinance as yf

    logger.info(f"[552#] yfinance BTC-JPY 1m ({period}) downloading...")
    ticker = yf.Ticker("BTC-JPY")
    hist = ticker.history(period=period, interval="1m")

    if hist.empty:
        raise RuntimeError("yfinance returned empty data for BTC-JPY 1m")

    ts = hist.index
    if ts.tz is not None:
        ts = ts.tz_convert("UTC")

    df = pd.DataFrame({
        "timestamp": ts,
        "open": hist["Open"].values,
        "high": hist["High"].values,
        "low": hist["Low"].values,
        "close": hist["Close"].values,
        "volume": hist["Volume"].values,
    })
    df = df.reset_index(drop=True)
    logger.info(
        f"[552#] Downloaded {len(df)} rows: "
        f"{df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}"
    )
    return df


def _compute_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """FeatureRegistry を使って特徴量を計算."""
    # Feature modules を import して register
    import ztb.features.scalping  # noqa: F401
    import ztb.features.market_theory  # noqa: F401
    import ztb.features.time.time_features  # noqa: F401

    try:
        import ztb.features.generators.technical.volume.chaikin_ad  # noqa: F401
    except ImportError:
        pass
    try:
        import ztb.features.volatility.normalized_atr  # noqa: F401
    except ImportError:
        pass

    from ztb.features.core.registry import FeatureRegistry

    df_indexed = df.copy()
    df_indexed.index = pd.DatetimeIndex(df_indexed["timestamp"])

    features_dict: dict[str, pd.Series] = {}
    for feat_name in feature_names:
        try:
            func = FeatureRegistry.get(feat_name)
            series = func(df_indexed)
            series.index = df.index
            features_dict[feat_name] = series
        except Exception as e:
            logger.warning(f"[552#] Feature {feat_name} computation failed: {e}")
            features_dict[feat_name] = pd.Series(
                np.zeros(len(df)), index=df.index, dtype=np.float32,
            )

    features_df = pd.DataFrame(features_dict)
    result = pd.concat([df[_OHLCV_COLS], features_df], axis=1)

    # float32 統一 (timestamp 除く)
    for col in result.columns:
        if col != "timestamp" and result[col].dtype == np.float64:
            result[col] = result[col].astype(np.float32)

    logger.info(f"[552#] Computed {len(feature_names)} features: shape={result.shape}")
    return result


def _merge_into_parquet(
    parquet_path: Path,
    new_data: pd.DataFrame,
) -> int:
    """新データを既存 parquet にマージ (重複排除 + ソート)."""
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        # timestamp を tz-naive UTC に統一
        if hasattr(existing["timestamp"].dtype, "tz") and existing["timestamp"].dt.tz is not None:
            existing["timestamp"] = existing["timestamp"].dt.tz_localize(None)
        n_before = len(existing)
    else:
        existing = pd.DataFrame()
        n_before = 0

    # new_data の timestamp も tz-naive に統一
    if hasattr(new_data["timestamp"].dtype, "tz") and new_data["timestamp"].dt.tz is not None:
        new_data = new_data.copy()
        new_data["timestamp"] = new_data["timestamp"].dt.tz_localize(None)

    # 新データに不足列があれば NaN で埋める
    if not existing.empty:
        for col in existing.columns:
            if col not in new_data.columns:
                new_data[col] = np.nan

    # 結合 + 重複排除
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = (
        combined
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    n_added = len(combined) - n_before
    logger.info(
        f"[552#] Merged: {n_before} → {len(combined)} rows (+{n_added}), "
        f"range: {combined['timestamp'].iloc[0]} ~ {combined['timestamp'].iloc[-1]}"
    )

    # 保存
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, index=False, compression="snappy")
    size_mb = parquet_path.stat().st_size / 1e6
    logger.info(f"[552#] Saved: {size_mb:.1f} MB")
    return n_added


def update_training_parquet(
    parquet_path: Path | None = None,
    period: str = "7d",
) -> int:
    """訓練用 parquet を yfinance データで更新.

    Returns:
        追加行数。
    """
    path = parquet_path or _PARQUET_PATH

    # 1. yfinance からダウンロード
    new_ohlcv = _download_ohlcv(period)

    # 2. 既存末尾をウォームアップに利用 (RSI等のインジケータ初期化)
    all_features = _get_all_parquet_features(path)

    if path.exists():
        existing = pd.read_parquet(path, columns=_OHLCV_COLS)
        if hasattr(existing["timestamp"].dtype, "tz") and existing["timestamp"].dt.tz is not None:
            existing["timestamp"] = existing["timestamp"].dt.tz_localize(None)
        # 新データの timestamp も tz-naive 化
        new_ohlcv_naive = new_ohlcv.copy()
        if hasattr(new_ohlcv_naive["timestamp"].dtype, "tz") and new_ohlcv_naive["timestamp"].dt.tz is not None:
            new_ohlcv_naive["timestamp"] = new_ohlcv_naive["timestamp"].dt.tz_localize(None)
        warmup = existing.tail(_WARMUP_ROWS).copy()
        warmup_with_new = pd.concat([warmup, new_ohlcv_naive], ignore_index=True)
        full = _compute_features(warmup_with_new, all_features)
        new_features = full.iloc[len(warmup):].reset_index(drop=True)
    else:
        new_features = _compute_features(new_ohlcv, all_features)

    # 3. マージ
    return _merge_into_parquet(path, new_features)


def _get_all_parquet_features(parquet_path: Path) -> list[str]:
    """parquet に存在する FeatureRegistry 計算可能な特徴量名を取得."""
    if not parquet_path.exists():
        return list(_SAC_FEATURES)

    existing_cols = set(pd.read_parquet(parquet_path, columns=[]).columns)
    non_feature_cols = set(_OHLCV_COLS)

    # FeatureRegistry から計算可能な名前のみ
    try:
        import ztb.features.scalping  # noqa: F401
        import ztb.features.market_theory  # noqa: F401
        import ztb.features.time.time_features  # noqa: F401
        try:
            import ztb.features.generators.technical.volume.chaikin_ad  # noqa: F401
        except ImportError:
            pass
        try:
            import ztb.features.volatility.normalized_atr  # noqa: F401
        except ImportError:
            pass
        from ztb.features.core.registry import FeatureRegistry
        registry_names = set(FeatureRegistry.list())
    except Exception:
        registry_names = set(_SAC_FEATURES)

    # 既存 parquet にある列 AND registry に登録されている特徴量
    computable = (existing_cols - non_feature_cols) & registry_names

    # SAC必須特徴量は必ず含める
    computable |= set(_SAC_FEATURES)

    return sorted(computable)


def ensure_data_fresh(
    parquet_path: Path | str | None = None,
    max_stale_hours: float = 48.0,
) -> bool:
    """データ鮮度チェック + 自動更新.

    retrain_scheduler から呼ばれることを想定。

    Returns:
        True if data was updated, False if already fresh.
    """
    path = Path(parquet_path) if parquet_path else _PARQUET_PATH
    stale_hours = _hours_since_last_update(path)

    if stale_hours <= max_stale_hours:
        logger.info(
            f"[552#] Data fresh: {stale_hours:.1f}h old "
            f"(<= {max_stale_hours}h threshold)"
        )
        return False

    logger.warning(
        f"[552#] Data STALE: {stale_hours:.1f}h old "
        f"(> {max_stale_hours}h threshold) — updating..."
    )
    try:
        n_added = update_training_parquet(path)
        logger.info(f"[552#] Data updated: {n_added} new rows added")
        return True
    except Exception as e:
        logger.error(f"[552#] Data update FAILED: {e}", exc_info=True)
        return False


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main() -> None:
    """CLI メイン."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    import argparse

    parser = argparse.ArgumentParser(
        description="552# SAC training data updater",
    )
    parser.add_argument(
        "--parquet", type=str, default=str(_PARQUET_PATH),
        help="Target parquet path",
    )
    parser.add_argument(
        "--period", type=str, default="7d",
        help="yfinance download period (default: 7d)",
    )
    args = parser.parse_args()

    path = Path(args.parquet)
    last_ts = _get_parquet_last_timestamp(path)
    stale = _hours_since_last_update(path)
    print(f"Current last timestamp: {last_ts}")
    print(f"Stale: {stale:.1f} hours")

    n_added = update_training_parquet(path, period=args.period)
    print(f"\nDone: {n_added} rows added")


if __name__ == "__main__":
    main()
