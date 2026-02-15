"""058# Feature Enricher: raw orderbook/trades → fill record 特徴量付与.

fill record の timestamp に対応する板・約定データから
マイクロストラクチャ特徴量を算出し、AS 分類器を強化する。

v459 K2 の非 RL 上限検証 + v460 マイクロストラクチャ特徴量を合わせた
「あるものだけで学習」パイプライン。
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_RAW_DIR = Path("data/v460/raw")

# 特徴量ウィンドウ (秒)
_TRADE_WINDOW_SEC = 60  # 直近 60 秒の約定統計
_OB_MATCH_TOLERANCE_SEC = 5  # 板スナップショットの許容誤差


def _read_jsonl_gz(path: Path) -> list[dict]:
    """gzip JSONL を読み込み."""
    records: list[dict] = []
    if not path.exists():
        return records
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_raw_orderbook(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """全日の板スナップショットを読み込み.

    Returns:
        columns: ts, best_bid, best_ask, mid_price, spread_bps,
                 bid_vol_5, ask_vol_5, depth_imbalance
    """
    d = raw_dir or _DEFAULT_RAW_DIR
    ob_dir = d / "orderbook"
    if not ob_dir.exists():
        return pd.DataFrame()

    all_records: list[dict] = []
    for f in sorted(ob_dir.glob("*.jsonl.gz")):
        all_records.extend(_read_jsonl_gz(f))

    if not all_records:
        return pd.DataFrame()

    rows: list[dict] = []
    for r in all_records:
        bids = r.get("bids", [])
        asks = r.get("asks", [])
        if not bids or not asks:
            continue
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        if mid <= 0:
            continue
        bid_vol_5 = sum(s for _, s in bids[:5])
        ask_vol_5 = sum(s for _, s in asks[:5])
        total_depth = bid_vol_5 + ask_vol_5
        rows.append({
            "ts": r["ts"],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": (best_ask - best_bid) / mid * 10000,
            "bid_vol_5": bid_vol_5,
            "ask_vol_5": ask_vol_5,
            "depth_imbalance": (
                (bid_vol_5 - ask_vol_5) / total_depth
                if total_depth > 0 else 0.0
            ),
        })

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} orderbook snapshots")
    return df


def load_raw_trades(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """全日の約定データを読み込み.

    Returns:
        columns: ts, price, amount, side
    """
    d = raw_dir or _DEFAULT_RAW_DIR
    tr_dir = d / "trades"
    if not tr_dir.exists():
        return pd.DataFrame()

    all_records: list[dict] = []
    for f in sorted(tr_dir.glob("*.jsonl.gz")):
        all_records.extend(_read_jsonl_gz(f))

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.sort_values("ts").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} trades")
    return df


def _compute_trade_features(
    trades_df: pd.DataFrame,
    ts: float,
    window_sec: int = _TRADE_WINDOW_SEC,
    *,
    _sorted_ts: np.ndarray | None = None,
) -> dict[str, float]:
    """指定時点の直前 window_sec 秒間の約定統計を算出.

    059# P1-7: searchsorted で O(log N) にフィルタ (呼び出し側で _sorted_ts を渡す).

    Returns:
        trade_count_60s, buy_ratio, trade_flow_imbalance_60s,
        avg_trade_size, price_velocity_60s, vpin_60s
    """
    _default = {
        "trade_count_60s": 0.0,
        "buy_ratio": 0.5,
        "trade_flow_imbalance_60s": 0.0,
        "avg_trade_size": 0.0,
        "price_velocity_60s": 0.0,
        "vpin_60s": 0.5,
    }
    if trades_df.empty:
        return _default

    t0 = ts - window_sec

    # 059# P1-7: searchsorted で O(log N) にスライス
    if _sorted_ts is not None:
        i_start = int(np.searchsorted(_sorted_ts, t0, side="left"))
        i_end = int(np.searchsorted(_sorted_ts, ts, side="left"))
        if i_start >= i_end:
            return _default
        window = trades_df.iloc[i_start:i_end]
    else:
        mask = (trades_df["ts"] >= t0) & (trades_df["ts"] < ts)
        window = trades_df.loc[mask]

    if window.empty:
        return _default

    n_trades = len(window)
    buy_mask = window["side"].str.lower() == "buy"
    buy_vol = float(window.loc[buy_mask, "amount"].sum())
    sell_vol = float(window.loc[~buy_mask, "amount"].sum())
    total_vol = buy_vol + sell_vol

    buy_ratio = buy_vol / total_vol if total_vol > 0 else 0.5
    tfi = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
    avg_size = total_vol / n_trades if n_trades > 0 else 0.0

    # Price velocity: (last - first) / first * 10000 bps
    first_price = float(window["price"].iloc[0])
    last_price = float(window["price"].iloc[-1])
    price_vel = (
        (last_price - first_price) / first_price * 10000
        if first_price > 0 else 0.0
    )

    # VPIN: |buy_vol - sell_vol| / total_vol
    vpin = abs(buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.5

    return {
        "trade_count_60s": float(n_trades),
        "buy_ratio": buy_ratio,
        "trade_flow_imbalance_60s": tfi,
        "avg_trade_size": avg_size,
        "price_velocity_60s": price_vel,
        "vpin_60s": vpin,
    }


def _find_nearest_ob(
    ob_df: pd.DataFrame,
    ts: float,
    tolerance_sec: int = _OB_MATCH_TOLERANCE_SEC,
) -> dict[str, float]:
    """指定時刻に最も近い板スナップショットから特徴量取得.

    Returns:
        spread_bps_ob, depth_imbalance_ob, bid_vol_5_ob, ask_vol_5_ob
    """
    default = {
        "spread_bps_ob": np.nan,
        "depth_imbalance_ob": np.nan,
        "bid_vol_5_ob": np.nan,
        "ask_vol_5_ob": np.nan,
    }
    if ob_df.empty:
        return default

    # Binary search for nearest
    idx = np.searchsorted(ob_df["ts"].values, ts)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(ob_df):
        candidates.append(idx)

    best_idx = -1
    best_diff = float("inf")
    for c in candidates:
        diff = abs(ob_df["ts"].iloc[c] - ts)
        if diff < best_diff:
            best_diff = diff
            best_idx = c

    if best_idx < 0 or best_diff > tolerance_sec:
        return default

    row = ob_df.iloc[best_idx]
    return {
        "spread_bps_ob": float(row["spread_bps"]),
        "depth_imbalance_ob": float(row["depth_imbalance"]),
        "bid_vol_5_ob": float(row["bid_vol_5"]),
        "ask_vol_5_ob": float(row["ask_vol_5"]),
    }


def enrich_fill_records(
    fill_df: pd.DataFrame,
    raw_dir: Optional[Path] = None,
    ob_tolerance_sec: int = _OB_MATCH_TOLERANCE_SEC,
    trade_window_sec: int = _TRADE_WINDOW_SEC,
) -> pd.DataFrame:
    """fill records にマイクロストラクチャ特徴量を付与.

    Args:
        fill_df: load_fill_records() の出力.
        raw_dir: raw data のディレクトリ.
        ob_tolerance_sec: 板スナップショットの許容誤差 (秒).
        trade_window_sec: 約定統計のウィンドウ (秒).

    Returns:
        enriched DataFrame. 新規カラム:
            spread_bps_ob, depth_imbalance_ob, bid_vol_5_ob, ask_vol_5_ob,
            trade_count_60s, buy_ratio, trade_flow_imbalance_60s,
            avg_trade_size, price_velocity_60s, vpin_60s
    """
    ob_df = load_raw_orderbook(raw_dir)
    trades_df = load_raw_trades(raw_dir)

    # 059# P1-7: 事前ソート + searchsorted で O(N_fill × log N_trades)
    if not trades_df.empty and "ts" in trades_df.columns:
        trades_df = trades_df.sort_values("ts").reset_index(drop=True)
        sorted_ts = trades_df["ts"].values
    else:
        sorted_ts = None

    enriched_rows: list[dict] = []
    for _, row in fill_df.iterrows():
        ts = float(row["timestamp"])

        # Orderbook features (nearest snapshot)
        ob_features = _find_nearest_ob(ob_df, ts, ob_tolerance_sec)

        # Trade features (rolling window)
        trade_features = _compute_trade_features(
            trades_df, ts, trade_window_sec, _sorted_ts=sorted_ts
        )

        enriched_rows.append({**ob_features, **trade_features})

    enriched = pd.DataFrame(enriched_rows, index=fill_df.index)
    n_ob_matched = enriched["spread_bps_ob"].notna().sum()

    logger.info(
        f"Enriched {len(fill_df)} records: "
        f"OB matched={n_ob_matched}/{len(fill_df)}, "
        f"trades available={not trades_df.empty}"
    )

    return pd.concat([fill_df, enriched], axis=1)


# ----- Enriched AS features builder -----

#: 新規マイクロストラクチャ特徴量のカラム名
MICRO_FEATURE_COLS = [
    "spread_bps_ob",
    "depth_imbalance_ob",
    "trade_count_60s",
    "buy_ratio",
    "trade_flow_imbalance_60s",
    "avg_trade_size",
    "price_velocity_60s",
    "vpin_60s",
]

#: side とのインタラクション特徴量 (AS 予測の本丸)
INTERACTION_FEATURE_COLS = [
    "side_aligned_imbalance",
    "side_aligned_tfi",
    "side_aligned_velocity",
]


def build_enriched_as_features(
    enriched_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """enriched fill records → AS 分類器用特徴量.

    既存 data_loader.build_as_features の出力 + マイクロストラクチャ特徴量
    + side-aligned インタラクション特徴量を返す.

    Args:
        enriched_df: enrich_fill_records() の出力.

    Returns:
        (X, y) タプル.
    """
    from scripts.v460.ml.data_loader import build_as_features

    # 1. 既存特徴量
    X_base, y = build_as_features(enriched_df)

    # 2. マイクロストラクチャ特徴量を追加
    # NOTE: NaN は保持。補完は CV fold 内で SimpleImputer が行う (059# P0-1)
    micro_features: dict[str, pd.Series] = {}
    for col in MICRO_FEATURE_COLS:
        if col in enriched_df.columns:
            micro_features[col] = enriched_df.loc[X_base.index, col].astype(float)

    # 3. side-aligned インタラクション特徴量
    #    buy 側: side_sign = +1, sell 側: side_sign = -1
    #    → "自分の注文に有利な方向" を正の値で表現
    side_sign = enriched_df.loc[X_base.index, "side"].map(
        {"buy": 1.0, "sell": -1.0}
    ).astype(float)

    # depth_imbalance × side_sign: bid 厚い → buy に有利 (+)
    if "depth_imbalance_ob" in enriched_df.columns:
        di = enriched_df.loc[X_base.index, "depth_imbalance_ob"].astype(float)
        aligned = di * side_sign
        micro_features["side_aligned_imbalance"] = aligned.fillna(0.0)

    # trade_flow_imbalance × side_sign: buy 優勢 → buy に有利 (+)
    if "trade_flow_imbalance_60s" in enriched_df.columns:
        tfi = enriched_df.loc[X_base.index, "trade_flow_imbalance_60s"].astype(float)
        aligned_tfi = tfi * side_sign
        micro_features["side_aligned_tfi"] = aligned_tfi.fillna(0.0)

    # price_velocity × side_sign: 上昇 → buy に有利 (+)
    if "price_velocity_60s" in enriched_df.columns:
        vel = enriched_df.loc[X_base.index, "price_velocity_60s"].astype(float)
        aligned_vel = vel * side_sign
        micro_features["side_aligned_velocity"] = aligned_vel.fillna(0.0)

    if micro_features:
        X_micro = pd.DataFrame(micro_features, index=X_base.index)
        X = pd.concat([X_base, X_micro], axis=1)
    else:
        X = X_base

    logger.info(
        f"Enriched AS features: {X.shape[1]} features "
        f"({X_base.shape[1]} base + {len(micro_features)} micro), "
        f"{len(X)} samples"
    )
    return X, y


def build_pnl_features(
    enriched_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """PnL 回帰用特徴量: post_fill_30s_pnl を直接予測.

    AS ラベルより sample 数が多く、目的変数が連続値なので
    リッジ回帰等で扱いやすい。

    Returns:
        (X, y) タプル. y は post_fill_30s_pnl (bps).
    """
    from scripts.v460.ml.data_loader import build_as_features as _build_base
    from datetime import datetime

    # filled かつ PnL 非 NaN
    mask = (
        enriched_df["filled"].astype(bool)
        & enriched_df["post_fill_30s_pnl"].notna()
    )
    data = enriched_df.loc[mask].copy()

    if len(data) < 20:
        raise ValueError(f"Insufficient PnL samples: {len(data)}")

    features: dict[str, pd.Series] = {}

    # --- 基本特徴量 (data_loader と同系列) ---
    features["side_buy"] = (data["side"] == "buy").astype(int)

    ts = data["timestamp"].astype(float)
    # 059# NEW-03: 小数時刻で統一 (skip_gate 推論側と同一粒度)
    hours = ts.apply(
        lambda t: (lambda d: d.hour + d.minute / 60.0)(datetime.fromtimestamp(t))
    )
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    if "spread_at_order" in data.columns:
        # NOTE: NaN は保持。補完は CV fold 内 (059# P0-1)
        features["spread_jpy"] = data["spread_at_order"].astype(float)

    if "spread_offset_ratio" in data.columns:
        features["offset_ratio"] = data["spread_offset_ratio"].astype(float)

    if "regime" in data.columns:
        regime = data["regime"].fillna("unknown")
        for val in ["trending", "ranging", "high_vol"]:
            features[f"regime_{val}"] = (regime == val).astype(int)

    # --- マイクロストラクチャ特徴量 ---
    # NOTE: NaN は保持。補完は CV fold 内 (059# P0-1)
    for col in MICRO_FEATURE_COLS:
        if col in data.columns:
            features[col] = data[col].astype(float)

    # --- インタラクション特徴量 ---
    side_sign = data["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)

    if "depth_imbalance_ob" in data.columns:
        di = data["depth_imbalance_ob"].astype(float)
        features["side_aligned_imbalance"] = (di * side_sign).fillna(0.0)
    if "trade_flow_imbalance_60s" in data.columns:
        tfi = data["trade_flow_imbalance_60s"].astype(float)
        features["side_aligned_tfi"] = (tfi * side_sign).fillna(0.0)
    if "price_velocity_60s" in data.columns:
        vel = data["price_velocity_60s"].astype(float)
        features["side_aligned_velocity"] = (vel * side_sign).fillna(0.0)

    X = pd.DataFrame(features, index=data.index)
    y = data["post_fill_30s_pnl"].astype(float)

    logger.info(
        f"PnL features: {X.shape[1]} features, {len(X)} samples, "
        f"mean PnL={y.mean():.2f} bps, std={y.std():.2f}"
    )
    return X, y
