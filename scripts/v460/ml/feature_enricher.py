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


def load_raw_orderbook(
    raw_dir: Optional[Path] = None,
    date_filter: Optional[set[str]] = None,
) -> pd.DataFrame:
    """板スナップショットを読み込み.

    Args:
        raw_dir: raw data ディレクトリ.
        date_filter: 130# 日付限定ロード. {"20260220", "20260221"} 形式.

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
        # 130# 日付限定ロード
        if date_filter is not None:
            stem = f.stem.replace(".jsonl", "")
            if stem not in date_filter:
                continue
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


def load_raw_trades(
    raw_dir: Optional[Path] = None,
    date_filter: Optional[set[str]] = None,
) -> pd.DataFrame:
    """約定データを読み込み.

    Args:
        raw_dir: raw data ディレクトリ.
        date_filter: 130# 日付限定ロード. {"20260220", "20260221"} 形式.
            None の場合は全日読み込み (後方互換).

    Returns:
        columns: ts, price, amount, side
    """
    d = raw_dir or _DEFAULT_RAW_DIR
    tr_dir = d / "trades"
    if not tr_dir.exists():
        return pd.DataFrame()

    all_records: list[dict] = []
    for f in sorted(tr_dir.glob("*.jsonl.gz")):
        # 130# 日付限定ロード: ファイル名 YYYYMMDD.jsonl.gz から日付抽出
        if date_filter is not None:
            stem = f.stem.replace(".jsonl", "")  # "20260220" etc.
            if stem not in date_filter:
                continue
        all_records.extend(_read_jsonl_gz(f))

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.sort_values("ts").reset_index(drop=True)
    n_days = len(date_filter) if date_filter else "all"
    logger.info(f"Loaded {len(df)} trades (days={n_days})")
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


# ----- 060# v2: Multi-timeframe & volatility features -----

_MULTI_TF_WINDOWS = [30, 300]  # 秒 (primary 60s は既存)


def _compute_multi_timeframe_trade_features(
    trades_df: pd.DataFrame,
    ts: float,
    *,
    _sorted_ts: np.ndarray | None = None,
) -> dict[str, float]:
    """060# v2: 30s/300s 窓の約定統計 (既存 60s を補完).

    AS 予測には時間スケール間の差分が重要:
    - 短期 (30s) vs 長期 (300s) の flow 乖離 → 情報トレーダー検知
    - 加速度 (trade rate の変化) → urgency
    """
    result: dict[str, float] = {}

    for w in _MULTI_TF_WINDOWS:
        suffix = f"_{w}s"
        feats = _compute_trade_features(
            trades_df, ts, w, _sorted_ts=_sorted_ts
        )
        # 主要指標のみ追加 (全部入れると次元が爆発)
        result[f"vpin{suffix}"] = feats["vpin_60s"]  # 命名は窓に合わせる
        result[f"tfi{suffix}"] = feats["trade_flow_imbalance_60s"]
        result[f"velocity{suffix}"] = feats["price_velocity_60s"]
        result[f"trade_count{suffix}"] = feats["trade_count_60s"]

    # Cross-timeframe: 30s vs 300s の加速度シグナル
    vpin_30 = result.get("vpin_30s", 0.5)
    vpin_300 = result.get("vpin_300s", 0.5)
    tfi_30 = result.get("tfi_30s", 0.0)
    tfi_300 = result.get("tfi_300s", 0.0)
    tc_30 = result.get("trade_count_30s", 0.0)
    tc_300 = result.get("trade_count_300s", 0.0)

    # VPIN 加速: 短期 VPIN が長期より高い → 直近に informed trading
    result["vpin_acceleration"] = vpin_30 - vpin_300

    # TFI 加速: 短期 flow が長期より偏っている → 方向性圧力の急変
    result["tfi_acceleration"] = tfi_30 - tfi_300

    # Trade rate 加速: 30s rate vs 300s rate (normalized to per-second)
    rate_30 = tc_30 / 30.0 if tc_30 > 0 else 0.0
    rate_300 = tc_300 / 300.0 if tc_300 > 0 else 0.0
    result["trade_rate_acceleration"] = rate_30 - rate_300

    return result


def _compute_return_momentum(
    ob_df: pd.DataFrame,
    ts: float,
    *,
    sorted_ts: np.ndarray | None = None,
    windows: tuple[int, ...] = (30, 60, 300),
) -> dict[str, float]:
    """060# v2: OB mid price ベースのリターンモメンタム + ボラティリティ.

    Args:
        ob_df: load_raw_orderbook() の出力 (ts, mid_price 列必須).
        ts: 基準時刻.
        sorted_ts: ob_df["ts"].values (事前準備).
        windows: lookback 窓 (秒).

    Returns:
        return_30s, return_60s, return_300s: 各窓のリターン (bps)
        realized_vol_300s: 300s 窓の実現ボラティリティ (bps)
        mid_price_at_order: 注文時mid (AS計算の基準)
    """
    default: dict[str, float] = {}
    for w in windows:
        default[f"return_{w}s"] = np.nan
    default["realized_vol_300s"] = np.nan

    if ob_df.empty or "ts" not in ob_df.columns:
        return default

    ts_arr = sorted_ts if sorted_ts is not None else ob_df["ts"].values
    mid_arr = ob_df["mid_price"].values

    # 現在の mid を取得 (最近傍)
    idx_now = int(np.searchsorted(ts_arr, ts, side="right")) - 1
    if idx_now < 0 or abs(ts_arr[idx_now] - ts) > 120:
        return default

    mid_now = mid_arr[idx_now]
    if mid_now <= 0:
        return default

    result: dict[str, float] = {}

    # 各窓のリターン (bps)
    for w in windows:
        t_past = ts - w
        idx_past = int(np.searchsorted(ts_arr, t_past, side="left"))
        if idx_past < len(ts_arr) and abs(ts_arr[idx_past] - t_past) < 120:
            mid_past = mid_arr[idx_past]
            if mid_past > 0:
                result[f"return_{w}s"] = (mid_now - mid_past) / mid_past * 10000
            else:
                result[f"return_{w}s"] = np.nan
        else:
            result[f"return_{w}s"] = np.nan

    # 実現ボラティリティ (300s 窓, returns の std)
    vol_window = 300
    t_start = ts - vol_window
    i_start = int(np.searchsorted(ts_arr, t_start, side="left"))
    i_end = int(np.searchsorted(ts_arr, ts, side="right"))
    if i_end - i_start >= 5:  # 最低 5 snapshots
        mids = mid_arr[i_start:i_end]
        returns_bps = np.diff(mids) / mids[:-1] * 10000
        result["realized_vol_300s"] = float(np.std(returns_bps))
    else:
        result["realized_vol_300s"] = np.nan

    return result


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
            avg_trade_size, price_velocity_60s, vpin_60s,
            + 060# v2 features (multi-timeframe, volatility, momentum)
    """
    # 130# 日付限定ロード: fill records のタイムスタンプから必要日を算出
    date_filter: set[str] | None = None
    if "timestamp" in fill_df.columns and len(fill_df) > 0:
        from datetime import datetime, timezone
        ts_min = float(fill_df["timestamp"].min())
        ts_max = float(fill_df["timestamp"].max())
        # 前後 trade_window_sec 分のマージンを確保
        margin = max(trade_window_sec, 300)
        d_start = datetime.fromtimestamp(ts_min - margin, tz=timezone.utc)
        d_end = datetime.fromtimestamp(ts_max + margin, tz=timezone.utc)
        # 日付セットを生成
        from datetime import timedelta
        date_filter = set()
        d = d_start.date()
        while d <= d_end.date():
            date_filter.add(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
        logger.info(f"130# Date filter: {sorted(date_filter)} ({len(date_filter)} days)")

    ob_df = load_raw_orderbook(raw_dir, date_filter=date_filter)
    trades_df = load_raw_trades(raw_dir, date_filter=date_filter)

    # 059# P1-7: 事前ソート + searchsorted で O(N_fill × log N_trades)
    if not trades_df.empty and "ts" in trades_df.columns:
        trades_df = trades_df.sort_values("ts").reset_index(drop=True)
        sorted_ts = trades_df["ts"].values
    else:
        sorted_ts = None

    # 060#: OB ts も事前準備 (multi-timeframe momentum 計算用)
    ob_sorted_ts = ob_df["ts"].values if not ob_df.empty else None

    enriched_rows: list[dict] = []
    for _, row in fill_df.iterrows():
        ts = float(row["timestamp"])

        # Orderbook features (nearest snapshot)
        ob_features = _find_nearest_ob(ob_df, ts, ob_tolerance_sec)

        # Trade features (rolling window — primary 60s)
        trade_features = _compute_trade_features(
            trades_df, ts, trade_window_sec, _sorted_ts=sorted_ts
        )

        # 060# v2: Multi-timeframe trade features (30s, 300s)
        multi_tf = _compute_multi_timeframe_trade_features(
            trades_df, ts, _sorted_ts=sorted_ts
        )

        # 060# v2: Return momentum & volatility (from OB mid prices)
        momentum = _compute_return_momentum(ob_df, ts, sorted_ts=ob_sorted_ts)

        enriched_rows.append(
            {**ob_features, **trade_features, **multi_tf, **momentum}
        )

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

#: 060# v2: Multi-timeframe & momentum 特徴量
V2_FEATURE_COLS = [
    # Multi-timeframe
    "vpin_30s",
    "tfi_30s",
    "velocity_30s",
    "trade_count_30s",
    "vpin_300s",
    "tfi_300s",
    "velocity_300s",
    "trade_count_300s",
    # Cross-timeframe acceleration
    "vpin_acceleration",
    "tfi_acceleration",
    "trade_rate_acceleration",
    # Return momentum
    "return_30s",
    "return_60s",
    "return_300s",
    # Realized volatility
    "realized_vol_300s",
]

#: side とのインタラクション特徴量 (AS 予測の本丸)
INTERACTION_FEATURE_COLS = [
    "side_aligned_imbalance",
    "side_aligned_tfi",
    "side_aligned_velocity",
]


def build_preorder_as_features(
    enriched_df: pd.DataFrame,
    *,
    require_spread: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """096# 注文前に観測可能な特徴量のみで AS 分類器用データを構築.

    build_features_from_market_state() (推論側) と完全に同じ特徴量契約。
    log_queue_wait / edge_bps / return momentum 等の事後特徴量を排除し、
    学習↔推論の不整合を解消する。

    Args:
        enriched_df: enrich_fill_records() の出力.
        require_spread: True の場合 spread_at_order 必須.

    Returns:
        (X, y) タプル. X のカラム = skip_gate._BASE_FEATURE_COLS と一致.
    """
    from datetime import datetime as _dt

    # filled かつ AS ラベル有りのみ
    mask = enriched_df["filled"].astype(bool) & enriched_df["adverse_selected_raw"].notna()
    data = enriched_df.loc[mask].copy()

    if require_spread:
        data = data.dropna(subset=["spread_at_order", "spread_offset_ratio"])

    if len(data) < 10:
        raise ValueError(f"Insufficient labeled samples: {len(data)}")

    features: dict[str, pd.Series] = {}

    # --- 注文前に観測可能な特徴量のみ ---
    # F1: side (binary)
    features["side_buy"] = (data["side"] == "buy").astype(int)

    # F2: hour (cyclic) — 注文時刻
    ts = data["timestamp"].astype(float)
    hours = ts.apply(
        lambda t: (lambda d: d.hour + d.minute / 60.0)(_dt.fromtimestamp(t))
    )
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # F3: spread_jpy — 注文時のスプレッド
    if "spread_at_order" in data.columns:
        features["spread_jpy"] = data["spread_at_order"].astype(float)

    # F4: offset_ratio — 注文時の設定値
    if "spread_offset_ratio" in data.columns:
        features["offset_ratio"] = data["spread_offset_ratio"].astype(float)

    # F5: regime (one-hot)
    if "regime" in data.columns:
        regime = data["regime"].fillna("unknown")
        for val in ["trending", "ranging", "high_vol"]:
            features[f"regime_{val}"] = (regime == val).astype(int)

    # F6: trade-based micro features (注文前に算出可能)
    for col in MICRO_FEATURE_COLS:
        if col in enriched_df.columns and col not in ("spread_bps_ob", "depth_imbalance_ob"):
            features[col] = enriched_df.loc[data.index, col].astype(float)

    # F7: side-aligned interaction features (trade-based only)
    side_sign = data["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)

    if "trade_flow_imbalance_60s" in enriched_df.columns:
        tfi = enriched_df.loc[data.index, "trade_flow_imbalance_60s"].astype(float)
        features["side_aligned_tfi"] = (tfi * side_sign).fillna(0.0)

    if "price_velocity_60s" in enriched_df.columns:
        vel = enriched_df.loc[data.index, "price_velocity_60s"].astype(float)
        features["side_aligned_velocity"] = (vel * side_sign).fillna(0.0)

    # NOTE: log_queue_wait, edge_bps は事後特徴量のため意図的に除外
    # NOTE: V2_FEATURE_COLS, side_aligned_return_* も推論側未実装のため除外

    X = pd.DataFrame(features, index=data.index)
    y = data["adverse_selected_raw"].astype(int)

    logger.info(
        f"Preorder AS features: {X.shape[1]} features, {len(X)} samples, "
        f"AS rate={y.mean():.1%} (096# feature contract aligned)"
    )
    return X, y


def build_enriched_as_features(
    enriched_df: pd.DataFrame,
    *,
    require_spread: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """enriched fill records → AS 分類器用特徴量.

    既存 data_loader.build_as_features の出力 + マイクロストラクチャ特徴量
    + side-aligned インタラクション特徴量を返す.

    WARNING: この関数は事後特徴量 (log_queue_wait, edge_bps) を含むため、
    SkipGate の学習には build_preorder_as_features() を使用すること (096#)。

    Args:
        enriched_df: enrich_fill_records() の出力.
        require_spread: True の場合 spread_at_order 必須 (件数減、全 fold で spread 有効).

    Returns:
        (X, y) タプル.
    """
    from scripts.v460.ml.data_loader import build_as_features

    # 1. 既存特徴量
    # 060# fix: enriched path では spread 必須 → 全 TSCV fold で clean features
    X_base, y = build_as_features(enriched_df, require_spread=require_spread)

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

    # 4. 060# v2: Multi-timeframe & momentum 特徴量
    for col in V2_FEATURE_COLS:
        if col in enriched_df.columns:
            micro_features[col] = enriched_df.loc[X_base.index, col].astype(float)

    # 5. 060# v2: side-aligned momentum (AS 予測のコア)
    #    リターンが自分に不利な方向 → AS リスク高
    for ret_col in ["return_30s", "return_60s", "return_300s"]:
        if ret_col in enriched_df.columns:
            ret = enriched_df.loc[X_base.index, ret_col].astype(float)
            # buy: 上昇 → 有利 (+), sell: 下降 → 有利 (+)
            aligned_ret = ret * side_sign
            micro_features[f"side_aligned_{ret_col}"] = aligned_ret

    if micro_features:
        X_micro = pd.DataFrame(micro_features, index=X_base.index)
        X = pd.concat([X_base, X_micro], axis=1)
    else:
        X = X_base

    n_v2 = sum(1 for c in X.columns if c in V2_FEATURE_COLS or c.startswith("side_aligned_return"))
    logger.info(
        f"Enriched AS features: {X.shape[1]} features "
        f"({X_base.shape[1]} base + {len(micro_features)} micro "
        f"[{n_v2} v2]), {len(X)} samples"
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

    # 060# NOTE: spread_jpy / offset_ratio は PnL パイプラインから除外.
    # 理由: 時系列的に後半にしか存在せず (Q1-Q2 は全 NaN),
    #   TSCV 初期 fold で SimpleImputer が全欠損→特徴量無効化.
    #   AS classifier では require_spread=True で対応済.
    #   PnL model の top features は velocity / vpin / hour 系であり影響なし.

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

    # --- 060# v2: Multi-timeframe & momentum ---
    for col in V2_FEATURE_COLS:
        if col in data.columns:
            features[col] = data[col].astype(float)

    # side-aligned momentum
    for ret_col in ["return_30s", "return_60s", "return_300s"]:
        if ret_col in data.columns:
            ret = data[ret_col].astype(float)
            features[f"side_aligned_{ret_col}"] = ret * side_sign

    X = pd.DataFrame(features, index=data.index)
    y = data["post_fill_30s_pnl"].astype(float)

    logger.info(
        f"PnL features: {X.shape[1]} features, {len(X)} samples, "
        f"mean PnL={y.mean():.2f} bps, std={y.std():.2f}"
    )
    return X, y
