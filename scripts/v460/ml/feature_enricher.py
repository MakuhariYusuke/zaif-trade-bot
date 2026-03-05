"""058# Feature Enricher: raw orderbook/trades → fill record 特徴量付与.

fill record の timestamp に対応する板・約定データから
マイクロストラクチャ特徴量を算出し、AS 分類器を強化する。

v459 K2 の非 RL 上限検証 + v460 マイクロストラクチャ特徴量を合わせた
「あるものだけで学習」パイプライン。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.v460.ml.frame_utils import compute_local_hour_cyclic
from ztb.io.jsonl_gz import read_jsonl_gz

logger = logging.getLogger(__name__)

_DEFAULT_RAW_DIR = Path("data/v460/raw")

# 特徴量ウィンドウ (秒)
_TRADE_WINDOW_SEC = 60  # 直近 60 秒の約定統計
_OB_MATCH_TOLERANCE_SEC = 5  # 板スナップショットの許容誤差
_MULTI_TF_WINDOWS = [30, 300]  # 秒 (primary 60s は既存)
_RAW_LOAD_CACHE_MAX_ENTRIES = 8


@dataclass(frozen=True)
class _RawLoadCacheEntry:
    """raw ロード結果の軽量キャッシュ."""

    file_signature: tuple[tuple[str, int, int], ...]
    df: pd.DataFrame


_RAW_ORDERBOOK_CACHE: dict[tuple[str, tuple[str, ...] | None], _RawLoadCacheEntry] = {}
_RAW_TRADES_CACHE: dict[tuple[str, tuple[str, ...] | None], _RawLoadCacheEntry] = {}


def _normalize_date_filter(
    date_filter: Optional[set[str]],
) -> tuple[str, ...] | None:
    """date_filter をハッシュ可能キーへ正規化."""
    if date_filter is None:
        return None
    return tuple(sorted(date_filter))


def _select_raw_files(
    data_dir: Path,
    date_filter: Optional[set[str]],
) -> list[Path]:
    """date_filter を適用した raw ファイル一覧を返す."""
    files: list[Path] = []
    for f in sorted(data_dir.glob("*.jsonl.gz")):
        if date_filter is not None:
            stem = f.stem.replace(".jsonl", "")
            if stem not in date_filter:
                continue
        files.append(f)
    return files


def _build_file_signature(files: list[Path]) -> tuple[tuple[str, int, int], ...]:
    """mtime+size ベースの軽量シグネチャを生成."""
    signature: list[tuple[str, int, int]] = []
    for f in files:
        try:
            st = f.stat()
            signature.append((f.name, st.st_mtime_ns, st.st_size))
        except OSError:
            signature.append((f.name, -1, -1))
    return tuple(signature)


def _load_cached_raw_df(
    cache: dict[tuple[str, tuple[str, ...] | None], _RawLoadCacheEntry],
    cache_key: tuple[str, tuple[str, ...] | None],
    file_signature: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame | None:
    """シグネチャ一致時のみ shallow copy を返す."""
    cached = cache.get(cache_key)
    if cached is None or cached.file_signature != file_signature:
        return None
    return cached.df.copy(deep=False)


def _store_raw_cache(
    cache: dict[tuple[str, tuple[str, ...] | None], _RawLoadCacheEntry],
    cache_key: tuple[str, tuple[str, ...] | None],
    file_signature: tuple[tuple[str, int, int], ...],
    df: pd.DataFrame,
) -> None:
    """キャッシュ保存。サイズ上限超過時は最古エントリを破棄."""
    cache[cache_key] = _RawLoadCacheEntry(file_signature=file_signature, df=df)
    if len(cache) <= _RAW_LOAD_CACHE_MAX_ENTRIES:
        return
    oldest = next(iter(cache))
    if oldest != cache_key:
        cache.pop(oldest, None)


@dataclass(frozen=True)
class _TradeFeatureContext:
    """約定特徴量の高速計算に使う前計算済み配列."""

    timestamps: np.ndarray
    prices: np.ndarray
    cumulative_total_volume: np.ndarray
    cumulative_buy_volume: np.ndarray


@dataclass(frozen=True)
class _OrderbookFeatureContext:
    """板特徴量の高速参照に使う前計算済み配列."""

    timestamps: np.ndarray
    mid_prices: np.ndarray
    spread_bps: np.ndarray
    depth_imbalance: np.ndarray
    bid_vol_5: np.ndarray
    ask_vol_5: np.ndarray


def _build_orderbook_feature_context(
    ob_df: pd.DataFrame,
) -> _OrderbookFeatureContext | None:
    """ob_df から高速参照用の配列を構築."""
    if ob_df.empty:
        return None
    return _OrderbookFeatureContext(
        timestamps=ob_df["ts"].to_numpy(dtype=np.float64, copy=False),
        mid_prices=ob_df["mid_price"].to_numpy(dtype=np.float64, copy=False),
        spread_bps=ob_df["spread_bps"].to_numpy(dtype=np.float64, copy=False),
        depth_imbalance=ob_df["depth_imbalance"].to_numpy(dtype=np.float64, copy=False),
        bid_vol_5=ob_df["bid_vol_5"].to_numpy(dtype=np.float64, copy=False),
        ask_vol_5=ob_df["ask_vol_5"].to_numpy(dtype=np.float64, copy=False),
    )


def _build_trade_feature_context(
    trades_df: pd.DataFrame,
    *,
    sorted_ts: np.ndarray | None = None,
) -> _TradeFeatureContext | None:
    """trades_df から高速窓計算用の前計算配列を構築."""
    if trades_df.empty:
        return None

    timestamps = (
        np.asarray(sorted_ts, dtype=np.float64)
        if sorted_ts is not None and len(sorted_ts) == len(trades_df)
        else trades_df["ts"].to_numpy(dtype=np.float64, copy=False)
    )
    if timestamps.size == 0:
        return None

    prices = trades_df["price"].to_numpy(dtype=np.float64, copy=False)
    amounts = trades_df["amount"].to_numpy(dtype=np.float64, copy=False)
    is_buy = trades_df["side"].astype(str).str.lower().eq("buy").to_numpy(dtype=bool, copy=False)
    buy_amounts = np.where(is_buy, amounts, 0.0)

    cumulative_total = np.empty(len(amounts) + 1, dtype=np.float64)
    cumulative_total[0] = 0.0
    np.cumsum(amounts, dtype=np.float64, out=cumulative_total[1:])

    cumulative_buy = np.empty(len(amounts) + 1, dtype=np.float64)
    cumulative_buy[0] = 0.0
    np.cumsum(buy_amounts, dtype=np.float64, out=cumulative_buy[1:])

    return _TradeFeatureContext(
        timestamps=timestamps,
        prices=prices,
        cumulative_total_volume=cumulative_total,
        cumulative_buy_volume=cumulative_buy,
    )


def _default_trade_features() -> dict[str, float]:
    """primary 60s 互換のデフォルト特徴量."""
    return {
        "trade_count_60s": 0.0,
        "buy_ratio": 0.5,
        "trade_flow_imbalance_60s": 0.0,
        "avg_trade_size": 0.0,
        "price_velocity_bps": 0.0,
        "vpin_60s": 0.5,
    }


def _default_multi_timeframe_trade_features(
    windows: tuple[int, ...] = tuple(_MULTI_TF_WINDOWS),
) -> dict[str, float]:
    """multi-timeframe 特徴量のデフォルト値."""
    result: dict[str, float] = {}
    for window_sec in windows:
        suffix = f"_{window_sec}s"
        result[f"vpin{suffix}"] = 0.5
        result[f"tfi{suffix}"] = 0.0
        result[f"velocity{suffix}"] = 0.0
        result[f"trade_count{suffix}"] = 0.0
    result["vpin_acceleration"] = 0.0
    result["tfi_acceleration"] = 0.0
    result["trade_rate_acceleration"] = 0.0
    return result


def _compute_trade_window_core(
    context: _TradeFeatureContext,
    *,
    end_index: int,
    ts: float,
    window_sec: int,
) -> dict[str, float]:
    """単一窓の約定統計コア計算."""
    start_index = int(np.searchsorted(context.timestamps, ts - window_sec, side="left"))
    if start_index >= end_index:
        return {
            "count": 0.0,
            "buy_ratio": 0.5,
            "tfi": 0.0,
            "avg_size": 0.0,
            "price_velocity": 0.0,
            "vpin": 0.5,
        }

    total_vol = float(
        context.cumulative_total_volume[end_index] - context.cumulative_total_volume[start_index]
    )
    buy_vol = float(
        context.cumulative_buy_volume[end_index] - context.cumulative_buy_volume[start_index]
    )
    count = end_index - start_index

    if total_vol > 0.0:
        buy_ratio = buy_vol / total_vol
        signed_flow = 2.0 * buy_vol - total_vol
        tfi = signed_flow / total_vol
        vpin = abs(signed_flow) / total_vol
    else:
        buy_ratio = 0.5
        tfi = 0.0
        vpin = 0.5

    first_price = float(context.prices[start_index])
    last_price = float(context.prices[end_index - 1])
    price_velocity = (
        (last_price - first_price) / first_price * 10000
        if first_price > 0.0
        else 0.0
    )

    return {
        "count": float(count),
        "buy_ratio": buy_ratio,
        "tfi": tfi,
        "avg_size": total_vol / count if count > 0 else 0.0,
        "price_velocity": price_velocity,
        "vpin": vpin,
    }


def _compute_trade_feature_bundle(
    ts: float,
    *,
    context: _TradeFeatureContext | None,
    primary_window_sec: int = _TRADE_WINDOW_SEC,
    multi_windows: tuple[int, ...] = tuple(_MULTI_TF_WINDOWS),
) -> tuple[dict[str, float], dict[str, float]]:
    """primary + multi-timeframe の trade 特徴量を一括計算."""
    primary = _default_trade_features()
    multi = _default_multi_timeframe_trade_features(multi_windows)
    if context is None:
        return primary, multi

    end_index = int(np.searchsorted(context.timestamps, ts, side="left"))
    if end_index <= 0:
        return primary, multi

    primary_core = _compute_trade_window_core(
        context,
        end_index=end_index,
        ts=ts,
        window_sec=primary_window_sec,
    )
    primary = {
        "trade_count_60s": primary_core["count"],
        "buy_ratio": primary_core["buy_ratio"],
        "trade_flow_imbalance_60s": primary_core["tfi"],
        "avg_trade_size": primary_core["avg_size"],
        "price_velocity_bps": primary_core["price_velocity"],
        "vpin_60s": primary_core["vpin"],
    }

    core_by_window: dict[int, dict[str, float]] = {}
    for window_sec in multi_windows:
        core = _compute_trade_window_core(
            context,
            end_index=end_index,
            ts=ts,
            window_sec=window_sec,
        )
        core_by_window[window_sec] = core
        suffix = f"_{window_sec}s"
        multi[f"vpin{suffix}"] = core["vpin"]
        multi[f"tfi{suffix}"] = core["tfi"]
        multi[f"velocity{suffix}"] = core["price_velocity"]
        multi[f"trade_count{suffix}"] = core["count"]

    short_core = core_by_window.get(30)
    long_core = core_by_window.get(300)
    if short_core is not None and long_core is not None:
        multi["vpin_acceleration"] = short_core["vpin"] - long_core["vpin"]
        multi["tfi_acceleration"] = short_core["tfi"] - long_core["tfi"]
        rate_30 = short_core["count"] / 30.0 if short_core["count"] > 0.0 else 0.0
        rate_300 = long_core["count"] / 300.0 if long_core["count"] > 0.0 else 0.0
        multi["trade_rate_acceleration"] = rate_30 - rate_300

    return primary, multi


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

    target_files = _select_raw_files(ob_dir, date_filter)
    date_filter_key = _normalize_date_filter(date_filter)
    cache_key = (str(ob_dir.resolve()), date_filter_key)
    file_signature = _build_file_signature(target_files)
    cached_df = _load_cached_raw_df(_RAW_ORDERBOOK_CACHE, cache_key, file_signature)
    if cached_df is not None:
        return cached_df

    if not target_files:
        empty = pd.DataFrame()
        _store_raw_cache(_RAW_ORDERBOOK_CACHE, cache_key, file_signature, empty)
        return empty.copy(deep=False)

    rows: list[dict[str, float]] = []
    for f in target_files:
        for r in read_jsonl_gz(f):
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

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ts").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} orderbook snapshots")
    _store_raw_cache(_RAW_ORDERBOOK_CACHE, cache_key, file_signature, df)
    return df.copy(deep=False)


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

    target_files = _select_raw_files(tr_dir, date_filter)
    date_filter_key = _normalize_date_filter(date_filter)
    cache_key = (str(tr_dir.resolve()), date_filter_key)
    file_signature = _build_file_signature(target_files)
    cached_df = _load_cached_raw_df(_RAW_TRADES_CACHE, cache_key, file_signature)
    if cached_df is not None:
        return cached_df

    if not target_files:
        empty = pd.DataFrame()
        _store_raw_cache(_RAW_TRADES_CACHE, cache_key, file_signature, empty)
        return empty.copy(deep=False)

    all_records: list[dict[str, object]] = []
    for f in target_files:
        all_records.extend(read_jsonl_gz(f))

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.sort_values("ts").reset_index(drop=True)
    n_days = len(date_filter) if date_filter else "all"
    logger.info(f"Loaded {len(df)} trades (days={n_days})")
    _store_raw_cache(_RAW_TRADES_CACHE, cache_key, file_signature, df)
    return df.copy(deep=False)


def _compute_trade_features(
    trades_df: pd.DataFrame,
    ts: float,
    window_sec: int = _TRADE_WINDOW_SEC,
    *,
    _sorted_ts: np.ndarray | None = None,
    _context: _TradeFeatureContext | None = None,
) -> dict[str, float]:
    """指定時点の直前 window_sec 秒間の約定統計を算出.

    059# P1-7: searchsorted で O(log N) にフィルタ (呼び出し側で _sorted_ts を渡す).

    Returns:
        trade_count_60s, buy_ratio, trade_flow_imbalance_60s,
        avg_trade_size, price_velocity_bps, vpin_60s
    """
    context = _context or _build_trade_feature_context(trades_df, sorted_ts=_sorted_ts)
    primary, _ = _compute_trade_feature_bundle(
        ts,
        context=context,
        primary_window_sec=window_sec,
        multi_windows=(),
    )
    return primary


def _find_nearest_ob(
    ob_df: pd.DataFrame,
    ts: float,
    tolerance_sec: int = _OB_MATCH_TOLERANCE_SEC,
    *,
    _context: _OrderbookFeatureContext | None = None,
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
    context = _context or _build_orderbook_feature_context(ob_df)
    if context is None:
        return default

    # Binary search for nearest
    idx = int(np.searchsorted(context.timestamps, ts))
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(context.timestamps):
        candidates.append(idx)

    best_idx = -1
    best_diff = float("inf")
    for c in candidates:
        diff = abs(float(context.timestamps[c]) - ts)
        if diff < best_diff:
            best_diff = diff
            best_idx = c

    if best_idx < 0 or best_diff > tolerance_sec:
        return default

    return {
        "spread_bps_ob": float(context.spread_bps[best_idx]),
        "depth_imbalance_ob": float(context.depth_imbalance[best_idx]),
        "bid_vol_5_ob": float(context.bid_vol_5[best_idx]),
        "ask_vol_5_ob": float(context.ask_vol_5[best_idx]),
    }


# ----- 060# v2: Multi-timeframe & volatility features -----


def _compute_multi_timeframe_trade_features(
    trades_df: pd.DataFrame,
    ts: float,
    *,
    _sorted_ts: np.ndarray | None = None,
    _context: _TradeFeatureContext | None = None,
) -> dict[str, float]:
    """060# v2: 30s/300s 窓の約定統計 (既存 60s を補完).

    AS 予測には時間スケール間の差分が重要:
    - 短期 (30s) vs 長期 (300s) の flow 乖離 → 情報トレーダー検知
    - 加速度 (trade rate の変化) → urgency
    """
    context = _context or _build_trade_feature_context(trades_df, sorted_ts=_sorted_ts)
    _, result = _compute_trade_feature_bundle(
        ts,
        context=context,
    )
    return result


def _compute_return_momentum(
    ob_df: pd.DataFrame,
    ts: float,
    *,
    sorted_ts: np.ndarray | None = None,
    _context: _OrderbookFeatureContext | None = None,
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

    context = _context or _build_orderbook_feature_context(ob_df)
    if context is None:
        return default

    ts_arr = (
        np.asarray(sorted_ts, dtype=np.float64)
        if sorted_ts is not None and len(sorted_ts) == len(context.timestamps)
        else context.timestamps
    )
    mid_arr = context.mid_prices

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
    trades_fallback_recent_days: int = 1,
    **kwargs: object,
) -> pd.DataFrame:
    """fill records にマイクロストラクチャ特徴量を付与.

    Args:
        fill_df: load_fill_records() の出力.
        raw_dir: raw data のディレクトリ.
        ob_tolerance_sec: 板スナップショットの許容誤差 (秒).
        trade_window_sec: 約定統計のウィンドウ (秒).
        trades_fallback_recent_days: 130# Q3: trades fallback の窓幅 (±N日, default 1).

    Returns:
        enriched DataFrame. 新規カラム:
            spread_bps_ob, depth_imbalance_ob, bid_vol_5_ob, ask_vol_5_ob,
            trade_count_60s, buy_ratio, trade_flow_imbalance_60s,
            avg_trade_size, price_velocity_bps, vpin_60s,
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
    # 130# trades は date_filter 適用後にデータが空の場合は全量にフォールバック.
    # OB recorder は板情報のみ記録し trades ファイルは別系統のため、
    # 当日分の trades ファイルが存在しないケースがある.
    trades_df = load_raw_trades(raw_dir, date_filter=date_filter)
    if trades_df.empty and date_filter is not None:
        # 130# F7 + D.1 Q3: ±N日にフォールバック (デフォルト ±1日)。
        # 特徴量 window は 60s/300s なので、大きな窓は I/O の無駄。
        # trades_fallback_recent_days は YAML からオーバーライド可能。
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        _now = _dt.now(tz=_tz.utc)
        _fallback_days = trades_fallback_recent_days
        _fb_filter: set[str] = set()
        for _i in range(_fallback_days + 1):
            _fb_filter.add((_now - _td(days=_i)).strftime("%Y%m%d"))
            _fb_filter.add((_now + _td(days=_i)).strftime("%Y%m%d"))  # 未来日付も含む (TZ 差分)
        logger.info(
            f"130# F7: trades empty with date_filter, "
            f"falling back to recent ±{_fallback_days} days: {sorted(_fb_filter)}"
        )
        trades_df = load_raw_trades(raw_dir, date_filter=_fb_filter)
        if trades_df.empty:
            # 139# §9-#5: 全量フォールバックを廃止 (I/O爆発 + 時間整合リスク)。
            # 代わりに WARNING で検知可能にし、trades_df は空のまま進める。
            logger.warning(
                "139# trades still empty after ±day fallback. "
                "Skipping trades features (date_filter=None全量ロード廃止). "
                "trades データ収集系を確認してください。"
            )

    # 059# P1-7: 事前ソート + searchsorted で O(N_fill × log N_trades)
    if not trades_df.empty and "ts" in trades_df.columns:
        trades_df = trades_df.sort_values("ts").reset_index(drop=True)
        sorted_ts = trades_df["ts"].values
        trade_context = _build_trade_feature_context(trades_df, sorted_ts=sorted_ts)
    else:
        sorted_ts = None
        trade_context = None

    ob_context = _build_orderbook_feature_context(ob_df)
    # 060#: OB ts も事前準備 (multi-timeframe momentum 計算用)
    ob_sorted_ts = ob_df["ts"].values if not ob_df.empty else None

    timestamps = fill_df["timestamp"].to_numpy(dtype=np.float64, copy=False)
    enriched_rows: list[dict[str, float]] = []
    for ts in timestamps:

        # Orderbook features (nearest snapshot)
        ob_features = _find_nearest_ob(
            ob_df,
            ts,
            ob_tolerance_sec,
            _context=ob_context,
        )

        # Trade features (rolling window — primary 60s)
        trade_features, multi_tf = _compute_trade_feature_bundle(
            ts,
            context=trade_context,
            primary_window_sec=trade_window_sec,
        )

        # 060# v2: Return momentum & volatility (from OB mid prices)
        momentum = _compute_return_momentum(
            ob_df,
            ts,
            sorted_ts=ob_sorted_ts,
            _context=ob_context,
        )

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

    # 168# §10: 重複列回避 — fill_df に既存のマイクロ特徴量列があれば
    # enriched 側の再計算値を優先する (全レコード一貫性のため)
    overlap_cols = fill_df.columns.intersection(enriched.columns)
    if len(overlap_cols) > 0:
        logger.info(f"168# Dropping {len(overlap_cols)} overlapping columns from fill_df: {overlap_cols.tolist()}")
        fill_df = fill_df.drop(columns=overlap_cols)

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
    "price_velocity_bps",
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
    # filled かつ AS ラベル有りのみ
    mask = enriched_df["filled"].astype(bool) & enriched_df["adverse_selected_raw"].notna()
    data = enriched_df.loc[mask].copy()

    if require_spread:
        data = data.dropna(subset=["spread_at_order", "spread_offset_ratio"])

    if len(data) < 10:
        raise ValueError(f"Insufficient labeled samples: {len(data)}")

    features: dict[str, pd.Series] = {}
    aligned = data

    # --- 注文前に観測可能な特徴量のみ ---
    # F1: side (binary)
    features["side_buy"] = (aligned["side"] == "buy").astype(int)

    # F2: hour (cyclic) — 注文時刻
    hour_sin, hour_cos = compute_local_hour_cyclic(data["timestamp"])
    features["hour_sin"] = hour_sin
    features["hour_cos"] = hour_cos

    # F3: spread_jpy — 注文時のスプレッド
    if "spread_at_order" in data.columns:
        features["spread_jpy"] = aligned["spread_at_order"].astype(float)

    # F4: offset_ratio — 注文時の設定値
    if "spread_offset_ratio" in data.columns:
        features["offset_ratio"] = aligned["spread_offset_ratio"].astype(float)

    # F5: regime (one-hot)
    # 176# 横展開: trending_up/trending_down も regime_trending=1 に含める
    if "regime" in data.columns:
        regime = aligned["regime"].fillna("unknown")
        features["regime_trending"] = regime.str.startswith("trending").astype(int)
        features["regime_ranging"] = (regime == "ranging").astype(int)
        features["regime_high_vol"] = (regime == "high_vol").astype(int)

    # F6: trade-based micro features (注文前に算出可能)
    for col in MICRO_FEATURE_COLS:
        if col in aligned.columns and col not in ("spread_bps_ob", "depth_imbalance_ob"):
            features[col] = aligned[col].astype(float)

    # F7: side-aligned interaction features (trade-based only)
    side_sign = aligned["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)

    if "trade_flow_imbalance_60s" in aligned.columns:
        tfi = aligned["trade_flow_imbalance_60s"].astype(float)
        features["side_aligned_tfi"] = (tfi * side_sign).fillna(0.0)

    if "price_velocity_bps" in aligned.columns:
        vel = aligned["price_velocity_bps"].astype(float)
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
    aligned_rows = enriched_df.loc[X_base.index]

    # 2. マイクロストラクチャ特徴量を追加
    # NOTE: NaN は保持。補完は CV fold 内で SimpleImputer が行う (059# P0-1)
    micro_features: dict[str, pd.Series] = {}
    for col in MICRO_FEATURE_COLS:
        if col in aligned_rows.columns:
            micro_features[col] = aligned_rows[col].astype(float)

    # 3. side-aligned インタラクション特徴量
    #    buy 側: side_sign = +1, sell 側: side_sign = -1
    #    → "自分の注文に有利な方向" を正の値で表現
    side_sign = aligned_rows["side"].map(
        {"buy": 1.0, "sell": -1.0}
    ).astype(float)

    # depth_imbalance × side_sign: bid 厚い → buy に有利 (+)
    if "depth_imbalance_ob" in aligned_rows.columns:
        di = aligned_rows["depth_imbalance_ob"].astype(float)
        aligned_imbalance = di * side_sign
        micro_features["side_aligned_imbalance"] = aligned_imbalance.fillna(0.0)

    # trade_flow_imbalance × side_sign: buy 優勢 → buy に有利 (+)
    if "trade_flow_imbalance_60s" in aligned_rows.columns:
        tfi = aligned_rows["trade_flow_imbalance_60s"].astype(float)
        aligned_tfi = tfi * side_sign
        micro_features["side_aligned_tfi"] = aligned_tfi.fillna(0.0)

    # price_velocity × side_sign: 上昇 → buy に有利 (+)
    if "price_velocity_bps" in aligned_rows.columns:
        vel = aligned_rows["price_velocity_bps"].astype(float)
        aligned_vel = vel * side_sign
        micro_features["side_aligned_velocity"] = aligned_vel.fillna(0.0)

    # 4. 060# v2: Multi-timeframe & momentum 特徴量
    for col in V2_FEATURE_COLS:
        if col in aligned_rows.columns:
            micro_features[col] = aligned_rows[col].astype(float)

    # 5. 060# v2: side-aligned momentum (AS 予測のコア)
    #    リターンが自分に不利な方向 → AS リスク高
    for ret_col in ["return_30s", "return_60s", "return_300s"]:
        if ret_col in aligned_rows.columns:
            ret = aligned_rows[ret_col].astype(float)
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

    # 059# NEW-03: 小数時刻で統一 (skip_gate 推論側と同一粒度)
    hour_sin, hour_cos = compute_local_hour_cyclic(data["timestamp"])
    features["hour_sin"] = hour_sin
    features["hour_cos"] = hour_cos

    # 060# NOTE: spread_jpy / offset_ratio は PnL パイプラインから除外.
    # 理由: 時系列的に後半にしか存在せず (Q1-Q2 は全 NaN),
    #   TSCV 初期 fold で SimpleImputer が全欠損→特徴量無効化.
    #   AS classifier では require_spread=True で対応済.
    #   PnL model の top features は velocity / vpin / hour 系であり影響なし.

    # 176# 横展開: trending_up/trending_down も regime_trending=1 に含める
    if "regime" in data.columns:
        regime = data["regime"].fillna("unknown")
        features["regime_trending"] = regime.str.startswith("trending").astype(int)
        features["regime_ranging"] = (regime == "ranging").astype(int)
        features["regime_high_vol"] = (regime == "high_vol").astype(int)

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
    if "price_velocity_bps" in data.columns:
        vel = data["price_velocity_bps"].astype(float)
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
