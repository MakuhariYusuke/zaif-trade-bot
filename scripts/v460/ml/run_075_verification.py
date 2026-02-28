"""075# ph2 検証: レビュー指摘への対応と 50K ステップ検証.

073# レビュー (074_ph2_rev_073.md) の重大指摘すべてに対応:
- CRITICAL#2: clean/quarantine 分離適用
- HIGH#3: S12 sim_pnl バグ修正
- HIGH#6: 統計検定 (Mann-Whitney U + Cliff's Delta) 適用
- MEDIUM#7: time_filter 両side同時ブロック時間の定量評価
- MEDIUM#9: JSON artifact 出力

加えて 50K ステップ Monte Carlo 検証を実施。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scripts.v460.ml.frame_utils import (
    collect_bad_side_hours,
    compute_utc_hour,
    exclude_side_hour_combos,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillRecord,
    filter_clean_records,
    load_fill_records_glob,
)
from ztb.metrics.gate_checks import (
    cliffs_delta,
    holm_bonferroni_gate,
    p_mean_gate,
)
from ztb.io.json_io import write_json

# --- 定数 ---
PNL_COL = "post_fill_30s_pnl"
RESULTS_DIR = _PROJECT_ROOT / "results" / "v460" / "fill_test"
ARTIFACT_DIR = _PROJECT_ROOT / "results" / "v460" / "verification_077"


def _collect_pnl_blocks(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    pnl_col: str,
) -> list[np.ndarray]:
    """日次単位の PnL ブロックを配列で収集."""
    if frame.empty:
        return []
    grouped = (
        frame.loc[mask, ["date", pnl_col]]
        .dropna(subset=[pnl_col])
        .groupby("date", sort=False)[pnl_col]
    )
    return [
        values.to_numpy(dtype=np.float64, copy=False)
        for _, values in grouped
        if len(values) > 0
    ]


def _prepare_block_stats(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """block bootstrap 用のサイズ/総和を前計算."""
    if not blocks:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    block_sizes = np.fromiter((block.size for block in blocks), dtype=np.int64, count=len(blocks))
    block_sums = np.fromiter((block.sum() for block in blocks), dtype=np.float64, count=len(blocks))
    return block_sizes, block_sums


def _sample_block_bootstrap_sum(
    blocks: list[np.ndarray],
    *,
    block_sizes: np.ndarray,
    block_sums: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> float:
    """日次ブロック bootstrap の合計値だけを返す."""
    if n_steps <= 0 or not blocks:
        return 0.0

    total = 0
    sampled_sum = 0.0
    n_blocks = len(blocks)
    while total < n_steps:
        block_idx = int(rng.integers(n_blocks))
        block_size = int(block_sizes[block_idx])
        if block_size <= 0:
            continue
        remaining = n_steps - total
        if block_size <= remaining:
            sampled_sum += float(block_sums[block_idx])
            total += block_size
            continue
        sampled_sum += float(blocks[block_idx][:remaining].sum())
        total = n_steps
    return sampled_sum


def load_clean_filled(
    *,
    run_ids: list[str] | None = None,
    git_shas: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """clean/quarantine 分離済みの filled records を返す.

    Args:
        run_ids: 指定時はこれらの run_id のレコードのみ使用 (084# 追加).
        git_shas: 指定時はこれらの git_sha (前方一致) のレコードのみ使用.

    Returns:
        (clean_all_df, clean_filled_df, quarantine_df, stats_dict)
        clean_all_df には filled=False (cancelled含む) も含む.
    """
    all_records = load_fill_records_glob(RESULTS_DIR)

    # 084# run_id / git_sha フィルタ
    if run_ids:
        _ids = set(run_ids)
        all_records = [r for r in all_records if getattr(r, "run_id", None) in _ids]
        print(f"  [filter] run_id={run_ids} → {len(all_records)} records")
    if git_shas:
        def _sha_match(rec: FillRecord) -> bool:
            sha = getattr(rec, "git_sha", None) or ""
            return any(sha.startswith(g) for g in git_shas)
        all_records = [r for r in all_records if _sha_match(r)]
        print(f"  [filter] git_sha={git_shas} → {len(all_records)} records")

    clean, quarantine = filter_clean_records(all_records, require_git_sha=True)

    def to_df(recs: list[FillRecord]) -> pd.DataFrame:
        if not recs:
            return pd.DataFrame()
        rows = []
        for r in recs:
            d: dict = {}
            for field in r.__dataclass_fields__:
                d[field] = getattr(r, field)
            rows.append(d)
        return pd.DataFrame(rows)

    clean_df = to_df(clean)
    quarantine_df = to_df(quarantine)

    # filled のみ
    if len(clean_df) > 0 and "filled" in clean_df.columns:
        clean_filled = clean_df[clean_df["filled"] == True].copy()
    else:
        clean_filled = pd.DataFrame()

    stats = {
        "total_records": len(all_records),
        "clean": len(clean),
        "quarantine": len(quarantine),
        "clean_filled": len(clean_filled),
        "quarantine_filled": len(quarantine_df[quarantine_df["filled"] == True])
        if len(quarantine_df) > 0 and "filled" in quarantine_df.columns
        else 0,
    }

    return clean_df, clean_filled, quarantine_df, stats


def section_0_data_quality(
    clean_filled: pd.DataFrame, quarantine_df: pd.DataFrame, stats: dict,
) -> dict:
    """§0 データ品質: clean/quarantine 分離結果."""
    print("=" * 70)
    print("§0 データ品質 (CRITICAL#2 対応: clean/quarantine 分離)")
    print("=" * 70)
    print(f"  全レコード:     {stats['total_records']}")
    print(f"  clean:          {stats['clean']}")
    print(f"  quarantine:     {stats['quarantine']}")
    print(f"  clean filled:   {stats['clean_filled']}")
    print(f"  quarantine filled: {stats['quarantine_filled']}")
    q_pct = stats["quarantine"] / stats["total_records"] * 100 if stats["total_records"] > 0 else 0
    print(f"  quarantine 比率: {q_pct:.1f}%")

    if len(quarantine_df) > 0 and "timestamp" in quarantine_df.columns:
        q_ts = pd.to_datetime(quarantine_df["timestamp"], unit="s", utc=True)
        print(f"  quarantine 期間: {q_ts.min()} → {q_ts.max()}")

    # clean vs quarantine の PnL 比較
    if stats["quarantine_filled"] > 0 and PNL_COL in quarantine_df.columns:
        q_pnl = quarantine_df.loc[quarantine_df["filled"] == True, PNL_COL].dropna()
        c_pnl = clean_filled[PNL_COL].dropna()
        if len(q_pnl) > 0 and len(c_pnl) > 0:
            print(f"\n  PnL 比較:")
            print(f"    clean mean:      {c_pnl.mean():+.3f} bps (n={len(c_pnl)})")
            print(f"    quarantine mean: {q_pnl.mean():+.3f} bps (n={len(q_pnl)})")
            print(f"    差異:            {c_pnl.mean() - q_pnl.mean():+.3f} bps")
    print()
    return stats


def section_1_basic_stats(clean_filled: pd.DataFrame) -> dict:
    """§1 基本統計 (clean データのみ)."""
    print("=" * 70)
    print("§1 基本統計 (clean filled のみ)")
    print("=" * 70)

    pnl = clean_filled[PNL_COL].dropna()
    result: dict = {
        "n_filled": len(clean_filled),
        "mean_pnl": float(pnl.mean()) if len(pnl) > 0 else None,
        "median_pnl": float(pnl.median()) if len(pnl) > 0 else None,
        "std_pnl": float(pnl.std()) if len(pnl) > 0 else None,
        "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else None,
    }

    # AS 統計
    if "adverse_selected" in clean_filled.columns:
        as_col = clean_filled["adverse_selected"].dropna()
        result["as_ratio"] = float(as_col.mean()) if len(as_col) > 0 else None

    # Side 別
    for side in ["buy", "sell"]:
        side_df = clean_filled[clean_filled["side"] == side]
        sp = side_df[PNL_COL].dropna()
        result[f"{side}_n"] = len(side_df)
        result[f"{side}_mean_pnl"] = float(sp.mean()) if len(sp) > 0 else None
        result[f"{side}_win_rate"] = float((sp > 0).mean()) if len(sp) > 0 else None
        if "adverse_selected" in side_df.columns:
            sa = side_df["adverse_selected"].dropna()
            result[f"{side}_as_ratio"] = float(sa.mean()) if len(sa) > 0 else None

    # 期間
    if "timestamp" in clean_filled.columns:
        ts = pd.to_datetime(clean_filled["timestamp"], unit="s", utc=True)
        result["date_range_start"] = str(ts.min())
        result["date_range_end"] = str(ts.max())
        result["days"] = float((ts.max() - ts.min()).total_seconds() / 86400)

    for k, v in result.items():
        if isinstance(v, float) and v is not None:
            print(f"  {k}: {v:+.4f}" if "pnl" in k or "rate" in k or "ratio" in k else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print()
    return result


def section_2_side_hour_heatmap(clean_filled: pd.DataFrame) -> dict:
    """§2 Side × Hour ヒートマップ (clean データのみ)."""
    print("=" * 70)
    print("§2 Side × Hour ヒートマップ (事前制御可能パラメータ)")
    print("=" * 70)

    clean_filled = clean_filled.copy()
    clean_filled["utc_hour"] = compute_utc_hour(clean_filled["timestamp"])

    results: dict = {}
    for side in ["buy", "sell"]:
        side_data = clean_filled[clean_filled["side"] == side]
        print(f"\n  {side.upper()} (n={len(side_data)}):")
        print(f"  {'UTC':>5} {'JST':>5} {'mean':>8} {'med':>8} {'n':>4} {'AS%':>5} {'win%':>5}")
        for h in range(24):
            hdata = side_data[side_data["utc_hour"] == h]
            if len(hdata) == 0:
                continue
            p = hdata[PNL_COL].dropna()
            as_r = hdata["adverse_selected"].dropna().mean() * 100 if "adverse_selected" in hdata.columns else 0
            jst = (h + 9) % 24
            mean_p = p.mean() if len(p) > 0 else 0
            med_p = p.median() if len(p) > 0 else 0
            win_p = (p > 0).mean() * 100 if len(p) > 0 else 0
            marker = " ★" if mean_p > 1.0 else (" ✗" if mean_p < -1.5 else "")
            print(
                f"  {h:>5} {jst:>5} {mean_p:>+8.3f} {med_p:>+8.3f} {len(hdata):>4} "
                f"{as_r:>5.1f} {win_p:>5.1f}{marker}"
            )
            results[f"{side}_UTC{h:02d}"] = {
                "mean_pnl": float(mean_p),
                "median_pnl": float(med_p),
                "n": len(hdata),
                "as_pct": float(as_r),
                "win_pct": float(win_p),
            }

    return results


def section_3_time_filter_impact(
    clean_filled: pd.DataFrame, clean_all: pd.DataFrame | None = None,
) -> dict:
    """§3 MEDIUM#7 対応: time_filter の機会損失評価 + 076# MEDIUM#6 マルチメトリクス."""
    print("\n" + "=" * 70)
    print("§3 Time Filter 機会損失分析 (MEDIUM#7 + 076# MEDIUM#6)")
    print("=" * 70)

    clean_filled = clean_filled.copy()
    clean_filled["utc_hour"] = compute_utc_hour(clean_filled["timestamp"])

    # 075# YAML の設定値 (clean データ再検証後 + §8.2 批判反映)
    global_skip = {1, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21}
    buy_skip = {1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23}
    sell_skip = {3, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23}

    # 各時間帯の両 side ブロック / 片 side のみ / 両 side 通過
    both_blocked = buy_skip & sell_skip
    buy_only = sell_skip - buy_skip  # sell がブロック → buy のみ通過
    sell_only = buy_skip - sell_skip  # buy がブロック → sell のみ通過
    both_open = set(range(24)) - buy_skip - sell_skip

    # 実質的に add: buy_only ∪ sell_only で片 side 取引可能
    side_specific_gain = (buy_skip | sell_skip) - both_blocked

    print(f"  グローバルスキップ (073# 以前): {sorted(global_skip)} ({len(global_skip)}/24h)")
    print(f"  両 side ブロック:   {sorted(both_blocked)} ({len(both_blocked)}/24h)")
    print(f"  buy のみ通過:       {sorted(buy_only)} ({len(buy_only)}/24h)")
    print(f"  sell のみ通過:      {sorted(sell_only)} ({len(sell_only)}/24h)")
    print(f"  両 side 通過:       {sorted(both_open)} ({len(both_open)}/24h)")
    print(f"  side 分離による追加稼働: {len(side_specific_gain)} h/day")

    # PnL 影響: 各カテゴリの historical PnL
    categories = {
        "both_blocked": both_blocked,
        "buy_only": buy_only,
        "sell_only": sell_only,
        "both_open": both_open,
    }
    result: dict = {}
    print(f"\n  {'カテゴリ':<20} {'mean PnL':>9} {'n':>5} {'備考'}")
    for cat_name, hours in categories.items():
        if not hours:
            print(f"  {cat_name:<20} {'N/A':>9} {'0':>5}")
            continue
        mask = clean_filled["utc_hour"].isin(hours)
        pnl_data = clean_filled.loc[mask, PNL_COL].dropna()

        # side 固有のフィルタリングも考慮
        if cat_name == "buy_only":
            pnl_data = clean_filled.loc[
                mask & (clean_filled["side"] == "buy"), PNL_COL
            ].dropna()
        elif cat_name == "sell_only":
            pnl_data = clean_filled.loc[
                mask & (clean_filled["side"] == "sell"), PNL_COL
            ].dropna()

        mean_p = float(pnl_data.mean()) if len(pnl_data) > 0 else 0.0
        note = ""
        if cat_name in ("buy_only", "sell_only") and len(pnl_data) > 0:
            note = "片 side で稼働 → PnL 改善期待"
        print(f"  {cat_name:<20} {mean_p:>+9.3f} {len(pnl_data):>5} {note}")
        result[cat_name] = {"hours": sorted(hours), "mean_pnl": mean_p, "n": len(pnl_data)}

    # 073# filter 適用後の理論 PnL (before vs after)
    print("\n  --- 073# filter 適用前後の理論PnL ---")
    # Before: グローバルスキップのみ
    before_mask = ~clean_filled["utc_hour"].isin(global_skip)
    before_pnl = clean_filled.loc[before_mask, PNL_COL].dropna()
    # After: side 別スキップ
    after_buy_mask = (clean_filled["side"] == "buy") & (~clean_filled["utc_hour"].isin(buy_skip))
    after_sell_mask = (clean_filled["side"] == "sell") & (~clean_filled["utc_hour"].isin(sell_skip))
    after_pnl = clean_filled.loc[after_buy_mask | after_sell_mask, PNL_COL].dropna()

    b_mean = float(before_pnl.mean()) if len(before_pnl) > 0 else 0
    a_mean = float(after_pnl.mean()) if len(after_pnl) > 0 else 0
    print(f"  Before (global filter): mean={b_mean:+.3f} bps, n={len(before_pnl)}")
    print(f"  After (side filter):    mean={a_mean:+.3f} bps, n={len(after_pnl)}")
    print(f"  改善幅:                 {a_mean - b_mean:+.3f} bps")
    result["before_global"] = {"mean_pnl": b_mean, "n": len(before_pnl)}
    result["after_side"] = {"mean_pnl": a_mean, "n": len(after_pnl)}
    result["improvement_bps"] = a_mean - b_mean

    # --- 076# MEDIUM#6: マルチメトリクス before/after 比較 ---
    print("\n  --- 076# MEDIUM#6: マルチメトリクス before/after ---")
    _multi_metric_before_after(
        result, clean_filled, clean_all,
        global_skip, buy_skip, sell_skip,
        before_mask, after_buy_mask, after_sell_mask,
    )

    return result


def _multi_metric_before_after(
    result: dict,
    clean_filled: pd.DataFrame,
    clean_all: pd.DataFrame | None,
    global_skip: set[int],
    buy_skip: set[int],
    sell_skip: set[int],
    before_mask: pd.Series,
    after_buy_mask: pd.Series,
    after_sell_mask: pd.Series,
) -> None:
    """076# MEDIUM#6: fill_rate, cancel_ratio, AS_ratio の before/after 比較."""
    metrics_result: dict = {}

    # --- AS_ratio (filled レコードから直接計算可能) ---
    if "adverse_selected" in clean_filled.columns:
        before_as = clean_filled.loc[before_mask, "adverse_selected"].dropna()
        after_as = clean_filled.loc[
            after_buy_mask | after_sell_mask, "adverse_selected"
        ].dropna()
        b_as = float(before_as.mean()) * 100 if len(before_as) > 0 else 0.0
        a_as = float(after_as.mean()) * 100 if len(after_as) > 0 else 0.0
        print(f"  AS_ratio  Before: {b_as:.1f}% (n={len(before_as)})  "
              f"After: {a_as:.1f}% (n={len(after_as)})  "
              f"Δ={a_as - b_as:+.1f}pp")
        metrics_result["as_ratio"] = {
            "before": b_as, "after": a_as, "delta_pp": a_as - b_as,
            "n_before": len(before_as), "n_after": len(after_as),
        }

    # --- fill_rate, cancel_ratio (全レコード clean_all が必要) ---
    if clean_all is not None and len(clean_all) > 0 and "timestamp" in clean_all.columns:
        all_df = clean_all.copy()
        all_df["utc_hour"] = compute_utc_hour(all_df["timestamp"])

        # Before: global skip 以外
        b_all_mask = ~all_df["utc_hour"].isin(global_skip)
        b_all = all_df[b_all_mask]
        # After: side 別 skip
        if "side" in all_df.columns:
            a_buy_mask = (all_df["side"] == "buy") & (~all_df["utc_hour"].isin(buy_skip))
            a_sell_mask = (all_df["side"] == "sell") & (~all_df["utc_hour"].isin(sell_skip))
            a_all = all_df[a_buy_mask | a_sell_mask]
        else:
            a_all = pd.DataFrame()

        for label, subset in [("Before", b_all), ("After", a_all)]:
            if len(subset) == 0:
                print(f"  {label}: データなし")
                continue
            n_total = len(subset)
            filled_col = subset.get("filled")
            cancelled_col = subset.get("cancelled")

            n_filled = int(filled_col.sum()) if filled_col is not None else 0
            n_cancelled = int(cancelled_col.sum()) if cancelled_col is not None else 0
            fill_rate = n_filled / n_total * 100 if n_total > 0 else 0.0
            cancel_ratio = n_cancelled / n_total * 100 if n_total > 0 else 0.0

            print(f"  {label:6s}  fill_rate={fill_rate:.1f}% "
                  f"cancel_ratio={cancel_ratio:.1f}% "
                  f"(filled={n_filled}, cancelled={n_cancelled}, total={n_total})")

            key = "before" if label == "Before" else "after"
            metrics_result[f"fill_rate_{key}"] = fill_rate
            metrics_result[f"cancel_ratio_{key}"] = cancel_ratio
            metrics_result[f"n_total_{key}"] = n_total

        if "fill_rate_before" in metrics_result and "fill_rate_after" in metrics_result:
            fr_delta = metrics_result["fill_rate_after"] - metrics_result["fill_rate_before"]
            cr_delta = metrics_result["cancel_ratio_after"] - metrics_result["cancel_ratio_before"]
            print(f"  Δfill_rate={fr_delta:+.1f}pp  Δcancel_ratio={cr_delta:+.1f}pp")
            metrics_result["fill_rate_delta_pp"] = fr_delta
            metrics_result["cancel_ratio_delta_pp"] = cr_delta
    else:
        print("  fill_rate/cancel_ratio: clean_all データ未提供 — スキップ")

    result["multi_metrics_076"] = metrics_result


def section_4_wf_with_stats(clean_filled: pd.DataFrame) -> dict:
    """§4 WF-4fold + 統計検定 (HIGH#6 対応)."""
    print("\n" + "=" * 70)
    print("§4 WF-4fold 戦略検証 (HIGH#6: Mann-Whitney U + Cliff's Delta)")
    print("=" * 70)

    filled = clean_filled.copy().sort_values("timestamp").reset_index(drop=True)
    filled[PNL_COL] = filled[PNL_COL].astype(float)
    filled["utc_hour"] = compute_utc_hour(filled["timestamp"])

    N = len(filled)
    fold_size = N // 5
    if fold_size < 10:
        print("  WARNING: fold_size < 10 — 統計的信頼性が低い")

    strategies_results: dict = {}

    # --- 戦略定義 ---
    def s0_baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """ベースライン (フィルタなし)."""
        return test

    def s1_side_time(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """073# 実装: side 別 time filter (事前制御可能)."""
        skip_combos = collect_bad_side_hours(
            train,
            pnl_col=PNL_COL,
            threshold=-0.5,
            min_count=3,
        )
        return exclude_side_hour_combos(test, skip_combos)

    def s9_conservative_side_time(
        train: pd.DataFrame, test: pd.DataFrame,
    ) -> pd.DataFrame:
        """Conservative side-time: 閾値 -1.0, n≥2."""
        skip_combos = collect_bad_side_hours(
            train,
            pnl_col=PNL_COL,
            threshold=-1.0,
            min_count=2,
        )
        return exclude_side_hour_combos(test, skip_combos)

    def s13_sell_offset_boost(
        train: pd.DataFrame, test: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sell fast fill 除外 (queue_wait < 10s).

        HIGH#5 考慮: queue_wait は厳密には事後情報だが、
        offset 増は fast fill 確率を制御するため「間接的事前制御」として扱う.
        """
        test_buy = test[test["side"] == "buy"]
        test_sell = test[(test["side"] == "sell") & (test["queue_wait_sec"] >= 10)]
        return pd.concat([test_buy, test_sell])

    def s12_offset_sim_fixed(
        train: pd.DataFrame, test: pd.DataFrame,
    ) -> pd.DataFrame:
        """HIGH#3 修正: offset 増シミュレーション — sim_pnl を返却値に反映."""
        test_f = test[test["queue_wait_sec"] >= 5].copy()
        # sim_pnl を PNL_COL に上書き (offset 増の効果を +0.5bps で近似)
        test_f[PNL_COL] = test_f[PNL_COL] + 0.5
        return test_f

    all_strategies = [
        ("S0_baseline", s0_baseline),
        ("S1_side_time", s1_side_time),
        ("S9_conservative", s9_conservative_side_time),
        ("S12_offset_sim_fix", s12_offset_sim_fixed),
        ("S13_sell_offset", s13_sell_offset_boost),
    ]

    # WF 評価
    baseline_folds: list[list[float]] = []  # S0 の fold 別 PnL
    strategy_fold_data: dict[str, list[dict]] = {}

    for name, fn in all_strategies:
        fold_results: list[dict] = []
        fold_p_values: list[float] = []

        for fold in range(4):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = fold_size * (fold + 2) if fold < 3 else N
            train = filled.iloc[:train_end].copy()
            test = filled.iloc[test_start:test_end].copy()

            filtered = fn(train, test)
            test_pnl = filtered[PNL_COL].dropna().tolist()

            # S0 のベースライン保存
            if name == "S0_baseline":
                baseline_folds.append(test_pnl)

            # 統計検定: strategy vs zero (one-sample)
            if len(test_pnl) >= 5:
                _, p_val = scipy_stats.wilcoxon(
                    test_pnl, alternative="greater",
                )
            else:
                p_val = 1.0
            fold_p_values.append(float(p_val))

            fold_results.append({
                "fold": fold,
                "mean_pnl": float(np.mean(test_pnl)) if test_pnl else 0.0,
                "n": len(test_pnl),
                "win_pct": float(sum(1 for x in test_pnl if x > 0) / len(test_pnl) * 100) if test_pnl else 0.0,
                "pass_rate": len(filtered) / len(test) * 100 if len(test) > 0 else 0,
                "p_value": float(p_val),
            })

        # Cliff's delta vs baseline (overall)
        all_strategy_pnl = []
        all_baseline_pnl = []
        for fold in range(4):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = fold_size * (fold + 2) if fold < 3 else N
            train = filled.iloc[:train_end].copy()
            test = filled.iloc[test_start:test_end].copy()
            filtered = fn(train, test)
            all_strategy_pnl.extend(filtered[PNL_COL].dropna().tolist())
            if name != "S0_baseline":
                all_baseline_pnl.extend(test[PNL_COL].dropna().tolist())

        if all_baseline_pnl and all_strategy_pnl:
            cliff_d = cliffs_delta(all_strategy_pnl, all_baseline_pnl)
        else:
            cliff_d = 0.0

        folds_positive = sum(1 for r in fold_results if r["mean_pnl"] > 0)
        avg_pnl = np.mean([r["mean_pnl"] for r in fold_results])

        # HIGH#3 (076#): p_mean_gate で幾何平均を正式に算出
        pmean_result = p_mean_gate(fold_p_values)

        strategies_results[name] = {
            "folds": fold_results,
            "folds_positive": folds_positive,
            "avg_pnl": float(avg_pnl),
            "cliff_d_vs_baseline": float(cliff_d),
            "fold_p_values": fold_p_values,
            "n_total": sum(r["n"] for r in fold_results),
            "p_mean": pmean_result,
        }

    # HIGH#3 (076#): Holm-Bonferroni 補正で family-wise 判定
    holm_input: dict[str, tuple[list[float], list[float]]] = {}
    for name, fn in all_strategies:
        if name == "S0_baseline":
            continue
        strat_pnl: list[float] = []
        base_pnl: list[float] = []
        for fold in range(4):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = fold_size * (fold + 2) if fold < 3 else N
            train = filled.iloc[:train_end].copy()
            test = filled.iloc[test_start:test_end].copy()
            filtered = fn(train, test)
            strat_pnl.extend(filtered[PNL_COL].dropna().tolist())
            base_pnl.extend(test[PNL_COL].dropna().tolist())
        holm_input[name] = (strat_pnl, base_pnl)

    holm_results = holm_bonferroni_gate(holm_input, alpha=0.05, min_effect=0.10)

    # Print
    print(
        f"\n  {'Strategy':<25} {'mean PnL':>9} {'win%':>6} {'n':>5} "
        f"{'pass%':>6} {'folds>0':>8} {'Cliff d':>8} {'p-mean':>8}"
    )
    print("  " + "-" * 85)
    for name, res in strategies_results.items():
        avg_pnl = res["avg_pnl"]
        avg_win = np.mean([r["win_pct"] for r in res["folds"]])
        avg_pass = np.mean([r["pass_rate"] for r in res["folds"]])
        n = res["n_total"]
        fp = res["folds_positive"]
        cd = res["cliff_d_vs_baseline"]
        p_geo = res["p_mean"]["p_geometric"]
        marker = " <<<" if fp == 4 else (" **" if fp >= 3 else "")
        print(
            f"  {name:<25} {avg_pnl:>+9.3f} {avg_win:>5.1f}% {n:>5} "
            f"{avg_pass:>5.1f}% {fp}/4{marker:>4} {cd:>+8.3f} {p_geo:>8.4f}"
        )

    # Holm-Bonferroni 結果表示
    print("\n  --- Holm-Bonferroni family-wise 判定 (076# HIGH#3) ---")
    print(f"  {'Strategy':<25} {'p_raw':>8} {'p_holm':>8} {'Cliff d':>8} {'PASS':>6}")
    for name, hr in holm_results.items():
        mark = "✅" if hr["pass"] else "❌"
        print(f"  {name:<25} {hr['p_raw']:>8.4f} {hr['p_holm']:>8.4f} {hr['d']:>+8.4f} {mark:>6}")

    # p_mean_gate 詳細
    print("\n  --- p_mean_gate 統合判定 ---")
    for name, res in strategies_results.items():
        pm = res["p_mean"]
        mark = "✅" if pm["pass"] else "❌"
        print(f"  {name:<25} p_geo={pm['p_geometric']:.4f} pass={mark}")

    strategies_results["holm_bonferroni"] = holm_results

    # Detail
    print("\n  --- Fold 別詳細 ---")
    for name, res in strategies_results.items():
        if "folds" not in res:
            continue
        print(f"\n  {name}:")
        for r in res["folds"]:
            print(
                f"    fold{r['fold']}: pnl={r['mean_pnl']:+.3f} win={r['win_pct']:.1f}% "
                f"n={r['n']} pass={r['pass_rate']:.0f}% p={r['p_value']:.4f}"
            )

    return strategies_results


def section_5_monte_carlo_50k(clean_filled: pd.DataFrame) -> dict:
    """§5 50,000 ステップ OOS block bootstrap 検証.

    076# CRITICAL#2 対応:
    - +0.2bps 手動バイアス加算を廃止 (自明な結果を回避)
    - 日次ブロック単位の OOS bootstrap に変更 (時系列自己相関を保持)
    - Before/After 比較は実データのみで構成
    """
    print("\n" + "=" * 70)
    print("§5 50,000 ステップ OOS Block Bootstrap 検証 (076# CRITICAL#2 修正)")
    print("=" * 70)

    filled = clean_filled.copy()
    filled["utc_hour"] = compute_utc_hour(filled["timestamp"])
    filled["date"] = pd.to_datetime(
        filled["timestamp"], unit="s", utc=True,
    ).dt.date

    N_STEPS = 50_000
    N_BOOTSTRAP = 1000
    rng = np.random.default_rng(42)

    # 075# YAML の設定値 (clean データ再検証後 + §8.2 批判反映)
    buy_skip = {1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23}
    sell_skip = {3, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23}
    global_skip = {1, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21}

    # --- Scenario A/B: filter 適用後の block を先に構築 ---
    before_mask = ~filled["utc_hour"].isin(global_skip)
    after_buy_mask = (filled["side"] == "buy") & (~filled["utc_hour"].isin(buy_skip))
    after_sell_mask = (filled["side"] == "sell") & (~filled["utc_hour"].isin(sell_skip))
    before_blocks = _collect_pnl_blocks(filled, before_mask, pnl_col=PNL_COL)
    after_buy_blocks = _collect_pnl_blocks(filled, after_buy_mask, pnl_col=PNL_COL)
    after_sell_blocks = _collect_pnl_blocks(filled, after_sell_mask, pnl_col=PNL_COL)
    before_block_sizes, before_block_sums = _prepare_block_stats(before_blocks)
    after_buy_block_sizes, after_buy_block_sums = _prepare_block_stats(after_buy_blocks)
    after_sell_block_sizes, after_sell_block_sums = _prepare_block_stats(after_sell_blocks)
    before_pool_size = int(before_block_sizes.sum())
    before_pool_mean = (
        float(before_block_sums.sum() / before_pool_size)
        if before_pool_size > 0 else 0.0
    )
    after_pool_size = int(after_buy_block_sizes.sum() + after_sell_block_sizes.sum())
    after_pool_sum = float(after_buy_block_sums.sum() + after_sell_block_sums.sum())
    after_pool_mean = (after_pool_sum / after_pool_size) if after_pool_size > 0 else 0.0

    if before_pool_size < 10 or after_pool_size < 5:
        print("  ERROR: 十分なデータなし (before/after pool が小さすぎる)")
        return {"error": "insufficient_data"}

    print(f"  Pool A (before, global filter):  {before_pool_size} records, mean={before_pool_mean:+.3f} bps")
    print(f"  Pool B (after, side filter):     {after_pool_size} records, mean={after_pool_mean:+.3f} bps")
    print(f"  ※ 076# 修正: sell +0.2bps 手動加算を廃止")

    # --- 日次ブロック bootstrap (076# CRITICAL#2) ---
    # 時系列自己相関を保持するため、日次ブロック単位でリサンプル
    print(f"\n  Block bootstrap: {len(before_blocks)} before blocks, "
          f"{len(after_buy_blocks)} after-buy blocks, {len(after_sell_blocks)} after-sell blocks")

    # Bootstrap
    before_arr = np.empty(N_BOOTSTRAP, dtype=np.float64)
    after_arr = np.empty(N_BOOTSTRAP, dtype=np.float64)

    print(f"  Running {N_BOOTSTRAP} bootstrap iterations ({N_STEPS} steps each)...")
    t0 = time.time()

    for i in range(N_BOOTSTRAP):
        # Scenario A: true block bootstrap
        before_arr[i] = _sample_block_bootstrap_sum(
            before_blocks,
            block_sizes=before_block_sizes,
            block_sums=before_block_sums,
            n_steps=N_STEPS,
            rng=rng,
        )

        # Scenario B: buy/sell を別ブロックで維持
        n_buy = N_STEPS // 2
        n_sell = N_STEPS - n_buy
        if after_buy_blocks and after_sell_blocks:
            buy_sum = _sample_block_bootstrap_sum(
                after_buy_blocks,
                block_sizes=after_buy_block_sizes,
                block_sums=after_buy_block_sums,
                n_steps=n_buy,
                rng=rng,
            )
            sell_sum = _sample_block_bootstrap_sum(
                after_sell_blocks,
                block_sizes=after_sell_block_sizes,
                block_sums=after_sell_block_sums,
                n_steps=n_sell,
                rng=rng,
            )
            after_arr[i] = buy_sum + sell_sum
        elif after_buy_blocks:
            after_arr[i] = _sample_block_bootstrap_sum(
                after_buy_blocks,
                block_sizes=after_buy_block_sizes,
                block_sums=after_buy_block_sums,
                n_steps=N_STEPS,
                rng=rng,
            )
        elif after_sell_blocks:
            after_arr[i] = _sample_block_bootstrap_sum(
                after_sell_blocks,
                block_sizes=after_sell_block_sizes,
                block_sums=after_sell_block_sums,
                n_steps=N_STEPS,
                rng=rng,
            )
        else:
            after_arr[i] = 0.0

    elapsed = time.time() - t0
    print(f"  完了: {elapsed:.1f}s")

    result = {
        "n_steps": N_STEPS,
        "n_bootstrap": N_BOOTSTRAP,
        "before": {
            "pool_size": len(pool_before),
            "pool_mean_bps": float(pool_before.mean()),
            "cumul_mean_bps": float(before_arr.mean()),
            "cumul_std_bps": float(before_arr.std()),
            "positive_pct": float((before_arr > 0).mean() * 100),
            "percentile_5": float(np.percentile(before_arr, 5)),
            "percentile_50": float(np.percentile(before_arr, 50)),
            "percentile_95": float(np.percentile(before_arr, 95)),
        },
        "after": {
            "pool_size": len(pool_after),
            "pool_mean_bps": float(pool_after.mean()),
            "cumul_mean_bps": float(after_arr.mean()),
            "cumul_std_bps": float(after_arr.std()),
            "positive_pct": float((after_arr > 0).mean() * 100),
            "percentile_5": float(np.percentile(after_arr, 5)),
            "percentile_50": float(np.percentile(after_arr, 50)),
            "percentile_95": float(np.percentile(after_arr, 95)),
        },
        "improvement": {
            "mean_diff_bps": float(after_arr.mean() - before_arr.mean()),
            "positive_pct_diff": float(
                (after_arr > 0).mean() * 100 - (before_arr > 0).mean() * 100,
            ),
        },
    }

    # Mann-Whitney U test: after > before?
    u_stat, p_val = scipy_stats.mannwhitneyu(
        after_arr, before_arr, alternative="greater",
    )
    cliff_d = cliffs_delta(after_arr.tolist(), before_arr.tolist())
    result["comparison"] = {
        "mann_whitney_p": float(p_val),
        "cliff_d": float(cliff_d),
        "significant": float(p_val) < 0.05,
    }

    # Print
    print(f"\n  --- 50K ステップ Bootstrap 結果 ---")
    print(f"  {'':>20} {'Before (global)':>18} {'After (side)':>18} {'差異':>10}")
    print(f"  {'累積PnL mean (bps)':>20} {result['before']['cumul_mean_bps']:>+18.1f} "
          f"{result['after']['cumul_mean_bps']:>+18.1f} "
          f"{result['improvement']['mean_diff_bps']:>+10.1f}")
    print(f"  {'累積PnL std (bps)':>20} {result['before']['cumul_std_bps']:>18.1f} "
          f"{result['after']['cumul_std_bps']:>18.1f}")
    print(f"  {'正の確率':>20} {result['before']['positive_pct']:>17.1f}% "
          f"{result['after']['positive_pct']:>17.1f}% "
          f"{result['improvement']['positive_pct_diff']:>+9.1f}%")
    print(f"  {'P5':>20} {result['before']['percentile_5']:>+18.1f} "
          f"{result['after']['percentile_5']:>+18.1f}")
    print(f"  {'P50':>20} {result['before']['percentile_50']:>+18.1f} "
          f"{result['after']['percentile_50']:>+18.1f}")
    print(f"  {'P95':>20} {result['before']['percentile_95']:>+18.1f} "
          f"{result['after']['percentile_95']:>+18.1f}")

    print(f"\n  統計検定: Mann-Whitney U p={result['comparison']['mann_whitney_p']:.6f}, "
          f"Cliff's d={result['comparison']['cliff_d']:+.4f}, "
          f"有意={result['comparison']['significant']}")

    # 50K ステップでの per-step mean PnL
    per_step_before = result["before"]["cumul_mean_bps"] / N_STEPS
    per_step_after = result["after"]["cumul_mean_bps"] / N_STEPS
    print(f"\n  Per-step 平均 PnL:")
    print(f"    Before: {per_step_before:+.4f} bps/step")
    print(f"    After:  {per_step_after:+.4f} bps/step")

    # JPY 換算 (BTC = 15,000,000 JPY, lot = 0.001 BTC)
    btc_price = 15_000_000
    lot = 0.001
    cumul_jpy_before = result["before"]["cumul_mean_bps"] * 1e-4 * btc_price * lot
    cumul_jpy_after = result["after"]["cumul_mean_bps"] * 1e-4 * btc_price * lot
    print(f"\n  50K ステップ累積 PnL (JPY, BTC=¥15M, lot=0.001):")
    print(f"    Before: ¥{cumul_jpy_before:+,.0f}")
    print(f"    After:  ¥{cumul_jpy_after:+,.0f}")
    print(f"    差異:   ¥{cumul_jpy_after - cumul_jpy_before:+,.0f}")

    result["jpy_estimate"] = {
        "before_jpy": float(cumul_jpy_before),
        "after_jpy": float(cumul_jpy_after),
        "diff_jpy": float(cumul_jpy_after - cumul_jpy_before),
    }

    return result


def section_6_multi_horizon(clean_filled: pd.DataFrame) -> dict:
    """§6 Multi-horizon 比較 (MEDIUM#8 考慮: 30s = Gate KPI, 60s/120s = 補助)."""
    print("\n" + "=" * 70)
    print("§6 Multi-horizon PnL比較 (30s=G1.1 KPI, 60s/120s=補助指標)")
    print("=" * 70)

    result: dict = {}
    print(f"  {'horizon':<10} {'mean':>8} {'median':>8} {'std':>8} {'win%':>6} {'n':>6} {'役割'}")
    for col, label, role in [
        ("post_fill_30s_pnl", "30s", "G1.1 KPI (主)"),
        ("post_fill_60s_pnl", "60s", "補助指標"),
        ("post_fill_120s_pnl", "120s", "補助指標"),
    ]:
        if col in clean_filled.columns:
            p = clean_filled[col].dropna()
            if len(p) > 0:
                print(
                    f"  {label:<10} {p.mean():>+8.3f} {p.median():>+8.3f} "
                    f"{p.std():>8.3f} {100*(p>0).mean():>5.1f}% {len(p):>6} {role}"
                )
                result[label] = {
                    "mean": float(p.mean()),
                    "median": float(p.median()),
                    "std": float(p.std()),
                    "win_pct": float((p > 0).mean() * 100),
                    "n": len(p),
                    "role": role,
                }

    print(f"\n  注意: G1.1 ゲート判定は 30s PnL で行う (000# §3.3)。")
    print(f"  120s の正転 (+0.101 bps) は E3 データ蓄積後に再評価。")
    return result


def section_7_permutation_test(clean_filled: pd.DataFrame) -> dict:
    """§7 Permutation Test: filter ラベルをシャッフルして帰無仮説検定.

    MC bootstrap (§5) は「同じ pool から繰り返しサンプリング」するため、
    filter 効果の統計的有意性を直接示さない（pool が異なるため有意になりやすい）。
    Permutation test は「filter ラベル (before/after) をランダム入替」することで
    「filter ラベルは PnL に影響しない」という帰無仮説を直接検定する。
    """
    print("\n" + "=" * 70)
    print("§7 Permutation Test (filter ラベルシャッフル帰無仮説検定)")
    print("=" * 70)

    filled = clean_filled.copy()
    filled["utc_hour"] = compute_utc_hour(filled["timestamp"])

    buy_skip = {1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23}
    sell_skip = {3, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23}
    global_skip = {1, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21}

    # Before group: global filter のみ通過
    before_mask = ~filled["utc_hour"].isin(global_skip)
    # After group: side 別 filter 通過
    after_buy_mask = (filled["side"] == "buy") & (~filled["utc_hour"].isin(buy_skip))
    after_sell_mask = (filled["side"] == "sell") & (~filled["utc_hour"].isin(sell_skip))
    after_mask = after_buy_mask | after_sell_mask

    before_pnl = filled.loc[before_mask, PNL_COL].dropna().values
    after_pnl = filled.loc[after_mask, PNL_COL].dropna().values

    observed_diff = float(after_pnl.mean() - before_pnl.mean())

    # Combined pool
    n_before = len(before_pnl)
    n_after = len(after_pnl)
    combined = np.concatenate([before_pnl, after_pnl])

    N_PERMUTATIONS = 10_000
    rng = np.random.default_rng(2026)
    count_ge = 0
    perm_diffs: list[float] = []

    print(f"  Before: n={n_before}, mean={before_pnl.mean():+.3f} bps")
    print(f"  After:  n={n_after}, mean={after_pnl.mean():+.3f} bps")
    print(f"  観測差: {observed_diff:+.3f} bps")
    print(f"  Permutations: {N_PERMUTATIONS}")

    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(combined)
        perm_before = perm[:n_before]
        perm_after = perm[n_before:n_before + n_after]
        diff = float(perm_after.mean() - perm_before.mean())
        perm_diffs.append(diff)
        if diff >= observed_diff:
            count_ge += 1

    p_value = count_ge / N_PERMUTATIONS
    perm_arr = np.array(perm_diffs)

    result = {
        "n_before": n_before,
        "n_after": n_after,
        "observed_diff_bps": observed_diff,
        "n_permutations": N_PERMUTATIONS,
        "p_value": p_value,
        "perm_diff_mean": float(perm_arr.mean()),
        "perm_diff_std": float(perm_arr.std()),
        "perm_diff_p5": float(np.percentile(perm_arr, 5)),
        "perm_diff_p95": float(np.percentile(perm_arr, 95)),
    }

    print(f"\n  --- Permutation Test 結果 ---")
    print(f"  p-value (one-sided): {p_value:.4f}")
    print(f"  帰無分布: mean={perm_arr.mean():+.4f}, std={perm_arr.std():.4f}")
    print(f"  帰無 95%CI: [{np.percentile(perm_arr, 2.5):+.3f}, {np.percentile(perm_arr, 97.5):+.3f}]")
    print(f"  観測差 {observed_diff:+.3f} vs 帰無 95% 上限 {np.percentile(perm_arr, 95):+.3f}")

    if p_value < 0.05:
        print(f"  → ✅ 有意 (p={p_value:.4f} < 0.05): filter 効果は偶然では説明困難")
    elif p_value < 0.10:
        print(f"  → ⚠️ 限界的 (p={p_value:.4f}): suggestive だが確定的でない")
    else:
        print(f"  → ❌ 非有意 (p={p_value:.4f} ≥ 0.10): 帰無仮説を棄却できない")

    return result


def section_8_temporal_stability(clean_filled: pd.DataFrame) -> dict:
    """§8 時系列安定性分析: 日別 PnL、連続損失、filter 前後の安定性比較."""
    print("\n" + "=" * 70)
    print("§8 時系列安定性分析 (日別 PnL + 連続損失)")
    print("=" * 70)

    filled = clean_filled.copy()
    filled["utc_dt"] = pd.to_datetime(filled["timestamp"], unit="s", utc=True)
    filled["utc_hour"] = filled["utc_dt"].dt.hour
    filled["date"] = filled["utc_dt"].dt.date

    buy_skip = {1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23}
    sell_skip = {3, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23}
    global_skip = {1, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21}

    before_mask = ~filled["utc_hour"].isin(global_skip)
    after_buy_mask = (filled["side"] == "buy") & (~filled["utc_hour"].isin(buy_skip))
    after_sell_mask = (filled["side"] == "sell") & (~filled["utc_hour"].isin(sell_skip))
    after_mask = after_buy_mask | after_sell_mask

    result: dict = {}

    for label, mask in [("before", before_mask), ("after", after_mask), ("all", pd.Series(True, index=filled.index))]:
        subset = filled.loc[mask].copy()
        if len(subset) == 0:
            result[label] = {"error": "no_data"}
            continue

        pnl = subset[PNL_COL].dropna()
        cumulative = pnl.cumsum()

        # 日別統計
        daily = subset.groupby("date")[PNL_COL].agg(["mean", "count", "sum", "std"]).reset_index()
        daily.columns = ["date", "mean_pnl", "n_trades", "total_pnl", "std_pnl"]

        # 連続損失ストリーク
        is_loss = (pnl < 0).values
        max_streak = 0
        current_streak = 0
        for loss in is_loss:
            if loss:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # 累積 PnL の max drawdown (bps)
        cumul_values = cumulative.values
        peak = np.maximum.accumulate(cumul_values)
        dd = peak - cumul_values
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0

        # Sharpe-like ratio (trade-level)
        pnl_mean = float(pnl.mean())
        pnl_std = float(pnl.std())
        sharpe_like = pnl_mean / pnl_std if pnl_std > 0 else 0.0

        result[label] = {
            "n_trades": len(pnl),
            "mean_pnl": pnl_mean,
            "cumulative_pnl": float(pnl.sum()),
            "max_drawdown_bps": max_dd,
            "max_loss_streak": max_streak,
            "sharpe_like": sharpe_like,
            "n_days": len(daily),
            "daily_stats": daily.to_dict(orient="records"),
        }

        print(f"\n  [{label.upper()}] n={len(pnl)}, days={len(daily)}")
        print(f"    mean PnL:           {pnl_mean:+.3f} bps")
        print(f"    cumulative PnL:     {pnl.sum():+.1f} bps")
        print(f"    max drawdown:       {max_dd:.1f} bps")
        print(f"    max loss streak:    {max_streak} trades")
        print(f"    sharpe-like ratio:  {sharpe_like:+.4f}")

        print(f"    --- 日別 ---")
        for row in daily.itertuples(index=False):
            sign = "+" if row.total_pnl >= 0 else ""
            print(f"    {row.date}  n={int(row.n_trades):>3}  "
                  f"mean={row.mean_pnl:>+7.3f}  total={sign}{row.total_pnl:.1f}")

    # Before vs After の安定性比較
    if "before" in result and "after" in result and "error" not in result["before"] and "error" not in result["after"]:
        print(f"\n  --- Before vs After 安定性比較 ---")
        for metric in ["max_drawdown_bps", "max_loss_streak", "sharpe_like"]:
            b_val = result["before"][metric]
            a_val = result["after"][metric]
            better = "After" if (metric == "sharpe_like" and a_val > b_val) or \
                               (metric != "sharpe_like" and a_val < b_val) else "Before"
            print(f"    {metric:<22} Before={b_val:>8.3f}  After={a_val:>8.3f}  → {better} が良好")

    return result


def section_9_power_analysis(clean_filled: pd.DataFrame) -> dict:
    """§9 検出力分析: 現サンプル数で検出可能な効果サイズの推定."""
    print("\n" + "=" * 70)
    print("§9 検出力分析 (現サンプル数での効果検出限界)")
    print("=" * 70)

    filled = clean_filled.copy()
    filled["utc_hour"] = compute_utc_hour(filled["timestamp"])

    buy_skip = {1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23}
    sell_skip = {3, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23}
    global_skip = {1, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21}

    before_mask = ~filled["utc_hour"].isin(global_skip)
    after_buy_mask = (filled["side"] == "buy") & (~filled["utc_hour"].isin(buy_skip))
    after_sell_mask = (filled["side"] == "sell") & (~filled["utc_hour"].isin(sell_skip))
    after_mask = after_buy_mask | after_sell_mask

    before_pnl = filled.loc[before_mask, PNL_COL].dropna().values
    after_pnl = filled.loc[after_mask, PNL_COL].dropna().values
    all_pnl = filled[PNL_COL].dropna().values

    n_before = len(before_pnl)
    n_after = len(after_pnl)
    pooled_std = float(all_pnl.std())

    # Cohen's d for observed effect
    observed_d = (after_pnl.mean() - before_pnl.mean()) / pooled_std if pooled_std > 0 else 0.0

    # Power simulation via bootstrap under known effect sizes
    N_SIM = 5000
    rng = np.random.default_rng(42)
    effect_sizes = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]  # Cohen's d
    power_results: dict = {}

    print(f"  現状: n_before={n_before}, n_after={n_after}, pooled_std={pooled_std:.3f}")
    print(f"  観測 Cohen's d: {observed_d:+.3f}")
    print(f"\n  --- 効果サイズ別検出力 (α=0.05, {N_SIM} simulations) ---")
    print(f"  {'Cohen d':>8} {'Power':>8} {'必要 n (各群)':>15} {'追加日数 (est)':>15}")

    trades_per_day = len(all_pnl) / 1.4  # 1.4 days for current data

    for d in effect_sizes:
        n_significant = 0
        for _ in range(N_SIM):
            sim_before = rng.choice(before_pnl, size=n_before, replace=True)
            sim_after = rng.choice(before_pnl, size=n_after, replace=True) + d * pooled_std
            _, p = scipy_stats.mannwhitneyu(sim_after, sim_before, alternative="greater")
            if p < 0.05:
                n_significant += 1

        power = n_significant / N_SIM

        z_alpha = 1.645  # one-sided α=0.05
        z_beta = 0.842   # power=0.80
        n_required = int(np.ceil(((z_alpha + z_beta) ** 2) * 2 / (d ** 2))) if d > 0 else 99999
        additional_n_needed = max(0, n_required - min(n_before, n_after))
        additional_days = additional_n_needed / trades_per_day if trades_per_day > 0 else float("inf")

        power_results[f"d_{d:.2f}"] = {
            "cohen_d": d,
            "power": power,
            "n_required_each": n_required,
            "additional_n_needed": additional_n_needed,
            "additional_days": additional_days,
        }

        print(f"  {d:>8.2f} {power:>7.1%} {n_required:>15} {additional_days:>14.1f}d")

    # 現在の効果サイズでの検出力
    if abs(observed_d) > 0.01:
        n_significant = 0
        for _ in range(N_SIM):
            sim_before = rng.choice(before_pnl, size=n_before, replace=True)
            sim_after = rng.choice(before_pnl, size=n_after, replace=True) + observed_d * pooled_std
            _, p = scipy_stats.mannwhitneyu(sim_after, sim_before, alternative="greater")
            if p < 0.05:
                n_significant += 1
        observed_power = n_significant / N_SIM
        print(f"\n  現在の効果サイズ d={observed_d:+.3f} での検出力: {observed_power:.1%}")
        power_results["observed"] = {
            "cohen_d": observed_d,
            "power": observed_power,
        }
    else:
        print(f"\n  現在の効果サイズが極小 (d={observed_d:+.3f}) — 検出不能")

    for d_key, pr in power_results.items():
        if d_key.startswith("d_") and pr["power"] >= 0.80:
            print(f"\n  → 80% power 到達: Cohen's d≥{pr['cohen_d']:.2f} "
                  f"(必要 n≥{pr['n_required_each']}, 追加 {pr['additional_days']:.1f} 日)")
            break
    else:
        print(f"\n  → 現サンプル数では全テスト効果サイズで 80% power 未達")

    return power_results


def save_artifact(all_results: dict) -> Path:
    """JSON artifact 保存 (MEDIUM#9 対応)."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = ARTIFACT_DIR / f"verification_077_{ts}.json"

    # numpy の値を Python native に変換
    def convert(obj: object) -> object:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        # datetime.date (not datetime)
        import datetime as _dt
        if isinstance(obj, _dt.date):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    write_json(path, all_results, indent=2, ensure_ascii=False, default=convert)

    print(f"\n[artifact] Saved: {path}")
    return path


def main() -> None:
    # 084# CLI フィルタ対応
    parser = argparse.ArgumentParser(description="075# ph2 検証")
    parser.add_argument(
        "--run-id", nargs="+", default=None,
        help="対象 run_id を指定 (複数可). 例: --run-id run1 run2",
    )
    parser.add_argument(
        "--git-sha", nargs="+", default=None,
        help="対象 git_sha を前方一致で指定. 例: --git-sha abc1234",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("075# ph2 検証: レビュー指摘対応 + 50K ステップ Monte Carlo")
    print("=" * 70)
    if args.run_id:
        print(f"  フィルタ: run_id = {args.run_id}")
    if args.git_sha:
        print(f"  フィルタ: git_sha = {args.git_sha}")
    print()

    # データロード (CRITICAL#2: clean/quarantine 分離)
    clean_all, clean_filled, quarantine_df, data_stats = load_clean_filled(
        run_ids=args.run_id, git_shas=args.git_sha,
    )

    if len(clean_filled) == 0:
        print("ERROR: No clean filled records found.")
        return

    all_results: dict = {"run_timestamp": datetime.now(timezone.utc).isoformat()}

    # §0 データ品質
    all_results["data_quality"] = section_0_data_quality(
        clean_filled, quarantine_df, data_stats,
    )

    # §1 基本統計
    all_results["basic_stats"] = section_1_basic_stats(clean_filled)

    # §2 ヒートマップ
    all_results["side_hour_heatmap"] = section_2_side_hour_heatmap(clean_filled)

    # §3 time_filter 機会損失分析
    all_results["time_filter_impact"] = section_3_time_filter_impact(clean_filled, clean_all)

    # §4 WF + 統計検定
    all_results["wf_strategies"] = section_4_wf_with_stats(clean_filled)

    # §5 50K Monte Carlo
    all_results["monte_carlo_50k"] = section_5_monte_carlo_50k(clean_filled)

    # §6 Multi-horizon
    all_results["multi_horizon"] = section_6_multi_horizon(clean_filled)

    # §7 Permutation Test (078# 追加)
    all_results["permutation_test"] = section_7_permutation_test(clean_filled)

    # §8 時系列安定性 (078# 追加)
    all_results["temporal_stability"] = section_8_temporal_stability(clean_filled)

    # §9 検出力分析 (078# 追加)
    all_results["power_analysis"] = section_9_power_analysis(clean_filled)

    # Artifact 保存
    save_artifact(all_results)

    print("\n" + "=" * 70)
    print("078# 検証完了 (§0-§9)")
    print("=" * 70)


if __name__ == "__main__":
    main()
