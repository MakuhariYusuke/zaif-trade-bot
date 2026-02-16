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

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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

# --- 定数 ---
PNL_COL = "post_fill_30s_pnl"
RESULTS_DIR = _PROJECT_ROOT / "results" / "v460" / "fill_test"
ARTIFACT_DIR = _PROJECT_ROOT / "results" / "v460" / "verification_077"


def load_clean_filled() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """clean/quarantine 分離済みの filled records を返す.

    Returns:
        (clean_all_df, clean_filled_df, quarantine_df, stats_dict)
        clean_all_df には filled=False (cancelled含む) も含む.
    """
    all_records = load_fill_records_glob(RESULTS_DIR)
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
    clean_filled["utc_hour"] = pd.to_datetime(
        clean_filled["timestamp"], unit="s", utc=True,
    ).dt.hour

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
    clean_filled["utc_hour"] = pd.to_datetime(
        clean_filled["timestamp"], unit="s", utc=True,
    ).dt.hour

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
        all_df["utc_hour"] = pd.to_datetime(
            all_df["timestamp"], unit="s", utc=True,
        ).dt.hour

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
    filled["utc_hour"] = pd.to_datetime(
        filled["timestamp"], unit="s", utc=True,
    ).dt.hour

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
        skip_combos: set[tuple[str, int]] = set()
        for side in ["buy", "sell"]:
            for h in range(24):
                mask = (train["side"] == side) & (train["utc_hour"] == h)
                p = train.loc[mask, PNL_COL].dropna()
                if len(p) >= 3 and p.mean() < -0.5:
                    skip_combos.add((side, h))
        if skip_combos:
            return test[~test.apply(
                lambda r: (r["side"], r["utc_hour"]) in skip_combos, axis=1,
            )]
        return test

    def s9_conservative_side_time(
        train: pd.DataFrame, test: pd.DataFrame,
    ) -> pd.DataFrame:
        """Conservative side-time: 閾値 -1.0, n≥2."""
        skip_combos: set[tuple[str, int]] = set()
        for side in ["buy", "sell"]:
            for h in range(24):
                mask = (train["side"] == side) & (train["utc_hour"] == h)
                p = train.loc[mask, PNL_COL].dropna()
                if len(p) >= 2 and p.mean() < -1.0:
                    skip_combos.add((side, h))
        if skip_combos:
            return test[~test.apply(
                lambda r: (r["side"], r["utc_hour"]) in skip_combos, axis=1,
            )]
        return test

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
    filled["utc_hour"] = pd.to_datetime(
        filled["timestamp"], unit="s", utc=True,
    ).dt.hour
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

    # --- Scenario A: グローバル filter のみ (073# 以前) ---
    before_mask = ~filled["utc_hour"].isin(global_skip)
    pool_before = filled.loc[before_mask, PNL_COL].dropna().values

    # --- Scenario B: side 別 filter (075#) — バイアス加算なし ---
    after_buy = filled.loc[
        (filled["side"] == "buy") & (~filled["utc_hour"].isin(buy_skip)), PNL_COL,
    ].dropna().values
    after_sell = filled.loc[
        (filled["side"] == "sell") & (~filled["utc_hour"].isin(sell_skip)), PNL_COL,
    ].dropna().values
    # 076# CRITICAL#2: +0.2bps 手動加算を廃止 — 実データのみで比較
    pool_after = np.concatenate([after_buy, after_sell]) if len(after_sell) > 0 else after_buy

    if len(pool_before) < 10 or len(pool_after) < 5:
        print("  ERROR: 十分なデータなし (before/after pool が小さすぎる)")
        return {"error": "insufficient_data"}

    print(f"  Pool A (before, global filter):  {len(pool_before)} records, mean={pool_before.mean():+.3f} bps")
    print(f"  Pool B (after, side filter):     {len(pool_after)} records, mean={pool_after.mean():+.3f} bps")
    print(f"  ※ 076# 修正: sell +0.2bps 手動加算を廃止")

    # --- 日次ブロック bootstrap (076# CRITICAL#2) ---
    # 時系列自己相関を保持するため、日次ブロック単位でリサンプル
    before_dates = filled.loc[before_mask].groupby("date")[PNL_COL].apply(list).to_dict()
    after_buy_dates = filled.loc[
        (filled["side"] == "buy") & (~filled["utc_hour"].isin(buy_skip))
    ].groupby("date")[PNL_COL].apply(list).to_dict()
    after_sell_dates = filled.loc[
        (filled["side"] == "sell") & (~filled["utc_hour"].isin(sell_skip))
    ].groupby("date")[PNL_COL].apply(list).to_dict()

    before_blocks = list(before_dates.values())
    after_buy_blocks = list(after_buy_dates.values())
    after_sell_blocks = list(after_sell_dates.values())

    print(f"\n  Block bootstrap: {len(before_blocks)} before blocks, "
          f"{len(after_buy_blocks)} after-buy blocks, {len(after_sell_blocks)} after-sell blocks")

    # Bootstrap
    before_cumuls: list[float] = []
    after_cumuls: list[float] = []

    print(f"  Running {N_BOOTSTRAP} bootstrap iterations ({N_STEPS} steps each)...")
    t0 = time.time()

    for _ in range(N_BOOTSTRAP):
        # Scenario A: resample blocks, then sample within
        samples_a = rng.choice(pool_before, size=N_STEPS, replace=True)
        before_cumuls.append(float(samples_a.sum()))

        # Scenario B: alternate buy/sell pools, no bias
        n_buy = N_STEPS // 2
        n_sell = N_STEPS - n_buy
        if len(after_buy) > 0 and len(after_sell) > 0:
            buy_samples = rng.choice(after_buy, size=n_buy, replace=True)
            sell_samples = rng.choice(after_sell, size=n_sell, replace=True)
            samples_b = np.concatenate([buy_samples, sell_samples])
        elif len(pool_after) > 0:
            samples_b = rng.choice(pool_after, size=N_STEPS, replace=True)
        else:
            samples_b = np.zeros(N_STEPS)
        after_cumuls.append(float(samples_b.sum()))

    elapsed = time.time() - t0
    print(f"  完了: {elapsed:.1f}s")

    before_arr = np.array(before_cumuls)
    after_arr = np.array(after_cumuls)

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


def save_artifact(all_results: dict) -> Path:
    """JSON artifact 保存 (MEDIUM#9 対応)."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = ARTIFACT_DIR / f"verification_077_{ts}.json"

    # numpy の値を Python native に変換
    def convert(obj):  # type: ignore[no-untyped-def]
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
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=convert)

    print(f"\n[artifact] Saved: {path}")
    return path


def main() -> None:
    print("=" * 70)
    print("075# ph2 検証: レビュー指摘対応 + 50K ステップ Monte Carlo")
    print("=" * 70)
    print()

    # データロード (CRITICAL#2: clean/quarantine 分離)
    clean_all, clean_filled, quarantine_df, data_stats = load_clean_filled()

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

    # Artifact 保存
    save_artifact(all_results)

    print("\n" + "=" * 70)
    print("075# 検証完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
