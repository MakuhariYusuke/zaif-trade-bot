"""073# ph2 戦略分析: fill records から高収益パラメータ戦略を探索.

全 fill records を読み込み、以下を分析:
1. 基本統計 (PnL, AS, fill_rate, side/regime/hour 別)
2. 条件別セグメント PnL (queue_wait, spread, offset, regime, side×hour)
3. Walk-Forward シミュレーション (WF-4fold) で戦略候補を評価
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))


def load_all_records() -> pd.DataFrame:
    """全 fill_records_*.jsonl + emergency を読み込み."""
    results_dir = _PROJECT_ROOT / "results" / "v460" / "fill_test"
    records: list[dict] = []
    for pattern in ["fill_records_*.jsonl", "emergency/*.jsonl"]:
        for f in sorted(results_dir.glob(pattern)):
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    df = pd.DataFrame(records)
    # 重複排除 (cycle_id ベース)
    if "cycle_id" in df.columns:
        df = df.drop_duplicates(subset="cycle_id", keep="last")
    return df


def basic_stats(df: pd.DataFrame) -> None:
    """§1 基本統計."""
    print("=" * 70)
    print("§1 基本統計")
    print("=" * 70)

    total = len(df)
    filled = df[df["filled"] == True]
    cancelled = df[df["cancelled"] == True]
    skipped = df[df.get("skip_gate_skipped", pd.Series(dtype=bool)) == True] if "skip_gate_skipped" in df.columns else pd.DataFrame()

    print(f"全レコード:       {total}")
    print(f"  約定 (filled):  {len(filled)} ({len(filled)/total*100:.1f}%)")
    print(f"  キャンセル:     {len(cancelled)} ({len(cancelled)/total*100:.1f}%)")
    print(f"  SkipGate skip:  {len(skipped)} ({len(skipped)/total*100:.1f}%)")

    # PnL 統計 (filled のみ)
    pnl_30 = filled["post_fill_30s_pnl"].dropna()
    pnl_60 = filled["post_fill_60s_pnl"].dropna() if "post_fill_60s_pnl" in filled.columns else pd.Series(dtype=float)
    pnl_120 = filled["post_fill_120s_pnl"].dropna() if "post_fill_120s_pnl" in filled.columns else pd.Series(dtype=float)

    print(f"\n--- PnL (bps) ---")
    for label, pnl in [("30s", pnl_30), ("60s", pnl_60), ("120s", pnl_120)]:
        if len(pnl) > 0:
            print(f"  {label}: mean={pnl.mean():.3f}, median={pnl.median():.3f}, "
                  f"std={pnl.std():.3f}, win%={100*(pnl>0).mean():.1f}%, n={len(pnl)}")

    # AS 統計
    as_col = filled["adverse_selected"].dropna() if "adverse_selected" in filled.columns else pd.Series(dtype=bool)
    if len(as_col) > 0:
        print(f"\n--- AS ---")
        print(f"  AS ratio: {as_col.mean()*100:.1f}% (n={len(as_col)})")

    # side 別
    print(f"\n--- Side 別 PnL (30s) ---")
    for side in ["buy", "sell"]:
        s = filled[filled["side"] == side]["post_fill_30s_pnl"].dropna()
        if len(s) > 0:
            print(f"  {side}: mean={s.mean():.3f}, win%={100*(s>0).mean():.1f}%, n={len(s)}")

    # regime 別
    if "regime" in filled.columns:
        print(f"\n--- Regime 別 PnL (30s) ---")
        for regime, grp in filled.groupby("regime"):
            p = grp["post_fill_30s_pnl"].dropna()
            if len(p) > 0:
                print(f"  {regime}: mean={p.mean():.3f}, win%={100*(p>0).mean():.1f}%, n={len(p)}")


def segment_analysis(df: pd.DataFrame) -> None:
    """§2 セグメント別分析."""
    print("\n" + "=" * 70)
    print("§2 セグメント別分析")
    print("=" * 70)

    filled = df[df["filled"] == True].copy()
    if len(filled) == 0:
        print("No filled records.")
        return

    pnl_col = "post_fill_30s_pnl"

    # UTC hour 別
    if "timestamp" in filled.columns:
        filled["utc_hour"] = pd.to_datetime(filled["timestamp"], unit="s", utc=True).dt.hour
        print("\n--- UTC Hour 別 PnL ---")
        hour_stats = []
        for h in range(24):
            hdf = filled[filled["utc_hour"] == h]
            p = hdf[pnl_col].dropna()
            if len(p) >= 3:
                hour_stats.append({
                    "hour": h, "n": len(p), "mean": p.mean(),
                    "win_pct": (p > 0).mean() * 100,
                    "jst": (h + 9) % 24,
                })
        if hour_stats:
            hdf = pd.DataFrame(hour_stats)
            for _, row in hdf.iterrows():
                marker = "***" if row["mean"] > 0.5 else ("---" if row["mean"] < -1.0 else "   ")
                print(f"  UTC{int(row['hour']):02d} (JST{int(row['jst']):02d}): "
                      f"mean={row['mean']:+.3f} win={row['win_pct']:.0f}% n={int(row['n'])} {marker}")

    # side × hour
    print("\n--- Side × UTC Hour PnL ---")
    for side in ["buy", "sell"]:
        sf = filled[filled["side"] == side]
        print(f"\n  [{side.upper()}]")
        for h in range(24):
            hdf = sf[sf["utc_hour"] == h]
            p = hdf[pnl_col].dropna()
            if len(p) >= 2:
                marker = "***" if p.mean() > 0.5 else ("---" if p.mean() < -1.0 else "   ")
                print(f"    UTC{h:02d} (JST{(h+9)%24:02d}): "
                      f"mean={p.mean():+.3f} win={100*(p>0).mean():.0f}% n={len(p)} {marker}")

    # queue_wait 別
    if "queue_wait_sec" in filled.columns:
        print("\n--- Queue Wait 別 PnL ---")
        bins = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 300)]
        for lo, hi in bins:
            mask = (filled["queue_wait_sec"] >= lo) & (filled["queue_wait_sec"] < hi)
            p = filled.loc[mask, pnl_col].dropna()
            if len(p) >= 3:
                print(f"  {lo:3d}-{hi:3d}s: mean={p.mean():+.3f} win={100*(p>0).mean():.0f}% n={len(p)}")

    # spread 別
    if "spread_at_order" in filled.columns:
        print("\n--- Spread 別 PnL ---")
        sp = filled["spread_at_order"].dropna()
        if len(sp) > 0:
            for q_lo, q_hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
                lo_val = sp.quantile(q_lo)
                hi_val = sp.quantile(q_hi)
                mask = (filled["spread_at_order"] >= lo_val) & (filled["spread_at_order"] < hi_val + 0.01)
                p = filled.loc[mask, pnl_col].dropna()
                if len(p) >= 3:
                    print(f"  Q{int(q_lo*4)+1} ({lo_val:.0f}-{hi_val:.0f} JPY): "
                          f"mean={p.mean():+.3f} win={100*(p>0).mean():.0f}% n={len(p)}")


def walk_forward_strategies(df: pd.DataFrame) -> None:
    """§3 Walk-Forward 4-fold 戦略シミュレーション.

    全 filled records を時系列4分割し、各 fold で train→test.
    train 期間で最適パラメータを学習し、test 期間で OOS 評価。
    """
    print("\n" + "=" * 70)
    print("§3 Walk-Forward 4-fold 戦略シミュレーション")
    print("=" * 70)

    filled = df[df["filled"] == True].copy()
    if len(filled) < 40:
        print("Not enough filled records for WF analysis.")
        return

    filled = filled.sort_values("timestamp").reset_index(drop=True)
    pnl_col = "post_fill_30s_pnl"
    filled[pnl_col] = filled[pnl_col].astype(float)

    # hour 情報
    filled["utc_hour"] = pd.to_datetime(filled["timestamp"], unit="s", utc=True).dt.hour

    N = len(filled)
    fold_size = N // 5  # 5 parts: 4 folds (train=3, test=1)

    strategies = {}

    # ===== S0: Baseline (全パス) =====
    s0_results = []
    for fold in range(4):
        test_start = fold_size * (fold + 1)
        test_end = fold_size * (fold + 2) if fold < 3 else N
        test = filled.iloc[test_start:test_end]
        test_pnl = test[pnl_col].dropna()
        s0_results.append({
            "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
            "win_pct": (test_pnl > 0).mean() * 100,
        })
    strategies["S0_baseline"] = s0_results

    # ===== S1: Side-specific time filter (train で side×hour 別 PnL 学習) =====
    s1_results = []
    for fold in range(4):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = fold_size * (fold + 2) if fold < 3 else N
        train = filled.iloc[:train_end]
        test = filled.iloc[test_start:test_end]

        # Train: side×hour で negative PnL の組み合わせを特定
        skip_combos = set()
        for side in ["buy", "sell"]:
            for h in range(24):
                mask = (train["side"] == side) & (train["utc_hour"] == h)
                p = train.loc[mask, pnl_col].dropna()
                if len(p) >= 3 and p.mean() < -0.5:
                    skip_combos.add((side, h))

        # Test: skip_combos に該当しないレコードのみ
        if skip_combos:
            test_filtered = test[~test.apply(
                lambda r: (r["side"], r["utc_hour"]) in skip_combos, axis=1
            )]
        else:
            test_filtered = test
        test_pnl = test_filtered[pnl_col].dropna()
        if len(test_pnl) > 0:
            s1_results.append({
                "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
                "win_pct": (test_pnl > 0).mean() * 100,
                "skip_combos": len(skip_combos),
                "pass_rate": len(test_filtered) / len(test) * 100 if len(test) > 0 else 0,
            })
    strategies["S1_side_time_filter"] = s1_results

    # ===== S2: Queue wait filter (slow fills only) =====
    s2_results = []
    for fold in range(4):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = fold_size * (fold + 2) if fold < 3 else N
        train = filled.iloc[:train_end]
        test = filled.iloc[test_start:test_end]

        # Train: queue_wait の最適下限を探索
        best_thr, best_pnl = 0, train[pnl_col].dropna().mean()
        for thr in [5, 10, 15, 20, 30, 60]:
            p = train.loc[train["queue_wait_sec"] >= thr, pnl_col].dropna()
            if len(p) >= 5 and p.mean() > best_pnl:
                best_pnl = p.mean()
                best_thr = thr

        # Test
        if best_thr > 0:
            test_filtered = test[test["queue_wait_sec"] >= best_thr]
        else:
            test_filtered = test
        test_pnl = test_filtered[pnl_col].dropna()
        if len(test_pnl) > 0:
            s2_results.append({
                "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
                "win_pct": (test_pnl > 0).mean() * 100,
                "thr": best_thr,
                "pass_rate": len(test_filtered) / len(test) * 100 if len(test) > 0 else 0,
            })
    strategies["S2_queue_wait_filter"] = s2_results

    # ===== S3: Spread adaptive filter (wide spread → better edge) =====
    s3_results = []
    for fold in range(4):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = fold_size * (fold + 2) if fold < 3 else N
        train = filled.iloc[:train_end]
        test = filled.iloc[test_start:test_end]

        # Train: spread 上位 50% が良いか下位 50% が良いか
        sp_med = train["spread_at_order"].dropna().median() if "spread_at_order" in train.columns else None
        use_wide = False
        if sp_med is not None and sp_med > 0:
            wide_pnl = train.loc[train["spread_at_order"] >= sp_med, pnl_col].dropna().mean()
            narrow_pnl = train.loc[train["spread_at_order"] < sp_med, pnl_col].dropna().mean()
            if wide_pnl > narrow_pnl:
                use_wide = True

        # Test
        if sp_med is not None and use_wide:
            test_filtered = test[test["spread_at_order"] >= sp_med]
        else:
            test_filtered = test
        test_pnl = test_filtered[pnl_col].dropna()
        if len(test_pnl) > 0:
            s3_results.append({
                "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
                "win_pct": (test_pnl > 0).mean() * 100,
                "use_wide": use_wide,
                "pass_rate": len(test_filtered) / len(test) * 100 if len(test) > 0 else 0,
            })
    strategies["S3_spread_filter"] = s3_results

    # ===== S4: Regime filter (ranging only) =====
    s4_results = []
    if "regime" in filled.columns:
        for fold in range(4):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = fold_size * (fold + 2) if fold < 3 else N
            train = filled.iloc[:train_end]
            test = filled.iloc[test_start:test_end]

            # Train: regime 別 PnL で正のものだけ残す
            good_regimes = set()
            for regime, grp in train.groupby("regime"):
                p = grp[pnl_col].dropna()
                if len(p) >= 3 and p.mean() > -0.3:
                    good_regimes.add(regime)

            # Test
            if good_regimes:
                test_filtered = test[test["regime"].isin(good_regimes)]
            else:
                test_filtered = test
            test_pnl = test_filtered[pnl_col].dropna()
            if len(test_pnl) > 0:
                s4_results.append({
                    "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
                    "win_pct": (test_pnl > 0).mean() * 100,
                    "good_regimes": list(good_regimes),
                    "pass_rate": len(test_filtered) / len(test) * 100 if len(test) > 0 else 0,
                })
        strategies["S4_regime_filter"] = s4_results

    # ===== S5: Combined (S1 + S4) =====
    s5_results = []
    if "regime" in filled.columns:
        for fold in range(4):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = fold_size * (fold + 2) if fold < 3 else N
            train = filled.iloc[:train_end]
            test = filled.iloc[test_start:test_end]

            # Train S1 component
            skip_combos = set()
            for side in ["buy", "sell"]:
                for h in range(24):
                    mask = (train["side"] == side) & (train["utc_hour"] == h)
                    p = train.loc[mask, pnl_col].dropna()
                    if len(p) >= 3 and p.mean() < -0.5:
                        skip_combos.add((side, h))

            # Train S4 component
            good_regimes = set()
            for regime, grp in train.groupby("regime"):
                p = grp[pnl_col].dropna()
                if len(p) >= 3 and p.mean() > -0.3:
                    good_regimes.add(regime)

            # Test: both filters
            test_f = test.copy()
            if skip_combos:
                test_f = test_f[~test_f.apply(
                    lambda r: (r["side"], r["utc_hour"]) in skip_combos, axis=1
                )]
            if good_regimes:
                test_f = test_f[test_f["regime"].isin(good_regimes)]

            test_pnl = test_f[pnl_col].dropna()
            if len(test_pnl) > 0:
                s5_results.append({
                    "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
                    "win_pct": (test_pnl > 0).mean() * 100,
                    "pass_rate": len(test_f) / len(test) * 100 if len(test) > 0 else 0,
                })
        strategies["S5_combined_s1_s4"] = s5_results

    # ===== S6: Offset adaptive (side 別 offset) =====
    s6_results = []
    for fold in range(4):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = fold_size * (fold + 2) if fold < 3 else N
        train = filled.iloc[:train_end]
        test = filled.iloc[test_start:test_end]

        # Train: side 別に最適 offset を探索
        best_offsets = {}
        for side in ["buy", "sell"]:
            side_train = train[train["side"] == side]
            if "effective_offset_used" in side_train.columns:
                off_med = side_train["effective_offset_used"].dropna().median()
                high_off = side_train.loc[
                    side_train["effective_offset_used"] >= off_med, pnl_col
                ].dropna()
                low_off = side_train.loc[
                    side_train["effective_offset_used"] < off_med, pnl_col
                ].dropna()
                if len(high_off) >= 3 and len(low_off) >= 3:
                    best_offsets[side] = "high" if high_off.mean() > low_off.mean() else "low"

        # Test: (情報のみ、フィルタリングはなし — offset は注文前に決定)
        test_pnl = test[pnl_col].dropna()
        if len(test_pnl) > 0:
            s6_results.append({
                "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
                "win_pct": (test_pnl > 0).mean() * 100,
                "best_offsets": best_offsets,
            })
    strategies["S6_offset_hints"] = s6_results

    # ===== S7: AS-aware side rotation (AS 高い side を一時的にスキップ) =====
    s7_results = []
    for fold in range(4):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = fold_size * (fold + 2) if fold < 3 else N
        train = filled.iloc[:train_end]
        test = filled.iloc[test_start:test_end]

        # Train: side 別 AS rate
        skip_side = None
        if "adverse_selected" in train.columns:
            for side in ["buy", "sell"]:
                s_as = train.loc[train["side"] == side, "adverse_selected"].dropna()
                s_pnl = train.loc[train["side"] == side, pnl_col].dropna()
                if len(s_as) >= 10 and s_as.mean() > 0.4 and s_pnl.mean() < -1.0:
                    skip_side = side

        # Test
        if skip_side:
            test_filtered = test[test["side"] != skip_side]
        else:
            test_filtered = test
        test_pnl = test_filtered[pnl_col].dropna()
        if len(test_pnl) > 0:
            s7_results.append({
                "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
                "win_pct": (test_pnl > 0).mean() * 100,
                "skip_side": skip_side,
                "pass_rate": len(test_filtered) / len(test) * 100 if len(test) > 0 else 0,
            })
    strategies["S7_as_side_rotation"] = s7_results

    # ===== S8: Aggressive composite (S1 + S7 + fast_fill guard) =====
    s8_results = []
    for fold in range(4):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = fold_size * (fold + 2) if fold < 3 else N
        train = filled.iloc[:train_end]
        test = filled.iloc[test_start:test_end]

        # S1 component
        skip_combos = set()
        for side in ["buy", "sell"]:
            for h in range(24):
                mask = (train["side"] == side) & (train["utc_hour"] == h)
                p = train.loc[mask, pnl_col].dropna()
                if len(p) >= 3 and p.mean() < -0.5:
                    skip_combos.add((side, h))

        # Fast fill guard: queue_wait < 5s の PnL が negative なら除外
        fast_fill_pnl = train.loc[train["queue_wait_sec"] < 5, pnl_col].dropna()
        guard_fast = len(fast_fill_pnl) >= 5 and fast_fill_pnl.mean() < -0.5

        # Test
        test_f = test.copy()
        if skip_combos:
            test_f = test_f[~test_f.apply(
                lambda r: (r["side"], r["utc_hour"]) in skip_combos, axis=1
            )]
        if guard_fast:
            test_f = test_f[test_f["queue_wait_sec"] >= 5]

        test_pnl = test_f[pnl_col].dropna()
        if len(test_pnl) > 0:
            s8_results.append({
                "fold": fold, "mean_pnl": test_pnl.mean(), "n": len(test_pnl),
                "win_pct": (test_pnl > 0).mean() * 100,
                "skip_combos": len(skip_combos),
                "guard_fast": guard_fast,
                "pass_rate": len(test_f) / len(test) * 100 if len(test) > 0 else 0,
            })
    strategies["S8_aggressive_composite"] = s8_results

    # ===== Print results =====
    print("\n--- WF-4fold 戦略比較 ---")
    print(f"{'Strategy':<30} {'mean_pnl':>9} {'win%':>6} {'n_avg':>6} {'pass%':>6} {'folds>0':>8}")
    print("-" * 70)
    for name, results in strategies.items():
        if not results:
            continue
        avg_pnl = np.mean([r["mean_pnl"] for r in results])
        avg_win = np.mean([r["win_pct"] for r in results])
        avg_n = np.mean([r["n"] for r in results])
        avg_pass = np.mean([r.get("pass_rate", 100) for r in results])
        folds_positive = sum(1 for r in results if r["mean_pnl"] > 0)
        marker = " <<<" if folds_positive == 4 else (" **" if folds_positive >= 3 else "")
        print(f"{name:<30} {avg_pnl:>+9.3f} {avg_win:>5.1f}% {avg_n:>6.0f} {avg_pass:>5.1f}% {folds_positive}/4{marker}")

    # Detail per fold
    print("\n--- Fold 別詳細 ---")
    for name, results in strategies.items():
        if not results:
            continue
        print(f"\n  {name}:")
        for r in results:
            extra = ""
            if "skip_combos" in r:
                extra += f" skip_combos={r['skip_combos']}"
            if "thr" in r:
                extra += f" thr={r['thr']}s"
            if "good_regimes" in r:
                extra += f" regimes={r['good_regimes']}"
            if "skip_side" in r:
                extra += f" skip_side={r['skip_side']}"
            if "guard_fast" in r:
                extra += f" fast_guard={r['guard_fast']}"
            if "best_offsets" in r:
                extra += f" offsets={r['best_offsets']}"
            print(f"    fold{r['fold']}: pnl={r['mean_pnl']:+.3f} win={r['win_pct']:.1f}% "
                  f"n={r['n']} pass={r.get('pass_rate', 100):.0f}%{extra}")


def multi_horizon_analysis(df: pd.DataFrame) -> None:
    """§4 Multi-horizon PnL 比較."""
    print("\n" + "=" * 70)
    print("§4 Multi-horizon PnL 比較")
    print("=" * 70)

    filled = df[df["filled"] == True].copy()
    horizons = []
    for col, label in [("post_fill_30s_pnl", "30s"), ("post_fill_60s_pnl", "60s"), ("post_fill_120s_pnl", "120s")]:
        if col in filled.columns:
            p = filled[col].dropna()
            if len(p) > 0:
                horizons.append((label, p))

    if not horizons:
        print("No multi-horizon data.")
        return

    print(f"{'horizon':<10} {'mean':>8} {'median':>8} {'std':>8} {'win%':>6} {'n':>6}")
    for label, p in horizons:
        print(f"{label:<10} {p.mean():>+8.3f} {p.median():>+8.3f} {p.std():>8.3f} {100*(p>0).mean():>5.1f}% {len(p):>6}")

    # Best horizon suggestion
    best = max(horizons, key=lambda x: x[1].mean())
    print(f"\n推奨 horizon: {best[0]} (mean PnL = {best[1].mean():+.3f} bps)")


def main() -> None:
    df = load_all_records()
    if len(df) == 0:
        print("ERROR: No fill records found.")
        return

    print(f"Total records loaded: {len(df)}")
    print(f"Date range: {pd.to_datetime(df['timestamp'].min(), unit='s')} "
          f"→ {pd.to_datetime(df['timestamp'].max(), unit='s')}")
    print()

    basic_stats(df)
    segment_analysis(df)
    walk_forward_strategies(df)
    multi_horizon_analysis(df)


if __name__ == "__main__":
    main()
