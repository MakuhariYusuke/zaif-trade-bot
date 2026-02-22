"""070# Part 2: 深掘り分析.

Part A で発見された時間帯構造の walk-forward 安定性を検証し、
実運用に使える戦略を絞り込む。
"""

from __future__ import annotations

import glob
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.data_loader import build_as_features, load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_enriched_as_features,
    enrich_fill_records,
)
from ztb.io.json_io import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def analyze_temporal_structure(df: pd.DataFrame) -> dict:
    """時間帯別PnLの安定性をWalk-Forwardで検証."""
    filled = df[df["filled"].astype(bool)].copy()
    pnl = filled["post_fill_30s_pnl"].astype(float)
    ts = filled["timestamp"].astype(float)
    hours_utc = ts.apply(lambda t: datetime.fromtimestamp(t).hour)
    jst_hours = ts.apply(lambda t: (datetime.fromtimestamp(t).hour + 9) % 24)

    results = {}

    # --- Per-hour PnL stats ---
    hour_stats = []
    for h in range(24):
        mask = hours_utc == h
        n = int(mask.sum())
        if n < 5:
            continue
        p = pnl[mask]
        as_mask = filled.loc[mask, "adverse_selected_raw"].notna()
        as_rate = filled.loc[mask & as_mask, "adverse_selected_raw"].astype(float).mean() if as_mask.any() else None
        hour_stats.append({
            "hour_utc": h,
            "hour_jst": (h + 9) % 24,
            "n": n,
            "pnl_mean": round(float(p.mean()), 4),
            "pnl_std": round(float(p.std()), 4),
            "pnl_median": round(float(p.median()), 4),
            "pnl_positive_rate": round(float((p > 0).mean()), 4),
            "as_rate": round(float(as_rate), 4) if as_rate is not None else None,
        })
    results["hour_stats"] = hour_stats

    # Print hour breakdown
    logger.info("\n--- Per-Hour PnL (UTC → JST) ---")
    logger.info(f"{'UTC':>4} {'JST':>4} {'N':>5} {'PnL_mean':>10} {'PnL_med':>10} "
                f"{'PnL+%':>7} {'AS_rate':>8}")
    for hs in sorted(hour_stats, key=lambda x: x["pnl_mean"]):
        logger.info(
            f"{hs['hour_utc']:4d} {hs['hour_jst']:4d} {hs['n']:5d} "
            f"{hs['pnl_mean']:10.4f} {hs['pnl_median']:10.4f} "
            f"{hs['pnl_positive_rate']:7.1%} "
            f"{hs['as_rate'] if hs['as_rate'] is not None else 'N/A':>8}"
        )

    # --- Walk-Forward test of "Skip negative-PnL hours" ---
    logger.info("\n--- Walk-Forward: Time-based filtering ---")

    # Sort by timestamp
    filled_sorted = filled.sort_values("timestamp").reset_index(drop=True)
    pnl_sorted = filled_sorted["post_fill_30s_pnl"].astype(float)
    hours_sorted = filled_sorted["timestamp"].astype(float).apply(
        lambda t: datetime.fromtimestamp(t).hour
    )

    n = len(filled_sorted)
    # Use 3 temporal splits: first third trains, rest tests; first 2/3 trains, rest tests
    wf_results = []
    for train_frac, label in [(0.33, "33/67"), (0.5, "50/50"), (0.67, "67/33")]:
        split_idx = int(n * train_frac)
        train_hours = hours_sorted.iloc[:split_idx]
        train_pnl = pnl_sorted.iloc[:split_idx]
        test_hours = hours_sorted.iloc[split_idx:]
        test_pnl = pnl_sorted.iloc[split_idx:]

        # Learn: which hours have negative mean PnL in training data
        train_hour_pnl = train_pnl.groupby(train_hours).mean()
        bad_hours_learned = set(train_hour_pnl[train_hour_pnl < 0].index.tolist())

        # Apply to test
        test_skip_mask = test_hours.isin(bad_hours_learned)
        test_keep_mask = ~test_skip_mask
        n_keep = int(test_keep_mask.sum())
        n_skip = int(test_skip_mask.sum())

        if n_keep > 0:
            baseline_test = float(test_pnl.mean())
            kept_test = float(test_pnl[test_keep_mask].mean())
            improvement = kept_test - baseline_test
        else:
            baseline_test = float(test_pnl.mean())
            kept_test = 0.0
            improvement = 0.0

        wf_r = {
            "split": label,
            "train_n": split_idx,
            "test_n": n - split_idx,
            "bad_hours_learned": sorted(bad_hours_learned),
            "n_bad_hours": len(bad_hours_learned),
            "test_n_keep": n_keep,
            "test_n_skip": n_skip,
            "test_skip_rate": round(n_skip / (n - split_idx) if (n - split_idx) > 0 else 0, 3),
            "test_baseline_pnl": round(baseline_test, 4),
            "test_kept_pnl": round(kept_test, 4),
            "test_improvement": round(improvement, 4),
        }
        wf_results.append(wf_r)
        logger.info(
            f"  Split {label}: bad_hours={sorted(bad_hours_learned)} "
            f"  skip_rate={wf_r['test_skip_rate']:.1%} "
            f"  baseline={wf_r['test_baseline_pnl']:.4f} "
            f"  kept={wf_r['test_kept_pnl']:.4f} "
            f"  improvement={wf_r['test_improvement']:+.4f}"
        )

    results["wf_time_filter"] = wf_results

    # --- Side × Time analysis ---
    logger.info("\n--- Side × Time PnL ---")
    for side in ["buy", "sell"]:
        side_mask = filled_sorted["side"] == side
        side_pnl = pnl_sorted[side_mask]
        side_hours = hours_sorted[side_mask]

        hour_pnl = side_pnl.groupby(side_hours).agg(["mean", "count"])
        logger.info(f"\n  Side: {side}")
        for h in sorted(hour_pnl.index):
            m = hour_pnl.loc[h, "mean"]
            c = hour_pnl.loc[h, "count"]
            if c >= 3:
                logger.info(f"    UTC {h:2d} (JST {(h+9)%24:2d}): n={c:3.0f}, PnL={m:+.4f} bps")

    return results


def analyze_queue_wait_structure(df: pd.DataFrame) -> dict:
    """Queue wait vs PnL の関係分析."""
    filled = df[df["filled"].astype(bool)].copy()
    pnl = filled["post_fill_30s_pnl"].astype(float)
    qw = filled["queue_wait_sec"].astype(float)

    logger.info("\n--- Queue Wait × PnL ---")
    bins = [0, 5, 10, 20, 30, 60, 120, 600]
    labels = ["0-5s", "5-10s", "10-20s", "20-30s", "30-60s", "60-120s", "120s+"]
    filled["qw_bin"] = pd.cut(qw, bins=bins, labels=labels, right=False)

    results = []
    for label in labels:
        mask = filled["qw_bin"] == label
        n = int(mask.sum())
        if n < 3:
            continue
        p = pnl[mask]
        r = {
            "bin": label,
            "n": n,
            "pnl_mean": round(float(p.mean()), 4),
            "pnl_median": round(float(p.median()), 4),
            "pnl_positive_rate": round(float((p > 0).mean()), 4),
        }
        results.append(r)
        logger.info(f"  {label:10s}: n={n:4d}, PnL_mean={r['pnl_mean']:+.4f}, "
                     f"PnL_med={r['pnl_median']:+.4f}, positive={r['pnl_positive_rate']:.1%}")

    return {"queue_wait_bins": results}


def analyze_multi_horizon_pnl(df: pd.DataFrame) -> dict:
    """30s/60s/120s 各ホライズンでの PnL 分析."""
    filled = df[df["filled"].astype(bool)].copy()

    logger.info("\n--- Multi-Horizon PnL ---")
    results = {}
    for col, label in [
        ("post_fill_30s_pnl", "30s"),
        ("post_fill_60s_pnl", "60s"),
        ("post_fill_120s_pnl", "120s"),
    ]:
        if col not in filled.columns:
            continue
        vals = filled[col].dropna().astype(float)
        r = {
            "n": len(vals),
            "mean": round(float(vals.mean()), 4),
            "median": round(float(vals.median()), 4),
            "std": round(float(vals.std()), 4),
            "positive_rate": round(float((vals > 0).mean()), 4),
        }
        results[label] = r
        logger.info(f"  {label}: n={r['n']}, mean={r['mean']:+.4f} bps, "
                     f"median={r['median']:+.4f}, positive={r['positive_rate']:.1%}")

    # Side breakdown per horizon
    for side in ["buy", "sell"]:
        for col, label in [
            ("post_fill_30s_pnl", "30s"),
            ("post_fill_60s_pnl", "60s"),
            ("post_fill_120s_pnl", "120s"),
        ]:
            if col not in filled.columns:
                continue
            mask = filled["side"] == side
            vals = filled.loc[mask, col].dropna().astype(float)
            if len(vals) < 5:
                continue
            logger.info(f"  {side} {label}: n={len(vals)}, mean={vals.mean():+.4f}, "
                         f"positive={float((vals > 0).mean()):.1%}")

    return results


def analyze_spread_offset_sensitivity(df: pd.DataFrame) -> dict:
    """spread_offset_ratio と PnL の関係."""
    filled = df[df["filled"].astype(bool)].copy()
    pnl = filled["post_fill_30s_pnl"].astype(float)

    logger.info("\n--- Spread Offset Ratio × PnL ---")
    if "spread_offset_ratio" not in filled.columns:
        logger.info("  No spread_offset_ratio data")
        return {}

    sor = filled["spread_offset_ratio"].dropna().astype(float)
    if len(sor) < 20:
        logger.info("  Too few samples with spread_offset_ratio")
        return {}

    pnl_with_sor = pnl.loc[sor.index]

    # Correlation
    ic, p = spearmanr(sor.values, pnl_with_sor.values)
    logger.info(f"  Spearman IC (spread_offset_ratio vs PnL): {ic:.4f}, p={p:.4f}")

    # Bins
    bins = [0, 0.03, 0.05, 0.07, 0.10, 1.0]
    labels = ["0-3%", "3-5%", "5-7%", "7-10%", "10%+"]
    binned = pd.cut(sor, bins=bins, labels=labels, right=False)
    results = []
    for label in labels:
        mask = binned == label
        n = int(mask.sum())
        if n < 3:
            continue
        p_vals = pnl_with_sor[mask]
        r = {
            "bin": label,
            "n": n,
            "pnl_mean": round(float(p_vals.mean()), 4),
            "pnl_positive_rate": round(float((p_vals > 0).mean()), 4),
        }
        results.append(r)
        logger.info(f"  {label:8s}: n={n:4d}, PnL={r['pnl_mean']:+.4f}, "
                     f"positive={r['pnl_positive_rate']:.1%}")

    return {"ic": round(ic, 4), "p_value": round(p, 4), "bins": results}


def analyze_round_trip_detail(df: pd.DataFrame) -> dict:
    """ラウンドトリップ (buy→sell or sell→buy) の詳細分析."""
    filled = df[df["filled"].astype(bool)].copy()
    filled = filled.sort_values("timestamp").reset_index(drop=True)

    logger.info("\n--- Round-Trip Analysis ---")

    # Build round trips: pair consecutive opposite-side fills
    trips = []
    i = 0
    while i < len(filled) - 1:
        curr = filled.iloc[i]
        # Find next opposite side fill
        for j in range(i + 1, len(filled)):
            nxt = filled.iloc[j]
            if curr["side"] != nxt["side"]:
                # Round trip found
                if curr["side"] == "buy":
                    buy_price = float(curr["fill_price"])
                    sell_price = float(nxt["fill_price"])
                else:
                    sell_price = float(curr["fill_price"])
                    buy_price = float(nxt["fill_price"])
                mid_price = (buy_price + sell_price) / 2
                rt_pnl_bps = (sell_price - buy_price) / mid_price * 10000 if mid_price > 0 else 0
                dt = float(nxt["timestamp"]) - float(curr["timestamp"])
                trips.append({
                    "entry_side": curr["side"],
                    "pnl_bps": round(rt_pnl_bps, 4),
                    "duration_sec": round(dt, 1),
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "entry_hour_utc": datetime.fromtimestamp(float(curr["timestamp"])).hour,
                })
                i = j + 1
                break
        else:
            break

    if not trips:
        logger.info("  No round trips found")
        return {}

    trip_df = pd.DataFrame(trips)
    total_pnl = trip_df["pnl_bps"].sum()
    mean_pnl = trip_df["pnl_bps"].mean()
    win_rate = (trip_df["pnl_bps"] > 0).mean()
    median_duration = trip_df["duration_sec"].median()

    logger.info(f"  Total round trips: {len(trips)}")
    logger.info(f"  Total PnL: {total_pnl:+.1f} bps")
    logger.info(f"  Mean PnL: {mean_pnl:+.4f} bps")
    logger.info(f"  Win rate: {win_rate:.1%}")
    logger.info(f"  Median duration: {median_duration:.0f}s")

    # PnL distribution
    pnl_arr = trip_df["pnl_bps"].values
    p5 = float(np.percentile(pnl_arr, 5))
    p25 = float(np.percentile(pnl_arr, 25))
    p50 = float(np.percentile(pnl_arr, 50))
    p75 = float(np.percentile(pnl_arr, 75))
    p95 = float(np.percentile(pnl_arr, 95))
    logger.info(f"  PnL distribution: p5={p5:.1f}, p25={p25:.1f}, p50={p50:.1f}, "
                 f"p75={p75:.1f}, p95={p95:.1f}")

    # Win streaks and loss streaks
    wins = trip_df["pnl_bps"] > 0
    win_count = 0
    loss_count = 0
    max_win_streak = 0
    max_loss_streak = 0
    for w in wins:
        if w:
            win_count += 1
            loss_count = 0
        else:
            loss_count += 1
            win_count = 0
        max_win_streak = max(max_win_streak, win_count)
        max_loss_streak = max(max_loss_streak, loss_count)
    logger.info(f"  Max win streak: {max_win_streak}, Max loss streak: {max_loss_streak}")

    # Per entry-hour analysis
    logger.info("\n  Per-hour round-trip performance:")
    for h in sorted(trip_df["entry_hour_utc"].unique()):
        mask = trip_df["entry_hour_utc"] == h
        n = int(mask.sum())
        if n < 2:
            continue
        p = trip_df.loc[mask, "pnl_bps"]
        wr = float((p > 0).mean())
        logger.info(f"    UTC {h:2d}: n={n:3d}, mean={p.mean():+.4f}, win_rate={wr:.0%}")

    return {
        "n_trips": len(trips),
        "total_pnl_bps": round(total_pnl, 2),
        "mean_pnl_bps": round(mean_pnl, 4),
        "win_rate": round(float(win_rate), 4),
        "median_duration_sec": round(median_duration, 1),
        "pnl_percentiles": {
            "p5": round(p5, 2), "p25": round(p25, 2), "p50": round(p50, 2),
            "p75": round(p75, 2), "p95": round(p95, 2),
        },
    }


def analyze_regime_structure(df: pd.DataFrame) -> dict:
    """レジーム別のPnL分析."""
    filled = df[df["filled"].astype(bool)].copy()
    pnl = filled["post_fill_30s_pnl"].astype(float)

    logger.info("\n--- Regime × PnL ---")
    if "regime" not in filled.columns:
        logger.info("  No regime data")
        return {}

    regime = filled["regime"].fillna("unknown")
    results = []
    for r_name in regime.unique():
        mask = regime == r_name
        n = int(mask.sum())
        if n < 3:
            continue
        p = pnl[mask]
        r = {
            "regime": r_name,
            "n": n,
            "pnl_mean": round(float(p.mean()), 4),
            "pnl_std": round(float(p.std()), 4),
            "pnl_positive_rate": round(float((p > 0).mean()), 4),
        }
        results.append(r)
        logger.info(f"  {r_name:12s}: n={n:4d}, PnL={r['pnl_mean']:+.4f} ± {r['pnl_std']:.4f}, "
                     f"positive={r['pnl_positive_rate']:.1%}")

    return {"regime_stats": results}


def main() -> None:
    output_dir = Path("reports/v460/model_search_070")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_fill_records()
    logger.info(f"Loaded {len(df)} records")

    all_results = {}

    # 1. Temporal structure
    all_results["temporal"] = analyze_temporal_structure(df)

    # 2. Queue wait
    all_results["queue_wait"] = analyze_queue_wait_structure(df)

    # 3. Multi-horizon PnL
    all_results["multi_horizon"] = analyze_multi_horizon_pnl(df)

    # 4. Spread offset
    all_results["spread_offset"] = analyze_spread_offset_sensitivity(df)

    # 5. Round-trip detail
    all_results["round_trip"] = analyze_round_trip_detail(df)

    # 6. Regime
    all_results["regime"] = analyze_regime_structure(df)

    # Save
    out_file = output_dir / "deep_analysis_results.json"
    write_json(out_file, all_results, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
