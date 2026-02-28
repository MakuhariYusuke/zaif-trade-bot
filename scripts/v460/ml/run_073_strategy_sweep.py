"""073# ph2 追加実験: side 別 time filter の閾値感度 + 複合戦略.

分析結果から新戦略を設計:
- S9: Conservative side-time (threshold -1.0, n>=2)
- S10: Asymmetric offset (sell offset 引上げ + side-time filter)
- S11: Best-hour-only (正のPnL hour だけ残す)
- S12: Regime-aware side-time (regime 考慮)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.ml.frame_utils import (
    collect_bad_side_hours,
    collect_good_side_hours,
    compute_utc_hour,
    exclude_side_hour_combos,
    include_side_hour_combos,
)
from scripts.v460.ml.run_073_strategy_analysis import load_all_records


def wf_evaluate(filled: pd.DataFrame, strategy_fn, name: str) -> dict:
    """Walk-Forward 4-fold で戦略を評価."""
    N = len(filled)
    fold_size = N // 5
    pnl_col = "post_fill_30s_pnl"
    results = []

    for fold in range(4):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = fold_size * (fold + 2) if fold < 3 else N
        train = filled.iloc[:train_end]
        test = filled.iloc[test_start:test_end]

        test_filtered, info = strategy_fn(train, test)
        eval_col = "sim_pnl" if "sim_pnl" in test_filtered.columns else pnl_col
        test_pnl = test_filtered[eval_col].dropna()
        if len(test_pnl) > 0:
            results.append({
                "fold": fold,
                "mean_pnl": test_pnl.mean(),
                "n": len(test_pnl),
                "win_pct": (test_pnl > 0).mean() * 100,
                "pass_rate": len(test_filtered) / len(test) * 100,
                **info,
            })

    return {"name": name, "results": results}


def s9_conservative_side_time(train: pd.DataFrame, test: pd.DataFrame):
    """Side 別 time filter: 閾値 -1.0, 最低 n=2."""
    skip_combos = collect_bad_side_hours(
        train,
        pnl_col="post_fill_30s_pnl",
        threshold=-1.0,
        min_count=2,
    )
    test_f = exclude_side_hour_combos(test, skip_combos)
    return test_f, {"skip_combos": len(skip_combos)}


def s10_asymmetric_offset_sim(train: pd.DataFrame, test: pd.DataFrame):
    """Sell 側 PnL が -1.0 以下の hour はスキップ + buy は -2.0 でスキップ.

    sell は全体的に AS が大きいため厳しめ、buy は緩め。
    """
    skip_combos = set()

    skip_combos |= collect_bad_side_hours(
        train.loc[train["side"] == "buy"],
        pnl_col="post_fill_30s_pnl",
        threshold=-2.0,
        min_count=2,
    )
    skip_combos |= collect_bad_side_hours(
        train.loc[train["side"] == "sell"],
        pnl_col="post_fill_30s_pnl",
        threshold=-0.8,
        min_count=2,
    )

    test_f = exclude_side_hour_combos(test, skip_combos)
    return test_f, {"skip_combos": len(skip_combos),
                    "buy_skip": sum(1 for s, h in skip_combos if s == "buy"),
                    "sell_skip": sum(1 for s, h in skip_combos if s == "sell")}


def s11_best_hours_only(train: pd.DataFrame, test: pd.DataFrame):
    """PnL > 0 の side×hour のみ残す (ポジティブセレクション)."""
    good_combos = collect_good_side_hours(
        train,
        pnl_col="post_fill_30s_pnl",
        threshold=0.0,
        min_count=3,
    )
    test_f = include_side_hour_combos(test, good_combos)
    return test_f, {"good_combos": len(good_combos)}


def s12_offset_7pct_sim(train: pd.DataFrame, test: pd.DataFrame):
    """Offset 5% → 7% シミュレーション: fast fill (<5s) を除外.

    070# P1推奨: spread_offset 拡大で per-fill エッジ改善。
    近似: queue_wait が短い注文ほど offset 不足を示唆。
    """
    pnl_col = "post_fill_30s_pnl"
    # 5s 未満はおそらく offset 増で約定しない → 除外
    test_f = test[test["queue_wait_sec"] >= 5].copy()
    # さらに PnL をオフセット分改善 (概算 +0.5bps)
    test_f["sim_pnl"] = test_f[pnl_col] + 0.5

    return test_f, {"removed_fast": len(test) - len(test_f)}


def s13_sell_offset_boost_sim(train: pd.DataFrame, test: pd.DataFrame):
    """Sell のみ offset 10% → 12% シミュレーション.

    Sell PnL -0.958 vs Buy -0.301 → sell に追加保護。
    sell の fast fill を除外 + PnL 改善想定。
    """
    pnl_col = "post_fill_30s_pnl"
    # Buy はそのまま
    test_buy = test[test["side"] == "buy"]
    # Sell: queue_wait < 10s を除外 (offset 増で fill しなくなる想定)
    test_sell = test[(test["side"] == "sell") & (test["queue_wait_sec"] >= 10)]

    test_f = pd.concat([test_buy, test_sell])
    return test_f, {"sell_removed": len(test) - len(test_f)}


def s14_combined_best(train: pd.DataFrame, test: pd.DataFrame):
    """S10 (asymmetric side-time) + fast fill guard + 120s horizon hint.

    実装可能な複合戦略の最終形。
    """
    skip_combos = set()

    skip_combos |= collect_bad_side_hours(
        train.loc[train["side"] == "buy"],
        pnl_col="post_fill_30s_pnl",
        threshold=-2.0,
        min_count=2,
    )
    skip_combos |= collect_bad_side_hours(
        train.loc[train["side"] == "sell"],
        pnl_col="post_fill_30s_pnl",
        threshold=-0.8,
        min_count=2,
    )

    # Fast fill guard
    fast_pnl = train.loc[train["queue_wait_sec"] < 5, "post_fill_30s_pnl"].dropna()
    guard_fast = len(fast_pnl) >= 5 and fast_pnl.mean() < -0.5

    # Apply
    test_f = exclude_side_hour_combos(test, skip_combos)
    if guard_fast:
        test_f = test_f[test_f["queue_wait_sec"] >= 5]

    return test_f, {
        "skip_combos": len(skip_combos),
        "guard_fast": guard_fast,
    }


def main() -> None:
    df = load_all_records()
    filled = df[df["filled"] == True].copy().sort_values("timestamp").reset_index(drop=True)
    filled["post_fill_30s_pnl"] = filled["post_fill_30s_pnl"].astype(float)
    filled["utc_hour"] = compute_utc_hour(filled["timestamp"])

    print(f"全 filled: {len(filled)}")
    print(f"Date: {pd.to_datetime(filled['timestamp'].min(), unit='s')} "
          f"→ {pd.to_datetime(filled['timestamp'].max(), unit='s')}")

    strategies = [
        ("S9_conservative_side_time", s9_conservative_side_time),
        ("S10_asymmetric_side_time", s10_asymmetric_offset_sim),
        ("S11_best_hours_only", s11_best_hours_only),
        ("S12_offset_7pct_sim", s12_offset_7pct_sim),
        ("S13_sell_offset_boost_sim", s13_sell_offset_boost_sim),
        ("S14_combined_best", s14_combined_best),
    ]

    all_results = []
    for name, fn in strategies:
        result = wf_evaluate(filled, fn, name)
        all_results.append(result)

    # Summary
    print(f"\n{'Strategy':<30} {'mean_pnl':>9} {'win%':>6} {'n_avg':>6} {'pass%':>6} {'folds>0':>8}")
    print("-" * 70)
    for res in all_results:
        r = res["results"]
        if not r:
            continue
        avg_pnl = np.mean([x["mean_pnl"] for x in r])
        avg_win = np.mean([x["win_pct"] for x in r])
        avg_n = np.mean([x["n"] for x in r])
        avg_pass = np.mean([x["pass_rate"] for x in r])
        folds_pos = sum(1 for x in r if x["mean_pnl"] > 0)
        marker = " <<<" if folds_pos == 4 else (" **" if folds_pos >= 3 else "")
        print(f"{res['name']:<30} {avg_pnl:>+9.3f} {avg_win:>5.1f}% {avg_n:>6.0f} {avg_pass:>5.1f}% {folds_pos}/4{marker}")

    # Detail
    print("\n--- Fold 別詳細 ---")
    for res in all_results:
        print(f"\n  {res['name']}:")
        for r in res["results"]:
            extras = {k: v for k, v in r.items()
                      if k not in ("fold", "mean_pnl", "n", "win_pct", "pass_rate")}
            extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
            print(f"    fold{r['fold']}: pnl={r['mean_pnl']:+.3f} win={r['win_pct']:.1f}% "
                  f"n={r['n']} pass={r['pass_rate']:.0f}% {extra_str}")


if __name__ == "__main__":
    main()
