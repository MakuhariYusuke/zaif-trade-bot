#!/usr/bin/env python3
"""
Analyze recent training reports for balance exploration.

Finds the best action distributions and identifies patterns.
"""

import math
from pathlib import Path
from typing import TypedDict

from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    get_recent_training_reports,
    load_training_report,
)
from ztb.utils.safety import ensure_dict, safe_to_float


class BalanceReportSummary(TypedDict):
    file: str
    modified: float
    buy: float
    sell: float
    hold: float
    balance_score: float
    balance_shaping: object
    balance_penalty: object
    timesteps: float


def calculate_balance_score(buy: float, sell: float, hold: float) -> float:
    """
    Calculate balance score (minimum of the three actions).
    Higher score means more balanced distribution.
    """
    return min(buy, sell, hold)


def analyze_reports(limit: int = 20) -> list[BalanceReportSummary]:
    """Analyze recent training reports."""
    report_files = get_recent_training_reports(limit=limit, reports_dir=Path("reports"))

    results: list[BalanceReportSummary] = []

    for report_file in report_files:
        data = load_training_report(report_file)
        if data is None:
            print(f"Error processing {report_file.name}: could not load JSON")
            continue

        ad = extract_action_distribution_from_payload(data)
        if not ad:
            continue

        buy = safe_to_float(ad.get("BUY"), 0.0)
        sell = safe_to_float(ad.get("SELL"), 0.0)
        hold = safe_to_float(ad.get("HOLD"), 0.0)

        configuration = ensure_dict(data.get("configuration"))
        curriculum = ensure_dict(configuration.get("curriculum"))
        balance_shaping = curriculum.get("balance_shaping_value", "N/A")
        balance_penalty = curriculum.get("balance_penalty", "N/A")

        training_stats = ensure_dict(data.get("training_stats"))
        timesteps = safe_to_float(training_stats.get("total_timesteps"), 0.0)

        try:
            modified = report_file.stat().st_mtime
        except OSError:
            modified = 0.0

        balance_score = calculate_balance_score(buy, sell, hold)
        results.append(
            {
                "file": report_file.name,
                "modified": modified,
                "buy": buy,
                "sell": sell,
                "hold": hold,
                "balance_score": balance_score,
                "balance_shaping": balance_shaping,
                "balance_penalty": balance_penalty,
                "timesteps": timesteps,
            }
        )

    return results


def main() -> None:
    print("=" * 80)
    print("Training Report Balance Analysis")
    print("=" * 80)

    results = analyze_reports(limit=30)

    if not results:
        print("No reports with action distribution found")
        return

    # Sort by balance score
    results_sorted = sorted(results, key=lambda x: x["balance_score"], reverse=True)

    print("\n📊 Top 10 Most Balanced Configurations:")
    print("-" * 80)
    print(
        f"{'Rank':<5} {'Balance':<8} {'BUY':<8} {'SELL':<8} {'HOLD':<8} {'Shaping':<10} {'Penalty':<8}"
    )
    print("-" * 80)

    for i, r in enumerate(results_sorted[:10], 1):
        print(
            f"{i:<5} {r['balance_score']:<8.2f} "
            f"{r['buy']:<8.2f} {r['sell']:<8.2f} {r['hold']:<8.2f} "
            f"{r['balance_shaping']!s:<10} {r['balance_penalty']!s:<8}"
        )

    print("\n" + "=" * 80)
    print("📈 Analysis Summary:")
    print("=" * 80)

    # Find target-like distributions (BUY ~60%, SELL ~33%, HOLD ~7%)
    target_results = [
        r
        for r in results
        if 50 <= r["buy"] <= 70 and 25 <= r["sell"] <= 45 and 3 <= r["hold"] <= 10
    ]

    if target_results:
        print(f"\n✅ Found {len(target_results)} reports matching target distribution:")
        print("   (BUY: 50-70%, SELL: 25-45%, HOLD: 3-10%)")
        print("-" * 80)

        for r in sorted(target_results, key=lambda x: x["balance_score"], reverse=True)[
            :5
        ]:
            print(
                f"  BUY={r['buy']:.1f}%, SELL={r['sell']:.1f}%, HOLD={r['hold']:.1f}% "
                f"(Balance={r['balance_score']:.2f})"
            )
            if r["balance_shaping"] != "N/A":
                print(
                    f"    → balance_shaping={r['balance_shaping']}, penalty={r['balance_penalty']}"
                )
    else:
        print("\n⚠️  No reports found matching target distribution")
        print("   Target: BUY 50-70%, SELL 25-45%, HOLD 3-10%")

    # Statistics
    avg_buy = sum(r["buy"] for r in results) / len(results)
    avg_sell = sum(r["sell"] for r in results) / len(results)
    avg_hold = sum(r["hold"] for r in results) / len(results)
    avg_balance = sum(r["balance_score"] for r in results) / len(results)

    print(f"\n📊 Average across {len(results)} reports:")
    print(f"   BUY:  {avg_buy:.2f}%")
    print(f"   SELL: {avg_sell:.2f}%")
    print(f"   HOLD: {avg_hold:.2f}%")
    print(f"   Balance Score: {avg_balance:.2f}")

    # Best overall
    best = results_sorted[0]
    print(f"\n🏆 Best Balance Score: {best['balance_score']:.2f}")
    print(
        f"   BUY={best['buy']:.1f}%, SELL={best['sell']:.1f}%, HOLD={best['hold']:.1f}%"
    )
    if best["balance_shaping"] != "N/A":
        print(
            f"   Config: balance_shaping={best['balance_shaping']}, penalty={best['balance_penalty']}"
        )

    print("\n" + "=" * 80)
    print("💡 Recommendations:")
    print("=" * 80)

    if target_results:
        # Get balance_shaping values from target results
        shaping_values: list[float] = []
        for result in target_results:
            if result["balance_shaping"] == "N/A":
                continue
            parsed = safe_to_float(result["balance_shaping"], math.nan)
            if not math.isnan(parsed):
                shaping_values.append(parsed)
        if shaping_values:
            avg_shaping = sum(shaping_values) / len(shaping_values)
            print(
                f"1. Target distributions found with avg balance_shaping={avg_shaping:.3f}"
            )
            print(f"2. Explore range: {avg_shaping-0.01:.3f} to {avg_shaping+0.01:.3f}")
        else:
            print("1. Target distributions found but no balance_shaping info")
            print("2. Try balance_shaping values: 0.04, 0.05, 0.06")
    else:
        print("1. No target distributions found yet")
        print("2. Recommended exploration:")
        print("   - balance_shaping_value: 0.03, 0.04, 0.05, 0.06, 0.07")
        print("   - balance_penalty: 3.0, 4.0, 5.0")
        print(
            "3. Run: python tools/run_balance_ab_tests.py --balance-values 0.04 0.05 0.06 --run"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
