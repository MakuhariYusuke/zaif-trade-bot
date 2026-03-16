#!/usr/bin/env python3
"""
Analyze the correlation between profitability and BUY/SELL balance.

This script investigates whether balanced BUY/SELL ratios lead to higher profitability.
"""

import statistics

from pathlib import Path
from typing import TypedDict

from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    get_recent_training_reports,
    load_training_report,
)
from ztb.trading.environment.components.rewards.utils import RewardUtils
from ztb.utils.safety import ensure_dict, safe_to_float


class ProfitabilityDataPoint(TypedDict):
    file: str
    final_reward: float
    buy: float
    sell: float
    hold: float
    buy_sell_ratio: float
    buy_sell_diff: float
    buy_sell_balance: float


def analyze_profitability_balance() -> None:
    """Analyze correlation between portfolio value and action balance."""
    report_files = get_recent_training_reports(limit=50, reports_dir=Path("reports"))
    
    data_points: list[ProfitabilityDataPoint] = []
    
    for report_file in report_files:
        data = load_training_report(report_file)
        if data is None:
            print(f"Error processing {report_file}: could not load JSON")
            continue

        stats = ensure_dict(data.get("training_stats"))
        pv = safe_to_float(stats.get("final_reward"), 0.0)
        ad = extract_action_distribution_from_payload(data)

        if not ad or pv == 0.0:
            continue

        buy = safe_to_float(ad.get("BUY"), 0.0)
        sell = safe_to_float(ad.get("SELL"), 0.0)
        hold = safe_to_float(ad.get("HOLD"), 0.0)

        if sell > 0:
            buy_sell_ratio = buy / sell
        else:
            buy_sell_ratio = 999.0  # Very high if no SELL

        buy_sell_diff = RewardUtils.calculate_balance_deviation_from_ratios(
            [buy, sell], [0.5, 0.5]
        )
        buy_sell_balance = min(buy, sell)  # Perfect balance = 0.5 each

        data_points.append(
            {
                "file": report_file.name,
                "final_reward": pv,
                "buy": buy,
                "sell": sell,
                "hold": hold,
                "buy_sell_ratio": buy_sell_ratio,
                "buy_sell_diff": buy_sell_diff,
                "buy_sell_balance": buy_sell_balance,
            }
        )
    
    if not data_points:
        print("No valid data points found")
        return
    
    print(f"\n{'='*80}")
    print("PROFITABILITY (Final Reward) vs BUY/SELL BALANCE ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Total reports analyzed: {len(data_points)}\n")
    
    # Separate high-reward and low-reward
    avg_reward = sum(dp["final_reward"] for dp in data_points) / len(data_points)
    high_reward = [dp for dp in data_points if dp["final_reward"] > avg_reward]
    low_reward = [dp for dp in data_points if dp["final_reward"] <= avg_reward]
    
    print(f"✅ High Reward (> avg {avg_reward:.2f}): {len(high_reward)}")
    print(f"❌ Low Reward (≤ avg): {len(low_reward)}\n")
    
    # Top 10 by final reward
    print(f"{'='*80}")
    print("TOP 10 HIGHEST REWARD CONFIGURATIONS")
    print(f"{'='*80}\n")
    
    sorted_by_reward = sorted(data_points, key=lambda x: x["final_reward"], reverse=True)[:10]
    
    for i, dp in enumerate(sorted_by_reward, 1):
        ratio_str = f"{dp['buy_sell_ratio']:.2f}" if dp['buy_sell_ratio'] < 10 else "∞"
        print(f"Rank {i:2d}:")
        print(f"  Final Reward:    {dp['final_reward']:>10.2f}")
        print(f"  BUY:  {dp['buy']:>5.1%}  SELL: {dp['sell']:>5.1%}  HOLD: {dp['hold']:>5.1%}")
        print(f"  BUY/SELL Ratio:  {ratio_str}")
        print(f"  BUY-SELL Diff:   {dp['buy_sell_diff']:.1%}")
        print(f"  Balance Score:   {dp['buy_sell_balance']:.1%}")
        print()
    
    # Statistics on high-reward configs
    if high_reward:
        print(f"{'='*80}")
        print("HIGH REWARD CONFIGS - ACTION DISTRIBUTION STATISTICS")
        print(f"{'='*80}\n")
        
        buy_values = [dp["buy"] for dp in high_reward]
        sell_values = [dp["sell"] for dp in high_reward]
        hold_values = [dp["hold"] for dp in high_reward]
        ratio_values = [dp["buy_sell_ratio"] for dp in high_reward if dp["buy_sell_ratio"] < 10]
        diff_values = [dp["buy_sell_diff"] for dp in high_reward]
        
        print(f"BUY  - Mean: {statistics.mean(buy_values):.1%}, "
              f"Median: {statistics.median(buy_values):.1%}, "
              f"StdDev: {statistics.stdev(buy_values) if len(buy_values) > 1 else 0:.1%}")
        print(f"SELL - Mean: {statistics.mean(sell_values):.1%}, "
              f"Median: {statistics.median(sell_values):.1%}, "
              f"StdDev: {statistics.stdev(sell_values) if len(sell_values) > 1 else 0:.1%}")
        print(f"HOLD - Mean: {statistics.mean(hold_values):.1%}, "
              f"Median: {statistics.median(hold_values):.1%}, "
              f"StdDev: {statistics.stdev(hold_values) if len(hold_values) > 1 else 0:.1%}")
        
        if ratio_values:
            print(f"\nBUY/SELL Ratio - Mean: {statistics.mean(ratio_values):.2f}, "
                  f"Median: {statistics.median(ratio_values):.2f}")
        
        print(f"BUY-SELL Diff  - Mean: {statistics.mean(diff_values):.1%}, "
              f"Median: {statistics.median(diff_values):.1%}")
        print()
    
    # Find most balanced configs
    print(f"{'='*80}")
    print("TOP 10 MOST BALANCED BUY/SELL RATIOS")
    print(f"{'='*80}\n")
    
    sorted_by_balance = sorted(data_points, key=lambda x: x["buy_sell_diff"])[:10]
    
    for i, dp in enumerate(sorted_by_balance, 1):
        ratio_str = f"{dp['buy_sell_ratio']:.2f}" if dp['buy_sell_ratio'] < 10 else "∞"
        print(f"Rank {i:2d}:")
        print(f"  Final Reward:    {dp['final_reward']:>10.2f}")
        print(f"  BUY:  {dp['buy']:>5.1%}  SELL: {dp['sell']:>5.1%}  HOLD: {dp['hold']:>5.1%}")
        print(f"  BUY/SELL Ratio:  {ratio_str}")
        print(f"  BUY-SELL Diff:   {dp['buy_sell_diff']:.1%} ⭐")
        print()
    
    # Correlation analysis
    print(f"{'='*80}")
    print("CORRELATION INSIGHTS")
    print(f"{'='*80}\n")
    
    # Compare high vs low reward
    if high_reward and low_reward:
        high_ratio = [dp["buy_sell_ratio"] for dp in high_reward if dp["buy_sell_ratio"] < 10]
        low_ratio = [dp["buy_sell_ratio"] for dp in low_reward if dp["buy_sell_ratio"] < 10]
        
        if high_ratio and low_ratio:
            print("Average BUY/SELL Ratio:")
            print(f"  High Reward:  {statistics.mean(high_ratio):.2f}")
            print(f"  Low Reward:   {statistics.mean(low_ratio):.2f}")
            print()
        
        high_diff = [dp["buy_sell_diff"] for dp in high_reward]
        low_diff = [dp["buy_sell_diff"] for dp in low_reward]
        
        print("Average BUY-SELL Imbalance:")
        print(f"  High Reward:  {statistics.mean(high_diff):.1%}")
        print(f"  Low Reward:   {statistics.mean(low_diff):.1%}")
        print()
    
    # Find sweet spot
    balanced_high = [dp for dp in high_reward if abs(dp["buy_sell_diff"]) < 0.15]
    print(f"High reward configs with balanced BUY/SELL (diff < 15%): {len(balanced_high)}/{len(high_reward)}")
    
    if balanced_high:
        avg_reward = statistics.mean([dp["final_reward"] for dp in balanced_high])
        print(f"Average Final Reward (balanced): {avg_reward:.2f}")
        print()
        
        print("Sample balanced & high-reward configs:")
        for dp in balanced_high[:3]:
            print(f"  Reward={dp['final_reward']:.2f}, "
                  f"BUY={dp['buy']:.1%}, SELL={dp['sell']:.1%}, HOLD={dp['hold']:.1%}")
    
    print()
    print(f"{'='*80}")
    print("RECOMMENDATION FOR v448")
    print(f"{'='*80}\n")
    
    if high_reward:
        top_configs = sorted_by_reward[:5]
        avg_buy = statistics.mean([dp["buy"] for dp in top_configs])
        avg_sell = statistics.mean([dp["sell"] for dp in top_configs])
        avg_hold = statistics.mean([dp["hold"] for dp in top_configs])
        avg_diff = statistics.mean([dp["buy_sell_diff"] for dp in top_configs])
        
        print("Based on top 5 high-reward configs:")
        print(f"  Average BUY:  {avg_buy:.1%}")
        print(f"  Average SELL: {avg_sell:.1%}")
        print(f"  Average HOLD: {avg_hold:.1%}")
        print(f"  Average BUY-SELL Diff: {avg_diff:.1%}")
        print()
        
        # Calculate ideal targets
        ideal_buy = avg_buy
        ideal_sell = avg_sell
        ideal_hold = avg_hold
        
        # Normalize to ensure they sum to 1.0
        total = ideal_buy + ideal_sell + ideal_hold
        ideal_buy /= total
        ideal_sell /= total
        ideal_hold /= total
        
        print("Recommended targets for v448:")
        print(f"  buy_target:  {ideal_buy:.2f}")
        print(f"  sell_target: {ideal_sell:.2f}")
        print(f"  hold_target: {ideal_hold:.2f}")
        print()
        
        # Check if BUY ≈ SELL hypothesis holds
        if avg_diff < 0.10:
            print("✅ HYPOTHESIS CONFIRMED: Top profitable configs have balanced BUY/SELL")
        else:
            print(f"⚠️  HYPOTHESIS PARTIAL: BUY-SELL imbalance is {avg_diff:.1%}")
            print("   Consider: Market conditions may favor slight BUY bias")


if __name__ == "__main__":
    analyze_profitability_balance()
