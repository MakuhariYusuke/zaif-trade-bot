#!/usr/bin/env python3
"""Monitor ongoing AB tests and report progress."""

import time
from pathlib import Path
from datetime import datetime

from ztb.reporting.services.catalog import (
    extract_action_distribution_from_payload,
    list_training_reports,
    load_training_report,
)
from ztb.trading.environment.components.rewards.utils import RewardUtils
from ztb.utils.safety import ensure_dict


def check_recent_reports(minutes=10):
    """Check reports created in last N minutes."""
    reports_dir = Path("reports")
    cutoff_time = time.time() - (minutes * 60)
    
    recent_reports = []
    for report_file in list_training_reports(reports_dir=reports_dir):
        try:
            modified = report_file.stat().st_mtime
        except OSError:
            continue
        if modified > cutoff_time:
            recent_reports.append(report_file)
    
    return sorted(recent_reports, key=lambda x: x.stat().st_mtime, reverse=True)


def analyze_report(report_path):
    """Extract key metrics from a report."""
    try:
        data = load_training_report(report_path)
        if data is None:
            return None
        
        ts = ensure_dict(data.get("training_stats"))
        actions = extract_action_distribution_from_payload(data)
        
        # Calculate balance score (distance from 33/33/33)
        buy = actions.get("BUY", 0)
        sell = actions.get("SELL", 0)
        hold = actions.get("HOLD", 0)

        balance_score = RewardUtils.calculate_balance_deviation_from_ratios(
            [buy, sell, hold], [0.333, 0.333, 0.333]
        )
        
        return {
            "file": report_path.name,
            "timesteps": ts.get("total_timesteps", 0),
            "actions": actions,
            "balance_score": balance_score,
            "ab_tag": data.get("metadata", {}).get("ab_tag", "unknown"),
            "time": datetime.fromtimestamp(report_path.stat().st_mtime).strftime("%H:%M:%S"),
        }
    except Exception:
        return None


def main():
    """Monitor and report on recent training runs."""
    print("=" * 80)
    print("AB TEST MONITOR - Recent Training Reports (Last 10 minutes)")
    print("=" * 80)
    
    recent = check_recent_reports(minutes=10)
    
    if not recent:
        print("\n⚠️  No recent reports found in last 10 minutes")
        print("   Tests may still be running or not yet completed")
        return
    
    print(f"\n📊 Found {len(recent)} recent report(s):")
    print()
    
    results = []
    for report_path in recent:
        info = analyze_report(report_path)
        if info:
            results.append(info)
    
    # Sort by balance score (best first)
    results.sort(key=lambda x: x["balance_score"])
    
    for i, result in enumerate(results, 1):
        actions = result["actions"]
        buy = actions.get("BUY", 0) * 100
        sell = actions.get("SELL", 0) * 100
        hold = actions.get("HOLD", 0) * 100
        
        print(f"{i}. {result['time']} | Balance Score: {result['balance_score']:.3f}")
        print(f"   BUY={buy:.1f}%, SELL={sell:.1f}%, HOLD={hold:.1f}%")
        print(f"   Steps: {result['timesteps']}, Tag: {result['ab_tag'][:30]}")
        print()
    
    # Show best result
    if results:
        best = results[0]
        print("=" * 80)
        print(f"🏆 BEST BALANCE: Score {best['balance_score']:.3f}")
        actions = best["actions"]
        print(f"   BUY={actions.get('BUY', 0)*100:.1f}%, "
              f"SELL={actions.get('SELL', 0)*100:.1f}%, "
              f"HOLD={actions.get('HOLD', 0)*100:.1f}%")
        print(f"   File: {best['file']}")
        print("=" * 80)


if __name__ == "__main__":
    main()
