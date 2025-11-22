#!/usr/bin/env python3
"""
Enhanced AB Test Results Analyzer with reward_components visualization.

Analyzes AB test results and visualizes reward_components to understand
which reward shaping strategies are most effective.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def load_training_reports(reports_dir: Path, pattern: str = "training_report_*.json") -> List[Dict[str, Any]]:
    """Load all training reports matching pattern."""
    reports = []
    for report_path in reports_dir.glob(pattern):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_report_file'] = report_path.name
                reports.append(data)
        except Exception as e:
            print(f"Warning: Could not load {report_path.name}: {e}")
    return reports


def extract_reward_components(report: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract reward_components from training report."""
    # Try direct key
    if "reward_components" in report:
        return report["reward_components"]
    
    # Try nested in training_stats
    stats = report.get("training_stats", {})
    if "reward_components" in stats:
        return stats["reward_components"]
    
    return None


def analyze_action_balance(report: Dict[str, Any]) -> Dict[str, float]:
    """Calculate action balance metrics."""
    stats = report.get("training_stats", {})
    action_dist = stats.get("action_distribution", {})
    
    buy = action_dist.get("BUY", 0)
    sell = action_dist.get("SELL", 0)
    hold = action_dist.get("HOLD", 0)
    total = buy + sell + hold
    
    if total == 0:
        return {"balance_score": float('inf'), "buy_ratio": 0, "sell_ratio": 0, "hold_ratio": 0}
    
    buy_ratio = buy / total
    sell_ratio = sell / total
    hold_ratio = hold / total
    
    # Balance score: deviation from ideal 1/3 each
    balance_score = abs(buy_ratio - 0.333) + abs(sell_ratio - 0.333) + abs(hold_ratio - 0.333)
    
    return {
        "balance_score": balance_score,
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "hold_ratio": hold_ratio
    }


def correlate_components_with_balance(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Correlate reward_components with action balance."""
    data_points = []
    
    for report in reports:
        components = extract_reward_components(report)
        if not components:
            continue
        
        balance_metrics = analyze_action_balance(report)
        
        data_points.append({
            "report_file": report.get("_report_file", "unknown"),
            "reward_components": components,
            "balance_metrics": balance_metrics,
            "config": report.get("configuration", {})
        })
    
    if not data_points:
        return {"error": "No reports with reward_components found"}
    
    # Find best balanced configuration
    best_config = min(data_points, key=lambda x: x["balance_metrics"]["balance_score"])
    
    # Aggregate component statistics by balance quality
    good_balance = [dp for dp in data_points if dp["balance_metrics"]["balance_score"] < 0.2]
    poor_balance = [dp for dp in data_points if dp["balance_metrics"]["balance_score"] > 0.4]
    
    def aggregate_components(data_points: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Aggregate component statistics."""
        components = defaultdict(list)
        for dp in data_points:
            for key, value in dp["reward_components"].items():
                components[key].append(float(value))
        
        return {
            key: {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
            for key, values in components.items() if values
        }
    
    return {
        "total_reports": len(data_points),
        "best_balanced": {
            "report_file": best_config["report_file"],
            "balance_score": best_config["balance_metrics"]["balance_score"],
            "action_distribution": {
                "buy": best_config["balance_metrics"]["buy_ratio"],
                "sell": best_config["balance_metrics"]["sell_ratio"],
                "hold": best_config["balance_metrics"]["hold_ratio"]
            },
            "reward_components": best_config["reward_components"]
        },
        "good_balance_stats": {
            "count": len(good_balance),
            "components": aggregate_components(good_balance)
        },
        "poor_balance_stats": {
            "count": len(poor_balance),
            "components": aggregate_components(poor_balance)
        }
    }


def print_analysis_report(analysis: Dict[str, Any], output_file: Optional[Path] = None):
    """Print formatted analysis report."""
    if "error" in analysis:
        print(f"\n❌ {analysis['error']}")
        return
    
    print("\n" + "="*80)
    print("AB Test Results Analysis with reward_components")
    print("="*80)
    
    print(f"\nTotal reports analyzed: {analysis['total_reports']}")
    
    # Best balanced configuration
    best = analysis["best_balanced"]
    print("\n" + "-"*80)
    print("Best Balanced Configuration:")
    print("-"*80)
    print(f"  Report: {best['report_file']}")
    print(f"  Balance Score: {best['balance_score']:.4f}")
    print(f"  Action Distribution:")
    print(f"    BUY:  {best['action_distribution']['buy']:.2%}")
    print(f"    SELL: {best['action_distribution']['sell']:.2%}")
    print(f"    HOLD: {best['action_distribution']['hold']:.2%}")
    print(f"\n  Reward Components:")
    for key, value in best['reward_components'].items():
        print(f"    {key:20s}: {value:8.6f}")
    
    # Good vs Poor balance comparison
    good = analysis["good_balance_stats"]
    poor = analysis["poor_balance_stats"]
    
    print("\n" + "-"*80)
    print(f"Good Balance (score < 0.2): {good['count']} reports")
    print("-"*80)
    if good['components']:
        for key, stats in good['components'].items():
            print(f"  {key:20s}: mean={stats['mean']:8.6f}, min={stats['min']:8.6f}, max={stats['max']:8.6f}")
    
    print("\n" + "-"*80)
    print(f"Poor Balance (score > 0.4): {poor['count']} reports")
    print("-"*80)
    if poor['components']:
        for key, stats in poor['components'].items():
            print(f"  {key:20s}: mean={stats['mean']:8.6f}, min={stats['min']:8.6f}, max={stats['max']:8.6f}")
    
    # Insights
    print("\n" + "-"*80)
    print("Insights:")
    print("-"*80)
    
    if good['components'] and poor['components']:
        common_keys = set(good['components'].keys()) & set(poor['components'].keys())
        
        for key in common_keys:
            good_mean = good['components'][key]['mean']
            poor_mean = poor['components'][key]['mean']
            diff = good_mean - poor_mean
            
            if abs(diff) > 0.001:
                direction = "higher" if diff > 0 else "lower"
                print(f"  • Good balance has {direction} {key}: {diff:+.6f} difference")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze AB test results with reward_components")
    parser.add_argument("--reports-dir", default="reports", help="Directory containing training reports")
    parser.add_argument("--pattern", default="training_report_*.json", help="File pattern to match")
    parser.add_argument("--output", help="Output JSON file for analysis results")
    parser.add_argument("--filter-recent", type=int, help="Only analyze N most recent reports")
    
    args = parser.parse_args()
    
    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"Error: Directory not found: {reports_dir}")
        return
    
    print(f"Loading reports from: {reports_dir}")
    reports = load_training_reports(reports_dir, args.pattern)
    print(f"Loaded {len(reports)} reports")
    
    if args.filter_recent:
        # Sort by modification time and take most recent
        reports_with_time = [
            (r, (reports_dir / r["_report_file"]).stat().st_mtime)
            for r in reports
        ]
        reports_with_time.sort(key=lambda x: x[1], reverse=True)
        reports = [r for r, _ in reports_with_time[:args.filter_recent]]
        print(f"Filtered to {len(reports)} most recent reports")
    
    print("\nAnalyzing reward_components and action balance...")
    analysis = correlate_components_with_balance(reports)
    
    output_path = Path(args.output) if args.output else None
    print_analysis_report(analysis, output_path)


if __name__ == "__main__":
    main()
