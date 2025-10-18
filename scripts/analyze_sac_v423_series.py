#!/usr/bin/env python3
"""
SAC v423 Series Analysis Script

Analyzes all training results from SAC v423 series.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACv423SeriesAnalyzer:
    """Analyzer for SAC v423 series training results."""

    def __init__(self):
        self.reports_dir = Path("reports")

    def find_v423_training_results(self) -> List[Path]:
        """Find all SAC v423 series training results."""
        if not self.reports_dir.exists():
            return []

        # Look for v423 training reports
        report_files = list(self.reports_dir.glob("training_report_sac_sac_v423*"))
        # Sort by modification time, most recent first
        report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return report_files

    def load_training_report(self, report_path: Path) -> Optional[Dict[str, Any]]:
        """Load training report from file."""
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load report {report_path}: {e}")
            return None

    def analyze_series_results(self) -> None:
        """Analyze all SAC v423 series results."""
        print("🔍 Analyzing SAC v423 Series Training Results")
        print("=" * 55)

        report_paths = self.find_v423_training_results()
        if not report_paths:
            print("❌ No SAC v423 training results found in reports/")
            return

        print(f"📁 Found {len(report_paths)} SAC v423 training results:")
        for i, path in enumerate(report_paths, 1):
            print(f"   {i}. {path.name}")
        print()

        results_summary = []

        # Analyze each result
        for i, report_path in enumerate(report_paths, 1):
            print(f"📊 Analysis #{i}: {report_path.name}")
            print("-" * 40)

            report_data = self.load_training_report(report_path)
            if not report_data:
                print("❌ Failed to load report")
                continue

            # Extract key information
            metadata = report_data.get('metadata', {})
            config = report_data.get('configuration', {})
            training_stats = report_data.get('training_stats', {})
            performance_metrics = report_data.get('performance_metrics', {})

            model_name = config.get('model_name', 'Unknown')
            total_timesteps = training_stats.get('total_timesteps', 0)
            training_time = training_stats.get('training_time', 0)
            action_dist = training_stats.get('action_distribution', {})

            # Display basic information
            print("📋 Training Summary:")
            print(f"   Model: {model_name}")
            print(f"   Algorithm: SAC")
            print(f"   Total Timesteps: {total_timesteps:,}")
            print(f"   Training Time: {training_time:.2f} seconds")
            print(f"   Steps/Second: {performance_metrics.get('steps_per_second', 0):.2f}")

            # Analyze action distribution
            if action_dist:
                print("\n🎯 Action Distribution:")
                total_actions = sum(action_dist.values())
                for action, count in action_dist.items():
                    percentage = (count / total_actions) * 100
                    print(f"   {action}: {count:,} ({percentage:.1f}%)")

            # Performance metrics
            if performance_metrics:
                print("\n📈 Performance Metrics:")
                print(f"   Action Diversity: {performance_metrics.get('action_diversity', 0):.3f}")
                print(f"   Dominant Action: {performance_metrics.get('dominant_action', 'N/A')}")
                print(f"   Dominant Action Ratio: {performance_metrics.get('dominant_action_ratio', 0):.3f}")

            # Store summary for comparison
            results_summary.append({
                'model': model_name,
                'timesteps': total_timesteps,
                'training_time': training_time,
                'steps_per_second': performance_metrics.get('steps_per_second', 0),
                'action_distribution': action_dist,
                'action_diversity': performance_metrics.get('action_diversity', 0),
                'dominant_action': performance_metrics.get('dominant_action', 'N/A'),
                'dominant_ratio': performance_metrics.get('dominant_action_ratio', 0)
            })

            print()

        # Comparative analysis
        if len(results_summary) > 1:
            print("📊 Comparative Analysis")
            print("=" * 30)

            # Sort by timesteps
            results_summary.sort(key=lambda x: x['timesteps'])

            print("\nModel Performance Comparison:")
            print("Timesteps | Model | Time(s) | SPS | Diversity | Dominant")
            print("-" * 65)
            for result in results_summary:
                print(f"{result['timesteps']:8d} | {result['model'][:10]:10} | {result['training_time']:7.1f} | {result['steps_per_second']:4.1f} | {result['action_diversity']:.3f} | {result['dominant_action']}")

            # Action distribution trends
            print("\n🎯 Action Distribution Trends:")
            actions = ['HOLD', 'BUY', 'SELL']
            for action in actions:
                trend = []
                for result in results_summary:
                    dist = result['action_distribution']
                    if action in dist:
                        total = sum(dist.values())
                        percentage = (dist[action] / total) * 100
                        trend.append(f"{percentage:.1f}%")
                    else:
                        trend.append("N/A")

                print(f"   {action}: {' -> '.join(trend)}")

        print("\n✅ Series analysis complete!")
        print(f"📁 Analyzed {len(report_paths)} SAC v423 training results")


def main():
    """Main analysis function."""
    analyzer = SACv423SeriesAnalyzer()
    analyzer.analyze_series_results()


if __name__ == "__main__":
    main()