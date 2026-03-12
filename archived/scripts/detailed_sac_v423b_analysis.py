#!/usr/bin/env python3
"""
Detailed SAC v423b Analysis Script

Performs comprehensive analysis of SAC v423b training results including:
- Action distribution analysis
- Position holding interval estimation
- Trading frequency analysis
- Action sequence patterns
- Performance metrics breakdown
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DetailedSACv423bAnalyzer:
    """Detailed analyzer for SAC v423b training results."""

    def __init__(self, report_path: str):
        self.report_path = Path(report_path)
        self.report_data = None
        self.action_dist = {}
        self.total_timesteps = 0

    def load_report(self) -> bool:
        """Load training report data."""
        try:
            with open(self.report_path, "r") as f:
                self.report_data = json.load(f)

            # Extract key data
            training_stats = self.report_data.get("training_stats", {})
            self.action_dist = training_stats.get("action_distribution", {})
            self.total_timesteps = training_stats.get("total_timesteps", 0)

            return True
        except Exception as e:
            logger.error(f"Failed to load report {self.report_path}: {e}")
            return False

    def analyze_action_distribution(self) -> Dict[str, Any]:
        """Analyze action distribution in detail."""
        if not self.action_dist:
            return {}

        total_actions = sum(self.action_dist.values())
        analysis = {
            "total_actions": total_actions,
            "action_counts": self.action_dist.copy(),
            "action_percentages": {},
            "action_ratios": {},
            "dominant_action": max(self.action_dist.items(), key=lambda x: x[1])[0],
            "dominant_ratio": max(self.action_dist.values()) / total_actions,
            "action_diversity": self._calculate_diversity(self.action_dist),
            "trading_intensity": self._calculate_trading_intensity(self.action_dist),
        }

        # Calculate percentages and ratios
        for action, count in self.action_dist.items():
            analysis["action_percentages"][action] = (count / total_actions) * 100
            analysis["action_ratios"][action] = count / total_actions

        return analysis

    def estimate_position_holding_intervals(self) -> Dict[str, Any]:
        """
        Estimate position holding intervals using probabilistic approach.

        Since we don't have the actual action sequence, we use statistical
        estimation based on action distribution and expected behavior.
        """
        if not self.action_dist:
            return {}

        # Action transition probabilities (simplified model)
        # Based on typical trading behavior patterns
        hold_prob = self.action_dist.get("HOLD", 0) / sum(self.action_dist.values())
        trade_prob = 1 - hold_prob  # Probability of making a trade (BUY or SELL)

        # Estimate average holding intervals
        # Using geometric distribution: expected holding time = 1/(1-p_hold)
        avg_hold_interval = 1 / (1 - hold_prob) if hold_prob < 1 else float("inf")

        # Estimate trading frequency
        trades_per_timestep = trade_prob
        estimated_trades = int(trades_per_timestep * self.total_timesteps)

        # Position holding analysis
        analysis = {
            "hold_probability": hold_prob,
            "trade_probability": trade_prob,
            "avg_holding_interval_steps": avg_hold_interval,
            "avg_holding_interval_minutes": avg_hold_interval
            * 5,  # Assuming 5-min candles
            "estimated_total_trades": estimated_trades,
            "trading_frequency_per_1000_steps": trades_per_timestep * 1000,
            "position_stability": hold_prob,  # Higher = more stable positions
            "trading_aggressiveness": trade_prob,  # Higher = more aggressive trading
        }

        return analysis

    def analyze_action_patterns(self) -> Dict[str, Any]:
        """
        Analyze expected action patterns and sequences.
        """
        if not self.action_dist:
            return {}

        # Calculate action balance metrics
        buy_ratio = self.action_dist.get("BUY", 0) / sum(self.action_dist.values())
        sell_ratio = self.action_dist.get("SELL", 0) / sum(self.action_dist.values())
        hold_ratio = self.action_dist.get("HOLD", 0) / sum(self.action_dist.values())

        # Action balance analysis
        buy_sell_balance = abs(buy_ratio - sell_ratio)
        action_entropy = self._calculate_entropy(self.action_dist)

        # Pattern analysis
        patterns = {
            "buy_sell_balance": buy_sell_balance,
            "balance_interpretation": self._interpret_balance(buy_sell_balance),
            "action_entropy": action_entropy,
            "entropy_interpretation": self._interpret_entropy(action_entropy),
            "expected_consecutive_holds": self._estimate_consecutive_holds(hold_ratio),
            "trading_style": self._classify_trading_style(
                buy_ratio, sell_ratio, hold_ratio
            ),
        }

        return patterns

    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics in detail."""
        if not self.report_data:
            return {}

        training_stats = self.report_data.get("training_stats", {})
        perf_metrics = self.report_data.get("performance_metrics", {})

        analysis = {
            "training_efficiency": perf_metrics.get("training_efficiency", 0),
            "steps_per_second": training_stats.get("steps_per_second", 0),
            "total_training_time": training_stats.get("training_time", 0),
            "model_path": training_stats.get("model_path", "N/A"),
            "memory_usage": self.report_data.get("system_info", {}).get(
                "memory_available", 0
            ),
            "action_diversity_score": perf_metrics.get("action_diversity", 0),
            "dominant_action_ratio": perf_metrics.get("dominant_action_ratio", 0),
        }

        return analysis

    def _calculate_diversity(self, action_dist: Dict[str, float]) -> float:
        """Calculate action diversity using Shannon entropy normalized."""
        total = sum(action_dist.values())
        if total == 0:
            return 0

        entropy = 0
        for count in action_dist.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        # Normalize by maximum possible entropy (log2 of number of actions)
        max_entropy = np.log2(len(action_dist))
        return entropy / max_entropy if max_entropy > 0 else 0

    def _calculate_trading_intensity(self, action_dist: Dict[str, float]) -> float:
        """Calculate trading intensity (0-1 scale)."""
        hold_ratio = action_dist.get("HOLD", 0) / sum(action_dist.values())
        return 1 - hold_ratio  # Higher = more trading

    def _calculate_entropy(self, action_dist: Dict[str, float]) -> float:
        """Calculate Shannon entropy of action distribution."""
        total = sum(action_dist.values())
        if total == 0:
            return 0

        entropy = 0
        for count in action_dist.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    def _interpret_balance(self, balance: float) -> str:
        """Interpret buy-sell balance."""
        if balance < 0.1:
            return "Well-balanced (BUY/SELL actions are evenly distributed)"
        elif balance < 0.2:
            return "Moderately balanced"
        elif balance < 0.3:
            return "Somewhat imbalanced"
        else:
            return "Highly imbalanced (strong bias toward one action)"

    def _interpret_entropy(self, entropy: float) -> str:
        """Interpret action entropy."""
        if entropy < 1.0:
            return "Low diversity (predictable behavior)"
        elif entropy < 1.5:
            return "Moderate diversity"
        else:
            return "High diversity (unpredictable behavior)"

    def _estimate_consecutive_holds(self, hold_ratio: float) -> float:
        """Estimate expected consecutive HOLD actions."""
        if hold_ratio >= 1:
            return float("inf")
        return 1 / (1 - hold_ratio)

    def _classify_trading_style(
        self, buy_ratio: float, sell_ratio: float, hold_ratio: float
    ) -> str:
        """Classify overall trading style."""
        if hold_ratio > 0.5:
            return "Conservative (prefers holding positions)"
        elif buy_ratio > sell_ratio + 0.1:
            return "Bullish (aggressive buying)"
        elif sell_ratio > buy_ratio + 0.1:
            return "Bearish (aggressive selling)"
        elif hold_ratio < 0.2:
            return "Highly Active (frequent trading)"
        else:
            return "Balanced (moderate activity)"

    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive analysis report."""
        if not self.load_report():
            print("❌ Failed to load training report")
            return

        print("🔬 Detailed SAC v423b Analysis Report")
        print("=" * 60)

        # Basic information
        metadata = self.report_data.get("metadata", {})
        training_stats = self.report_data.get("training_stats", {})

        print("📊 Training Overview:")
        print(f"   Model: {metadata.get('model_name', 'N/A')}")
        print(f"   Algorithm: {metadata.get('algorithm', 'N/A')}")
        print(f"   Total Timesteps: {training_stats.get('total_timesteps', 0):,}")
        print(f"   Training Time: {training_stats.get('training_time', 0):.2f} seconds")
        print(f"   Steps/Second: {training_stats.get('steps_per_second', 0):.2f}")
        print()

        # Action distribution analysis
        action_analysis = self.analyze_action_distribution()
        if action_analysis:
            print("🎯 Action Distribution Analysis:")
            print(f"   Total Actions Recorded: {action_analysis['total_actions']:,}")
            print("   Action Breakdown:")
            for action, count in action_analysis["action_counts"].items():
                pct = action_analysis["action_percentages"][action]
                print(f"     {action}: {count:,} ({pct:.1f}%)")
            print(
                f"   Dominant Action: {action_analysis['dominant_action']} ({action_analysis['dominant_ratio']:.3f})"
            )
            print(f"   Action Diversity: {action_analysis['action_diversity']:.3f}")
            print(f"   Trading Intensity: {action_analysis['trading_intensity']:.3f}")
            print()

        # Position holding analysis
        holding_analysis = self.estimate_position_holding_intervals()
        if holding_analysis:
            print("📈 Position Holding Analysis:")
            print(f"   Hold Probability: {holding_analysis['hold_probability']:.3f}")
            print(f"   Trade Probability: {holding_analysis['trade_probability']:.3f}")
            print(
                f"   Average Holding Interval: {holding_analysis['avg_holding_interval_steps']:.1f} steps"
            )
            print(
                f"   Average Holding Time: {holding_analysis['avg_holding_interval_minutes']:.1f} minutes"
            )
            print(
                f"   Estimated Total Trades: {holding_analysis['estimated_total_trades']:,}"
            )
            print(
                f"   Trading Frequency: {holding_analysis['trading_frequency_per_1000_steps']:.1f} trades/1000 steps"
            )
            print(
                f"   Position Stability: {holding_analysis['position_stability']:.3f}"
            )
            print(
                f"   Trading Aggressiveness: {holding_analysis['trading_aggressiveness']:.3f}"
            )
            print()

        # Action patterns
        pattern_analysis = self.analyze_action_patterns()
        if pattern_analysis:
            print("🔄 Action Pattern Analysis:")
            print(f"   Buy-Sell Balance: {pattern_analysis['buy_sell_balance']:.3f}")
            print(
                f"   Balance Assessment: {pattern_analysis['balance_interpretation']}"
            )
            print(f"   Action Entropy: {pattern_analysis['action_entropy']:.3f}")
            print(
                f"   Entropy Assessment: {pattern_analysis['entropy_interpretation']}"
            )
            print(
                f"   Expected Consecutive Holds: {pattern_analysis['expected_consecutive_holds']:.1f}"
            )
            print(f"   Trading Style: {pattern_analysis['trading_style']}")
            print()

        # Performance metrics
        perf_analysis = self.analyze_performance_metrics()
        if perf_analysis:
            print("⚡ Performance Analysis:")
            print(f"   Training Efficiency: {perf_analysis['training_efficiency']:.4f}")
            print(
                f"   Memory Usage: {perf_analysis['memory_usage'] / (1024**3):.1f} GB available"
            )
            print(
                f"   Action Diversity Score: {perf_analysis['action_diversity_score']:.3f}"
            )
            print()

        # Summary and insights
        print("💡 Key Insights:")
        if action_analysis and holding_analysis:
            dominant = action_analysis["dominant_action"]
            stability = holding_analysis["position_stability"]
            intensity = action_analysis["trading_intensity"]

            if dominant == "BUY" and intensity > 0.8:
                print("   • Highly aggressive BUY-focused strategy")
                print("   • Very active trading with minimal position holding")
            elif dominant == "HOLD" and stability > 0.5:
                print("   • Conservative position-holding strategy")
                print("   • Prefers maintaining existing positions")
            elif intensity > 0.7:
                print("   • Active trading strategy with frequent position changes")
                print("   • Lower position stability indicates dynamic market response")
            else:
                print("   • Balanced trading approach")
                print("   • Moderate activity with reasonable position management")

        print()
        print("✅ Detailed analysis complete!")


def main():
    """Main analysis function."""
    report_path = "reports/training_report_sac_sac_v423b_step_test_20251018_172150.json"

    if not os.path.exists(report_path):
        print(f"❌ Report file not found: {report_path}")
        print("Available reports:")
        reports_dir = Path("reports")
        if reports_dir.exists():
            for report in reports_dir.glob("training_report_sac_*.json"):
                print(f"  - {report.name}")
        return

    analyzer = DetailedSACv423bAnalyzer(report_path)
    analyzer.generate_comprehensive_report()


if __name__ == "__main__":
    main()
