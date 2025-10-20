#!/usr/bin/env python3
"""
SAC v426 vs v427 Comprehensive Comparison Analysis

Compare v426 and v427 performance metrics and identify improvements.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_backtest_results(file_path: str) -> Dict[str, Any]:
    """Load backtest results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_v424_v427(v424_path: str, v427_path: str) -> Dict[str, Any]:
    """
    Compare v424 and v427 backtest results.

    Args:
        v424_path: Path to v424 backtest results
        v427_path: Path to v427 backtest results

    Returns:
        Comparison analysis
    """
    logger.info("Comparing v424 vs v427 backtest results...")

    # Load results
    v424_results = load_backtest_results(v424_path)
    v427_results = load_backtest_results(v427_path)

    comparison = {
        "v424_summary": v424_results,
        "v427_summary": v427_results["summary"],
        "performance_comparison": {},
        "improvement_analysis": {},
        "recommendations": [],
    }

    # Performance comparison
    v424_total_return = v424_results.get("total_return_pct", 0) / 100
    v427_total_return = v427_results["summary"]["total_return"]

    v424_sharpe = v424_results.get("sharpe_ratio", 0)
    v427_sharpe = v427_results["summary"]["sharpe_ratio"]

    v424_max_dd = v424_results.get("max_drawdown", 0)
    v427_max_dd = v427_results["summary"]["max_drawdown"]

    v424_win_rate = v424_results.get("win_rate", 0)
    v427_win_rate = v427_results["summary"]["win_rate"]

    comparison["performance_comparison"] = {
        "total_return": {
            "v424": v424_total_return,
            "v427": v427_total_return,
            "difference": v427_total_return - v424_total_return,
            "improvement_pct": (
                (v427_total_return - v424_total_return) / abs(v424_total_return)
            )
            * 100
            if v424_total_return != 0
            else 0,
        },
        "sharpe_ratio": {
            "v424": v424_sharpe,
            "v427": v427_sharpe,
            "difference": v427_sharpe - v424_sharpe,
            "improvement_pct": ((v427_sharpe - v424_sharpe) / abs(v424_sharpe)) * 100
            if v424_sharpe != 0
            else 0,
        },
        "max_drawdown": {
            "v424": v424_max_dd,
            "v427": v427_max_dd,
            "difference": v427_max_dd - v424_max_dd,  # More negative is worse
            "improvement_pct": ((v427_max_dd - v424_max_dd) / abs(v424_max_dd)) * 100
            if v424_max_dd != 0
            else 0,
        },
        "win_rate": {
            "v424": v424_win_rate,
            "v427": v427_win_rate,
            "difference": v427_win_rate - v424_win_rate,
            "improvement_pct": ((v427_win_rate - v424_win_rate) / abs(v424_win_rate))
            * 100
            if v424_win_rate != 0
            else 0,
        },
    }

    # Action distribution comparison
    v424_actions = v424_results.get("action_distribution", {})
    v427_actions = v427_results.get("action_distribution", {})

    comparison["action_distribution_comparison"] = {
        "v424": v424_actions,
        "v427": v427_actions,
        "analysis": analyze_action_distribution(v424_actions, v427_actions),
    }

    # Improvement analysis
    comparison["improvement_analysis"] = analyze_improvements(comparison)

    # Generate recommendations
    comparison["recommendations"] = generate_recommendations(comparison)

    return comparison


def analyze_action_distribution(
    v424_actions: Dict[str, float], v427_actions: Dict[str, float]
) -> Dict[str, Any]:
    """Analyze differences in action distribution."""
    analysis = {}

    # Convert action keys for consistency
    v424_normalized = {}
    for key, value in v424_actions.items():
        if key.upper() == "HOLD":
            v424_normalized["0"] = value
        elif key.upper() == "BUY":
            v424_normalized["1"] = value
        elif key.upper() == "SELL":
            v424_normalized["-1"] = value

    # Compare distributions
    for action in ["-1", "0", "1"]:
        v424_pct = v424_normalized.get(action, 0)
        v427_pct = v427_actions.get(action, 0)
        difference = v427_pct - v424_pct

        analysis[action] = {
            "v424": v424_pct,
            "v427": v427_pct,
            "difference": difference,
            "assessment": "overtrading"
            if difference > 0.1
            else "undertrading"
            if difference < -0.1
            else "balanced",
        }

    return analysis


def analyze_improvements(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze what improvements were achieved."""
    perf_comp = comparison["performance_comparison"]
    action_comp = comparison["action_distribution_comparison"]

    analysis = {
        "overall_assessment": "",
        "key_findings": [],
        "technical_improvements": [],
        "issues_identified": [],
    }

    # Overall assessment
    return_improvement = perf_comp["total_return"]["improvement_pct"]
    if return_improvement > 10:
        analysis["overall_assessment"] = "significant_improvement"
    elif return_improvement > 0:
        analysis["overall_assessment"] = "moderate_improvement"
    elif return_improvement > -20:
        analysis["overall_assessment"] = "slight_decline"
    else:
        analysis["overall_assessment"] = "significant_decline"

    # Key findings
    if perf_comp["win_rate"]["difference"] > 0:
        analysis["key_findings"].append(
            "Improved win rate indicates better trade selection"
        )
    else:
        analysis["key_findings"].append(
            "Declining win rate suggests trade quality issues"
        )

    if perf_comp["sharpe_ratio"]["difference"] > 0:
        analysis["key_findings"].append("Better risk-adjusted returns")
    else:
        analysis["key_findings"].append("Worse risk-adjusted performance")

    # Action distribution issues
    hold_trading = action_comp["analysis"]["0"]
    if hold_trading["v427"] < 0.05:
        analysis["issues_identified"].append(
            "Extremely low HOLD percentage indicates overtrading"
        )

    buy_sell_balance = abs(
        action_comp["analysis"]["1"]["v427"] - action_comp["analysis"]["-1"]["v427"]
    )
    if buy_sell_balance > 0.3:
        analysis["issues_identified"].append(
            "Imbalanced BUY/SELL actions suggest directional bias"
        )

    # Technical improvements
    analysis["technical_improvements"] = [
        "Market regime awareness features",
        "Correlation-aware reward system",
        "Ensemble prediction signals",
        "Adaptive feature selection",
        "Meta-learning integration",
        "Federated learning capabilities",
        "Continual learning for knowledge retention",
    ]

    return analysis


def generate_recommendations(comparison: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations for improvement."""
    recommendations = []
    perf_comp = comparison["performance_comparison"]
    action_comp = comparison["action_distribution_comparison"]

    # Performance-based recommendations
    if perf_comp["total_return"]["improvement_pct"] < 0:
        recommendations.append(
            "Investigate reward function calibration - returns declined significantly"
        )

    if perf_comp["sharpe_ratio"]["improvement_pct"] < -50:
        recommendations.append(
            "Address risk management - Sharpe ratio deteriorated substantially"
        )

    if perf_comp["win_rate"]["difference"] < -0.1:
        recommendations.append(
            "Improve trade entry/exit logic - win rate dropped significantly"
        )

    # Action distribution recommendations
    if action_comp["analysis"]["0"]["v427"] < 0.05:
        recommendations.append(
            "Reduce overtrading by implementing stricter HOLD thresholds"
        )
        recommendations.append(
            "Add position sizing limits to prevent excessive trading"
        )

    if (
        abs(
            action_comp["analysis"]["1"]["v427"] - action_comp["analysis"]["-1"]["v427"]
        )
        > 0.2
    ):
        recommendations.append(
            "Balance BUY/SELL actions through reward function adjustments"
        )

    # Technical recommendations
    recommendations.extend(
        [
            "Validate feature engineering pipeline - ensure all v427 features are properly integrated",
            "Test ensemble system components individually before integration",
            "Implement proper regime detection and adaptation logic",
            "Add comprehensive hyperparameter tuning for v427-specific parameters",
            "Consider gradual rollout of advanced features rather than all-at-once integration",
        ]
    )

    return recommendations


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="SAC v424 vs v427 Comparison Analysis")
    parser.add_argument(
        "--v424",
        default="results/backtest_v424_cost_aware.json",
        help="v424 backtest results",
    )
    parser.add_argument(
        "--v427",
        default="results/backtest_v427_complete_fixed.json",
        help="v427 backtest results",
    )
    parser.add_argument(
        "--output",
        default="results/v424_v427_comparison.json",
        help="Output comparison file",
    )

    args = parser.parse_args()

    # Perform comparison
    comparison = compare_v424_v427(args.v424, args.v427)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("SAC v424 vs v427 COMPARISON ANALYSIS")
    print("=" * 80)

    perf = comparison["performance_comparison"]
    print("\nPERFORMANCE METRICS:")
    print(f"v424 Total Return: {perf['total_return']['v424']*100:.2f}%")
    print(f"v427 Total Return: {perf['total_return']['v427']*100:.2f}%")
    print(f"v424 Sharpe Ratio: {perf['sharpe_ratio']['v424']:.2f}")
    print(f"v427 Sharpe Ratio: {perf['sharpe_ratio']['v427']:.2f}")

    print("\nACTION DISTRIBUTION:")
    actions = comparison["action_distribution_comparison"]
    print(f"v424 HOLD %: {actions['v424'].get('HOLD', 0)*100:.1f}%")
    print(f"v427 HOLD %: {actions['v427'].get('0', 0)*100:.1f}%")

    print("\nANALYSIS:")
    analysis = comparison["improvement_analysis"]
    print(
        f"Overall Assessment: {analysis['overall_assessment'].replace('_', ' ').title()}"
    )

    print("\nKEY FINDINGS:")
    for finding in analysis["key_findings"]:
        print(f"• {finding}")

    print("\nISSUES IDENTIFIED:")
    for issue in analysis["issues_identified"]:
        print(f"• {issue}")

    print("\nRECOMMENDATIONS:")
    for rec in comparison["recommendations"]:
        print(f"• {rec}")

    print("=" * 80)
    print(f"Detailed results saved to: {args.output}")


if __name__ == "__main__":
    sys.exit(main())
