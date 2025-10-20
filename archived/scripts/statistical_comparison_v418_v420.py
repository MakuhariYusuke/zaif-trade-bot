#!/usr/bin/env python3
"""
Statistical Comparison Script for SAC v418 vs v420

Runs multiple paper trading simulations and performs statistical analysis.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger


def run_single_paper_trade(script_path: str, model_version: str) -> Dict[str, Any]:
    """Run a single paper trading simulation."""
    logger = get_logger(__name__)

    try:
        # Run the paper trading script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode != 0:
            logger.error(f"Paper trading failed for {model_version}: {result.stderr}")
            return None

        # Parse the output to extract key metrics
        output = result.stdout

        # Extract metrics from output
        metrics = {}

        # Look for portfolio value
        if "Final Portfolio:" in output:
            # v418 format
            lines = output.split("\n")
            for line in lines:
                if "Final Portfolio:" in line:
                    try:
                        value_str = line.split(":")[1].strip().split()[0]
                        metrics["final_portfolio"] = float(value_str.replace(",", ""))
                    except:
                        pass
                elif "Total Return:" in line:
                    try:
                        return_str = line.split(":")[1].strip().replace("%", "")
                        metrics["total_return"] = float(return_str)
                    except:
                        pass
        else:
            # v420 format
            lines = output.split("\n")
            for line in lines:
                if "Average Portfolio Value:" in line:
                    try:
                        value_str = line.split(":")[1].strip().split()[0]
                        metrics["final_portfolio"] = float(value_str.replace(",", ""))
                    except:
                        pass
                elif "Average Reward:" in line:
                    try:
                        reward_str = line.split(":")[1].strip().split()[0]
                        metrics["avg_reward"] = float(reward_str.replace(",", ""))
                    except:
                        pass

        # Extract action distribution
        action_dist = {"HOLD": 0, "BUY": 0, "SELL": 0}
        if "Action Distribution:" in output:
            dist_line = None
            for line in output.split("\n"):
                if "Action Distribution:" in line:
                    dist_line = line
                    break

            if dist_line:
                try:
                    # Parse action distribution
                    dist_part = dist_line.split(":")[1].strip()
                    if "HOLD:" in dist_part:
                        # v420 format: HOLD: 33.1%, BUY: 66.8%, SELL: 0.0%
                        parts = dist_part.split(",")
                        for part in parts:
                            if "HOLD:" in part:
                                action_dist["HOLD"] = float(
                                    part.split(":")[1].strip().replace("%", "")
                                )
                            elif "BUY:" in part:
                                action_dist["BUY"] = float(
                                    part.split(":")[1].strip().replace("%", "")
                                )
                            elif "SELL:" in part:
                                action_dist["SELL"] = float(
                                    part.split(":")[1].strip().replace("%", "")
                                )
                    else:
                        # v418 format: {0: 1, 1: 2726, 2: 2272}
                        # This is more complex to parse, skip for now
                        pass
                except:
                    pass

        metrics["action_distribution"] = action_dist
        metrics["model_version"] = model_version
        metrics["timestamp"] = time.time()

        logger.info(
            f"Completed {model_version}: Portfolio={metrics.get('final_portfolio', 'N/A')}"
        )

        return metrics

    except Exception as e:
        logger.error(f"Failed to run paper trading for {model_version}: {e}")
        return None


def run_multiple_simulations(
    script_path: str, model_version: str, num_runs: int = 5
) -> List[Dict[str, Any]]:
    """Run multiple paper trading simulations."""
    logger = get_logger(__name__)
    logger.info(f"Running {num_runs} simulations for {model_version}")

    results = []

    for i in range(num_runs):
        logger.info(f"Run {i+1}/{num_runs} for {model_version}")
        result = run_single_paper_trade(script_path, model_version)
        if result:
            results.append(result)
        else:
            logger.warning(f"Run {i+1} failed for {model_version}")

        # Small delay between runs
        time.sleep(1)

    return results


def perform_statistical_analysis(
    v418_results: List[Dict[str, Any]], v420_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Perform statistical analysis comparing v418 and v420 results."""
    logger = get_logger(__name__)

    # Extract portfolio values
    v418_portfolios = [r.get("final_portfolio", 200000) for r in v418_results if r]
    v420_portfolios = [r.get("final_portfolio", 200000) for r in v420_results if r]

    if not v418_portfolios or not v420_portfolios:
        return {"error": "Insufficient data for statistical analysis"}

    # Basic statistics
    analysis = {
        "v418": {
            "count": len(v418_portfolios),
            "mean": np.mean(v418_portfolios),
            "std": np.std(v418_portfolios),
            "min": np.min(v418_portfolios),
            "max": np.max(v418_portfolios),
        },
        "v420": {
            "count": len(v420_portfolios),
            "mean": np.mean(v420_portfolios),
            "std": np.std(v420_portfolios),
            "min": np.min(v420_portfolios),
            "max": np.max(v420_portfolios),
        },
    }

    # T-test for portfolio values
    try:
        t_stat, p_value = stats.ttest_ind(v418_portfolios, v420_portfolios)
        analysis["t_test"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
    except Exception as e:
        logger.warning(f"T-test failed: {e}")
        analysis["t_test"] = {"error": str(e)}

    # Calculate effect size (Cohen's d)
    try:
        pooled_std = np.sqrt(
            (analysis["v418"]["std"] ** 2 + analysis["v420"]["std"] ** 2) / 2
        )
        cohen_d = (
            (analysis["v418"]["mean"] - analysis["v420"]["mean"]) / pooled_std
            if pooled_std > 0
            else 0
        )
        analysis["effect_size"] = {
            "cohen_d": cohen_d,
            "interpretation": "large"
            if abs(cohen_d) > 0.8
            else "medium"
            if abs(cohen_d) > 0.5
            else "small",
        }
    except Exception as e:
        logger.warning(f"Effect size calculation failed: {e}")
        analysis["effect_size"] = {"error": str(e)}

    return analysis


def main():
    """Main function for statistical comparison."""
    logger = get_logger(__name__)

    # Configuration
    num_runs = 5  # Number of simulation runs per model

    print("=" * 80)
    print("STATISTICAL COMPARISON: SAC v418 vs v420")
    print("=" * 80)

    # Run simulations for v418
    print(f"\n🧪 Running {num_runs} simulations for SAC v418...")
    v418_results = run_multiple_simulations(
        "scripts/paper_trade_sac_v418.py", "v418", num_runs
    )

    # Run simulations for v420
    print(f"\n🧪 Running {num_runs} simulations for SAC v420...")
    v420_results = run_multiple_simulations(
        "scripts/paper_trade_sac_v420.py", "v420", num_runs
    )

    # Perform statistical analysis
    print("\n📊 Performing statistical analysis...")
    analysis = perform_statistical_analysis(v418_results, v420_results)

    # Save detailed results
    timestamp = int(time.time())
    results_file = f"reports/statistical_comparison_v418_v420_{timestamp}.json"

    full_results = {
        "metadata": {
            "timestamp": timestamp,
            "num_runs_per_model": num_runs,
            "description": "Statistical comparison of SAC v418 vs v420 using paper trading",
        },
        "v418_results": v418_results,
        "v420_results": v420_results,
        "statistical_analysis": analysis,
    }

    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n📈 PERFORMANCE SUMMARY:")
    print(
        f"   SAC v418: {analysis['v418']['count']} runs, "
        f"Mean={analysis['v418']['mean']:.2f}, Std={analysis['v418']['std']:.2f}"
    )
    print(
        f"   SAC v420: {analysis['v420']['count']} runs, "
        f"Mean={analysis['v420']['mean']:.2f}, Std={analysis['v420']['std']:.2f}"
    )
    print(f"   Difference: {analysis['v418']['mean'] - analysis['v420']['mean']:.2f}")

    print("\n📊 STATISTICAL TESTS:")
    if "t_test" in analysis and "p_value" in analysis["t_test"]:
        p_val = analysis["t_test"]["p_value"]
        significant = analysis["t_test"]["significant"]
        print(f"   T-test p-value: {p_val:.4f}")
        print(f"   Significant difference: {'YES' if significant else 'NO'} (p < 0.05)")

    if "effect_size" in analysis and "cohen_d" in analysis["effect_size"]:
        d = analysis["effect_size"]["cohen_d"]
        interpretation = analysis["effect_size"]["interpretation"]
        print(f"   Cohen's d: {d:.3f}")
        print(f"   Effect size interpretation: {interpretation}")

    # Recommendation
    print("\n🎯 RECOMMENDATION:")
    if analysis.get("t_test", {}).get("significant", False):
        winner = (
            "v418" if analysis["v418"]["mean"] > analysis["v420"]["mean"] else "v420"
        )
        print("   Statistically significant difference found!")
        print(f"   🏆 {winner.upper()} shows superior performance")
    else:
        print("   No statistically significant difference found")
        print("   Both models perform similarly")

    print("=" * 80)


if __name__ == "__main__":
    main()
