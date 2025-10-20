#!/usr/bin/env python3
"""
SAC v420 Parameter Tuning Results Analyzer

Analyzes parameter tuning results and generates recommendations for optimal SAC hyperparameters.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Ensure we're using the correct Python environment
if sys.version_info < (3, 11):
    print("Error: Python 3.11+ required")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TuningResultsAnalyzer:
    """Analyzer for SAC parameter tuning results."""

    def __init__(self):
        self.results_dir = Path("results/sac_v420_tuning")
        self.analysis_dir = self.results_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def load_tuning_results(self) -> Optional[Dict[str, Any]]:
        """Load tuning results from summary file."""
        summary_file = self.results_dir / "tuning_summary.json"

        if not summary_file.exists():
            print(f"❌ Tuning results not found: {summary_file}")
            return None

        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
            return None

    def analyze_baseline_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze baseline configuration performance."""
        baseline_results = results.get("results_by_group", {}).get("baseline", {})
        baseline_runs = baseline_results.get("results", [])

        analysis = {
            "total_runs": len(baseline_runs),
            "successful_runs": sum(1 for r in baseline_runs if r["status"] == "success"),
            "avg_duration": baseline_results.get("avg_duration", 0),
            "performance_stability": self._calculate_stability_score(baseline_runs)
        }

        return analysis

    def analyze_parameter_sweeps(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance across parameter sweeps."""
        parameter_analysis = {}

        parameter_groups = [
            "learning_rate", "buffer_size", "batch_size",
            "entropy_coef", "reward_scale", "gamma"
        ]

        for param_group in parameter_groups:
            group_results = results.get("results_by_group", {}).get(param_group, {})
            runs = group_results.get("results", [])

            if not runs:
                continue

            analysis = {
                "configs_tested": len(runs),
                "successful_runs": sum(1 for r in runs if r["status"] == "success"),
                "performance_comparison": self._compare_parameter_configs(runs, param_group),
                "recommendation": self._generate_parameter_recommendation(runs, param_group)
            }

            parameter_analysis[param_group] = analysis

        return parameter_analysis

    def _calculate_stability_score(self, runs: List[Dict[str, Any]]) -> float:
        """Calculate stability score based on successful runs and duration consistency."""
        if not runs:
            return 0.0

        successful_runs = [r for r in runs if r["status"] == "success"]
        success_rate = len(successful_runs) / len(runs)

        if len(successful_runs) < 2:
            return success_rate

        durations = [r.get("duration", 0) for r in successful_runs]
        duration_std = np.std(durations)
        duration_mean = np.mean(durations)

        # Normalize duration variability (lower is better)
        duration_stability = 1.0 / (1.0 + duration_std / max(duration_mean, 1.0))

        return (success_rate + duration_stability) / 2.0

    def _compare_parameter_configs(self, runs: List[Dict[str, Any]], param_group: str) -> Dict[str, Any]:
        """Compare performance between different parameter values."""
        successful_runs = [r for r in runs if r["status"] == "success"]

        if len(successful_runs) < 2:
            return {"comparison": "insufficient_data"}

        # Extract parameter values from config files
        param_values = []
        durations = []

        for run in successful_runs:
            config_file = run.get("config_file", "")
            duration = run.get("duration", 0)

            # Parse parameter value from filename
            param_value = self._extract_parameter_value(config_file, param_group)
            if param_value is not None:
                param_values.append(param_value)
                durations.append(duration)

        if len(param_values) < 2:
            return {"comparison": "insufficient_data"}

        # Calculate performance metrics
        comparison = {
            "parameter_values": param_values,
            "durations": durations,
            "best_performer": param_values[np.argmin(durations)] if durations else None,
            "worst_performer": param_values[np.argmax(durations)] if durations else None,
            "performance_range": max(durations) - min(durations) if durations else 0
        }

        return comparison

    def _extract_parameter_value(self, config_file: str, param_group: str) -> Optional[float]:
        """Extract parameter value from config filename."""
        try:
            # Parse parameter value from filename patterns
            if param_group == "learning_rate":
                if "0.0001" in config_file:
                    return 0.0001
                elif "0.001" in config_file:
                    return 0.001
            elif param_group == "buffer_size":
                if "100k" in config_file:
                    return 100000
                elif "200k" in config_file:
                    return 200000
            elif param_group == "batch_size":
                if "64" in config_file:
                    return 64
                elif "256" in config_file:
                    return 256
            elif param_group == "entropy_coef":
                if "0.01" in config_file:
                    return 0.01
                elif "0.1" in config_file:
                    return 0.1
            elif param_group == "reward_scale":
                if "0.1" in config_file:
                    return 0.1
                elif "10.0" in config_file:
                    return 10.0
            elif param_group == "gamma":
                if "0.95" in config_file:
                    return 0.95
                elif "0.999" in config_file:
                    return 0.999
        except:
            pass

        return None

    def _generate_parameter_recommendation(self, runs: List[Dict[str, Any]], param_group: str) -> str:
        """Generate recommendation for parameter value."""
        comparison = self._compare_parameter_configs(runs, param_group)

        if comparison.get("comparison") == "insufficient_data":
            return "Insufficient data for recommendation"

        best_performer = comparison.get("best_performer")
        if best_performer is None:
            return "Unable to determine best performer"

        # Generate human-readable recommendation
        param_name_map = {
            "learning_rate": "learning rate",
            "buffer_size": "buffer size",
            "batch_size": "batch size",
            "entropy_coef": "entropy coefficient",
            "reward_scale": "reward scale",
            "gamma": "gamma (discount factor)"
        }

        param_display = param_name_map.get(param_group, param_group)

        return f"Recommended {param_display}: {best_performer}"

    def generate_analysis_report(self) -> None:
        """Generate comprehensive analysis report."""
        results = self.load_tuning_results()
        if not results:
            return

        print("🔍 Analyzing SAC v420 Parameter Tuning Results")
        print("=" * 50)

        # Analyze baseline performance
        baseline_analysis = self.analyze_baseline_performance(results)
        print("📊 Baseline Performance:")
        print(f"   Runs: {baseline_analysis['total_runs']}")
        print(f"   Success Rate: {baseline_analysis['successful_runs']}/{baseline_analysis['total_runs']}")
        print(".2f"        print(".3f"
        # Analyze parameter sweeps
        parameter_analysis = self.analyze_parameter_sweeps(results)

        print("\n📈 Parameter Sweep Analysis:")
        for param_group, analysis in parameter_analysis.items():
            print(f"\n{param_group.replace('_', ' ').title()}:")
            print(f"   Configurations tested: {analysis['configs_tested']}")
            print(f"   Successful runs: {analysis['successful_runs']}")

            comparison = analysis['performance_comparison']
            if comparison.get('comparison') != 'insufficient_data':
                print(f"   Best performer: {comparison.get('best_performer')}")
                print(".2f"
            print(f"   Recommendation: {analysis['recommendation']}")

        # Generate recommendations summary
        self._generate_recommendations_summary(parameter_analysis)

        print("
✅ Analysis completed!"        print(f"📄 Detailed report saved to: {self.analysis_dir}")

    def _generate_recommendations_summary(self, parameter_analysis: Dict[str, Dict[str, Any]]) -> None:
        """Generate final recommendations summary."""
        recommendations_file = self.analysis_dir / "parameter_recommendations.txt"

        with open(recommendations_file, 'w', encoding='utf-8') as f:
            f.write("SAC v420 Parameter Tuning Recommendations\n")
            f.write("=" * 50 + "\n\n")

            f.write("Based on parameter sweep analysis:\n\n")

            for param_group, analysis in parameter_analysis.items():
                recommendation = analysis.get('recommendation', 'No recommendation available')
                f.write(f"{param_group.replace('_', ' ').title()}: {recommendation}\n")

            f.write("\nNext Steps:\n")
            f.write("1. Implement recommended parameters in production config\n")
            f.write("2. Run extended training (10k-100k steps) with optimal parameters\n")
            f.write("3. Validate performance on held-out test data\n")
            f.write("4. Consider additional parameter combinations if needed\n")

        print(f"📋 Recommendations saved to: {recommendations_file}")


def main():
    """Main entry point for analysis."""
    analyzer = TuningResultsAnalyzer()
    analyzer.generate_analysis_report()


if __name__ == "__main__":
    main()
