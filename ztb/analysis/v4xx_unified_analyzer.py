#!/usr/bin/env python3
"""
Unified Analysis Framework for v4XX Series

A lightweight, focused analysis system that supports all v4XX versions
through unified configuration and minimal interface.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ztb.training.core.unified_base import UnifiedBase
from ztb.utils.analysis_formatters import print_formatted_metrics


class V4XXUnifiedAnalyzer(UnifiedBase):
    """Lightweight unified analyzer for all v4XX series backtest results."""

    def __init__(self, results_path: str, version: Optional[str] = None):
        """
        Initialize unified analyzer.

        Args:
            results_path: Path to backtest results file or directory
            version: Version identifier (auto-detected if None)
        """
        super().__init__(version=version)
        self.results_path = Path(results_path)
        self.version = version or self._detect_version()
        self.data = self._load_data()
        self.metrics = {}

        self.logger.info(f"Initialized analyzer for v{self.version}")

    def _detect_version(self) -> str:
        """Detect version from results path or content."""
        path_str = str(self.results_path).lower()

        if "v440" in path_str:
            return "440"
        elif "v437" in path_str:
            return "437"
        elif "v435" in path_str:
            return "435"
        elif "v427" in path_str:
            return "427"

        # Try to detect from content
        try:
            if self.results_path.is_file():
                with open(self.results_path, "r", encoding="utf-8") as f:
                    content = f.read(500)  # Read first 500 chars
                    if '"config_version": "4.4.0"' in content:
                        return "440"
                    elif "v437" in content:
                        return "437"
                    elif "v435" in content:
                        return "435"
                    elif "v427" in content:
                        return "427"
        except:
            pass

        return "unknown"

    def _load_data(self) -> Dict[str, Any]:
        """Load backtest results data."""
        try:
            if self.results_path.is_file():
                # Single results file
                return self.load_config(str(self.results_path))
            else:
                # Directory with multiple results
                return self._load_directory_results()

        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            raise

    def _load_directory_results(self) -> Dict[str, Any]:
        """Load results from directory structure."""
        results = {}

        # Look for common result files
        result_files = [
            "backtest_results.json",
            "backtest_results_v440.json",
            "sac_v435_detailed_analysis.json",
            "v437_detailed_profit_analysis.json",
        ]

        for filename in result_files:
            filepath = self.results_path / filename
            if filepath.exists():
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        results[filename] = json.load(f)
                    self.logger.info(f"Loaded results from: {filename}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {filename}: {e}")

        if not results:
            raise FileNotFoundError(f"No result files found in {self.results_path}")

        return results

    def calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic trading metrics."""
        metrics = {}

        try:
            if "summary" in self.data:
                # Single summary format (v440 style)
                summary = self.data["summary"]
                metrics.update(
                    {
                        "total_episodes": summary.get("total_episodes", 0),
                        "average_reward": summary.get("average_reward", 0),
                        "average_trades": summary.get("average_trades", 0),
                        "win_rate": summary.get("win_rate", 0),
                        "total_return": summary.get("total_return", 0),
                        "sharpe_ratio": summary.get("sharpe_ratio", 0),
                        "max_drawdown": summary.get("max_drawdown", 0),
                    }
                )
            elif isinstance(self.data, dict) and len(self.data) > 1:
                # Multiple variants format (v435 style)
                metrics = self._calculate_multi_variant_metrics()
            else:
                # Episodes format
                metrics = self._calculate_episode_metrics()

        except Exception as e:
            self.logger.error(f"Failed to calculate basic metrics: {e}")

        self.metrics.update(metrics)
        return metrics

    def _calculate_multi_variant_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for multiple variants."""
        metrics = {}
        variants = list(self.data.keys())

        for variant in variants:
            variant_data = self.data[variant]
            if "summary" in variant_data:
                summary = variant_data["summary"]
                metrics[variant] = {
                    "total_return": summary.get("total_return", 0),
                    "sharpe_ratio": summary.get("sharpe_ratio", 0),
                    "max_drawdown": summary.get("max_drawdown", 0),
                    "win_rate": summary.get("win_rate", 0),
                    "total_trades": summary.get("total_trades", 0),
                }

        return metrics

    def _calculate_episode_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from episodes data."""
        if "episodes" not in self.data:
            return {}

        episodes = self.data["episodes"]
        if not episodes:
            return {}

        # Extract metrics from episodes
        rewards = [ep.get("reward", 0) for ep in episodes]
        returns = [ep.get("return_pct", 0) for ep in episodes]
        trades = [ep.get("trades", 0) for ep in episodes]

        metrics = {
            "total_episodes": len(episodes),
            "average_reward": np.mean(rewards),
            "average_trades": np.mean(trades),
            "total_return": np.mean(returns),
            "return_std": np.std(returns),
            "max_return": max(returns),
            "min_return": min(returns),
        }

        return metrics

    def calculate_advanced_metrics(self) -> Dict[str, Any]:
        """Calculate advanced statistical metrics."""
        advanced_metrics = {}

        try:
            # Risk-adjusted metrics
            if "total_return" in self.metrics and "return_std" in self.metrics:
                total_return = self.metrics["total_return"]
                return_std = self.metrics["return_std"]

                if return_std > 0:
                    advanced_metrics["sharpe_ratio"] = total_return / return_std
                    advanced_metrics["sortino_ratio"] = (
                        total_return / return_std
                    )  # Simplified

            # Drawdown analysis
            advanced_metrics.update(self._calculate_drawdown_metrics())

        except Exception as e:
            self.logger.error(f"Failed to calculate advanced metrics: {e}")

        self.metrics.update(advanced_metrics)
        return advanced_metrics

    def _calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """Calculate drawdown-related metrics."""
        drawdown_metrics = {}

        try:
            if "episodes" in self.data:
                returns = [ep.get("return_pct", 0) for ep in self.data["episodes"]]
                if returns:
                    cumulative = np.cumprod(1 + np.array(returns) / 100)
                    peak = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - peak) / peak

                    drawdown_metrics["max_drawdown"] = np.min(drawdown) * 100
                    drawdown_metrics["average_drawdown"] = (
                        np.mean(drawdown[drawdown < 0]) * 100
                    )

        except Exception as e:
            self.logger.warning(f"Failed to calculate drawdown metrics: {e}")

        return drawdown_metrics

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report = {
            "version": self.version,
            "analysis_timestamp": datetime.now().isoformat(),
            "results_path": str(self.results_path),
            "metrics": self.metrics,
            "summary": {},
        }

        # Generate summary
        if self.metrics:
            report["summary"] = {
                "total_metrics_calculated": len(self.metrics),
                "performance_score": self._calculate_performance_score(),
            }

        return report

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        try:
            score = 0.0

            # Return component (40%)
            if "total_return" in self.metrics:
                ret = self.metrics["total_return"]
                score += min(max(ret / 100, -1), 1) * 0.4

            # Sharpe ratio component (30%)
            if "sharpe_ratio" in self.metrics:
                sharpe = self.metrics["sharpe_ratio"]
                score += min(max(sharpe / 5, 0), 1) * 0.3

            # Win rate component (20%)
            if "win_rate" in self.metrics:
                win_rate = self.metrics["win_rate"]
                score += win_rate * 0.2

            # Drawdown penalty (10%)
            if "max_drawdown" in self.metrics:
                dd = abs(self.metrics["max_drawdown"])
                score -= min(dd / 50, 0.1)

            return max(0.0, min(1.0, score))

        except:
            return 0.0

    def print_report(self):
        """Print formatted analysis report."""
        try:
            # Ensure metrics are calculated
            if not self.metrics:
                self.calculate_basic_metrics()
                self.calculate_advanced_metrics()

            report = self.generate_report()

            print(f"\n📊 V{self.version} Analysis Report")
            print("=" * 60)
            print(f"Results Path: {self.results_path}")
            print(f"Analysis Time: {report['analysis_timestamp']}")
            print()

            # Print metrics using formatter
            print_formatted_metrics(
                self.metrics, f"V{self.version} Performance Metrics"
            )

            # Print summary
            summary = report["summary"]
            print("\n📈 Summary:")
            print(
                f"  - Metrics Calculated: {summary.get('total_metrics_calculated', 0)}"
            )
            print(".2%")

        except Exception as e:
            self.logger.error(f"Failed to print report: {e}")

    def save_report(self, output_path: Optional[str] = None):
        """Save analysis report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"v{self.version}_analysis_report_{timestamp}.json"

        super().save_config(self.generate_report(), output_path)

    def run(self):
        """Execute the main functionality (alias for analyze)."""
        self.calculate_basic_metrics()
        self.calculate_advanced_metrics()
        self.print_report()


def analyze_v4xx_results(
    results_path: str, version: Optional[str] = None, save_report: bool = True
):
    """
    Convenience function to analyze v4XX results.

    Args:
        results_path: Path to results file or directory
        version: Version override (optional)
        save_report: Whether to save report to file
    """
    try:
        analyzer = V4XXUnifiedAnalyzer(results_path, version)
        analyzer.calculate_basic_metrics()
        analyzer.calculate_advanced_metrics()
        analyzer.print_report()

        if save_report:
            analyzer.save_report()

        return analyzer

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise
