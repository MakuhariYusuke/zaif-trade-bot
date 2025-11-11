#!/usr/bin/env python3
"""
Unified Analysis Framework for v4XX Series

A lightweight, focused analysis system that supports all v4XX versions
through unified configuration and minimal interface.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

# Import dependencies with fallbacks
class UnifiedBase:
    """Fallback base class for unified functionality."""
    def __init__(self, version: Optional[str] = None) -> None:
        self.version = version or "unknown"
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except:
            return {}

    def save_config(self, data: Dict[str, Any], path: str) -> None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except:
            pass

try:
    from ztb.utils.analysis_formatters import print_formatted_metrics
except ImportError:
    def print_formatted_metrics(metrics: Dict[str, Any], title: str = "") -> None:
        pass

from ztb.analysis.adaptive_confidence_adjuster import MarketRegimeDetector, MarketRegime


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
        self.results_path: Path = Path(results_path)
        self.version = version or self._detect_version()
        self.data = self._load_data()
        self.metrics: Dict[str, Any] = {}

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
                data = self.load_config(str(self.results_path))
                return data if isinstance(data, dict) else {}
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

    def print_report(self) -> None:
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

    def save_report(self, output_path: Optional[str] = None) -> None:
        """Save analysis report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"v{self.version}_analysis_report_{timestamp}.json"

        super().save_config(self.generate_report(), output_path)

    def run(self) -> None:
        """Execute the main functionality (alias for analyze)."""
        self.calculate_basic_metrics()
        self.calculate_advanced_metrics()
        self.print_report()

    def analyze_multi_period_backtest(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze multi-period backtest results with market regime insights.

        This method integrates the multi_period_analysis_sac_v445_3.py analysis
        functionality into the unified analyzer framework.

        Args:
            backtest_results: Results from multi-period backtest

        Returns:
            Dict containing comprehensive analysis
        """
        analysis: Dict[str, Any] = {
            "overall_performance": {},
            "regime_performance": {},
            "timeframe_comparison": {},
            "recommendations": {},
            "error": None
        }

        try:
            # Analyze overall performance
            analysis["overall_performance"] = self._analyze_overall_performance(backtest_results)

            # Analyze performance by market regime
            analysis["regime_performance"] = self._analyze_regime_performance(backtest_results)

            # Compare different timeframes
            analysis["timeframe_comparison"] = self._analyze_timeframe_comparison(backtest_results)

            # Generate recommendations
            analysis["recommendations"] = self._generate_trading_recommendations(analysis)

            self.logger.info("Multi-period backtest analysis completed")

        except Exception as e:
            self.logger.error(f"Failed to analyze multi-period backtest: {e}")
            analysis["error"] = str(e)

        return analysis

    def _analyze_overall_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall performance across all periods."""
        overall = {}

        for window_key, window_data in results.items():
            if "summary" in window_data and "overall" in window_data["summary"]:
                summary = window_data["summary"]["overall"]
                timeframe = window_key.replace("h_windows", "h")

                overall[timeframe] = {
                    "total_periods": summary.get("total_periods", 0),
                    "avg_return": summary.get("avg_return", 0),
                    "win_rate": summary.get("win_rate", 0),
                    "sharpe_ratio": summary.get("sharpe_ratio", 0)
                }

        return overall

    def _analyze_regime_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance by market regime."""
        regime_analysis = {}

        for window_key, window_data in results.items():
            if "summary" in window_data and "by_trend_type" in window_data["summary"]:
                timeframe = window_key.replace("h_windows", "h")
                regime_analysis[timeframe] = window_data["summary"]["by_trend_type"]

        return regime_analysis

    def _analyze_timeframe_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across different timeframes."""
        comparison: Dict[str, Any] = {
            "best_timeframe": {},
            "regime_suitability": {},
            "risk_adjusted_performance": {}
        }

        # Find best performing timeframe
        timeframe_performance = {}
        for window_key, window_data in results.items():
            if "summary" in window_data and "overall" in window_data["summary"]:
                timeframe = window_key.replace("h_windows", "h")
                summary = window_data["summary"]["overall"]
                timeframe_performance[timeframe] = {
                    "avg_return": summary.get("avg_return", 0),
                    "win_rate": summary.get("win_rate", 0),
                    "sharpe_ratio": summary.get("sharpe_ratio", 0)
                }

        if timeframe_performance:
            # Best by return
            best_return = max(timeframe_performance.items(), key=lambda x: x[1]["avg_return"])
            comparison["best_timeframe"]["return"] = best_return[0]

            # Best by win rate
            best_win_rate = max(timeframe_performance.items(), key=lambda x: x[1]["win_rate"])
            comparison["best_timeframe"]["win_rate"] = best_win_rate[0]

            # Best by Sharpe ratio
            best_sharpe = max(timeframe_performance.items(), key=lambda x: x[1]["sharpe_ratio"])
            comparison["best_timeframe"]["sharpe_ratio"] = best_sharpe[0]

        return comparison

    def _generate_trading_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendations based on analysis."""
        recommendations: Dict[str, Any] = {
            "optimal_timeframe": "",
            "regime_strategy": {},
            "risk_management": {},
            "implementation_priority": []
        }

        # Type hints for mypy
        regime_strategy: Dict[str, Dict[str, Any]] = recommendations["regime_strategy"]

        try:
            overall_perf = analysis.get("overall_performance", {})
            timeframe_comp = analysis.get("timeframe_comparison", {})

            # Determine optimal timeframe
            if "best_timeframe" in timeframe_comp:
                best_tf = timeframe_comp["best_timeframe"]
                # Prioritize Sharpe ratio for risk-adjusted performance
                if "sharpe_ratio" in best_tf:
                    recommendations["optimal_timeframe"] = best_tf["sharpe_ratio"]

            # Generate regime-specific strategies
            regime_perf = analysis.get("regime_performance", {})
            if regime_perf:
                for timeframe, regimes in regime_perf.items():
                    regime_strategy[timeframe] = {}
                    for regime, perf in regimes.items():
                        win_rate = perf.get("win_rate", 0)
                        avg_return = perf.get("avg_return", 0)

                        if win_rate > 60 and avg_return > 0:
                            strategy = "積極的取引"
                        elif win_rate > 40 and avg_return > 0:
                            strategy = "標準取引"
                        elif win_rate < 30 or avg_return < 0:
                            strategy = "取引回避"
                        else:
                            strategy = "慎重取引"

                        regime_strategy[timeframe][regime] = {
                            "recommended_strategy": strategy,
                            "expected_win_rate": win_rate,
                            "expected_return": avg_return
                        }

            # Risk management recommendations
            recommendations["risk_management"] = {
                "max_drawdown_limit": 0.1,  # 10%
                "position_size_limit": 0.05,  # 5% per trade
                "daily_loss_limit": 0.02,  # 2%
                "regime_based_position_sizing": True
            }

            # Implementation priority
            recommendations["implementation_priority"] = [
                "レジーム適応機能の実装",
                "最適タイムフレームの採用",
                "リスク管理機能の強化",
                "バックテストの精密化"
            ]

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations["error"] = str(e)

        return recommendations


def analyze_multi_period_backtest(
    self,
    periods: List[Dict[str, Any]],
    model_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze backtest results across multiple time periods.

    Args:
        periods: List of period definitions with start/end dates
        model_path: Path to trained model (optional)
        config_path: Path to config file (optional)

    Returns:
        Multi-period analysis results
    """
    results = {
        "period_analysis": [],
        "overall_metrics": {},
        "regime_performance": {},
        "recommendations": []
    }

    try:
        for period in periods:
            period_result = self._analyze_single_period(period, model_path, config_path)
            results["period_analysis"].append(period_result)

        # Calculate overall metrics
        results["overall_metrics"] = self._calculate_overall_metrics(results["period_analysis"])

        # Analyze regime performance
        results["regime_performance"] = self._analyze_regime_performance(results["period_analysis"])

        # Generate recommendations
        results["recommendations"] = self._generate_multi_period_recommendations(results)

    except Exception as e:
        self.logger.error(f"Multi-period analysis failed: {e}")
        results["error"] = str(e)

    return results


def _analyze_single_period(
    self,
    period: Dict[str, Any],
    model_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze a single time period."""
    # Placeholder implementation - would integrate with actual backtest logic
    return {
        "period_name": period.get("name", "unknown"),
        "start_date": period.get("start_date"),
        "end_date": period.get("end_date"),
        "metrics": {
            "total_return": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_trades": 0
        },
        "regime_distribution": {},
        "performance_by_regime": {}
    }


def _calculate_overall_metrics(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall metrics across all periods."""
    if not period_results:
        return {}

    # Aggregate metrics across periods
    total_return = sum(p["metrics"]["total_return"] for p in period_results)
    total_trades = sum(p["metrics"]["total_trades"] for p in period_results)

    # Weighted average for rates
    weighted_win_rate = 0.0
    total_weight = 0.0

    for p in period_results:
        weight = p["metrics"]["total_trades"]
        weighted_win_rate += p["metrics"]["win_rate"] * weight
        total_weight += weight

    avg_win_rate = weighted_win_rate / total_weight if total_weight > 0 else 0.0

    return {
        "total_periods": len(period_results),
        "total_return": total_return,
        "average_win_rate": avg_win_rate,
        "total_trades": total_trades,
        "average_return_per_period": total_return / len(period_results) if period_results else 0.0
    }


def _analyze_regime_performance(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance by market regime."""
    regime_performance = {}

    for period in period_results:
        for regime, perf in period.get("performance_by_regime", {}).items():
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(perf)

    # Calculate averages for each regime
    for regime, performances in regime_performance.items():
        if performances:
            avg_return = sum(p.get("return", 0) for p in performances) / len(performances)
            avg_win_rate = sum(p.get("win_rate", 0) for p in performances) / len(performances)
            regime_performance[regime] = {
                "average_return": avg_return,
                "average_win_rate": avg_win_rate,
                "period_count": len(performances)
            }

    return regime_performance


def _generate_multi_period_recommendations(self, results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on multi-period analysis."""
    recommendations = []

    overall_metrics = results.get("overall_metrics", {})
    regime_performance = results.get("regime_performance", {})

    # Analyze overall performance
    if overall_metrics.get("average_win_rate", 0) > 0.6:
        recommendations.append("全体的に良好なパフォーマンス - 現在の戦略を維持")
    elif overall_metrics.get("average_win_rate", 0) < 0.4:
        recommendations.append("パフォーマンス改善が必要 - 戦略の見直しを検討")

    # Analyze regime-specific performance
    strong_regimes = []
    weak_regimes = []

    for regime, perf in regime_performance.items():
        if perf.get("average_win_rate", 0) > 0.6:
            strong_regimes.append(regime)
        elif perf.get("average_win_rate", 0) < 0.4:
            weak_regimes.append(regime)

    if strong_regimes:
        recommendations.append(f"強いパフォーマンスを示すレジーム: {', '.join(strong_regimes)}")

    if weak_regimes:
        recommendations.append(f"改善が必要なレジーム: {', '.join(weak_regimes)}")

    return recommendations


def analyze_v4xx_results(
    results_path: str, version: Optional[str] = None, save_report: bool = True
) -> V4XXUnifiedAnalyzer:
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
