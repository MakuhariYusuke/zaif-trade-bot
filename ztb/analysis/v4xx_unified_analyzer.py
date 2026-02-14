#!/usr/bin/env python3
"""
Unified Analysis Framework for v4XX Series.

A lightweight analysis system that supports v4XX results through one
backward-compatible interface.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, cast

import numpy as np

from ztb.io.json_io import read_json, write_json
from ztb.io.text_io import read_text
from ztb.types.common import ObjectMap, ObjectRecords


def _as_object_map(value: object) -> ObjectMap:
    return cast(ObjectMap, value) if isinstance(value, dict) else {}


def _as_object_records(value: object) -> ObjectRecords:
    if not isinstance(value, list):
        return []
    return [cast(ObjectMap, item) for item in value if isinstance(item, dict)]


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class UnifiedBase:
    """Fallback base class for unified functionality."""

    def __init__(self, version: Optional[str] = None) -> None:
        self.version = version or "unknown"
        import logging

        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(self, path: str) -> ObjectMap:
        try:
            data = read_json(path)
            return _as_object_map(data)
        except Exception:
            return {}

    def save_config(self, data: ObjectMap, path: str) -> None:
        try:
            write_json(path, data, indent=2, ensure_ascii=False)
        except Exception:
            return


try:
    from ztb.utils.analysis_formatters import print_formatted_metrics
except ImportError:

    def print_formatted_metrics(metrics: ObjectMap, title: str = "") -> None:
        """Fallback implementation for print_formatted_metrics."""
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print(f"{'=' * 60}")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print(f"{'=' * 60}")


class V4XXUnifiedAnalyzer(UnifiedBase):
    """Lightweight unified analyzer for all v4XX series backtest results."""

    def __init__(self, results_path: str, version: Optional[str] = None):
        super().__init__(version=version)
        self.results_path: Path = Path(results_path)
        self.version = version or self._detect_version()
        self.data: ObjectMap = self._load_data()
        self.metrics: ObjectMap = {}
        self.logger.info("Initialized analyzer for v%s", self.version)

    def _detect_version(self) -> str:
        """Detect version from results path or content."""
        path_str = str(self.results_path).lower()

        if "v440" in path_str:
            return "440"
        if "v437" in path_str:
            return "437"
        if "v435" in path_str:
            return "435"
        if "v427" in path_str:
            return "427"

        try:
            if self.results_path.is_file():
                content = read_text(self.results_path)
                snippet = content[:500]
                if '"config_version": "4.4.0"' in snippet:
                    return "440"
                if "v437" in snippet:
                    return "437"
                if "v435" in snippet:
                    return "435"
                if "v427" in snippet:
                    return "427"
        except Exception:
            return "unknown"

        return "unknown"

    def _load_data(self) -> ObjectMap:
        """Load backtest results data."""
        if self.results_path.is_file():
            return self.load_config(str(self.results_path))
        return self._load_directory_results()

    def _load_directory_results(self) -> ObjectMap:
        """Load results from directory structure."""
        results: ObjectMap = {}
        result_files = [
            "backtest_results.json",
            "backtest_results_v440.json",
            "sac_v435_detailed_analysis.json",
            "v437_detailed_profit_analysis.json",
        ]

        for filename in result_files:
            filepath = self.results_path / filename
            if not filepath.exists():
                continue
            try:
                payload = read_json(filepath)
                results[filename] = _as_object_map(payload)
                self.logger.info("Loaded results from: %s", filename)
            except Exception as exc:
                self.logger.warning("Failed to load %s: %s", filename, exc)

        if not results:
            raise FileNotFoundError(f"No result files found in {self.results_path}")
        return results

    def calculate_basic_metrics(self) -> ObjectMap:
        """Calculate basic trading metrics."""
        metrics: ObjectMap = {}
        try:
            summary = _as_object_map(self.data.get("summary"))
            if summary:
                metrics.update(
                    {
                        "total_episodes": _as_int(summary.get("total_episodes")),
                        "average_reward": _as_float(summary.get("average_reward")),
                        "average_trades": _as_float(summary.get("average_trades")),
                        "win_rate": _as_float(summary.get("win_rate")),
                        "total_return": _as_float(summary.get("total_return")),
                        "sharpe_ratio": _as_float(summary.get("sharpe_ratio")),
                        "max_drawdown": _as_float(summary.get("max_drawdown")),
                    }
                )
            elif len(self.data) > 1:
                metrics = self._calculate_multi_variant_metrics()
            else:
                metrics = self._calculate_episode_metrics()
        except Exception as exc:
            self.logger.error("Failed to calculate basic metrics: %s", exc)

        self.metrics.update(metrics)
        return metrics

    def _calculate_multi_variant_metrics(self) -> ObjectMap:
        metrics: ObjectMap = {}
        for variant, variant_payload in self.data.items():
            variant_data = _as_object_map(variant_payload)
            summary = _as_object_map(variant_data.get("summary"))
            if not summary:
                continue
            metrics[variant] = {
                "total_return": _as_float(summary.get("total_return")),
                "sharpe_ratio": _as_float(summary.get("sharpe_ratio")),
                "max_drawdown": _as_float(summary.get("max_drawdown")),
                "win_rate": _as_float(summary.get("win_rate")),
                "total_trades": _as_int(summary.get("total_trades")),
            }
        return metrics

    def _calculate_episode_metrics(self) -> ObjectMap:
        episodes = _as_object_records(self.data.get("episodes"))
        if not episodes:
            return {}

        rewards = [_as_float(ep.get("reward")) for ep in episodes]
        returns = [
            _as_float(ep.get("return_pct", ep.get("return", 0.0))) for ep in episodes
        ]
        trades = [_as_float(ep.get("trades")) for ep in episodes]

        return {
            "total_episodes": len(episodes),
            "average_reward": float(np.mean(rewards)),
            "average_trades": float(np.mean(trades)),
            "total_return": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "max_return": float(max(returns)),
            "min_return": float(min(returns)),
        }

    def calculate_advanced_metrics(self) -> ObjectMap:
        """Calculate advanced statistical metrics."""
        advanced_metrics: ObjectMap = {}
        try:
            total_return = _as_float(self.metrics.get("total_return"))
            return_std = _as_float(self.metrics.get("return_std"))
            if return_std > 0:
                sharpe = total_return / return_std
                advanced_metrics["sharpe_ratio"] = sharpe
                advanced_metrics["sortino_ratio"] = sharpe
            advanced_metrics.update(self._calculate_drawdown_metrics())
        except Exception as exc:
            self.logger.error("Failed to calculate advanced metrics: %s", exc)

        self.metrics.update(advanced_metrics)
        return advanced_metrics

    def _calculate_drawdown_metrics(self) -> ObjectMap:
        drawdown_metrics: ObjectMap = {}
        try:
            episodes = _as_object_records(self.data.get("episodes"))
            returns = [
                _as_float(ep.get("return_pct", ep.get("return", 0.0)))
                for ep in episodes
            ]
            if returns:
                cumulative = np.cumprod(1 + np.array(returns, dtype=float) / 100.0)
                peak = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - peak) / peak
                drawdown_metrics["max_drawdown"] = float(np.min(drawdown) * 100.0)
                negative_drawdowns = drawdown[drawdown < 0]
                drawdown_metrics["average_drawdown"] = (
                    float(np.mean(negative_drawdowns) * 100.0)
                    if negative_drawdowns.size
                    else 0.0
                )
        except Exception as exc:
            self.logger.warning("Failed to calculate drawdown metrics: %s", exc)
        return drawdown_metrics

    def generate_report(self) -> ObjectMap:
        """Generate comprehensive analysis report."""
        report: ObjectMap = {
            "version": self.version,
            "analysis_timestamp": datetime.now().isoformat(),
            "results_path": str(self.results_path),
            "metrics": self.metrics,
            "summary": {},
        }
        if self.metrics:
            report["summary"] = {
                "total_metrics_calculated": len(self.metrics),
                "performance_score": self._calculate_performance_score(),
            }
        return report

    def _calculate_performance_score(self) -> float:
        try:
            score = 0.0
            ret = _as_float(self.metrics.get("total_return"))
            sharpe = _as_float(self.metrics.get("sharpe_ratio"))
            win_rate = _as_float(self.metrics.get("win_rate"))
            max_drawdown = abs(_as_float(self.metrics.get("max_drawdown")))

            score += min(max(ret / 100.0, -1.0), 1.0) * 0.4
            score += min(max(sharpe / 5.0, 0.0), 1.0) * 0.3
            score += win_rate * 0.2
            score -= min(max_drawdown / 50.0, 0.1)
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    def print_report(self) -> None:
        """Print formatted analysis report."""
        try:
            if not self.metrics:
                self.calculate_basic_metrics()
                self.calculate_advanced_metrics()

            report = self.generate_report()
            summary = _as_object_map(report.get("summary"))
            performance_score = _as_float(summary.get("performance_score"))

            print(f"\nV{self.version} Analysis Report")
            print("=" * 60)
            print(f"Results Path: {self.results_path}")
            print(f"Analysis Time: {report.get('analysis_timestamp', '')}")
            print()
            print_formatted_metrics(self.metrics, f"V{self.version} Performance Metrics")
            print("\nSummary:")
            print(
                f"  - Metrics Calculated: {summary.get('total_metrics_calculated', 0)}"
            )
            print(f"  - Performance Score: {performance_score:.2%}")
        except Exception as exc:
            self.logger.error("Failed to print report: %s", exc)

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

    def analyze_multi_period_backtest(self, backtest_results: object) -> ObjectMap:
        """
        Backward-compatible multi-period analysis.

        Accepts:
        - `list[dict]`: period definitions (legacy tests and scripts)
        - `dict[str, dict]`: precomputed window results
        """
        if isinstance(backtest_results, list):
            return self._analyze_period_definitions(_as_object_records(backtest_results))

        if isinstance(backtest_results, dict):
            return self._analyze_window_results(_as_object_map(backtest_results))

        return {
            "period_analysis": [],
            "overall_metrics": {},
            "regime_performance": {},
            "recommendations": [],
            "error": f"Unsupported backtest_results type: {type(backtest_results).__name__}",
        }

    def _analyze_window_results(self, backtest_results: ObjectMap) -> ObjectMap:
        analysis: ObjectMap = {
            "overall_performance": {},
            "regime_performance": {},
            "timeframe_comparison": {},
            "recommendations": {},
            "error": None,
        }
        try:
            analysis["overall_performance"] = self._analyze_overall_performance(
                backtest_results
            )
            analysis["regime_performance"] = self._analyze_window_regime_performance(
                backtest_results
            )
            analysis["timeframe_comparison"] = self._analyze_timeframe_comparison(
                backtest_results
            )
            analysis["recommendations"] = self._generate_trading_recommendations(
                analysis
            )
        except Exception as exc:
            self.logger.error("Failed to analyze multi-period backtest: %s", exc)
            analysis["error"] = str(exc)
        return analysis

    def _analyze_period_definitions(
        self, periods: ObjectRecords, model_path: Optional[str] = None, config_path: Optional[str] = None
    ) -> ObjectMap:
        results: ObjectMap = {
            "period_analysis": [],
            "overall_metrics": {},
            "regime_performance": {},
            "recommendations": [],
        }
        try:
            period_analysis: ObjectRecords = []
            for period in periods:
                period_analysis.append(
                    self._analyze_single_period(period, model_path, config_path)
                )
            results["period_analysis"] = period_analysis
            results["overall_metrics"] = self._calculate_overall_metrics(period_analysis)
            results["regime_performance"] = self._analyze_regime_performance(
                period_analysis
            )
            results["recommendations"] = self._generate_multi_period_recommendations(
                results
            )
        except Exception as exc:
            self.logger.error("Multi-period analysis failed: %s", exc)
            results["error"] = str(exc)
        return results

    def _analyze_overall_performance(self, results: ObjectMap) -> ObjectMap:
        overall: ObjectMap = {}
        for window_key, window_payload in results.items():
            window_data = _as_object_map(window_payload)
            summary = _as_object_map(window_data.get("summary"))
            overall_summary = _as_object_map(summary.get("overall"))
            if not overall_summary:
                continue
            timeframe = str(window_key).replace("h_windows", "h")
            overall[timeframe] = {
                "total_periods": _as_int(overall_summary.get("total_periods")),
                "avg_return": _as_float(overall_summary.get("avg_return")),
                "win_rate": _as_float(overall_summary.get("win_rate")),
                "sharpe_ratio": _as_float(overall_summary.get("sharpe_ratio")),
            }
        return overall

    def _analyze_window_regime_performance(self, results: ObjectMap) -> ObjectMap:
        regime_analysis: ObjectMap = {}
        for window_key, window_payload in results.items():
            window_data = _as_object_map(window_payload)
            summary = _as_object_map(window_data.get("summary"))
            by_trend = _as_object_map(summary.get("by_trend_type"))
            if by_trend:
                timeframe = str(window_key).replace("h_windows", "h")
                regime_analysis[timeframe] = by_trend
        return regime_analysis

    def _analyze_timeframe_comparison(self, results: ObjectMap) -> ObjectMap:
        comparison: ObjectMap = {
            "best_timeframe": {},
            "regime_suitability": {},
            "risk_adjusted_performance": {},
        }
        timeframe_performance: ObjectMap = {}

        for window_key, window_payload in results.items():
            window_data = _as_object_map(window_payload)
            summary = _as_object_map(window_data.get("summary"))
            overall = _as_object_map(summary.get("overall"))
            if not overall:
                continue
            timeframe = str(window_key).replace("h_windows", "h")
            timeframe_performance[timeframe] = {
                "avg_return": _as_float(overall.get("avg_return")),
                "win_rate": _as_float(overall.get("win_rate")),
                "sharpe_ratio": _as_float(overall.get("sharpe_ratio")),
            }

        tf_items = list(timeframe_performance.items())
        if tf_items:
            best_return = max(
                tf_items, key=lambda item: _as_float(_as_object_map(item[1]).get("avg_return"))
            )
            best_win_rate = max(
                tf_items, key=lambda item: _as_float(_as_object_map(item[1]).get("win_rate"))
            )
            best_sharpe = max(
                tf_items,
                key=lambda item: _as_float(_as_object_map(item[1]).get("sharpe_ratio")),
            )
            best_timeframe = _as_object_map(comparison.get("best_timeframe"))
            best_timeframe["return"] = best_return[0]
            best_timeframe["win_rate"] = best_win_rate[0]
            best_timeframe["sharpe_ratio"] = best_sharpe[0]
            comparison["best_timeframe"] = best_timeframe

        return comparison

    def _generate_trading_recommendations(self, analysis: ObjectMap) -> ObjectMap:
        recommendations: ObjectMap = {
            "optimal_timeframe": "",
            "regime_strategy": {},
            "risk_management": {},
            "implementation_priority": [],
        }
        regime_strategy = _as_object_map(recommendations.get("regime_strategy"))
        try:
            timeframe_comp = _as_object_map(analysis.get("timeframe_comparison"))
            best_timeframe = _as_object_map(timeframe_comp.get("best_timeframe"))
            sharpe_best = best_timeframe.get("sharpe_ratio")
            if sharpe_best is not None:
                recommendations["optimal_timeframe"] = sharpe_best

            regime_perf = _as_object_map(analysis.get("regime_performance"))
            for timeframe, regimes_payload in regime_perf.items():
                regimes = _as_object_map(regimes_payload)
                timeframe_map: ObjectMap = {}
                for regime_name, perf_payload in regimes.items():
                    perf = _as_object_map(perf_payload)
                    win_rate = _as_float(perf.get("win_rate"))
                    avg_return = _as_float(perf.get("avg_return"))
                    if win_rate > 60.0 and avg_return > 0:
                        strategy = "積極的取引"
                    elif win_rate > 40.0 and avg_return > 0:
                        strategy = "標準取引"
                    elif win_rate < 30.0 or avg_return < 0:
                        strategy = "取引回避"
                    else:
                        strategy = "慎重取引"
                    timeframe_map[str(regime_name)] = {
                        "recommended_strategy": strategy,
                        "expected_win_rate": win_rate,
                        "expected_return": avg_return,
                    }
                regime_strategy[str(timeframe)] = timeframe_map

            recommendations["risk_management"] = {
                "max_drawdown_limit": 0.1,
                "position_size_limit": 0.05,
                "daily_loss_limit": 0.02,
                "regime_based_position_sizing": True,
            }
            recommendations["implementation_priority"] = [
                "レジーム適応機能の実装",
                "最適タイムフレームの採用",
                "リスク管理機能の強化",
                "バックテストの精密化",
            ]
        except Exception as exc:
            self.logger.error("Failed to generate recommendations: %s", exc)
            recommendations["error"] = str(exc)
        return recommendations

    def _analyze_single_period(
        self,
        period: ObjectMap,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> ObjectMap:
        return {
            "period_name": period.get("name", "unknown"),
            "start_date": period.get("start_date"),
            "end_date": period.get("end_date"),
            "metrics": {
                "total_return": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "total_trades": 0,
            },
            "regime_distribution": {},
            "performance_by_regime": {},
            "model_path": model_path,
            "config_path": config_path,
        }

    def _calculate_overall_metrics(self, period_results: ObjectRecords) -> ObjectMap:
        if not period_results:
            return {}

        total_return = 0.0
        total_trades = 0
        weighted_win_rate = 0.0
        total_weight = 0.0

        for period in period_results:
            metrics = _as_object_map(period.get("metrics"))
            period_return = _as_float(metrics.get("total_return"))
            period_trades = _as_int(metrics.get("total_trades"))
            period_win_rate = _as_float(metrics.get("win_rate"))

            total_return += period_return
            total_trades += period_trades
            weighted_win_rate += period_win_rate * period_trades
            total_weight += float(period_trades)

        avg_win_rate = weighted_win_rate / total_weight if total_weight > 0 else 0.0
        average_return = total_return / len(period_results) if period_results else 0.0

        return {
            "total_periods": len(period_results),
            "total_return": total_return,
            "average_return": average_return,
            "average_return_per_period": average_return,
            "average_win_rate": avg_win_rate,
            "total_trades": total_trades,
        }

    def _analyze_regime_performance(self, period_results: ObjectRecords) -> ObjectMap:
        regime_performance: ObjectMap = {}

        for period in period_results:
            perf_map = _as_object_map(period.get("performance_by_regime"))
            for regime_name, regime_payload in perf_map.items():
                regime_key = str(regime_name)
                collected = _as_object_records(regime_performance.get(regime_key))
                collected.append(_as_object_map(regime_payload))
                regime_performance[regime_key] = collected

        for regime_name, performances_payload in list(regime_performance.items()):
            performances = _as_object_records(performances_payload)
            if not performances:
                continue
            avg_return = sum(_as_float(p.get("return")) for p in performances) / len(
                performances
            )
            avg_win_rate = sum(
                _as_float(p.get("win_rate")) for p in performances
            ) / len(performances)
            regime_performance[regime_name] = {
                "average_return": avg_return,
                "average_win_rate": avg_win_rate,
                "period_count": len(performances),
            }

        return regime_performance

    def _generate_multi_period_recommendations(self, results: ObjectMap) -> list[str]:
        recommendations: list[str] = []
        overall_metrics = _as_object_map(results.get("overall_metrics"))
        regime_performance = _as_object_map(results.get("regime_performance"))

        avg_win_rate = _as_float(overall_metrics.get("average_win_rate"))
        if avg_win_rate > 0.6:
            recommendations.append("全体的に良好なパフォーマンス - 現在の戦略を維持")
        elif avg_win_rate < 0.4:
            recommendations.append("パフォーマンス改善が必要 - 戦略の見直しを検討")

        strong_regimes: list[str] = []
        weak_regimes: list[str] = []
        for regime_name, perf_payload in regime_performance.items():
            perf = _as_object_map(perf_payload)
            regime_win_rate = _as_float(perf.get("average_win_rate"))
            if regime_win_rate > 0.6:
                strong_regimes.append(str(regime_name))
            elif regime_win_rate < 0.4:
                weak_regimes.append(str(regime_name))

        if strong_regimes:
            recommendations.append(
                f"強いパフォーマンスを示すレジーム: {', '.join(strong_regimes)}"
            )
        if weak_regimes:
            recommendations.append(
                f"改善が必要なレジーム: {', '.join(weak_regimes)}"
            )

        return recommendations


def analyze_multi_period_backtest(
    self: V4XXUnifiedAnalyzer,
    periods: ObjectRecords,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> ObjectMap:
    """Backward-compatible wrapper for legacy call sites."""
    return self._analyze_period_definitions(periods, model_path, config_path)


def _analyze_single_period(
    self: V4XXUnifiedAnalyzer,
    period: ObjectMap,
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> ObjectMap:
    return self._analyze_single_period(period, model_path, config_path)


def _calculate_overall_metrics(
    self: V4XXUnifiedAnalyzer, period_results: ObjectRecords
) -> ObjectMap:
    return self._calculate_overall_metrics(period_results)


def _analyze_regime_performance(
    self: V4XXUnifiedAnalyzer, period_results: ObjectRecords
) -> ObjectMap:
    return self._analyze_regime_performance(period_results)


def _generate_multi_period_recommendations(
    self: V4XXUnifiedAnalyzer, results: ObjectMap
) -> list[str]:
    return self._generate_multi_period_recommendations(results)


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
    analyzer = V4XXUnifiedAnalyzer(results_path, version)
    analyzer.calculate_basic_metrics()
    analyzer.calculate_advanced_metrics()
    analyzer.print_report()

    if save_report:
        analyzer.save_report()

    return analyzer
