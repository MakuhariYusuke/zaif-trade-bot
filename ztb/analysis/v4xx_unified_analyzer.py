"""Unified analyzer for v4xx backtest artifacts.

This module provides a lightweight implementation used by tests and by
integration utilities that expect the historical `V4XXUnifiedAnalyzer` API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class V4XXUnifiedAnalyzer:
    """Analyze single and multi-period backtest result payloads."""

    def __init__(self, results_path: str | Path) -> None:
        self.results_path = Path(results_path)
        self.results = self._load_results()
        self.version = str(self.results.get("version", "v4xx"))
        self.metrics: dict[str, Any] = {}

    def _load_results(self) -> dict[str, Any]:
        if not self.results_path.exists():
            return {}
        try:
            payload = json.loads(self.results_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def calculate_basic_metrics(self) -> dict[str, Any]:
        summary = self.results.get("summary", {})
        if not isinstance(summary, dict):
            return {}
        metrics = {
            "total_episodes": int(summary.get("total_episodes", 0)),
            "average_reward": float(summary.get("average_reward", 0.0)),
            "average_trades": float(summary.get("average_trades", 0.0)),
            "win_rate": float(summary.get("win_rate", 0.0)),
            "total_return": float(summary.get("total_return", 0.0)),
            "sharpe_ratio": float(summary.get("sharpe_ratio", 0.0)),
            "max_drawdown": float(summary.get("max_drawdown", 0.0)),
        }
        self.metrics = metrics
        return metrics

    def _analyze_single_period(self, period: dict[str, Any]) -> dict[str, Any]:
        name = str(period.get("name", "period"))
        metrics = self.calculate_basic_metrics()
        return {
            "period_name": name,
            "metrics": {
                "total_return": float(metrics.get("total_return", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "total_trades": int(metrics.get("average_trades", 0.0)),
            },
            "performance_by_regime": {},
        }

    def analyze_multi_period_backtest(
        self, periods: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not periods:
            return {
                "period_analysis": [],
                "overall_metrics": {},
                "regime_performance": {},
                "recommendations": [],
            }

        period_analysis = [self._analyze_single_period(p) for p in periods]
        results = {
            "period_analysis": period_analysis,
            "overall_metrics": self._calculate_overall_metrics(period_analysis),
            "regime_performance": self._analyze_regime_performance(period_analysis),
        }
        results["recommendations"] = self._generate_multi_period_recommendations(results)
        return results

    def _calculate_overall_metrics(
        self, period_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not period_results:
            return {}

        total_periods = len(period_results)
        returns: list[float] = []
        win_rates: list[float] = []
        total_trades = 0
        for result in period_results:
            metrics = result.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            returns.append(float(metrics.get("total_return", 0.0)))
            win_rates.append(float(metrics.get("win_rate", 0.0)))
            total_trades += int(metrics.get("total_trades", 0))
        if not returns:
            return {}

        return {
            "total_periods": total_periods,
            "average_return": sum(returns) / len(returns),
            "average_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
            "total_trades": total_trades,
        }

    def _analyze_regime_performance(
        self, period_results: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        buckets: dict[str, dict[str, list[float]]] = {}
        for result in period_results:
            perf = result.get("performance_by_regime", {})
            if not isinstance(perf, dict):
                continue
            for regime, regime_metrics in perf.items():
                if not isinstance(regime_metrics, dict):
                    continue
                item = buckets.setdefault(regime, {"returns": [], "win_rates": []})
                item["returns"].append(float(regime_metrics.get("return", 0.0)))
                item["win_rates"].append(float(regime_metrics.get("win_rate", 0.0)))

        output: dict[str, dict[str, float]] = {}
        for regime, values in buckets.items():
            returns = values["returns"]
            win_rates = values["win_rates"]
            output[regime] = {
                "average_return": sum(returns) / len(returns) if returns else 0.0,
                "average_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
            }
        return output

    def _generate_multi_period_recommendations(
        self, results: dict[str, Any]
    ) -> list[str]:
        recommendations: list[str] = []
        overall = results.get("overall_metrics", {})
        regime_perf = results.get("regime_performance", {})

        if isinstance(overall, dict):
            avg_win_rate = float(overall.get("average_win_rate", 0.0))
            if avg_win_rate < 0.5:
                recommendations.append("Overall win rate is low; reduce risk exposure.")

        if isinstance(regime_perf, dict):
            for regime, stats in regime_perf.items():
                if not isinstance(stats, dict):
                    continue
                win_rate = float(stats.get("average_win_rate", 0.0))
                if win_rate >= 0.6:
                    recommendations.append(
                        f"Regime '{regime}' performs strongly; consider allocation boost."
                    )
                elif win_rate <= 0.4:
                    recommendations.append(
                        f"Regime '{regime}' underperforms; tighten filters or reduce sizing."
                    )

        if not recommendations:
            recommendations.append("Performance is stable across evaluated periods.")
        return recommendations

