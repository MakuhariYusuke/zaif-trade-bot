#!/usr/bin/env python3
"""
Experimental Features Weekly Report Generator

This script generates comprehensive weekly reports for experimental features,
including wave1-3 features, with detailed performance analytics and recommendations.
"""

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ztb.evaluation.re_evaluate_features import (
    ComprehensiveFeatureReEvaluator,
    discover_feature_classes,
    evaluate_feature_class,
)


@dataclass
class ExperimentalFeatureMetrics:
    """Data class for experimental feature performance metrics"""

    feature_name: str
    module_name: str
    nan_rate: float
    delta_sharpe: float
    computation_time_ms: float
    stability_score: float
    recommendation: str
    reason_code: str


class ExperimentalWeeklyReporter:
    """Weekly performance report generator for experimental features"""

    def __init__(self, output_dir: str = "docs/reports/weekly"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.feature_evaluator = ComprehensiveFeatureReEvaluator()

        # Report thresholds
        self.thresholds = {
            "excellent_delta_sharpe": 0.15,
            "good_delta_sharpe": 0.05,
            "acceptable_nan_rate": 0.10,
            "high_nan_rate": 0.25,
            "fast_computation_ms": 100,
            "slow_computation_ms": 1000,
        }

        # Database for tracking historical performance
        self.db_path = self.output_dir / "experimental_history.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for historical tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experimental_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date TEXT,
                    feature_name TEXT,
                    module_name TEXT,
                    nan_rate REAL,
                    delta_sharpe REAL,
                    computation_time_ms REAL,
                    stability_score REAL,
                    recommendation TEXT,
                    reason_code TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_report_date_feature 
                ON experimental_history(report_date, feature_name)
            """
            )
            conn.commit()

    def evaluate_all_experimental_features(self) -> List[ExperimentalFeatureMetrics]:
        """Evaluate all experimental features and return metrics"""
        all_metrics = []

        experimental_modules = [
            "src.trading.features.experimental",
            "src.trading.features.wave1",
            "src.trading.features.wave2",
            "src.trading.features.wave3",
        ]

        for module_name in experimental_modules:
            print(f"\n🔍 Evaluating module: {module_name}")

            try:
                # Discover feature classes in module
                feature_classes = discover_feature_classes(module_name)

                for feature_name, feature_class in feature_classes.items():
                    print(f"  📊 Evaluating {feature_name}...")

                    if self.feature_evaluator.price_data is None:
                        print(f"  ❌ No price data available for {feature_name}")
                        continue

                    # Evaluate feature performance
                    result = evaluate_feature_class(
                        feature_class, self.feature_evaluator.price_data, feature_name
                    )

                    if result.get("status") == "success":
                        # Ensure result is treated as dict
                        result_dict = cast(Dict[str, Any], result)

                        # Calculate derived metrics
                        stability_score = self._calculate_stability_score(result_dict)
                        recommendation, reason_code = self._generate_recommendation(
                            result_dict
                        )

                        metrics = ExperimentalFeatureMetrics(
                            feature_name=feature_name,
                            module_name=module_name,
                            nan_rate=result.get("nan_rate", 0.0),
                            delta_sharpe=result.get("best_delta_sharpe", 0.0),
                            computation_time_ms=result.get("computation_time_ms", 0.0),
                            stability_score=stability_score,
                            recommendation=recommendation,
                            reason_code=reason_code,
                        )

                        all_metrics.append(metrics)
                        print(
                            f"    ✅ {feature_name}: {recommendation} (Δ Sharpe: {metrics.delta_sharpe:.3f})"
                        )
                    else:
                        print(
                            f"    ❌ {feature_name}: {result.get('status', 'unknown')}"
                        )

            except Exception as e:
                print(f"  🚫 Error evaluating {module_name}: {e}")

        return all_metrics

    def _calculate_stability_score(self, result: Dict[str, Any]) -> float:
        """Calculate stability score based on feature evaluation results"""
        stability_factors = []

        # Low NaN rate contributes to stability
        nan_stability = max(0, 1 - (result["nan_rate"] * 2))
        stability_factors.append(nan_stability)

        # Consistent performance across correlations
        correlations = result.get("feature_correlations", {})
        if correlations:
            corr_values = list(correlations.values())
            corr_std = np.std([abs(c) for c in corr_values if not np.isnan(c)])
            corr_stability = max(0, 1 - corr_std)
            stability_factors.append(corr_stability)

        # Performance consistency
        performances = result.get("feature_performances", {})
        if performances:
            perf_values = []
            for perf in performances.values():
                if perf["type"] == "signal":
                    perf_values.append(abs(perf.get("delta_sharpe", 0)))
                elif perf["type"] == "quantile":
                    perf_values.append(abs(perf.get("quantile_spread", 0)))

            if perf_values:
                perf_stability = 1 - (
                    np.std(perf_values) / (np.mean(perf_values) + 1e-6)
                )
                stability_factors.append(max(0, min(1, perf_stability)))

        return float(np.mean(stability_factors)) if stability_factors else 0.5

    def _generate_recommendation(self, result: Dict[str, Any]) -> tuple[str, str]:
        """Generate recommendation and reason code based on evaluation results"""
        nan_rate = result["nan_rate"]
        delta_sharpe = result.get("best_delta_sharpe", 0.0)
        computation_time = result["computation_time_ms"]

        # High NaN rate - needs data quality improvement
        if nan_rate > self.thresholds["high_nan_rate"]:
            return "🔴 REJECT", "high_nan_rate"

        # Excellent performance
        if delta_sharpe >= self.thresholds["excellent_delta_sharpe"]:
            if computation_time <= self.thresholds["fast_computation_ms"]:
                return "🟢 PROMOTE", "strong_performance"
            else:
                return "🟡 OPTIMIZE", "needs_speed_optimization"

        # Good performance
        elif delta_sharpe >= self.thresholds["good_delta_sharpe"]:
            if nan_rate <= self.thresholds["acceptable_nan_rate"]:
                return "🟢 APPROVE", "moderate_performance"
            else:
                return "🟡 IMPROVE", "needs_data_quality"

        # Poor performance
        elif delta_sharpe < 0.01:
            if computation_time > self.thresholds["slow_computation_ms"]:
                return "🔴 REJECT", "weak_performance_slow"
            else:
                return "🟡 INVESTIGATE", "weak_performance"

        # Needs tuning
        else:
            return "🟡 TUNE", "needs_tuning"

    def store_metrics(
        self, metrics: List[ExperimentalFeatureMetrics], report_date: str
    ) -> None:
        """Store metrics in historical database"""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute(
                    """
                    INSERT INTO experimental_history 
                    (report_date, feature_name, module_name, nan_rate, delta_sharpe, 
                     computation_time_ms, stability_score, recommendation, reason_code)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        report_date,
                        metric.feature_name,
                        metric.module_name,
                        metric.nan_rate,
                        metric.delta_sharpe,
                        metric.computation_time_ms,
                        metric.stability_score,
                        metric.recommendation,
                        metric.reason_code,
                    ),
                )
            conn.commit()

    def get_historical_trends(
        self, feature_name: str, weeks: int = 4
    ) -> List[Dict[str, Any]]:
        """Get historical performance trends for a feature"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT report_date, delta_sharpe, nan_rate, computation_time_ms, stability_score
                FROM experimental_history 
                WHERE feature_name = ? 
                ORDER BY report_date DESC 
                LIMIT ?
            """,
                (feature_name, weeks),
            )

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate comprehensive weekly report"""
        report_date = datetime.now().strftime("%Y-%m-%d")

        print(f"🚀 Generating experimental features weekly report for {report_date}")

        # Evaluate all experimental features
        metrics = self.evaluate_all_experimental_features()

        # Store metrics in database
        self.store_metrics(metrics, report_date)

        # Generate report sections
        report = {
            "report_date": report_date,
            "total_features_evaluated": len(metrics),
            "summary": self._generate_summary(metrics),
            "recommendations": self._generate_recommendations(metrics),
            "performance_analysis": self._generate_performance_analysis(metrics),
            "module_breakdown": self._generate_module_breakdown(metrics),
            "trending_analysis": self._generate_trending_analysis(metrics),
            "detailed_metrics": [self._metric_to_dict(m) for m in metrics],
        }

        return report

    def _generate_summary(
        self, metrics: List[ExperimentalFeatureMetrics]
    ) -> Dict[str, Any]:
        """Generate executive summary"""
        if not metrics:
            return {"message": "No metrics available"}

        promote_count = len([m for m in metrics if "🟢 PROMOTE" in m.recommendation])
        approve_count = len([m for m in metrics if "🟢 APPROVE" in m.recommendation])
        needs_work_count = len([m for m in metrics if "🟡" in m.recommendation])
        reject_count = len([m for m in metrics if "🔴" in m.recommendation])

        avg_delta_sharpe = np.mean([m.delta_sharpe for m in metrics])
        avg_nan_rate = np.mean([m.nan_rate for m in metrics])
        avg_computation_time = np.mean([m.computation_time_ms for m in metrics])

        return {
            "recommendation_breakdown": {
                "promote": promote_count,
                "approve": approve_count,
                "needs_work": needs_work_count,
                "reject": reject_count,
            },
            "average_metrics": {
                "delta_sharpe": round(avg_delta_sharpe, 4),
                "nan_rate": round(avg_nan_rate, 4),
                "computation_time_ms": round(avg_computation_time, 2),
            },
            "top_performers": sorted(
                metrics, key=lambda x: x.delta_sharpe, reverse=True
            )[:3],
            "areas_of_concern": [
                m for m in metrics if "🔴" in m.recommendation or m.nan_rate > 0.2
            ],
        }

    def _generate_recommendations(
        self, metrics: List[ExperimentalFeatureMetrics]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []

        # Promote recommendations
        promotable = [m for m in metrics if "🟢 PROMOTE" in m.recommendation]
        if promotable:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "action": "Promote to production",
                    "features": [m.feature_name for m in promotable],
                    "rationale": "Strong performance with fast computation",
                }
            )

        # Data quality improvements
        high_nan = [m for m in metrics if m.nan_rate > self.thresholds["high_nan_rate"]]
        if high_nan:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "action": "Improve data quality",
                    "features": [m.feature_name for m in high_nan],
                    "rationale": f"NaN rates above {self.thresholds['high_nan_rate'] * 100}% threshold",
                }
            )

        # Performance optimization
        slow_features = [
            m
            for m in metrics
            if m.computation_time_ms > self.thresholds["slow_computation_ms"]
        ]
        if slow_features:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "action": "Optimize computation speed",
                    "features": [m.feature_name for m in slow_features],
                    "rationale": f"Computation time exceeds {self.thresholds['slow_computation_ms']}ms",
                }
            )

        return recommendations

    def _generate_performance_analysis(
        self, metrics: List[ExperimentalFeatureMetrics]
    ) -> Dict[str, Any]:
        """Generate detailed performance analysis"""
        if not metrics:
            return {}

        delta_sharpes = [m.delta_sharpe for m in metrics]
        nan_rates = [m.nan_rate for m in metrics]
        computation_times = [m.computation_time_ms for m in metrics]

        return {
            "delta_sharpe_distribution": {
                "mean": round(np.mean(delta_sharpes), 4),
                "median": round(np.median(delta_sharpes), 4),
                "std": round(np.std(delta_sharpes), 4),
                "min": round(np.min(delta_sharpes), 4),
                "max": round(np.max(delta_sharpes), 4),
                "percentiles": {
                    "25th": round(np.percentile(delta_sharpes, 25), 4),
                    "75th": round(np.percentile(delta_sharpes, 75), 4),
                },
            },
            "data_quality_metrics": {
                "avg_nan_rate": round(np.mean(nan_rates), 4),
                "features_with_high_nan": len([m for m in metrics if m.nan_rate > 0.1]),
                "worst_nan_rate": round(np.max(nan_rates), 4),
            },
            "computational_efficiency": {
                "avg_computation_time_ms": round(np.mean(computation_times), 2),
                "fastest_feature": min(
                    metrics, key=lambda x: x.computation_time_ms
                ).feature_name,
                "slowest_feature": max(
                    metrics, key=lambda x: x.computation_time_ms
                ).feature_name,
            },
        }

    def _generate_module_breakdown(
        self, metrics: List[ExperimentalFeatureMetrics]
    ) -> Dict[str, Any]:
        """Generate module-specific breakdown"""
        modules: Dict[str, Dict[str, Any]] = {}

        for metric in metrics:
            module = metric.module_name
            if module not in modules:
                modules[module] = {
                    "feature_count": 0,
                    "avg_delta_sharpe": 0,
                    "avg_nan_rate": 0,
                    "recommendations": {},
                }

            modules[module]["feature_count"] += 1
            modules[module]["avg_delta_sharpe"] += metric.delta_sharpe
            modules[module]["avg_nan_rate"] += metric.nan_rate

            rec = (
                metric.recommendation.split()[1]
                if " " in metric.recommendation
                else metric.recommendation
            )
            modules[module]["recommendations"][rec] = (
                modules[module]["recommendations"].get(rec, 0) + 1
            )

        # Calculate averages
        for module_data in modules.values():
            if module_data["feature_count"] > 0:
                module_data["avg_delta_sharpe"] /= module_data["feature_count"]
                module_data["avg_nan_rate"] /= module_data["feature_count"]
                module_data["avg_delta_sharpe"] = round(
                    module_data["avg_delta_sharpe"], 4
                )
                module_data["avg_nan_rate"] = round(module_data["avg_nan_rate"], 4)

        return modules

    def _generate_trending_analysis(
        self, metrics: List[ExperimentalFeatureMetrics]
    ) -> Dict[str, Any]:
        """Generate trending analysis comparing to historical data"""
        trending = {}

        for metric in metrics:
            historical = self.get_historical_trends(metric.feature_name)
            if len(historical) > 1:
                current_sharpe = metric.delta_sharpe
                prev_sharpe = (
                    historical[1]["delta_sharpe"]
                    if len(historical) > 1
                    else current_sharpe
                )

                trend_direction = (
                    "improving" if current_sharpe > prev_sharpe else "declining"
                )
                trend_magnitude = abs(current_sharpe - prev_sharpe)

                trending[metric.feature_name] = {
                    "trend_direction": trend_direction,
                    "trend_magnitude": round(trend_magnitude, 4),
                    "current_sharpe": round(current_sharpe, 4),
                    "previous_sharpe": round(prev_sharpe, 4),
                    "weeks_tracked": len(historical),
                }

        return trending

    def _metric_to_dict(self, metric: ExperimentalFeatureMetrics) -> Dict[str, Any]:
        """Convert ExperimentalFeatureMetrics to dictionary"""
        return {
            "feature_name": metric.feature_name,
            "module_name": metric.module_name,
            "nan_rate": round(metric.nan_rate, 4),
            "delta_sharpe": round(metric.delta_sharpe, 4),
            "computation_time_ms": round(metric.computation_time_ms, 2),
            "stability_score": round(metric.stability_score, 3),
            "recommendation": metric.recommendation,
            "reason_code": metric.reason_code,
        }

    def export_report(
        self, report: Dict[str, Any], format: str = "json"
    ) -> Optional[Path]:
        """Export report to file"""
        report_date = report["report_date"]
        output_file: Optional[Path] = None

        # Check if format is supported
        if format not in ["json", "yaml", "markdown"]:
            raise ValueError(
                f"Unsupported export format: {format}. Supported formats: json, yaml, markdown"
            )

        if format == "json":
            output_file = self.output_dir / f"experimental_report_{report_date}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        elif format == "yaml":
            output_file = self.output_dir / f"experimental_report_{report_date}.yaml"
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(report, f, default_flow_style=False, allow_unicode=True)

        elif format == "markdown":
            output_file = self.output_dir / f"experimental_report_{report_date}.md"
            self._export_markdown_report(report, output_file)

        if output_file is not None:
            print(f"📊 Report exported to: {output_file}")
            return output_file
        return None  # Fallback, though raise should prevent this

    def _export_markdown_report(
        self, report: Dict[str, Any], output_file: Path
    ) -> None:
        """Export report as Markdown"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Experimental Features Weekly Report\n")
            f.write(f"**Report Date:** {report['report_date']}\n")
            f.write(
                f"**Total Features Evaluated:** {report['total_features_evaluated']}\n\n"
            )

            # Summary
            summary = report["summary"]
            f.write("## 📈 Executive Summary\n\n")
            f.write(
                f"- **Promote:** {summary['recommendation_breakdown']['promote']} features\n"
            )
            f.write(
                f"- **Approve:** {summary['recommendation_breakdown']['approve']} features\n"
            )
            f.write(
                f"- **Needs Work:** {summary['recommendation_breakdown']['needs_work']} features\n"
            )
            f.write(
                f"- **Reject:** {summary['recommendation_breakdown']['reject']} features\n\n"
            )

            # Performance metrics
            avg = summary["average_metrics"]
            f.write(f"**Average Performance:**\n")
            f.write(f"- Delta Sharpe: {avg['delta_sharpe']}\n")
            f.write(f"- NaN Rate: {avg['nan_rate'] * 100:.1f}%\n")
            f.write(f"- Computation Time: {avg['computation_time_ms']:.1f}ms\n\n")

            # Recommendations
            f.write("## 🎯 Key Recommendations\n\n")
            for rec in report["recommendations"]:
                f.write(f"### {rec['action']} ({rec['priority']} Priority)\n")
                f.write(f"**Features:** {', '.join(rec['features'])}\n\n")
                f.write(f"**Rationale:** {rec['rationale']}\n\n")

            # Detailed metrics
            f.write("## 📊 Detailed Feature Analysis\n\n")
            f.write(
                "| Feature | Module | Delta Sharpe | NaN Rate | Comp Time (ms) | Recommendation |\n"
            )
            f.write(
                "|---------|--------|--------------|----------|----------------|----------------|\n"
            )

            for metric in report["detailed_metrics"]:
                f.write(
                    f"| {metric['feature_name']} | {metric['module_name'].split('.')[-1]} |"
                )
                f.write(
                    f" {metric['delta_sharpe']} | {metric['nan_rate'] * 100:.1f}% |"
                )
                f.write(
                    f" {metric['computation_time_ms']} | {metric['recommendation']} |\n"
                )


def main() -> None:
    """Main execution function"""
    reporter = ExperimentalWeeklyReporter()

    # Generate weekly report
    report = reporter.generate_weekly_report()

    # Export in multiple formats
    reporter.export_report(report, "json")
    reporter.export_report(report, "markdown")

    print("\n🎉 Experimental features weekly report generated successfully!")
    print(f"Total features evaluated: {report['total_features_evaluated']}")

    # Print summary
    summary = report["summary"]
    print(f"Recommendations: {summary['recommendation_breakdown']}")


if __name__ == "__main__":
    main()
