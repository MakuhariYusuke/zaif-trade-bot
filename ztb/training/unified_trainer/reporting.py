#!/usr/bin/env python3
"""
Training reporting and logging utilities.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ztb.metrics.metrics import sharpe_ratio, sortino_ratio
from ztb.reporting.services.training_reports import (
    save_ensemble_report,
    save_training_report,
)
from ztb.types.common import ConfigDict, ObjectMap, ObjectRecords
from ztb.utils.logging_utils import get_logger

# Magic number constants for reporting
STABILITY_WINDOWS = [10, 50, 100, 500]
DEFAULT_PORTFOLIO_BASE = 10000
FORECAST_PERIOD = 50


class TrainingReporter:
    """Generate comprehensive training reports."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger: logging.Logger = logger or get_logger(__name__)
        self._event_logger = TrainingEventLogger(self.logger)
        # Keep a direct reference for generate_report's event capture.
        self.events = self._event_logger.events

    def generate_report(
        self, config: ConfigDict, stats: ObjectMap, success: bool
    ) -> ObjectMap:
        """Generate a comprehensive training report."""
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "algorithm": config.get("algorithm", "unknown"),
                "model_name": config.get("model_name", "unknown"),
                "ab_tag": config.get("ab_tag") if isinstance(config, dict) else None,
                "success": success,
            },
            "configuration": config,
            "training_stats": stats,
            "performance_metrics": self._calculate_performance_metrics(stats),
            "system_info": self._get_system_info(),
        }

        # Include any logged events if present (provides step-level diagnostics such as action_distribution)
        if hasattr(self, "events") and self.events:
            report["training_events"] = self.events.copy()

        # Include reward_components if present in stats for AB analysis
        if "reward_components" in stats:
            report["reward_components"] = stats["reward_components"]

        return report

    def save_report(self, report: ObjectMap, output_dir: str = "reports") -> str:
        """Save training report to file."""
        try:
            filepath = save_training_report(report, output_dir=output_dir)
            self.logger.info(f"Training report saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save training report: {e}")
            return ""

    def print_summary(self, report: ObjectMap) -> None:
        """Print a human-readable summary of the training report."""
        meta = report.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        stats = report.get("training_stats", {})
        if not isinstance(stats, dict):
            stats = {}
        perf = report.get("performance_metrics", {})
        if not isinstance(perf, dict):
            perf = {}

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING REPORT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Algorithm: {str(meta.get('algorithm', 'unknown')).upper()}")
        self.logger.info(f"Model: {meta.get('model_name', 'unknown')}")
        self.logger.info(
            f"Status: {'SUCCESS' if bool(meta.get('success', False)) else 'FAILED'}"
        )
        self.logger.info(f"Timestamp: {meta.get('timestamp', 'n/a')}")

        if stats:
            self.logger.info("TRAINING STATISTICS")
            self.logger.info("-" * 30)
            for key, value in stats.items():
                if isinstance(value, float):
                    if "time" in str(key).lower():
                        self.logger.info(f"{key}: {value:.2f}s")
                    elif "rate" in str(key).lower() or "ratio" in str(key).lower():
                        self.logger.info(f"{key}: {value:.4f}")
                    else:
                        self.logger.info(f"{key}: {value:.2f}")
                elif isinstance(value, int):
                    self.logger.info(f"{key}: {value:,}")
                else:
                    self.logger.info(f"{key}: {value}")

        if perf:
            self.logger.info("PERFORMANCE METRICS")
            self.logger.info("-" * 30)
            for key, value in perf.items():
                if isinstance(value, float):
                    self.logger.info(f"{key}: {value:.4f}")
                else:
                    self.logger.info(f"{key}: {value}")

        self.logger.info("=" * 60)

    def log_training_start(self, algorithm: str, config: ConfigDict) -> None:
        """Log training start (delegated to TrainingEventLogger)."""
        self._event_logger.log_training_start(algorithm, config)

    def log_training_progress(
        self, step: int, total_steps: int, stats: ObjectMap
    ) -> None:
        """Log training progress (delegated to TrainingEventLogger)."""
        self._event_logger.log_training_progress(step, total_steps, stats)

    def log_training_complete(self, success: bool, stats: ObjectMap) -> None:
        """Log training completion (delegated to TrainingEventLogger)."""
        self._event_logger.log_training_complete(success, stats)

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log a training error (delegated to TrainingEventLogger)."""
        self._event_logger.log_error(error, context)

    def get_events(self) -> list:
        """Get logged training events."""
        return self._event_logger.get_events()

    def save_events(self, filepath: str) -> None:
        """Save logged events to file."""
        self._event_logger.save_events(filepath)

    def generate_ensemble_report(
        self, ensemble_stats: ObjectMap, decision_log: ObjectRecords
    ) -> ObjectMap:
        """Generate ensemble analysis report (delegated to TrainingEventLogger)."""
        return self._event_logger.generate_ensemble_report(ensemble_stats, decision_log)

    def save_ensemble_report(
        self, report: ObjectMap, output_dir: str = "reports"
    ) -> str:
        """Save ensemble report (delegated to TrainingEventLogger)."""
        return self._event_logger.save_ensemble_report(report, output_dir=output_dir)

    def _calculate_performance_metrics(self, stats: ObjectMap) -> ObjectMap:
        """Calculate additional performance metrics from training stats."""
        # metrics can contain floats and occasional strings (e.g. dominant_action)
        metrics: ObjectMap = {}

        if not stats:
            return metrics

        # Training efficiency
        total_timesteps = stats.get("total_timesteps", 0)
        training_time = stats.get("training_time", 0)
        if training_time > 0:
            metrics["steps_per_second"] = total_timesteps / training_time
            metrics["training_efficiency"] = total_timesteps / (
                training_time * 1000
            )  # steps per ms

        # Action distribution analysis
        action_dist: Dict[str, float] = stats.get("action_distribution", {})
        if action_dist:
            # Calculate action diversity (1.0 = perfectly balanced, 0.0 = single action)
            actions = list(action_dist.values())
            if actions:
                ideal_ratio = 1.0 / len(actions)
                diversity = 1.0 - sum(abs(r - ideal_ratio) for r in actions) / 2.0
                metrics["action_diversity"] = diversity

                # Most used action
                most_used = max(action_dist.items(), key=lambda x: x[1])
                metrics["dominant_action"] = most_used[0]
                metrics["dominant_action_ratio"] = most_used[1]

        return metrics

    def _get_system_info(self) -> ObjectMap:
        """Get basic system information."""
        try:
            import platform

            import psutil

            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}


class TrainingEventLogger:
    """Enhanced logging for training processes."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger: logging.Logger = logger or get_logger(__name__)
        self.events: ObjectRecords = []

    def log_event(
        self, event_type: str, message: str, data: Optional[ObjectMap] = None
    ) -> None:
        """Log a training event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "data": data or {},
        }

        self.events.append(event)
        self.logger.info(f"[{event_type}] {message}")

    def log_training_start(self, algorithm: str, config: ConfigDict) -> None:
        """Log training start."""
        total_timesteps = None
        if isinstance(config, dict):
            training_cfg = config.get("training", {})
            if isinstance(training_cfg, dict):
                total_timesteps = training_cfg.get("total_timesteps")
        self.log_event(
            "training_start",
            f"Starting {algorithm} training",
            {
                "algorithm": algorithm,
                "config_summary": {
                    "total_timesteps": total_timesteps,
                    "model_name": config.get("model_name")
                    if isinstance(config, dict)
                    else None,
                },
            },
        )

    def log_training_progress(
        self, step: int, total_steps: int, stats: ObjectMap
    ) -> None:
        """Log training progress."""
        progress = step / total_steps if total_steps > 0 else 0
        self.log_event(
            "training_progress", f"Step {step}/{total_steps} ({progress:.1%})", stats
        )

    def log_training_complete(self, success: bool, stats: ObjectMap) -> None:
        """Log training completion."""
        status = "success" if success else "failure"
        self.log_event("training_complete", f"Training {status}", stats)

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error."""
        self.log_event(
            "error",
            f"Error in {context}: {str(error)}",
            {"error_type": type(error).__name__, "context": context},
        )

    def get_events(self) -> list:
        """Get all logged events."""
        return self.events.copy()

    def save_events(self, filepath: str) -> None:
        """Save events to file."""
        try:
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            from ztb.io.json_io import write_json

            write_json(save_path, self.events, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Training events saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save training events: {e}")

    def generate_ensemble_report(
        self, ensemble_stats: ObjectMap, decision_log: ObjectRecords
    ) -> ObjectMap:
        """Generate comprehensive ensemble analysis report."""
        report = {
            "ensemble_analysis": {
                "timestamp": datetime.now().isoformat(),
                "summary": self._analyze_ensemble_performance(
                    ensemble_stats, decision_log
                ),
                "member_analysis": self._analyze_member_performance(ensemble_stats),
                "decision_analysis": self._analyze_decision_patterns(decision_log),
                "diversity_analysis": self._analyze_ensemble_diversity(ensemble_stats),
                "stability_analysis": self._analyze_ensemble_stability(decision_log),
                "market_adaptation": self._analyze_market_adaptation(decision_log),
                "risk_analysis": self._analyze_ensemble_risk(decision_log),
                "performance_forecast": self._generate_performance_forecast(
                    ensemble_stats, decision_log
                ),
            }
        }

        return report

    def _analyze_ensemble_performance(
        self, ensemble_stats: ObjectMap, decision_log: ObjectRecords
    ) -> ObjectMap:
        """Analyze overall ensemble performance."""
        if not decision_log:
            return {"error": "no_decision_data"}

        overall_stats = ensemble_stats.get("overall_stats", {})

        # 決定の統計
        total_decisions = len(decision_log)
        action_distribution: Dict[int, int] = {}
        confidence_trends: List[float] = []

        for decision in decision_log:
            predictions = decision.get("predictions", {})
            final_action = decision.get("final_action", 0)

            # アクション分布
            action_distribution[final_action] = (
                action_distribution.get(final_action, 0) + 1
            )

            # 平均信頼度
            if predictions:
                avg_conf = sum(p["confidence"] for p in predictions.values()) / len(
                    predictions
                )
                confidence_trends.append(avg_conf)

        return {
            "total_decisions": total_decisions,
            "action_distribution": action_distribution,
            "avg_confidence": overall_stats.get("avg_confidence", 0),
            "confidence_trend": confidence_trends[-100:]
            if confidence_trends
            else [],  # 最新100件
            "performance_score": overall_stats.get("avg_performance", 0),
            "stability_score": overall_stats.get("avg_stability", 0),
        }

    def _analyze_member_performance(
        self, ensemble_stats: ObjectMap
    ) -> ObjectMap:
        """Analyze individual member performance."""
        member_stats = ensemble_stats.get("member_stats", {})

        if not member_stats:
            return {"error": "no_member_data"}

        specialization_performance: Dict[str, List[float]] = {}
        member_details: Dict[str, ObjectMap] = {}

        for member_id, stats in member_stats.items():
            spec = stats.get("specialization", "unknown")
            perf = stats.get("performance_score", 0)
            stab = stats.get("stability_score", 0)
            conf = stats.get("confidence", 0)

            member_details[member_id] = {
                "specialization": spec,
                "performance": perf,
                "stability": stab,
                "confidence": conf,
                "composite_score": (perf + stab + conf) / 3,
            }

            # 専門化ごとの集計
            if spec not in specialization_performance:
                specialization_performance[spec] = []
            specialization_performance[spec].append(perf)

        # 専門化ごとの平均パフォーマンス
        spec_avg_performance = {}
        for spec, performances in specialization_performance.items():
            spec_avg_performance[spec] = {
                "avg_performance": sum(performances) / len(performances),
                "member_count": len(performances),
                "best_performance": max(performances),
                "worst_performance": min(performances),
            }

        return {
            "member_details": member_details,
            "specialization_performance": spec_avg_performance,
            "top_performer": max(
                member_details.keys(),
                key=lambda x: member_details[x]["composite_score"],
            ),
            "needs_improvement": [
                mid
                for mid, stats in member_details.items()
                if stats["composite_score"] < 0.5
            ],
        }

    def _analyze_decision_patterns(
        self, decision_log: ObjectRecords
    ) -> ObjectMap:
        """Analyze decision patterns and voting behavior."""
        if not decision_log:
            return {"error": "no_decision_data"}

        voting_methods: Dict[str, int] = {}
        consensus_rates = {"reached": 0, "failed": 0}
        action_sequences = []
        confidence_patterns = []

        for decision in decision_log:
            analysis = decision.get("analysis", {})
            method = analysis.get("method", "unknown")
            final_action = decision.get("final_action", 0)

            # 投票方法の集計
            voting_methods[method] = voting_methods.get(method, 0) + 1

            # 合意率の集計
            if method == "consensus":
                if analysis.get("consensus_reached", False):
                    consensus_rates["reached"] += 1
                else:
                    consensus_rates["failed"] += 1

            # アクションシーケンス
            action_sequences.append(final_action)

            # 信頼度パターン
            predictions = decision.get("predictions", {})
            if predictions:
                pattern = {
                    "avg_confidence": sum(p["confidence"] for p in predictions.values())
                    / len(predictions),
                    "confidence_variance": np.var(
                        [p["confidence"] for p in predictions.values()]
                    ),
                    "member_agreement": len(
                        set(p["action"] for p in predictions.values())
                    )
                    == 1,
                }
                confidence_patterns.append(pattern)

        # アクション遷移分析
        transitions: Dict[str, int] = {}
        for i in range(1, len(action_sequences)):
            prev_action = action_sequences[i - 1]
            curr_action = action_sequences[i]
            key = f"{prev_action}->{curr_action}"
            transitions[key] = transitions.get(key, 0) + 1

        return {
            "voting_method_distribution": voting_methods,
            "consensus_rate": consensus_rates["reached"]
            / (consensus_rates["reached"] + consensus_rates["failed"])
            if (consensus_rates["reached"] + consensus_rates["failed"]) > 0
            else 0,
            "action_transitions": transitions,
            "decision_stability": self._calculate_decision_stability(action_sequences),
            "confidence_patterns": confidence_patterns[-50:]
            if confidence_patterns
            else [],  # 最新50件
        }

    def _analyze_ensemble_diversity(
        self, ensemble_stats: ObjectMap
    ) -> ObjectMap:
        """Analyze ensemble diversity."""
        member_stats = ensemble_stats.get("member_stats", {})

        if not member_stats:
            return {"error": "no_member_data"}

        # 専門化の多様性
        specializations = [
            stats.get("specialization", "unknown") for stats in member_stats.values()
        ]
        specialization_diversity: float = len(set(specializations)) / len(
            specializations
        )

        # パフォーマンスの多様性
        performances = [
            stats.get("performance_score", 0) for stats in member_stats.values()
        ]
        performance_diversity: float = float(
            np.std(performances) if len(performances) > 1 else 0.0
        )

        # 信頼度の多様性
        confidences = [stats.get("confidence", 0) for stats in member_stats.values()]
        confidence_diversity: float = float(
            np.std(confidences) if len(confidences) > 1 else 0.0
        )

        return {
            "specialization_diversity": specialization_diversity,
            "performance_diversity": performance_diversity,
            "confidence_diversity": confidence_diversity,
            "overall_diversity_score": (
                specialization_diversity + performance_diversity + confidence_diversity
            )
            / 3,
            "diversity_recommendations": self._generate_diversity_recommendations(
                specialization_diversity, performance_diversity, confidence_diversity
            ),
        }

    def _analyze_ensemble_stability(
        self, decision_log: ObjectRecords
    ) -> ObjectMap:
        """Analyze ensemble stability over time."""
        if not decision_log:
            return {"error": "no_decision_data"}

        # 時間経過による安定性分析
        stability_windows = STABILITY_WINDOWS
        stability_analysis: Dict[str, ObjectMap] = {}

        for window_size in stability_windows:
            if len(decision_log) < window_size:
                continue

            # ローリングウィンドウでの安定性計算
            stabilities = []
            for i in range(window_size, len(decision_log), window_size // 2):
                window = decision_log[i - window_size : i]

                # 安定性の指標
                actions = [d.get("final_action", 0) for d in window]
                action_stability = 1 - (
                    len(set(actions)) / len(actions)
                )  # アクションの一貫性

                confidences = []
                for d in window:
                    predictions = d.get("predictions", {})
                    if predictions:
                        avg_conf = sum(
                            p["confidence"] for p in predictions.values()
                        ) / len(predictions)
                        confidences.append(avg_conf)

                confidence_stability = np.std(confidences) if confidences else 1.0
                confidence_stability = 1 / (1 + confidence_stability)  # 標準偏差の逆数

                overall_stability = (action_stability + confidence_stability) / 2
                stabilities.append(overall_stability)

            stability_analysis[f"window_{window_size}"] = {
                "avg_stability": np.mean(stabilities) if stabilities else 0,
                "stability_trend": stabilities[-10:]
                if stabilities
                else [],  # 最新10ウィンドウ
                "stability_volatility": np.std(stabilities) if stabilities else 0,
            }

        return stability_analysis

    def _analyze_market_adaptation(
        self, decision_log: ObjectRecords
    ) -> ObjectMap:
        """Analyze how well the ensemble adapts to market conditions."""
        if not decision_log:
            return {"error": "no_decision_data"}

        # 市場条件ごとのパフォーマンス分析
        market_conditions: Dict[str, ObjectMap] = {}
        adaptation_trends: ObjectRecords = []

        for decision in decision_log:
            market_state = decision.get("market_state", {})
            condition = market_state.get("regime", "unknown")
            volatility = float(market_state.get("volatility", 0.5))
            trend = market_state.get("trend", 0)

            if condition not in market_conditions:
                market_conditions[condition] = {
                    "decisions": 0,
                    "avg_confidence": [],
                    "performance": [],
                    "volatility_range": [],
                }

            market_conditions[condition]["decisions"] += 1
            market_conditions[condition]["avg_confidence"].append(
                decision.get("avg_confidence", 0)
            )
            market_conditions[condition]["volatility_range"].append(volatility)

            # 適応性のトレンド
            predictions = decision.get("predictions", {})
            if predictions:
                member_agreement = (
                    len(set(p["action"] for p in predictions.values())) == 1
                )
                adaptation_trends.append(
                    {
                        "condition": condition,
                        "volatility": volatility,
                        "trend": trend,
                        "agreement": member_agreement,
                        "confidence": sum(p["confidence"] for p in predictions.values())
                        / len(predictions),
                    }
                )

        # 市場条件ごとの集計
        condition_summary = {}
        for condition, data in market_conditions.items():
            condition_summary[condition] = {
                "total_decisions": data["decisions"],
                "avg_confidence": np.mean(data["avg_confidence"])
                if data["avg_confidence"]
                else 0,
                "confidence_std": np.std(data["avg_confidence"])
                if len(data["avg_confidence"]) > 1
                else 0,
                "volatility_range": f"{min(data['volatility_range']):.3f}-{max(data['volatility_range']):.3f}",
                "adaptation_score": self._calculate_adaptation_score(data),
            }

        return {
            "condition_performance": condition_summary,
            "adaptation_trends": adaptation_trends[-100:],  # 最新100件
            "overall_adaptation_score": self._calculate_overall_adaptation(
                condition_summary
            ),
            "market_regime_coverage": len(condition_summary),
            "recommendations": self._generate_adaptation_recommendations(
                condition_summary
            ),
        }

    def _analyze_ensemble_risk(
        self, decision_log: ObjectRecords
    ) -> ObjectMap:
        """Analyze ensemble risk metrics."""
        if not decision_log:
            return {"error": "no_decision_data"}

        # リスク指標の計算
        drawdown_analysis: List[float] = []
        volatility_analysis: List[float] = []
        portfolio_values: List[float] = []

        current_drawdown: float = 0.0
        peak_value: float = 0.0
        portfolio_value: float = float(DEFAULT_PORTFOLIO_BASE)  # 仮定の初期値

        for decision in decision_log:
            # 簡易的なポートフォリオ価値のシミュレーション
            action = decision.get("final_action", 0)
            confidence = decision.get("avg_confidence", 0.5)

            # アクションに基づく価値変化（簡易モデル）
            if action == 0:  # BUY
                change = confidence * 0.01  # 1%の上昇期待
            elif action == 1:  # HOLD
                change = 0  # 変化なし
            else:  # SELL
                change = -confidence * 0.01  # 1%の下落期待

            portfolio_value *= 1 + change
            portfolio_values.append(portfolio_value)

            # ドローダウン計算
            if portfolio_value > peak_value:
                peak_value = portfolio_value
                current_drawdown = 0
            else:
                current_drawdown = (peak_value - portfolio_value) / peak_value

            drawdown_analysis.append(current_drawdown)

            # ボラティリティ分析
            if len(drawdown_analysis) > 10:
                volatility_analysis.append(float(np.std(drawdown_analysis[-10:])))

        # リスク調整リターン
        # Use DEFAULT_PORTFOLIO_BASE for base calculation
        total_return = (portfolio_value - float(DEFAULT_PORTFOLIO_BASE)) / float(
            DEFAULT_PORTFOLIO_BASE
        )
        max_drawdown = max(drawdown_analysis) if drawdown_analysis else 0
        avg_volatility: float = (
            float(np.mean(volatility_analysis)) if volatility_analysis else 0.0
        )

        # Calculate returns for Sharpe and Sortino ratios
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio_val = sharpe_ratio(returns)
            sortino_ratio_val = sortino_ratio(returns)
        else:
            sharpe_ratio_val = 0.0
            sortino_ratio_val = 0.0

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "avg_volatility": avg_volatility,
            "sharpe_ratio": sharpe_ratio_val,
            "sortino_ratio": sortino_ratio_val,
            "risk_score": self._calculate_risk_score(max_drawdown, avg_volatility),
            "drawdown_analysis": drawdown_analysis[-50:],  # 最新50件
            "volatility_trend": volatility_analysis[-50:]
            if volatility_analysis
            else [],
        }

    def _generate_performance_forecast(
        self, ensemble_stats: ObjectMap, decision_log: ObjectRecords
    ) -> ObjectMap:
        """Generate performance forecast based on current trends."""
        if not decision_log:
            return {"error": "insufficient_data"}

        # トレンド分析
        recent_decisions = decision_log[-100:]  # 最新100件
        confidence_trend = []
        performance_trend = []

        for decision in recent_decisions:
            confidence_trend.append(decision.get("avg_confidence", 0.5))
            # パフォーマンスの推定（簡易モデル）
            action = decision.get("final_action", 0)
            confidence = decision.get("avg_confidence", 0.5)
            estimated_perf = confidence * (1 if action in [0, 1] else -1)
            performance_trend.append(estimated_perf)

        # トレンド予測
        conf_slope = (
            np.polyfit(range(len(confidence_trend)), confidence_trend, 1)[0]
            if len(confidence_trend) > 1
            else 0
        )
        perf_slope = (
            np.polyfit(range(len(performance_trend)), performance_trend, 1)[0]
            if len(performance_trend) > 1
            else 0
        )

        # 予測期間（次の決定数）
        forecast_period = FORECAST_PERIOD
        confidence_forecast: List[float] = []
        performance_forecast: List[float] = []

        for i in range(forecast_period):
            conf_pred = confidence_trend[-1] + conf_slope * (i + 1)
            conf_pred = max(0, min(1, conf_pred))  # 0-1の範囲に制限
            confidence_forecast.append(conf_pred)

            perf_pred = performance_trend[-1] + perf_slope * (i + 1)
            performance_forecast.append(perf_pred)

        return {
            "confidence_trend_slope": conf_slope,
            "performance_trend_slope": perf_slope,
            "forecast_period": forecast_period,
            "confidence_forecast": confidence_forecast,
            "performance_forecast": performance_forecast,
            "forecast_confidence": self._calculate_forecast_confidence(
                conf_slope, perf_slope
            ),
            "recommendations": self._generate_forecast_recommendations(
                conf_slope, perf_slope
            ),
        }

    def _calculate_adaptation_score(self, condition_data: ObjectMap) -> float:
        """Calculate adaptation score for a market condition."""
        decisions = condition_data["decisions"]
        avg_conf = (
            np.mean(condition_data["avg_confidence"])
            if condition_data["avg_confidence"]
            else 0
        )
        conf_std = (
            np.std(condition_data["avg_confidence"])
            if len(condition_data["avg_confidence"]) > 1
            else 0
        )

        # 適応スコア = 決定数 × 平均信頼度 × (1 - 信頼度の標準偏差)
        adaptation_score = decisions * avg_conf * (1 - min(conf_std, 0.5))
        return float(adaptation_score)

    def _calculate_overall_adaptation(self, condition_summary: ObjectMap) -> float:
        """Calculate overall adaptation score across all conditions."""
        if not condition_summary:
            return 0.0

        total_score = sum(
            data["adaptation_score"] for data in condition_summary.values()
        )
        avg_score = total_score / len(condition_summary)

        # 正規化（0-1の範囲）
        return float(min(avg_score / 1000, 1.0))  # 1000は経験的なスケーリング係数

    def _calculate_risk_score(
        self, max_drawdown: float, avg_volatility: float
    ) -> float:
        """Calculate overall risk score."""
        # リスクスコア = (最大ドローダウン + 平均ボラティリティ) / 2
        # 低いスコアほどリスクが低い
        risk_score = (max_drawdown + avg_volatility) / 2
        return min(risk_score, 1.0)  # 0-1の範囲に制限

    def _calculate_forecast_confidence(
        self, conf_slope: float, perf_slope: float
    ) -> float:
        """Calculate confidence in the forecast."""
        # 予測の信頼性 = 1 - |トレンド勾配|（安定したトレンドほど信頼性が高い）
        conf_stability = 1 - min(abs(conf_slope), 1.0)
        perf_stability = 1 - min(abs(perf_slope), 1.0)

        return (conf_stability + perf_stability) / 2

    def _generate_adaptation_recommendations(
        self, condition_summary: ObjectMap
    ) -> List[str]:
        """Generate adaptation improvement recommendations."""
        recommendations = []

        if not condition_summary:
            return ["市場適応データを収集する必要があります"]

        # 最も弱い市場条件を特定
        weak_conditions = sorted(
            condition_summary.items(), key=lambda x: x[1]["adaptation_score"]
        )[:2]

        for condition, data in weak_conditions:
            if data["adaptation_score"] < 500:  # 経験的な閾値
                recommendations.append(
                    f"{condition}市場条件での適応を改善するため、専門家を追加または再訓練"
                )

        if len(condition_summary) < 3:
            recommendations.append("より多様な市場条件でのテストを実施")

        if not recommendations:
            recommendations.append("市場適応性は良好です")

        return recommendations

    def _generate_forecast_recommendations(
        self, conf_slope: float, perf_slope: float
    ) -> List[str]:
        """Generate forecast-based recommendations."""
        recommendations = []

        if abs(conf_slope) > 0.01:
            if conf_slope > 0:
                recommendations.append("信頼度が上昇傾向 - 現在の戦略を維持")
            else:
                recommendations.append("信頼度が低下傾向 - モデル再訓練を検討")

        if abs(perf_slope) > 0.005:
            if perf_slope > 0:
                recommendations.append(
                    "パフォーマンスが改善傾向 - ポジションサイズを増加"
                )
            else:
                recommendations.append("パフォーマンスが低下傾向 - リスク管理を強化")

        if abs(conf_slope) <= 0.01 and abs(perf_slope) <= 0.005:
            recommendations.append("安定したパフォーマンス - 現在の設定を維持")

        return recommendations

    def _calculate_decision_stability(self, action_sequence: List[int]) -> float:
        """Calculate decision stability from action sequence."""
        if len(action_sequence) < 2:
            return 1.0

        # 連続する同じアクションの割合
        same_action_count = sum(
            1
            for i in range(1, len(action_sequence))
            if action_sequence[i] == action_sequence[i - 1]
        )

        stability = same_action_count / (len(action_sequence) - 1)
        return stability

    def _generate_diversity_recommendations(
        self, spec_div: float, perf_div: float, conf_div: float
    ) -> List[str]:
        """Generate diversity improvement recommendations."""
        recommendations = []

        if spec_div < 0.8:
            recommendations.append(
                "専門化の多様性を高めるため、新しい市場レジームの専門家を追加"
            )

        if perf_div < 0.2:
            recommendations.append(
                "メンバーのパフォーマンス差が小さいため、個別最適化を実施"
            )

        if conf_div < 0.15:
            recommendations.append(
                "信頼度のばらつきが小さいため、メンバーの専門性を再評価"
            )

        if not recommendations:
            recommendations.append("アンサンブルの多様性は良好です")

        return recommendations

    def save_ensemble_report(
        self, report: ObjectMap, output_dir: str = "reports"
    ) -> str:
        """Save ensemble report to file."""
        try:
            filepath = save_ensemble_report(report, output_dir=output_dir)
            self.logger.info(f"Ensemble report saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save ensemble report: {e}")
            return ""
