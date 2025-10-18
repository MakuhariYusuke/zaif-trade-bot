#!/usr/bin/env python3
"""
Training reporting and logging utilities.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ztb.utils.logging_utils import get_logger


class TrainingReporter:
    """Generate comprehensive training reports."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)

    def generate_report(self, config: Dict[str, Any], stats: Dict[str, Any], success: bool) -> Dict[str, Any]:
        """Generate a comprehensive training report."""
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "algorithm": config.get("algorithm", "unknown"),
                "model_name": config.get("model_name", "unknown"),
                "success": success
            },
            "configuration": config,
            "training_stats": stats,
            "performance_metrics": self._calculate_performance_metrics(stats),
            "system_info": self._get_system_info()
        }

        return report

    def save_report(self, report: Dict[str, Any], output_dir: str = "reports") -> str:
        """Save training report to file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm = report["metadata"]["algorithm"]
        model_name = report["metadata"]["model_name"]

        filename = f"training_report_{algorithm}_{model_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Training report saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to save training report: {e}")
            return ""

    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the training report."""
        meta = report["metadata"]
        stats = report["training_stats"]
        perf = report["performance_metrics"]

        print("\n" + "="*60)
        print("TRAINING REPORT SUMMARY")
        print("="*60)

        print(f"Algorithm: {meta['algorithm'].upper()}")
        print(f"Model: {meta['model_name']}")
        print(f"Status: {'✅ SUCCESS' if meta['success'] else '❌ FAILED'}")
        print(f"Timestamp: {meta['timestamp']}")

        if stats:
            print("\n📊 TRAINING STATISTICS:")
            print("-" * 30)

            for key, value in stats.items():
                if isinstance(value, float):
                    if 'time' in key.lower():
                        print(f"{key}: {value:.2f}s")
                    elif 'rate' in key.lower() or 'ratio' in key.lower():
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value:.2f}")
                elif isinstance(value, int):
                    print(f"{key}: {value:,}")
                else:
                    print(f"{key}: {value}")

        if perf:
            print("\n📈 PERFORMANCE METRICS:")
            print("-" * 30)

            for key, value in perf.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

        print("="*60)

    def _calculate_performance_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional performance metrics from training stats."""
        metrics = {}

        if not stats:
            return metrics

        # Training efficiency
        total_timesteps = stats.get('total_timesteps', 0)
        training_time = stats.get('training_time', 0)
        if training_time > 0:
            metrics['steps_per_second'] = total_timesteps / training_time
            metrics['training_efficiency'] = total_timesteps / (training_time * 1000)  # steps per ms

        # Action distribution analysis
        action_dist = stats.get('action_distribution', {})
        if action_dist:
            # Calculate action diversity (1.0 = perfectly balanced, 0.0 = single action)
            actions = list(action_dist.values())
            if actions:
                ideal_ratio = 1.0 / len(actions)
                diversity = 1.0 - sum(abs(r - ideal_ratio) for r in actions) / 2.0
                metrics['action_diversity'] = diversity

                # Most used action
                most_used = max(action_dist.items(), key=lambda x: x[1])
                metrics['dominant_action'] = most_used[0]
                metrics['dominant_action_ratio'] = most_used[1]

        return metrics

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            import platform
            import psutil

            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}


class TrainingLogger:
    """Enhanced logging for training processes."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.events = []

    def log_event(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a training event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "data": data or {}
        }

        self.events.append(event)
        self.logger.info(f"[{event_type}] {message}")

    def log_training_start(self, algorithm: str, config: Dict[str, Any]):
        """Log training start."""
        self.log_event("training_start", f"Starting {algorithm} training", {
            "algorithm": algorithm,
            "config_summary": {
                "total_timesteps": config.get("total_timesteps"),
                "model_name": config.get("model_name")
            }
        })

    def log_training_progress(self, step: int, total_steps: int, stats: Dict[str, Any]):
        """Log training progress."""
        progress = step / total_steps if total_steps > 0 else 0
        self.log_event("training_progress", f"Step {step}/{total_steps} ({progress:.1%})", stats)

    def log_training_complete(self, success: bool, stats: Dict[str, Any]):
        """Log training completion."""
        status = "success" if success else "failure"
        self.log_event("training_complete", f"Training {status}", stats)

    def log_error(self, error: Exception, context: str = ""):
        """Log an error."""
        self.log_event("error", f"Error in {context}: {str(error)}", {
            "error_type": type(error).__name__,
            "context": context
        })

    def get_events(self) -> list:
        """Get all logged events."""
        return self.events.copy()

    def save_events(self, filepath: str):
        """Save events to file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.events, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Training events saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save training events: {e}")

    def generate_ensemble_report(self, ensemble_stats: Dict[str, Any],
                               decision_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive ensemble analysis report."""
        report = {
            "ensemble_analysis": {
                "timestamp": datetime.now().isoformat(),
                "summary": self._analyze_ensemble_performance(ensemble_stats, decision_log),
                "member_analysis": self._analyze_member_performance(ensemble_stats),
                "decision_analysis": self._analyze_decision_patterns(decision_log),
                "diversity_analysis": self._analyze_ensemble_diversity(ensemble_stats),
                "stability_analysis": self._analyze_ensemble_stability(decision_log)
            }
        }

        return report

    def _analyze_ensemble_performance(self, ensemble_stats: Dict[str, Any],
                                    decision_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall ensemble performance."""
        if not decision_log:
            return {"error": "no_decision_data"}

        overall_stats = ensemble_stats.get("overall_stats", {})

        # 決定の統計
        total_decisions = len(decision_log)
        action_distribution = {}
        confidence_trends = []

        for decision in decision_log:
            predictions = decision.get("predictions", {})
            final_action = decision.get("final_action", 0)

            # アクション分布
            action_distribution[final_action] = action_distribution.get(final_action, 0) + 1

            # 平均信頼度
            if predictions:
                avg_conf = sum(p["confidence"] for p in predictions.values()) / len(predictions)
                confidence_trends.append(avg_conf)

        return {
            "total_decisions": total_decisions,
            "action_distribution": action_distribution,
            "avg_confidence": overall_stats.get("avg_confidence", 0),
            "confidence_trend": confidence_trends[-100:] if confidence_trends else [],  # 最新100件
            "performance_score": overall_stats.get("avg_performance", 0),
            "stability_score": overall_stats.get("avg_stability", 0)
        }

    def _analyze_member_performance(self, ensemble_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual member performance."""
        member_stats = ensemble_stats.get("member_stats", {})

        if not member_stats:
            return {"error": "no_member_data"}

        specialization_performance = {}
        member_details = {}

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
                "composite_score": (perf + stab + conf) / 3
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
                "worst_performance": min(performances)
            }

        return {
            "member_details": member_details,
            "specialization_performance": spec_avg_performance,
            "top_performer": max(member_details.keys(),
                               key=lambda x: member_details[x]["composite_score"]),
            "needs_improvement": [mid for mid, stats in member_details.items()
                                if stats["composite_score"] < 0.5]
        }

    def _analyze_decision_patterns(self, decision_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze decision patterns and voting behavior."""
        if not decision_log:
            return {"error": "no_decision_data"}

        voting_methods = {}
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
                    "avg_confidence": sum(p["confidence"] for p in predictions.values()) / len(predictions),
                    "confidence_variance": np.var([p["confidence"] for p in predictions.values()]),
                    "member_agreement": len(set(p["action"] for p in predictions.values())) == 1
                }
                confidence_patterns.append(pattern)

        # アクション遷移分析
        transitions = {}
        for i in range(1, len(action_sequences)):
            prev_action = action_sequences[i-1]
            curr_action = action_sequences[i]
            key = f"{prev_action}->{curr_action}"
            transitions[key] = transitions.get(key, 0) + 1

        return {
            "voting_method_distribution": voting_methods,
            "consensus_rate": consensus_rates["reached"] / (consensus_rates["reached"] + consensus_rates["failed"])
                               if (consensus_rates["reached"] + consensus_rates["failed"]) > 0 else 0,
            "action_transitions": transitions,
            "decision_stability": self._calculate_decision_stability(action_sequences),
            "confidence_patterns": confidence_patterns[-50:] if confidence_patterns else []  # 最新50件
        }

    def _analyze_ensemble_diversity(self, ensemble_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ensemble diversity."""
        member_stats = ensemble_stats.get("member_stats", {})

        if not member_stats:
            return {"error": "no_member_data"}

        # 専門化の多様性
        specializations = [stats.get("specialization", "unknown") for stats in member_stats.values()]
        specialization_diversity = len(set(specializations)) / len(specializations)

        # パフォーマンスの多様性
        performances = [stats.get("performance_score", 0) for stats in member_stats.values()]
        performance_diversity = np.std(performances) if len(performances) > 1 else 0

        # 信頼度の多様性
        confidences = [stats.get("confidence", 0) for stats in member_stats.values()]
        confidence_diversity = np.std(confidences) if len(confidences) > 1 else 0

        return {
            "specialization_diversity": specialization_diversity,
            "performance_diversity": performance_diversity,
            "confidence_diversity": confidence_diversity,
            "overall_diversity_score": (specialization_diversity +
                                      performance_diversity +
                                      confidence_diversity) / 3,
            "diversity_recommendations": self._generate_diversity_recommendations(
                specialization_diversity, performance_diversity, confidence_diversity
            )
        }

    def _analyze_ensemble_stability(self, decision_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze ensemble stability over time."""
        if not decision_log:
            return {"error": "no_decision_data"}

        # 時間経過による安定性分析
        stability_windows = [10, 50, 100, 500]
        stability_analysis = {}

        for window_size in stability_windows:
            if len(decision_log) < window_size:
                continue

            # ローリングウィンドウでの安定性計算
            stabilities = []
            for i in range(window_size, len(decision_log), window_size // 2):
                window = decision_log[i-window_size:i]

                # 安定性の指標
                actions = [d.get("final_action", 0) for d in window]
                action_stability = 1 - (len(set(actions)) / len(actions))  # アクションの一貫性

                confidences = []
                for d in window:
                    predictions = d.get("predictions", {})
                    if predictions:
                        avg_conf = sum(p["confidence"] for p in predictions.values()) / len(predictions)
                        confidences.append(avg_conf)

                confidence_stability = np.std(confidences) if confidences else 1.0
                confidence_stability = 1 / (1 + confidence_stability)  # 標準偏差の逆数

                overall_stability = (action_stability + confidence_stability) / 2
                stabilities.append(overall_stability)

            stability_analysis[f"window_{window_size}"] = {
                "avg_stability": np.mean(stabilities) if stabilities else 0,
                "stability_trend": stabilities[-10:] if stabilities else [],  # 最新10ウィンドウ
                "stability_volatility": np.std(stabilities) if stabilities else 0
            }

        return stability_analysis

    def _calculate_decision_stability(self, action_sequence: List[int]) -> float:
        """Calculate decision stability from action sequence."""
        if len(action_sequence) < 2:
            return 1.0

        # 連続する同じアクションの割合
        same_action_count = sum(1 for i in range(1, len(action_sequence))
                              if action_sequence[i] == action_sequence[i-1])

        stability = same_action_count / (len(action_sequence) - 1)
        return stability

    def _generate_diversity_recommendations(self, spec_div: float, perf_div: float,
                                          conf_div: float) -> List[str]:
        """Generate diversity improvement recommendations."""
        recommendations = []

        if spec_div < 0.8:
            recommendations.append("専門化の多様性を高めるため、新しい市場レジームの専門家を追加")

        if perf_div < 0.2:
            recommendations.append("メンバーのパフォーマンス差が小さいため、個別最適化を実施")

        if conf_div < 0.15:
            recommendations.append("信頼度のばらつきが小さいため、メンバーの専門性を再評価")

        if not recommendations:
            recommendations.append("アンサンブルの多様性は良好です")

        return recommendations

    def save_ensemble_report(self, report: Dict[str, Any], output_dir: str = "reports") -> str:
        """Save ensemble report to file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ensemble_analysis_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Ensemble report saved to {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Failed to save ensemble report: {e}")
            return ""