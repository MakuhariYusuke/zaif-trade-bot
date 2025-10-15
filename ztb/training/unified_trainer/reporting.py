#!/usr/bin/env python3
"""
Training reporting and logging utilities.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

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