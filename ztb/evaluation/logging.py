"""
Evaluation logging and persistence utilities.

This module provides functionality for logging evaluation results
and maintaining evaluation history with latest/best result tracking.
"""

import gzip
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np


@dataclass
class EvaluationRecord:
    """Data class for evaluation records"""

    timestamp: str
    feature_name: str
    status: str
    computation_time_ms: float
    nan_rate: float
    total_columns: int
    aligned_periods: int
    baseline_sharpe: float = 0.0
    cv_results: Dict[str, Any] = field(default_factory=dict)
    feature_correlations: Dict[str, float] = field(default_factory=dict)
    feature_performances: Dict[str, Any] = field(default_factory=dict)
    best_delta_sharpe: Optional[float] = None
    best_feature_name: str = ""
    avg_correlation: float = 0.0
    error: Optional[str] = None


class EvaluationLogger:
    """Logger for evaluation results with persistence"""

    def __init__(self, log_dir: str = "logs/evaluation"):
        """
        Initialize evaluation logger

        Args:
            log_dir: Directory to store evaluation logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.history_file = self.log_dir / "evaluation_history.jsonl.gz"
        self.latest_file = self.log_dir / "latest_results.json"
        self.best_file = self.log_dir / "best_results.json"

    def log_evaluation(self, evaluation_result: Dict[str, Any]) -> None:
        """
        Log a single evaluation result

        Args:
            evaluation_result: Evaluation result dictionary
        """
        # Create evaluation record
        record = EvaluationRecord(
            timestamp=datetime.now().isoformat(), **evaluation_result
        )

        # Append to history file
        self._append_to_history(record)

        # Update latest results
        self._update_latest_results(record)

        # Update best results if this is better
        self._update_best_results(record)

    def _append_to_history(self, record: EvaluationRecord) -> None:
        """Append record to compressed history file"""
        record_dict = asdict(record)

        with gzip.open(self.history_file, "at", encoding="utf-8") as f:
            f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

    def _update_latest_results(self, record: EvaluationRecord) -> None:
        """Update latest results file"""
        # Load existing latest results
        latest_results = self._load_json_file(self.latest_file)

        # Update with new record
        latest_results[record.feature_name] = asdict(record)

        # Save updated results
        self._save_json_file(self.latest_file, latest_results)

    def _update_best_results(self, record: EvaluationRecord) -> None:
        """Update best results if current record is better"""
        if record.status != "success" or record.best_delta_sharpe is None:
            return

        # Load existing best results
        best_results = self._load_json_file(self.best_file)

        current_best = best_results.get(record.feature_name, {})

        # Check if this is better (higher delta sharpe)
        current_best_score = current_best.get("best_delta_sharpe", 0)
        if record.best_delta_sharpe > current_best_score:
            best_results[record.feature_name] = asdict(record)
            self._save_json_file(self.best_file, best_results)

    def get_evaluation_history(
        self, feature_name: Optional[str] = None, days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation history

        Args:
            feature_name: Filter by feature name (optional)
            days: Get results from last N days (optional)

        Returns:
            List of evaluation records
        """
        if not self.history_file.exists():
            return []

        records = []
        cutoff_date = None

        if days:
            cutoff_date = datetime.now() - timedelta(days=days)

        try:
            with gzip.open(self.history_file, "rt", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())

                    # Filter by feature name
                    if feature_name and record["feature_name"] != feature_name:
                        continue

                    # Filter by date
                    if cutoff_date:
                        record_date = datetime.fromisoformat(record["timestamp"])
                        if record_date < cutoff_date:
                            continue

                    records.append(record)

        except Exception as e:
            print(f"Error reading evaluation history: {e}")

        return records

    def get_latest_results(self, feature_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get latest evaluation results

        Args:
            feature_name: Get results for specific feature (optional)

        Returns:
            Latest results dictionary
        """
        latest_results = self._load_json_file(self.latest_file)

        if feature_name:
            return cast(Dict[str, Any], latest_results.get(feature_name, {}))

        return latest_results

    def get_best_results(self, feature_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get best evaluation results

        Args:
            feature_name: Get results for specific feature (optional)

        Returns:
            Best results dictionary
        """
        best_results = self._load_json_file(self.best_file)

        if feature_name:
            return cast(Dict[str, Any], best_results.get(feature_name, {}))

        return best_results

    def get_evaluation_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get evaluation summary statistics

        Args:
            days: Number of days to include in summary

        Returns:
            Summary statistics
        """
        history = self.get_evaluation_history(days=days)

        if not history:
            return {"total_evaluations": 0}

        # Basic statistics
        total_evaluations = len(history)
        successful_evaluations = len([r for r in history if r["status"] == "success"])
        failed_evaluations = total_evaluations - successful_evaluations

        # Performance statistics
        successful_records = [r for r in history if r["status"] == "success"]
        avg_computation_time = (
            np.mean([r["computation_time_ms"] for r in successful_records])
            if successful_records
            else 0
        )
        avg_nan_rate = (
            np.mean([r["nan_rate"] for r in successful_records])
            if successful_records
            else 0
        )

        # Sharpe ratio statistics
        sharpe_ratios = [
            r["baseline_sharpe"]
            for r in successful_records
            if r.get("baseline_sharpe") is not None
        ]
        avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0

        # Feature performance
        delta_sharpes = [
            r["best_delta_sharpe"]
            for r in successful_records
            if r.get("best_delta_sharpe") is not None
        ]
        avg_delta_sharpe = np.mean(delta_sharpes) if delta_sharpes else 0

        # Feature counts by status
        status_counts: Dict[str, int] = {}
        for record in history:
            status = record["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "success_rate": (
                successful_evaluations / total_evaluations
                if total_evaluations > 0
                else 0
            ),
            "avg_computation_time_ms": avg_computation_time,
            "avg_nan_rate": avg_nan_rate,
            "avg_baseline_sharpe": avg_sharpe,
            "avg_best_delta_sharpe": avg_delta_sharpe,
            "status_counts": status_counts,
            "period_days": days,
        }

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file safely"""
        if not file_path.exists():
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return cast(Dict[str, Any], json.load(f))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return cast(Dict[str, Any], {})

    def _save_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save JSON file safely"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving {file_path}: {e}")

    def cleanup_old_logs(self, days_to_keep: int = 90) -> int:
        """
        Clean up old log entries

        Args:
            days_to_keep: Number of days of logs to keep

        Returns:
            Number of entries removed
        """
        if not self.history_file.exists():
            return 0

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        temp_file = self.history_file.with_suffix(".tmp")

        removed_count = 0
        kept_records = []

        try:
            with gzip.open(self.history_file, "rt", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    record_date = datetime.fromisoformat(record["timestamp"])

                    if record_date >= cutoff_date:
                        kept_records.append(record)
                    else:
                        removed_count += 1

            # Write kept records back
            with gzip.open(temp_file, "wt", encoding="utf-8") as f:
                for record in kept_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Replace original file
            temp_file.replace(self.history_file)

        except Exception as e:
            print(f"Error during log cleanup: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return 0

        return removed_count
