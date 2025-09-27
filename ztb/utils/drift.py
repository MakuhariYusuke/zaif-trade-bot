"""
Drift monitoring for data and model quality.
データとモデルのドリフト監視
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, cast
from scipy.stats import entropy, skew, kurtosis
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class DriftMonitor:
    """Monitor data drift and model performance drift"""

    def __init__(self, baseline_path: str = "data/baseline_stats.json",
                 drift_threshold: float = 0.1,
                 history_window_days: int = 30):
        """
        Initialize drift monitor

        Args:
            baseline_path: Path to baseline statistics JSON file
            drift_threshold: Threshold for drift detection (KL divergence)
            history_window_days: Days to keep drift history
        """
        self.baseline_path = Path(baseline_path)
        self.drift_threshold = drift_threshold
        self.history_window_days = history_window_days
        self.baseline_stats = self._load_baseline_stats()
        self.drift_history: List[Dict[str, Any]] = []

    def _load_baseline_stats(self) -> Dict[str, Any]:
        """Load baseline statistics from file"""
        if self.baseline_path.exists():
            with open(self.baseline_path, 'r') as f:
                return cast(Dict[str, Any], json.load(f))
        else:
            # Create default baseline if file doesn't exist
            return {
                "features": {},
                "pnl_distribution": {"mean": 0.0, "std": 1.0, "skew": 0.0, "kurtosis": 3.0},
                "created_at": datetime.now().isoformat()
            }

    def _save_baseline_stats(self) -> None:
        """Save current baseline statistics"""
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, 'w') as f:
            json.dump(self.baseline_stats, f, indent=2, default=str)

    def update_baseline(self, features_df: pd.DataFrame, pnl_series: Optional[pd.Series] = None) -> None:
        """Update baseline statistics with current data"""
        print("Updating baseline statistics...")

        # Update feature statistics
        self.baseline_stats["features"] = {}
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'int64']:
                series = features_df[col].dropna()
                if len(series) > 0:
                    self.baseline_stats["features"][col] = {
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "skew": float(skew(series)),
                        "kurtosis": float(kurtosis(series)),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "histogram_bins": self._compute_histogram(series)
                    }

        # Update PnL distribution if provided
        if pnl_series is not None and len(pnl_series) > 0:
            self.baseline_stats["pnl_distribution"] = {
                "mean": float(pnl_series.mean()),
                "std": float(pnl_series.std()),
                "skew": float(skew(pnl_series)),
                "kurtosis": float(kurtosis(pnl_series))
            }

        self.baseline_stats["updated_at"] = datetime.now().isoformat()
        self._save_baseline_stats()
        print("Baseline statistics updated successfully")

    def _compute_histogram(self, series: pd.Series, bins: int = 20) -> List[float]:
        """Compute histogram bin edges for distribution comparison"""
        hist, bin_edges = np.histogram(series.dropna(), bins=bins, density=True)
        return cast(List[float], bin_edges.tolist())

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """Compute KL divergence between two distributions"""
        # Add small epsilon to avoid division by zero
        p = p + epsilon
        q = q + epsilon

        # Normalize to probability distributions
        p = p / p.sum()
        q = q / q.sum()

        return float(entropy(p, q))

    def detect_data_drift(self, current_features_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift in feature distributions"""
        drift_scores = {}
        max_drift = 0.0
        drifted_features = []

        for feature_name, baseline_stats in self.baseline_stats.get("features", {}).items():
            if feature_name not in current_features_df.columns:
                continue

            current_series = current_features_df[feature_name].dropna()
            if len(current_series) == 0:
                continue

            # Compute current statistics
            current_stats = {
                "mean": float(current_series.mean()),
                "std": float(current_series.std()),
                "skew": float(skew(current_series)),
                "kurtosis": float(kurtosis(current_series))
            }

            # Compare distributions using histogram
            baseline_hist_bins = baseline_stats.get("histogram_bins", [])
            if baseline_hist_bins:
                # Create histograms with same bins
                current_hist, _ = np.histogram(current_series, bins=baseline_hist_bins, density=True)
                baseline_hist = np.ones(len(current_hist)) / len(current_hist)  # Uniform baseline

                # Compute KL divergence
                kl_score = self._kl_divergence(baseline_hist, current_hist)
                drift_scores[feature_name] = kl_score

                if kl_score > self.drift_threshold:
                    drifted_features.append(feature_name)
                    max_drift = max(max_drift, kl_score)

        return {
            "drift_scores": drift_scores,
            "max_drift": max_drift,
            "drifted_features": drifted_features,
            "is_drifted": max_drift > self.drift_threshold,
            "drift_threshold": self.drift_threshold
        }

    def detect_model_drift(self, current_pnl: pd.Series) -> Dict[str, Any]:
        """Detect model performance drift using PnL distribution"""
        if not current_pnl.dropna().any():
            return {
                "is_drifted": False,
                "drift_score": 0.0,
                "message": "No PnL data available"
            }

        baseline_pnl = self.baseline_stats.get("pnl_distribution", {})
        if not baseline_pnl:
            return {
                "is_drifted": False,
                "drift_score": 0.0,
                "message": "No baseline PnL distribution available"
            }

        # Compare statistical properties
        current_stats = {
            "mean": float(current_pnl.mean()),
            "std": float(current_pnl.std()),
            "skew": float(skew(current_pnl.dropna())),
            "kurtosis": float(kurtosis(current_pnl.dropna()))
        }

        # Compute drift score as weighted difference in statistics
        weights = {"mean": 0.4, "std": 0.3, "skew": 0.15, "kurtosis": 0.15}
        drift_score = 0.0

        for stat_name, weight in weights.items():
            if stat_name in baseline_pnl and stat_name in current_stats:
                baseline_val = baseline_pnl[stat_name]
                current_val = current_stats[stat_name]

                # Normalize by baseline std if available
                if baseline_pnl.get("std", 1.0) > 0:
                    normalized_diff = abs(current_val - baseline_val) / baseline_pnl["std"]
                else:
                    normalized_diff = abs(current_val - baseline_val)

                drift_score += weight * normalized_diff

        return {
            "is_drifted": drift_score > self.drift_threshold,
            "drift_score": drift_score,
            "current_stats": current_stats,
            "baseline_stats": baseline_pnl,
            "drift_threshold": self.drift_threshold
        }

    def check_quality_gates(self, features_df: pd.DataFrame,
                          pnl_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Run all quality checks"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_drift": self.detect_data_drift(features_df),
            "model_drift": self.detect_model_drift(pnl_series) if pnl_series is not None else None,
            "quality_gates": {}
        }

        # Quality gates
        results["quality_gates"] = {
            "data_completeness": self._check_data_completeness(features_df),
            "feature_validity": self._check_feature_validity(features_df),
            "pnl_reasonableness": self._check_pnl_reasonableness(pnl_series) if pnl_series is not None else True
        }

        # Overall quality gate pass rate
        gate_results = list(cast(Dict[str, bool], results["quality_gates"]).values())
        results["quality_gate_pass_rate"] = cast(Any, sum(gate_results) / len(gate_results) if gate_results else 0.0)

        # Store in history
        self.drift_history.append(results)

        # Clean old history
        cutoff_date = datetime.now() - timedelta(days=self.history_window_days)
        self.drift_history = [
            entry for entry in self.drift_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]

        return results

    def _check_data_completeness(self, df: pd.DataFrame, threshold: float = 0.8) -> bool:
        """Check if data completeness meets threshold"""
        completeness_scores = []
        for col in df.columns:
            non_null_ratio = df[col].notna().mean()
            completeness_scores.append(non_null_ratio)

        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
        return bool(avg_completeness >= threshold)

    def _check_feature_validity(self, df: pd.DataFrame) -> bool:  # type: ignore[no-any-return]
        """Check if features contain valid values (no inf, nan in critical places)"""
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                series = df[col]
                # Check for inf or -inf values
                if np.any(np.isinf(series)):
                    return False
                # Check for all NaN columns
                if series.isna().all():
                    return False
        return True

    def _check_pnl_reasonableness(self, pnl: pd.Series, max_reasonable_pnl: float = 1.0) -> bool:
        """Check if PnL values are within reasonable bounds"""
        if pnl is None or len(pnl) == 0:
            return True

        # Check for extreme values
        max_pnl = pnl.abs().max()
        return max_pnl <= max_reasonable_pnl

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of recent drift history"""
        if not self.drift_history:
            return {"message": "No drift history available"}

        recent_entries = self.drift_history[-10:]  # Last 10 entries

        summary = {
            "total_checks": len(recent_entries),
            "data_drift_incidents": sum(1 for entry in recent_entries if entry["data_drift"]["is_drifted"]),
            "model_drift_incidents": sum(1 for entry in recent_entries
                                       if entry.get("model_drift") and entry["model_drift"]["is_drifted"]),
            "avg_quality_gate_pass_rate": np.mean([entry["quality_gate_pass_rate"] for entry in recent_entries]),
            "latest_check": recent_entries[-1] if recent_entries else None
        }

        return summary

    def export_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        summary = self.get_drift_summary()

        metrics = []

        if "latest_check" in summary and summary["latest_check"]:
            latest = summary["latest_check"]

            # Data drift score
            if "max_drift" in latest["data_drift"]:
                metrics.append(f'# HELP data_drift_score Maximum KL divergence score for data drift detection')
                metrics.append(f'# TYPE data_drift_score gauge')
                metrics.append(f'data_drift_score {latest["data_drift"]["max_drift"]}')

            # Model drift score
            if latest.get("model_drift") and "drift_score" in latest["model_drift"]:
                metrics.append(f'# HELP model_drift_score Drift score for model performance')
                metrics.append(f'# TYPE model_drift_score gauge')
                metrics.append(f'model_drift_score {latest["model_drift"]["drift_score"]}')

            # Quality gate pass rate
            metrics.append(f'# HELP quality_gate_pass_rate Overall quality gate pass rate (0.0-1.0)')
            metrics.append(f'# TYPE quality_gate_pass_rate gauge')
            metrics.append(f'quality_gate_pass_rate {latest["quality_gate_pass_rate"]}')

        return "\n".join(metrics)