"""
Adaptive Threshold Manager for dynamic quality gates
"""
from typing import Dict, Any
import numpy as np
from pathlib import Path
from ztb.evaluation.status import CoverageValidator


class AdaptiveThresholdManager:
    """Manages adaptive thresholds based on historical market data"""

    def __init__(self, historical_data_path: str):
        self.historical_data_path = Path(historical_data_path)
        self.thresholds_cache: Dict[str, Any] = {}
        self.historical_successes: Dict[str, Any] = {}
        self._load_historical_data()

    def _load_historical_data(self) -> None:
        """Load historical evaluation results"""
        if not self.historical_data_path.exists():
            return

        # Load historical coverage data
        coverage_data = CoverageValidator.load_coverage_files(str(self.historical_data_path))

        # Extract successful feature metrics
        successful_features = {}
        for event in coverage_data.get("events", []):
            if event.get("type") == "feature_promoted" and event.get("to_status") == "verified":
                feature = event.get("feature")
                if "details" in event and "criterion_details" in event["details"]:
                    successful_features[feature] = event["details"]

        self.historical_successes = successful_features

    def get_adaptive_threshold(self, metric_name: str, percentile: float = 20.0) -> float:
        """Get adaptive threshold based on historical successful features"""
        if metric_name not in self.thresholds_cache:
            if not self.historical_successes:
                # Fallback to default values
                defaults = {
                    "sharpe_ratio": 0.3,
                    "max_drawdown": -0.2,
                    "win_rate": 0.5
                }
                self.thresholds_cache[metric_name] = defaults.get(metric_name, 0.0)
            else:
                # Calculate percentile from successful features
                values = []
                for feature_data in self.historical_successes.values():
                    for criterion in feature_data.get("criterion_details", []):
                        if criterion.get("name") == metric_name and criterion.get("passed"):
                            values.append(criterion.get("actual", 0))

                if values:
                    threshold = float(np.percentile(values, percentile))
                    self.thresholds_cache[metric_name] = threshold
                else:
                    self.thresholds_cache[metric_name] = 0.0

        return self.thresholds_cache[metric_name]

    def get_adaptive_gates(self) -> Dict[str, float]:
        """Get adaptive quality gates based on historical data"""
        gates = {
            'nan_rate_threshold': 0.8,
            'correlation_threshold': 0.05,
            'skew_threshold': 3.0,
            'kurtosis_threshold': 8.0
        }

        # Try to adapt based on historical harmful features
        # Analyze failed features to adjust thresholds
        failed_features = {}
        coverage_data = CoverageValidator.load_coverage_files(str(self.historical_data_path))
        for event in coverage_data.get("events", []):
            # Exclude synthetic data events from learning
            if event.get("details", {}).get("dataset") == "synthetic":
                continue
            if event.get("type") == "feature_tested" and event.get("to_status") == "discarded":
                feature = event.get("feature")
                if "details" in event and "quality_results" in event["details"]:
                    failed_features[feature] = event["details"]["quality_results"]

        # Require minimum events for adaptation
        if len(failed_features) < 10:
            return gates  # Return default if insufficient data

        if failed_features:
            # Adjust thresholds based on failed features
            nan_rates = [f.get('nan_rate', 0) for f in failed_features.values() if f.get('nan_rate', 0) > 0]
            if nan_rates:
                # Set threshold slightly above the maximum failed NaN rate
                gates['nan_rate_threshold'] = min(0.9, max(nan_rates) + 0.05)

            correlations = [f.get('correlation', 0) for f in failed_features.values() if f.get('correlation') is not None]
            if correlations:
                # Set threshold slightly above the maximum failed correlation
                gates['correlation_threshold'] = min(0.1, max(correlations) + 0.01)

            skews = [abs(f.get('skew', 0)) for f in failed_features.values() if f.get('skew') is not None]
            if skews:
                # Set threshold slightly above the maximum failed skew, with bounds
                adaptive_skew = max(skews) + 0.5
                gates['skew_threshold'] = max(1.5, min(5.0, adaptive_skew))

            kurtoses = [abs(f.get('kurtosis', 0)) for f in failed_features.values() if f.get('kurtosis') is not None]
            if kurtoses:
                # Set threshold slightly above the maximum failed kurtosis, with bounds
                adaptive_kurtosis = max(kurtoses) + 1.0
                gates['kurtosis_threshold'] = max(3.0, min(12.0, adaptive_kurtosis))

        return gates

    def update_thresholds(self, new_evaluation_results: Dict[str, Any]) -> None:
        """Update thresholds with new evaluation results"""
        # This would be called after each evaluation cycle
        # Implementation depends on how we want to update the adaptive thresholds
        pass