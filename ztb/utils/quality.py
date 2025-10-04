"""
Quality Gates for feature validation with adaptive thresholds
"""

from typing import Any, Dict, Optional

import pandas as pd

from ztb.utils.core.stats import calculate_kurtosis, calculate_skew, nan_ratio
from ztb.utils.thresholds import AdaptiveThresholdManager


class QualityGates:
    """Quality Gates for feature validation with adaptive thresholds"""

    def __init__(self, adaptive_manager: Optional[AdaptiveThresholdManager] = None):
        super().__init__()
        self.adaptive_manager = adaptive_manager
        self.base_gates = {
            "nan_rate_threshold": 0.8,
            "correlation_threshold": 0.05,
            "skew_threshold": 3.0,
            "kurtosis_threshold": 8.0,
        }

    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """Get adaptive thresholds from historical data"""
        if self.adaptive_manager:
            return self.adaptive_manager.get_adaptive_gates()
        return self.base_gates

    def evaluate(
        self,
        feature_data: pd.Series,
        price_data: pd.Series,
        mode: str = "normal",
        dataset: str = "synthetic",
    ) -> Dict[str, Any]:
        """Evaluate feature against quality gates"""
        gates = self.get_adaptive_thresholds()

        # Bootstrap mode: relax correlation threshold or skip correlation gate
        if mode == "bootstrap":
            gates = gates.copy()
            gates["correlation_threshold"] = 0.01  # Relax correlation threshold

        # CoinGecko dataset: adjust correlation threshold for real market data
        if dataset == "coingecko":
            gates = gates.copy()
            gates["correlation_threshold"] = 0.005  # More lenient for real market data

        results = {}

        # NaN rate
        nan_rate = nan_ratio(feature_data)
        results["nan_rate"] = nan_rate
        results["nan_rate_pass"] = nan_rate <= gates["nan_rate_threshold"]

        # Correlation with price (long-term: full period, short-term: last 1000 samples)
        valid_data = pd.concat([feature_data, price_data], axis=1).dropna()
        if len(valid_data) > 10:
            # Long-term correlation (full period)
            corr_long = abs(valid_data.iloc[:, 0].corr(valid_data.iloc[:, 1]))
            results["corr_long"] = corr_long

            # Short-term correlation (last 1000 samples)
            short_data = valid_data.tail(1000)
            if len(short_data) > 10:
                corr_short = abs(short_data.iloc[:, 0].corr(short_data.iloc[:, 1]))
                results["corr_short"] = corr_short
            else:
                results["corr_short"] = (
                    corr_long  # Fallback to long if insufficient data
                )

            # Correlation pass: either short or long meets threshold
            results["correlation_pass"] = (
                results["corr_short"] >= gates["correlation_threshold"]
                or results["corr_long"] >= gates["correlation_threshold"]
            )
            results["correlation"] = max(
                results["corr_short"], results["corr_long"]
            )  # For backward compatibility
        else:
            results["corr_long"] = None  # type: ignore[assignment]
            results["corr_short"] = None  # type: ignore[assignment]
            results["correlation"] = None  # type: ignore[assignment]
            results["correlation_pass"] = False

        # Skewness
        if len(feature_data.dropna()) > 10:
            skew_val = calculate_skew(feature_data.dropna())
            if pd.notna(skew_val):
                try:
                    skew = float(skew_val)
                    results["skew"] = skew
                    results["skew_pass"] = abs(skew) <= gates["skew_threshold"]
                except (ValueError, TypeError):
                    results["skew"] = None  # type: ignore
                    results["skew_pass"] = False
            else:
                results["skew"] = None  # type: ignore
                results["skew_pass"] = False
        else:
            results["skew"] = None  # type: ignore[assignment]
            results["skew_pass"] = False

        # Kurtosis
        if len(feature_data.dropna()) > 10:
            kurtosis_val = calculate_kurtosis(feature_data.dropna())
            if pd.notna(kurtosis_val):
                try:
                    kurtosis = float(kurtosis_val)
                    results["kurtosis"] = kurtosis
                    results["kurtosis_pass"] = (
                        abs(kurtosis) <= gates["kurtosis_threshold"]
                    )
                except (ValueError, TypeError):
                    results["kurtosis"] = None  # type: ignore
                    results["kurtosis_pass"] = False
            else:
                results["kurtosis"] = None  # type: ignore
                results["kurtosis_pass"] = False
        else:
            results["kurtosis"] = None  # type: ignore
            results["kurtosis_pass"] = False

        # Overall pass
        results["overall_pass"] = all(
            [
                results["nan_rate_pass"],
                results["correlation_pass"],
                results["skew_pass"],
                results["kurtosis_pass"],
            ]
        )

        return results
