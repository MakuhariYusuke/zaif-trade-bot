"""
Data Processing Utilities

Provides utilities for processing and validating data used in
feature weight adjustment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Utility class for processing weight adjustment data.

    Provides methods for data validation, normalization, and feature engineering.
    """

    def __init__(self):
        """Initialize DataProcessor."""
        self.scaler = StandardScaler()

    def validate_performance_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate performance data structure and values.

        Args:
            data: Performance data to validate

        Returns:
            True if data is valid, False otherwise
        """
        required_keys = ['total_return', 'win_rate', 'trade_count']

        if not all(key in data for key in required_keys):
            logger.warning(f"Missing required keys in performance data: {required_keys}")
            return False

        # Check value ranges
        if not (0 <= data.get('win_rate', 0) <= 1):
            logger.warning("Win rate must be between 0 and 1")
            return False

        if data.get('trade_count', 0) < 0:
            logger.warning("Trade count cannot be negative")
            return False

        return True

    def validate_feature_weights(self, weights: Dict[str, float]) -> bool:
        """
        Validate feature weights.

        Args:
            weights: Feature weights to validate

        Returns:
            True if weights are valid, False otherwise
        """
        if not weights:
            logger.warning("Weights dictionary is empty")
            return False

        # Check that all weights are numeric and finite
        for feature, weight in weights.items():
            if not isinstance(weight, (int, float)) or not np.isfinite(weight):
                logger.warning(f"Invalid weight for feature {feature}: {weight}")
                return False

        # Check weight range (allow negative weights for some strategies)
        weight_values = list(weights.values())
        if not all(-1.0 <= w <= 1.0 for w in weight_values):
            logger.warning("Weights must be between -1.0 and 1.0")
            return False

        return True

    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0.

        Args:
            weights: Feature weights to normalize

        Returns:
            Normalized weights
        """
        if not weights:
            return {}

        weight_values = np.array(list(weights.values()))
        weight_sum = np.sum(np.abs(weight_values))  # Use absolute values for normalization

        if weight_sum == 0:
            # If all weights are zero, distribute equally
            normalized_values = np.ones(len(weights)) / len(weights)
        else:
            # Normalize by absolute sum
            normalized_values = weight_values / weight_sum

        return dict(zip(weights.keys(), normalized_values))

    def calculate_feature_correlations(
        self,
        feature_data: pd.DataFrame,
        target_variable: str
    ) -> Dict[str, float]:
        """
        Calculate correlations between features and target variable.

        Args:
            feature_data: DataFrame with features and target
            target_variable: Name of target variable column

        Returns:
            Dictionary of feature correlations
        """
        if target_variable not in feature_data.columns:
            logger.warning(f"Target variable {target_variable} not found in data")
            return {}

        correlations = {}
        target_data = feature_data[target_variable]

        for column in feature_data.columns:
            if column != target_variable:
                try:
                    if feature_data[column].dtype in ['int64', 'float64']:
                        corr = target_data.corr(feature_data[column])
                        if np.isfinite(corr):
                            correlations[column] = float(corr)
                except Exception as e:
                    logger.warning(f"Failed to calculate correlation for {column}: {e}")

        return correlations

    def detect_outliers(self, data: pd.Series, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in data.

        Args:
            data: Data series to check for outliers
            method: Outlier detection method ('iqr' or 'zscore')

        Returns:
            Boolean series indicating outliers
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            return z_scores > 3
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def smooth_time_series(self, data: pd.Series, window: int = 5) -> pd.Series:
        """
        Apply smoothing to time series data.

        Args:
            data: Time series data to smooth
            window: Smoothing window size

        Returns:
            Smoothed data
        """
        return data.rolling(window=window, center=True).mean()

    def calculate_rolling_statistics(
        self,
        data: pd.Series,
        window: int = 20
    ) -> Dict[str, pd.Series]:
        """
        Calculate rolling statistics for time series data.

        Args:
            data: Time series data
            window: Rolling window size

        Returns:
            Dictionary of rolling statistics
        """
        return {
            'mean': data.rolling(window=window).mean(),
            'std': data.rolling(window=window).std(),
            'min': data.rolling(window=window).min(),
            'max': data.rolling(window=window).max(),
            'trend': data.rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        }

    def prepare_feature_matrix(
        self,
        raw_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        scale_features: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare feature matrix for analysis.

        Args:
            raw_data: Raw data DataFrame
            feature_columns: Columns to use as features
            target_column: Target column (optional)
            scale_features: Whether to scale features

        Returns:
            Tuple of (feature_matrix, target_series)
        """
        # Select features
        feature_data = raw_data[feature_columns].copy()

        # Handle missing values
        feature_data = feature_data.fillna(feature_data.mean())

        # Scale features if requested
        if scale_features:
            feature_data = pd.DataFrame(
                self.scaler.fit_transform(feature_data),
                columns=feature_columns,
                index=feature_data.index
            )

        target_data = None
        if target_column and target_column in raw_data.columns:
            target_data = raw_data[target_column].copy()

        return feature_data, target_data

    def calculate_performance_metrics(
        self,
        predictions: pd.Series,
        actuals: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for predictions.

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            Dictionary of performance metrics
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        # R-squared
        ss_res = np.sum((predictions - actuals) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r_squared': float(r_squared),
        }