"""
Multi-Timeframe Feature Engineering System

Comprehensive feature engineering system that supports multiple timeframes:
- 1 minute (1m)
- 5 minutes (5m)
- 15 minutes (15m)
- 1 hour (1h)
- 4 hours (4h)
- 1 day (1d)

This system integrates features from multiple timeframes to provide richer
market context for reinforcement learning agents.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ztb.features.models.sac.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.features.timeframe import Timeframe, get_timeframe_params
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class MultiTimeframeFeatureEngineer:
    """
    Multi-timeframe feature engineering system.

    Generates comprehensive features across multiple timeframes and integrates
    them for enhanced market analysis and reinforcement learning.
    """

    # Supported timeframes
    SUPPORTED_TIMEFRAMES = [
        Timeframe.M1,
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.H1,
        Timeframe.H4,
        Timeframe.D1,
    ]

    def __init__(
        self,
        market_system: Any | None = None,
        config_path: str | None = None,
        base_timeframe: Timeframe = Timeframe.M5,  # Primary timeframe for training
    ):
        """
        Initialize multi-timeframe feature engineer.

        Args:
            market_system: Market adaptive system instance
            config_path: Path to feature configuration
            base_timeframe: Primary timeframe for the system
        """
        self.base_timeframe = base_timeframe
        self.market_system = market_system

        # Initialize feature engineers for each timeframe
        self.timeframe_engineers: dict[Timeframe, SACv427FeatureEngineer] = {}

        for timeframe in self.SUPPORTED_TIMEFRAMES:
            engineer = SACv427FeatureEngineer(
                market_system=market_system,
                config_path=config_path,
            )
            self.timeframe_engineers[timeframe] = engineer

        # Timeframe-specific parameters
        self.timeframe_params = {
            timeframe: get_timeframe_params(timeframe)
            for timeframe in self.SUPPORTED_TIMEFRAMES
        }

        logger.info(
            f"Initialized MultiTimeframeFeatureEngineer with base timeframe: {base_timeframe.value}"
        )

    def generate_multi_timeframe_features(
        self,
        data_dict: dict[Timeframe, pd.DataFrame],
        feature_set: str | None = None,
        include_timeframe_indicators: bool = True,
    ) -> pd.DataFrame:
        """
        Generate features across multiple timeframes and integrate them.

        Args:
            data_dict: Dictionary mapping timeframes to their dataframes
            feature_set: Feature set to use ('full', 'minimal', etc.)
            include_timeframe_indicators: Whether to include timeframe identification features

        Returns:
            Integrated dataframe with multi-timeframe features
        """
        if not data_dict:
            raise ValueError("No data provided for any timeframe")

        # Validate that we have data for at least the base timeframe
        if self.base_timeframe not in data_dict:
            logger.warning(
                f"Base timeframe {self.base_timeframe.value} not found in data, using first available"
            )
            self.base_timeframe = list(data_dict.keys())[0]

        # Generate features for each timeframe
        timeframe_features = {}
        base_df = data_dict[self.base_timeframe]

        for timeframe, df in data_dict.items():
            logger.info(f"Generating features for timeframe: {timeframe.value}")

            try:
                # Get timeframe-specific window sizes
                params = self.timeframe_params[timeframe]
                window_sizes = [
                    params["short_period"],
                    params["medium_period"],
                    params["long_period"],
                ]

                # Generate features for this timeframe
                features_df = self.timeframe_engineers[
                    timeframe
                ].generate_v427_features(
                    df=df,
                    window_sizes=window_sizes,
                    feature_set=feature_set,
                )

                # Add timeframe prefix to column names (except base columns)
                base_columns = ["open", "high", "low", "close", "volume", "timestamp"]
                prefixed_features = {}

                for col in features_df.columns:
                    if col in base_columns:
                        prefixed_features[col] = features_df[col]
                    else:
                        prefixed_features[f"{timeframe.value}_{col}"] = features_df[col]

                timeframe_features[timeframe] = pd.DataFrame(prefixed_features)

            except Exception as e:
                logger.error(f"Failed to generate features for {timeframe.value}: {e}")
                continue

        # Integrate features from all timeframes
        integrated_df = self._integrate_timeframe_features(
            timeframe_features,
            base_df,
            include_timeframe_indicators,
        )

        logger.info(
            f"Generated integrated features with {len(integrated_df.columns)} total columns"
        )
        return integrated_df

    def _integrate_timeframe_features(
        self,
        timeframe_features: dict[Timeframe, pd.DataFrame],
        base_df: pd.DataFrame,
        include_timeframe_indicators: bool,
    ) -> pd.DataFrame:
        """
        Integrate features from multiple timeframes into a single dataframe.

        Args:
            timeframe_features: Features for each timeframe
            base_df: Base timeframe dataframe
            include_timeframe_indicators: Whether to add timeframe indicators

        Returns:
            Integrated dataframe
        """
        # Start with base timeframe features
        base_timeframe = self.base_timeframe
        integrated_df = timeframe_features[base_timeframe].copy()

        # Add features from other timeframes
        # Use a single concat operation to avoid repeated DataFrame.insert/assignment
        # which fragments the DataFrame and can cause ordering/consistency issues.
        dfs_to_concat = [integrated_df]

        for timeframe, features_df in timeframe_features.items():
            if timeframe == base_timeframe:
                continue

            # Align timestamps (this is a simplified approach - in practice,
            # you'd want proper time alignment). Here we only add columns that
            # don't already exist in the integrated frame to match previous behavior.
            try:
                cols_to_add = [
                    c for c in features_df.columns if c not in integrated_df.columns
                ]
                if not cols_to_add:
                    continue

                dfs_to_concat.append(features_df[cols_to_add])
            except Exception as e:
                logger.warning(
                    f"Failed to integrate features from {timeframe.value}: {e}"
                )

        # Concatenate once to produce a defragmented, consistent DataFrame
        if len(dfs_to_concat) > 1:
            try:
                integrated_df = pd.concat(dfs_to_concat, axis=1)
                # Make an explicit copy to defragment memory if pandas warns
                integrated_df = integrated_df.copy()
            except Exception as e:
                logger.warning(f"Failed to concat timeframe feature frames: {e}")

        # Add timeframe identification features if requested
        if include_timeframe_indicators:
            integrated_df = self._add_timeframe_indicators(integrated_df)

        return integrated_df

    def _add_timeframe_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add timeframe identification and relationship features.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with timeframe indicators
        """
        # Add timeframe metadata columns
        df = df.copy()

        # Timeframe hierarchy indicators (higher timeframes are more stable)
        timeframe_hierarchy = {
            Timeframe.M1: 1,
            Timeframe.M5: 2,
            Timeframe.M15: 3,
            Timeframe.H1: 4,
            Timeframe.H4: 5,
            Timeframe.D1: 6,
        }

        df["timeframe_hierarchy"] = timeframe_hierarchy.get(self.base_timeframe, 1)

        # Add timeframe-specific volatility ratios
        # (This would be enhanced with actual volatility calculations)
        df["timeframe_volatility_ratio"] = 1.0  # Placeholder

        return df

    def get_timeframe_feature_counts(
        self, data_dict: dict[Timeframe, pd.DataFrame]
    ) -> dict[str, int]:
        """
        Get feature counts for each timeframe.

        Args:
            data_dict: Dictionary of timeframe data

        Returns:
            Dictionary with feature counts per timeframe
        """
        counts = {}

        for timeframe, df in data_dict.items():
            try:
                features_df = self.timeframe_engineers[
                    timeframe
                ].generate_v427_features(df)
                counts[timeframe.value] = len(features_df.columns)
            except Exception as e:
                logger.error(f"Failed to count features for {timeframe.value}: {e}")
                counts[timeframe.value] = 0

        return counts

    def validate_timeframe_data(
        self, data_dict: dict[Timeframe, pd.DataFrame]
    ) -> list[str]:
        """
        Validate timeframe data quality and consistency.

        Args:
            data_dict: Dictionary of timeframe data

        Returns:
            list of validation warnings/errors
        """
        warnings = []

        required_columns = ["open", "high", "low", "close", "volume"]

        for timeframe, df in data_dict.items():
            # Check required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                warnings.append(
                    f"{timeframe.value}: Missing required columns: {missing_cols}"
                )

            # Check data quality
            if df.empty:
                warnings.append(f"{timeframe.value}: Dataframe is empty")
                continue

            # Check for NaN values
            nan_rate = df.isnull().mean().mean()
            if nan_rate > 0.1:
                warnings.append(".1f")

            # Check data length
            if len(df) < 100:
                warnings.append(
                    f"{timeframe.value}: Insufficient data points ({len(df)})"
                )

        return warnings
