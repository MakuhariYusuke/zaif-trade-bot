"""
Multi-Timeframe Feature Engineering Interface

Main interface for the multi-timeframe feature engineering system.
Provides high-level API for generating features across multiple timeframes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ztb.features.feature_set_config import get_feature_config
from ztb.features.generators.multi_timeframe.config import MultiTimeframeConfig
from ztb.features.generators.multi_timeframe.data_pipeline import MultiTimeframeDataPipeline
from ztb.features.generators.multi_timeframe.engine import MultiTimeframeFeatureEngineer
from ztb.features.timeframe import Timeframe
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MultiTimeframeFeatureSystem:
    """
    Complete multi-timeframe feature engineering system.

    Integrates data loading, feature generation, and configuration management
    for comprehensive multi-timeframe analysis.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        data_base_path: Optional[str] = None,
        market_system: Optional[Any] = None,
    ):
        """
        Initialize multi-timeframe feature system.

        Args:
            config_path: Path to configuration file
            data_base_path: Base path for data files
            market_system: Market adaptive system instance
        """
        # Check if multi-timeframe features are enabled in global feature config
        feature_config = get_feature_config()
        feature_flags = feature_config.get_feature_flags()

        if not feature_flags.get("include_multi_timeframe_features", True):
            logger.info(
                "Multi-timeframe features disabled in global feature configuration"
            )
            self.config = None
            self.data_pipeline = None
            self.feature_engineer = None
            self._enabled = False
            return

        self.config = MultiTimeframeConfig(config_path)
        self.data_pipeline = MultiTimeframeDataPipeline(data_base_path)
        self.feature_engineer = MultiTimeframeFeatureEngineer(
            market_system=market_system,
            config_path=config_path,
            base_timeframe=self.config.get_base_timeframe(),
        )
        self._enabled = True

        logger.info("Initialized MultiTimeframeFeatureSystem")

    def process_multi_timeframe_data(
        self,
        data_or_files: Optional[Union[pd.DataFrame, Dict[Timeframe, str]]] = None,
        feature_set: Optional[str] = None,
        synchronize_data: bool = True,
        generate_missing_timeframes: bool = True,
    ) -> pd.DataFrame:
        """
        Process data across multiple timeframes and generate integrated features.

        Args:
            data_or_files: Either a single DataFrame (for base timeframe) or mapping of timeframes to data file paths
            feature_set: Feature set to use for generation
            synchronize_data: Whether to synchronize timestamps across timeframes
            generate_missing_timeframes: Whether to generate missing timeframes through resampling

        Returns:
            Integrated dataframe with multi-timeframe features
        """
        if not self._enabled:
            logger.warning("Multi-timeframe features are disabled")
            return pd.DataFrame()

        # Handle DataFrame input (single timeframe)
        if isinstance(data_or_files, pd.DataFrame):
            # Standardize the input DataFrame columns
            df = data_or_files.copy()

            # Standardize column names
            column_mapping = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Timestamp": "timestamp",
                "Date": "timestamp",
                "Time": "timestamp",
            }
            df = df.rename(columns=column_mapping)

            # Ensure required columns exist
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Process timestamps
            if "timestamp" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df["timestamp"] = df.index
                else:
                    raise ValueError("No timestamp column found")

            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Basic validation
            df = df.dropna(subset=["open", "high", "low", "close"], how="all")
            df = df[
                (df["open"] > 0)
                & (df["high"] > 0)
                & (df["low"] > 0)
                & (df["close"] > 0)
            ]

            # Use the provided DataFrame as base timeframe data
            base_timeframe = self.config.get_base_timeframe()
            raw_data = {base_timeframe: df}
            data_files = None
        else:
            # Handle file paths input
            data_files = data_or_files
            raw_data = {}

        # Get enabled timeframes from config
        enabled_timeframes = self.config.get_enabled_timeframes()

        # Load data for enabled timeframes (if not already provided)
        if not raw_data:
            logger.info(f"Loading data for {len(enabled_timeframes)} timeframes")
            raw_data = self.data_pipeline.load_timeframe_data(
                timeframes=enabled_timeframes,
                data_files=data_files,
            )

        if not raw_data:
            raise ValueError("No data loaded for any timeframe")

        # Generate missing timeframes if requested
        if generate_missing_timeframes:
            raw_data = self.data_pipeline.generate_missing_timeframes(
                raw_data, enabled_timeframes
            )

        # Synchronize data if requested
        if synchronize_data:
            base_timeframe = self.config.get_base_timeframe()
            raw_data = self.data_pipeline.synchronize_timeframes(
                raw_data, base_timeframe
            )

        # Generate integrated features
        logger.info("Generating multi-timeframe features")
        integrated_features = self.feature_engineer.generate_multi_timeframe_features(
            data_dict=raw_data,
            feature_set=feature_set,
        )

        # Clear raw data to free memory
        raw_data.clear()
        gc.collect()

        logger.info(
            f"Generated {len(integrated_features)} rows with {len(integrated_features.columns)} features"
        )
        return integrated_features

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get data quality report for loaded timeframes."""
        if not self._enabled:
            return {
                "status": "disabled",
                "message": "Multi-timeframe features are disabled",
            }

        return self.data_pipeline.get_data_quality_report(
            self.data_pipeline.timeframe_data
        )

    def get_feature_counts(self) -> Dict[str, int]:
        """Get feature counts for each timeframe."""
        if not self._enabled:
            return {}

        return self.feature_engineer.get_timeframe_feature_counts(
            self.data_pipeline.timeframe_data
        )

    def validate_system(self) -> List[str]:
        """Validate the entire multi-timeframe system."""
        if not self._enabled:
            return ["Multi-timeframe features are disabled in global configuration"]

        issues = []

        # Validate configuration
        config_issues = self.config.validate_config()
        issues.extend([f"Config: {issue}" for issue in config_issues])

        # Validate data
        if self.data_pipeline.timeframe_data:
            data_issues = self.feature_engineer.validate_timeframe_data(
                self.data_pipeline.timeframe_data
            )
            issues.extend([f"Data: {issue}" for issue in data_issues])

        return issues

    def update_configuration(
        self,
        timeframe: Timeframe,
        config_updates: Dict[str, Any],
    ) -> None:
        """Update configuration for specific timeframe."""
        if not self._enabled:
            logger.warning("Multi-timeframe features are disabled")
            return

        self.config.update_timeframe_config(timeframe, config_updates)

    def save_configuration(self, path: Optional[str] = None) -> None:
        """Save current configuration."""
        if not self._enabled:
            logger.warning("Multi-timeframe features are disabled")
            return

        self.config.save_config(path)

    def enable_timeframe(self, timeframe: Timeframe) -> None:
        """Enable a timeframe."""
        if not self._enabled:
            logger.warning("Multi-timeframe features are disabled")
            return

        self.config.enable_timeframe(timeframe)

    def disable_timeframe(self, timeframe: Timeframe) -> None:
        """Disable a timeframe."""
        if not self._enabled:
            logger.warning("Multi-timeframe features are disabled")
            return

        self.config.disable_timeframe(timeframe)

    def set_base_timeframe(self, timeframe: Timeframe) -> None:
        """Set base timeframe for the system."""
        if not self._enabled:
            logger.warning("Multi-timeframe features are disabled")
            return

        self.config.set_base_timeframe(timeframe)
        # Reinitialize feature engineer with new base timeframe
        self.feature_engineer.base_timeframe = timeframe

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status."""
        if not self._enabled:
            return {
                "status": "disabled",
                "message": "Multi-timeframe features are disabled",
            }

        return {
            "enabled_timeframes": [
                tf.value for tf in self.config.get_enabled_timeframes()
            ],
            "base_timeframe": self.config.get_base_timeframe().value,
            "loaded_data": list(self.data_pipeline.timeframe_data.keys()),
            "config_path": self.config.config_path,
            "data_base_path": str(self.data_pipeline.data_base_path),
        }


# Convenience functions for easy usage


def create_multi_timeframe_system(
    config_path: Optional[str] = None,
    data_base_path: Optional[str] = None,
) -> MultiTimeframeFeatureSystem:
    """
    Create a multi-timeframe feature system with default settings.

    Args:
        config_path: Path to configuration file
        data_base_path: Base path for data files

    Returns:
        Configured MultiTimeframeFeatureSystem instance
    """
    return MultiTimeframeFeatureSystem(
        config_path=config_path,
        data_base_path=data_base_path,
    )


def process_multi_timeframe_features(
    data_files: Optional[Dict[Timeframe, str]] = None,
    config_path: Optional[str] = None,
    feature_set: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to process multi-timeframe features in one call.

    Args:
        data_files: Mapping of timeframes to data file paths
        config_path: Path to configuration file
        feature_set: Feature set to use

    Returns:
        Integrated feature dataframe
    """
    system = create_multi_timeframe_system(config_path=config_path)
    return system.process_multi_timeframe_data(
        data_files=data_files,
        feature_set=feature_set,
    )


__all__ = [
    # Main classes
    "MultiTimeframeFeatureSystem",
    "MultiTimeframeFeatureEngineer",
    "MultiTimeframeConfig",
    "MultiTimeframeDataPipeline",
    # Convenience functions
    "create_multi_timeframe_system",
    "process_multi_timeframe_features",
]
