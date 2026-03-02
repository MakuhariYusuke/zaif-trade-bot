"""
Multi-Timeframe Configuration Management

Configuration system for multi-timeframe feature engineering,
including timeframe-specific parameters and feature set management.
"""

from __future__ import annotations

from pathlib import Path

from ztb.features.timeframe import Timeframe
from ztb.io.common import PathLike
from ztb.io.json_io import read_json_object, write_json
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class MultiTimeframeConfig:
    """
    Configuration manager for multi-timeframe feature engineering.

    Handles configuration of multiple timeframes, their parameters,
    and feature generation settings.
    """

    def __init__(self, config_path: PathLike | None = None):
        """
        Initialize multi-timeframe configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = (
            str(config_path) if config_path is not None else self._get_default_config_path()
        )
        self.config = self._load_config()
        self._valid_timeframe_values = {tf.value for tf in Timeframe}

    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        return str(Path(__file__).parent / "config" / "multi_timeframe_config.json")

    @staticmethod
    def _as_object_map(value: object) -> dict[str, object]:
        """Safely coerce value into object map."""
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_string_list(value: object) -> list[str]:
        """Safely coerce value into list[str]."""
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    def _get_section(self, key: str) -> dict[str, object]:
        """Get top-level config section as object map."""
        return self._as_object_map(self.config.get(key, {}))

    def _load_config(self) -> dict[str, object]:
        """Load configuration from file."""
        try:
            config = read_json_object(self.config_path)
            logger.info(f"Loaded multi-timeframe config from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Config file not found at {self.config_path}, using defaults"
            )
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, object]:
        """Get default configuration."""
        return {
            "enabled_timeframes": ["1min", "5min", "15min", "1hour", "4hour", "1day"],
            "base_timeframe": "5min",
            "feature_sets": {
                "1min": {
                    "feature_set": "minimal",
                    "window_sizes": [3, 5, 7, 10],
                    "max_features": 50,
                },
                "5min": {
                    "feature_set": "full",
                    "window_sizes": [5, 10, 15, 20, 30],
                    "max_features": 100,
                },
                "15min": {
                    "feature_set": "full",
                    "window_sizes": [10, 20, 30, 50, 100],
                    "max_features": 150,
                },
                "1hour": {
                    "feature_set": "full",
                    "window_sizes": [20, 50, 100, 200],
                    "max_features": 200,
                },
                "4hour": {
                    "feature_set": "high_quality",
                    "window_sizes": [50, 100, 200, 400],
                    "max_features": 250,
                },
                "1day": {
                    "feature_set": "high_quality",
                    "window_sizes": [100, 200, 400, 800],
                    "max_features": 300,
                },
            },
            "integration": {
                "include_timeframe_indicators": True,
                "timeframe_alignment_method": "forward_fill",
                "max_timeframe_lag": "4hour",
                "feature_prefixing": True,
            },
            "quality_control": {
                "max_nan_rate": 0.10,
                "min_variance": 1e-8,
                "max_correlation": 0.95,
                "remove_outliers": True,
            },
            "performance": {
                "parallel_processing": True,
                "cache_features": True,
                "memory_limit_mb": 2048,
            },
        }

    def get_enabled_timeframes(self) -> list[Timeframe]:
        """Get list of enabled timeframes."""
        enabled = self._as_string_list(self.config.get("enabled_timeframes", []))
        return [
            Timeframe(tf) for tf in enabled if tf in self._valid_timeframe_values
        ]

    def get_base_timeframe(self) -> Timeframe:
        """Get base timeframe for the system."""
        base_tf_obj = self.config.get("base_timeframe", "5min")
        base_tf = (
            base_tf_obj
            if isinstance(base_tf_obj, str) and base_tf_obj in self._valid_timeframe_values
            else "5min"
        )
        return Timeframe(base_tf)

    def get_timeframe_config(self, timeframe: Timeframe) -> dict[str, object]:
        """Get configuration for specific timeframe."""
        feature_sets = self._get_section("feature_sets")
        return self._as_object_map(feature_sets.get(timeframe.value, {}))

    def get_integration_config(self) -> dict[str, object]:
        """Get integration configuration."""
        return self._get_section("integration")

    def get_quality_config(self) -> dict[str, object]:
        """Get quality control configuration."""
        return self._get_section("quality_control")

    def get_performance_config(self) -> dict[str, object]:
        """Get performance configuration."""
        return self._get_section("performance")

    def update_timeframe_config(
        self, timeframe: Timeframe, config_updates: dict[str, object]
    ) -> None:
        """Update configuration for specific timeframe."""
        feature_sets = self._get_section("feature_sets")
        timeframe_config = self._as_object_map(feature_sets.get(timeframe.value, {}))
        timeframe_config.update(config_updates)
        feature_sets[timeframe.value] = timeframe_config
        self.config["feature_sets"] = feature_sets
        logger.info(f"Updated config for {timeframe.value}: {config_updates}")

    def enable_timeframe(self, timeframe: Timeframe) -> None:
        """Enable a timeframe."""
        enabled = self._as_string_list(self.config.get("enabled_timeframes", []))
        if timeframe.value not in enabled:
            enabled.append(timeframe.value)
            self.config["enabled_timeframes"] = enabled
            logger.info(f"Enabled timeframe: {timeframe.value}")

    def disable_timeframe(self, timeframe: Timeframe) -> None:
        """Disable a timeframe."""
        enabled = self._as_string_list(self.config.get("enabled_timeframes", []))
        if timeframe.value in enabled:
            enabled.remove(timeframe.value)
            self.config["enabled_timeframes"] = enabled
            logger.info(f"Disabled timeframe: {timeframe.value}")

    def set_base_timeframe(self, timeframe: Timeframe) -> None:
        """set base timeframe."""
        self.config["base_timeframe"] = timeframe.value
        logger.info(f"set base timeframe to: {timeframe.value}")

    def save_config(self, path: PathLike | None = None) -> None:
        """Save configuration to file."""
        save_path = Path(path) if path is not None else Path(self.config_path)

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            write_json(save_path, self.config, indent=2, ensure_ascii=False)
            logger.info(f"Saved config to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues: list[str] = []

        # Check enabled timeframes
        enabled_timeframes = self._as_string_list(
            self.config.get("enabled_timeframes", [])
        )
        valid_timeframes = self._valid_timeframe_values

        for tf in enabled_timeframes:
            if tf not in valid_timeframes:
                issues.append(f"Invalid timeframe: {tf}")

        # Check base timeframe
        base_tf = self.config.get("base_timeframe")
        if base_tf and base_tf not in valid_timeframes:
            issues.append(f"Invalid base timeframe: {base_tf}")
        elif base_tf and base_tf not in enabled_timeframes:
            issues.append(f"Base timeframe {base_tf} not in enabled timeframes")

        # Check feature sets
        feature_sets = self._get_section("feature_sets")
        for tf in enabled_timeframes:
            if tf not in feature_sets:
                issues.append(f"Missing feature set config for: {tf}")

        return issues
