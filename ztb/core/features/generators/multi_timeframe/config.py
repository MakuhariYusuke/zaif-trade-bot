"""
Multi-Timeframe Configuration Management

Configuration system for multi-timeframe feature engineering,
including timeframe-specific parameters and feature set management.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ztb.features.timeframe import Timeframe
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MultiTimeframeConfig:
    """
    Configuration manager for multi-timeframe feature engineering.

    Handles configuration of multiple timeframes, their parameters,
    and feature generation settings.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize multi-timeframe configuration.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        return str(Path(__file__).parent / "config" / "multi_timeframe_config.json")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
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

    def _get_default_config(self) -> Dict[str, Any]:
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

    def get_enabled_timeframes(self) -> List[Timeframe]:
        """Get list of enabled timeframes."""
        enabled = self.config.get("enabled_timeframes", [])
        return [Timeframe(tf) for tf in enabled if tf in [tf.value for tf in Timeframe]]

    def get_base_timeframe(self) -> Timeframe:
        """Get base timeframe for the system."""
        base_tf = self.config.get("base_timeframe", "5min")
        return Timeframe(base_tf)

    def get_timeframe_config(self, timeframe: Timeframe) -> Dict[str, Any]:
        """Get configuration for specific timeframe."""
        feature_sets = self.config.get("feature_sets", {})
        return feature_sets.get(timeframe.value, {})

    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration configuration."""
        return self.config.get("integration", {})

    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality control configuration."""
        return self.config.get("quality_control", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.config.get("performance", {})

    def update_timeframe_config(
        self, timeframe: Timeframe, config_updates: Dict[str, Any]
    ) -> None:
        """Update configuration for specific timeframe."""
        if "feature_sets" not in self.config:
            self.config["feature_sets"] = {}

        if timeframe.value not in self.config["feature_sets"]:
            self.config["feature_sets"][timeframe.value] = {}

        self.config["feature_sets"][timeframe.value].update(config_updates)
        logger.info(f"Updated config for {timeframe.value}: {config_updates}")

    def enable_timeframe(self, timeframe: Timeframe) -> None:
        """Enable a timeframe."""
        enabled = self.config.setdefault("enabled_timeframes", [])
        if timeframe.value not in enabled:
            enabled.append(timeframe.value)
            logger.info(f"Enabled timeframe: {timeframe.value}")

    def disable_timeframe(self, timeframe: Timeframe) -> None:
        """Disable a timeframe."""
        enabled = self.config.get("enabled_timeframes", [])
        if timeframe.value in enabled:
            enabled.remove(timeframe.value)
            logger.info(f"Disabled timeframe: {timeframe.value}")

    def set_base_timeframe(self, timeframe: Timeframe) -> None:
        """Set base timeframe."""
        self.config["base_timeframe"] = timeframe.value
        logger.info(f"Set base timeframe to: {timeframe.value}")

    def save_config(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved config to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check enabled timeframes
        enabled_timeframes = self.config.get("enabled_timeframes", [])
        valid_timeframes = [tf.value for tf in Timeframe]

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
        feature_sets = self.config.get("feature_sets", {})
        for tf in enabled_timeframes:
            if tf not in feature_sets:
                issues.append(f"Missing feature set config for: {tf}")

        return issues
