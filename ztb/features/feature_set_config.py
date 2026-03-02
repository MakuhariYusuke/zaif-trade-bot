"""
Feature set Configuration for SAC v427

Configurable feature sets for easy swapping and customization.
"""

from pathlib import Path
from typing import cast

from ztb.io.json_io import read_json_object, write_json

class FeatureSetConfig:
    """
    Configuration class for managing different feature sets.

    Provides predefined feature sets and custom filtering capabilities.
    """

    # Predefined feature sets
    FEATURE_SETS = {
        "default": {
            "name": "Default Feature set",
            "description": "Standard feature set with basic filtering",
            "excluded_features": [
                "dividends",
                "stock splits",
            ],
            "include_regime_features": True,
            "include_correlation_features": True,
            "include_ensemble_features": True,
            "include_risk_features": True,
            "include_multi_timeframe_features": True,
        },
        "full": {
            "name": "Full Feature set",
            "description": "Complete SAC v427 feature set (150+ dimensions)",
            "excluded_features": [],
            "include_regime_features": True,
            "include_correlation_features": True,
            "include_ensemble_features": True,
            "include_risk_features": True,
            "include_multi_timeframe_features": True,
        },
        "minimal": {
            "name": "Minimal Feature set",
            "description": "Core features only (30-50 dimensions)",
            "excluded_features": [
                # Exclude complex derived features
                "regime_*",
                "correlation_*",
                "ensemble_*",
                "risk_*",
            ],
            "include_regime_features": False,
            "include_correlation_features": False,
            "include_ensemble_features": False,
            "include_risk_features": False,
            "include_multi_timeframe_features": False,
        },
        "no_harmful": {
            "name": "No Harmful Features",
            "description": "Full features with critical harmful features removed",
            "excluded_features": [
                "dividends",
                "stock splits",
            ],  # Critical harmful features
            "include_regime_features": True,
            "include_correlation_features": True,
            "include_ensemble_features": True,
            "include_risk_features": True,
            "include_multi_timeframe_features": True,
        },
        "high_quality": {
            "name": "High Quality Only",
            "description": "Only excellent quality features (correlation-filtered)",
            "excluded_features": [
                "dividends",
                "stock splits",  # Critical harmful
                # Add other harmful features identified by analysis
                "open",
                "high",
                "low",
                # "close",  # OHLCV base - needed for feature engineering
                "volume",  # Volume (high correlation)
                "returns",
                "log_returns",  # Simple returns (high correlation)
            ],
            "include_regime_features": True,
            "include_correlation_features": False,  # Avoid correlation issues
            "include_ensemble_features": True,
            "include_risk_features": True,
            "include_multi_timeframe_features": True,
        },
        "v435_risk_managed": {
            "name": "SAC v435 Risk Managed Features",
            "description": "Optimized feature set for SAC v435 with risk management",
            "excluded_features": [
                "dividends",
                "stock splits",
            ],
            "include_regime_features": True,
            "include_correlation_features": True,
            "include_ensemble_features": True,
            "include_risk_features": True,
            "include_multi_timeframe_features": True,
        },
        "v435_risk_managed_no_multi_timeframe": {
            "name": "SAC v435 Risk Managed Features (No MTF)",
            "description": "Optimized feature set for SAC v435 with risk management (No Multi-Timeframe)",
            "excluded_features": [
                "dividends",
                "stock splits",
            ],
            "include_regime_features": True,
            "include_correlation_features": True,
            "include_ensemble_features": True,
            "include_risk_features": True,
            "include_multi_timeframe_features": False,
        },
        "v451": {
            "name": "SAC v451 Regime Aware",
            "description": "Phase 7 feature set with temporal and regime awareness",
            "excluded_features": [],
            "include_regime_features": True,
            "include_correlation_features": True,
            "include_ensemble_features": True,
            "include_risk_features": True,
            "include_multi_timeframe_features": False,
        },
        "v454": {
            "name": "SAC v454 Signal Quality",
            "description": "Phase 8 feature set with noise filtering and signal quality improvements",
            "excluded_features": [],
            "include_regime_features": True,
            "include_correlation_features": True,
            "include_ensemble_features": True,
            "include_risk_features": True,
            "include_multi_timeframe_features": False,
        },
    }

    def __init__(self, config_path: str | None = None):
        self.config_path = Path(config_path) if config_path else None
        self.current_config = self.FEATURE_SETS[
            "no_harmful"
        ].copy()  # Default to safe set

        if self.config_path and self.config_path.exists():
            self.load_config()

    def load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            if self.config_path and self.config_path.exists():
                loaded_config = read_json_object(self.config_path)
                self.current_config.update(loaded_config)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")

    def save_config(self) -> None:
        """Save current configuration to JSON file."""
        if self.config_path:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.config_path, self.current_config, indent=2, ensure_ascii=False)

    def set_feature_set(self, set_name: str) -> None:
        """set predefined feature set."""
        if set_name not in self.FEATURE_SETS:
            available = list(self.FEATURE_SETS.keys())
            raise ValueError(
                f"Unknown feature set '{set_name}'. Available: {available}"
            )

        self.current_config = self.FEATURE_SETS[set_name].copy()

    def get_excluded_features(self) -> list[str]:
        """Get list of features to exclude."""
        return cast(list[str], self.current_config.get("excluded_features", []))

    def add_excluded_feature(self, feature: str) -> None:
        """Add a feature to the exclusion list."""
        excluded = cast(
            list[str], self.current_config.setdefault("excluded_features", [])
        )
        if feature not in excluded:
            excluded.append(feature)

    def remove_excluded_feature(self, feature: str) -> None:
        """Remove a feature from the exclusion list."""
        if "excluded_features" in self.current_config:
            excluded = cast(list[str], self.current_config["excluded_features"])
            self.current_config["excluded_features"] = [
                f for f in excluded if f != feature
            ]

    def get_feature_flags(self) -> dict[str, bool]:
        """Get feature category flags."""
        return {
            "include_regime_features": cast(
                bool, self.current_config.get("include_regime_features", True)
            ),
            "include_correlation_features": cast(
                bool, self.current_config.get("include_correlation_features", True)
            ),
            "include_ensemble_features": cast(
                bool, self.current_config.get("include_ensemble_features", True)
            ),
            "include_risk_features": cast(
                bool, self.current_config.get("include_risk_features", True)
            ),
            "include_multi_timeframe_features": cast(
                bool, self.current_config.get("include_multi_timeframe_features", True)
            ),
        }

    def list_available_sets(self) -> dict[str, dict]:
        """list all available predefined feature sets."""
        return self.FEATURE_SETS.copy()

    def get_current_config(self) -> dict:
        """Get current configuration."""
        return self.current_config.copy()

# Global configuration instance
_feature_config = None

def get_feature_config(config_path: str | None = None) -> FeatureSetConfig:
    """Get global feature configuration instance."""
    global _feature_config
    if _feature_config is None:
        _feature_config = FeatureSetConfig(config_path)
    return _feature_config
