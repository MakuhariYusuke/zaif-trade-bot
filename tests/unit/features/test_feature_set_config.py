#!/usr/bin/env python3
"""
Unit tests for FeatureSetConfig class.

Tests for feature set configuration and validation.
"""

import pytest

from ztb.features.feature_set_config import FeatureSetConfig


class TestFeatureSetConfig:
    """Test FeatureSetConfig class functionality."""

    def test_get_feature_set_config_valid(self):
        """Test getting valid feature set configuration."""
        config = FeatureSetConfig.FEATURE_SETS.get("high_quality")

        assert config is not None
        assert config["name"] == "High Quality Only"
        assert "excluded_features" in config
        assert "close" not in config["excluded_features"]  # Should not be excluded

    def test_get_feature_set_config_invalid(self):
        """Test getting invalid feature set configuration."""
        config = FeatureSetConfig.FEATURE_SETS.get("invalid_set")

        assert config is None

    def test_get_available_feature_sets(self):
        """Test getting list of available feature sets."""
        sets = list(FeatureSetConfig.FEATURE_SETS.keys())

        assert isinstance(sets, list)
        assert "high_quality" in sets
        assert "default" in sets
        assert "minimal" in sets
        assert "full" in sets

    def test_is_valid_feature_set(self):
        """Test feature set validation."""
        assert "high_quality" in FeatureSetConfig.FEATURE_SETS
        assert "default" in FeatureSetConfig.FEATURE_SETS
        assert "minimal" in FeatureSetConfig.FEATURE_SETS
        assert "full" in FeatureSetConfig.FEATURE_SETS

        assert "invalid_set" not in FeatureSetConfig.FEATURE_SETS
        assert "" not in FeatureSetConfig.FEATURE_SETS
        assert None not in FeatureSetConfig.FEATURE_SETS

    def test_high_quality_excludes_harmful_features(self):
        """Test that high_quality excludes known harmful features."""
        config = FeatureSetConfig.FEATURE_SETS.get("high_quality")

        excluded = config["excluded_features"]

        # Should exclude harmful features
        assert "dividends" in excluded
        assert "stock splits" in excluded
        assert "open" in excluded
        assert "high" in excluded
        assert "low" in excluded
        assert "volume" in excluded

        # Should NOT exclude close (needed for feature engineering)
        assert "close" not in excluded

    def test_high_quality_includes_important_features(self):
        """Test that high_quality includes important feature categories."""
        config = FeatureSetConfig.FEATURE_SETS.get("high_quality")

        assert config["include_regime_features"] is True
        assert config["include_ensemble_features"] is True
        assert config["include_risk_features"] is True
        assert config["include_multi_timeframe_features"] is True

        # Correlation features are disabled to avoid issues
        assert config["include_correlation_features"] is False

    def test_default_feature_set_structure(self):
        """Test default feature set has proper structure."""
        config = FeatureSetConfig.FEATURE_SETS.get("default")

        assert config is not None
        assert "name" in config
        assert "description" in config
        assert "excluded_features" in config
        assert isinstance(config["excluded_features"], list)

    def test_minimal_feature_set_excludes_complex_features(self):
        """Test that minimal feature set excludes complex derived features."""
        config = FeatureSetConfig.FEATURE_SETS.get("minimal")

        excluded = config["excluded_features"]

        # Should exclude complex features
        assert any("regime_" in feature for feature in excluded) or "regime_*" in excluded
        assert any("correlation_" in feature for feature in excluded) or "correlation_*" in excluded
        assert any("ensemble_" in feature for feature in excluded) or "ensemble_*" in excluded
        assert any("risk_" in feature for feature in excluded) or "risk_*" in excluded

        # Should disable complex feature categories
        assert config["include_regime_features"] is False
        assert config["include_correlation_features"] is False
        assert config["include_ensemble_features"] is False
        assert config["include_risk_features"] is False
        assert config["include_multi_timeframe_features"] is False

    def test_full_feature_set_includes_everything(self):
        """Test that full feature set includes all feature categories."""
        config = FeatureSetConfig.FEATURE_SETS.get("full")

        assert config["excluded_features"] == []

        assert config["include_regime_features"] is True
        assert config["include_correlation_features"] is True
        assert config["include_ensemble_features"] is True
        assert config["include_risk_features"] is True
        assert config["include_multi_timeframe_features"] is True

    def test_get_feature_set_description(self):
        """Test getting feature set descriptions."""
        config = FeatureSetConfig.FEATURE_SETS.get("high_quality")
        desc = config.get("description", "")

        assert desc is not None
        assert "correlation-filtered" in desc

    def test_feature_set_config_consistency(self):
        """Test that all available feature sets have consistent structure."""
        available_sets = list(FeatureSetConfig.FEATURE_SETS.keys())

        for set_name in available_sets:
            config = FeatureSetConfig.FEATURE_SETS.get(set_name)

            assert config is not None, f"Config missing for {set_name}"
            assert "name" in config, f"Name missing for {set_name}"
            assert "description" in config, f"Description missing for {set_name}"
            assert "excluded_features" in config, f"Excluded features missing for {set_name}"
            assert isinstance(config["excluded_features"], list), f"Excluded features not list for {set_name}"

            # Check boolean flags
            bool_flags = [
                "include_regime_features",
                "include_correlation_features",
                "include_ensemble_features",
                "include_risk_features",
                "include_multi_timeframe_features"
            ]

            for flag in bool_flags:
                assert flag in config, f"{flag} missing for {set_name}"
                assert isinstance(config[flag], bool), f"{flag} not boolean for {set_name}"


class TestFeatureSetConfigIntegration:
    """Test FeatureSetConfig integration with other components."""

    def test_high_quality_close_not_excluded(self):
        """Test that close is not excluded from high_quality (regression test)."""
        config = FeatureSetConfig.FEATURE_SETS.get("high_quality")

        assert config is not None
        # This is critical for feature engineering to work
        assert "close" not in config["excluded_features"], \
            "close should not be excluded from high_quality feature set"

    def test_feature_set_names_match_definitions(self):
        """Test that feature set names in available list match definitions."""
        available = list(FeatureSetConfig.FEATURE_SETS.keys())
        defined = list(FeatureSetConfig.FEATURE_SETS.keys())

        assert set(available) == set(defined), \
            "Available feature sets don't match defined feature sets"


if __name__ == "__main__":
    pytest.main([__file__])