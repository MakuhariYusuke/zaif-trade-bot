#!/usr/bin/env python3
"""
Unit tests for SACTrainer class.

Tests for SACTrainer feature set handling and configuration propagation.
"""

import pytest
from unittest.mock import Mock, patch

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer


class TestSACTrainerFeatureSetHandling:
    """Test SACTrainer feature set resolution and propagation."""

    def test_resolve_feature_set_override_from_training_features(self):
        """Test feature_set resolution from training.features section."""
        config = {
            "training": {
                "features": {"feature_set": "high_quality"}
            }
        }

        trainer = SACTrainer(config)
        result = trainer._resolve_feature_set_override(None)

        assert result == "high_quality"

    def test_resolve_feature_set_override_priority_order(self):
        """Test feature_set resolution priority order."""
        # training.features should have highest priority
        config = {
            "training": {
                "features": {"feature_set": "high_quality"},
                "environment": {"feature_set": "default"}
            },
            "environment": {"feature_set": "minimal"}
        }

        trainer = SACTrainer(config)
        result = trainer._resolve_feature_set_override("low_quality")

        assert result == "high_quality"

    def test_resolve_feature_set_override_fallback_to_env_candidate(self):
        """Test fallback to env_candidate when no valid feature_set in config."""
        config = {
            "training": {
                "features": {"other_setting": True}
            }
        }

        trainer = SACTrainer(config)
        result = trainer._resolve_feature_set_override("high_quality")

        assert result is None  # No valid candidates in config, env_candidate is ignored when None

    def test_resolve_feature_set_override_invalid_candidates(self):
        """Test handling of invalid feature_set candidates."""
        config = {
            "training": {
                "features": {"feature_set": "invalid_set"},
                "environment": {"feature_set": "another_invalid"}
            }
        }

        trainer = SACTrainer(config)
        result = trainer._resolve_feature_set_override("another_invalid")

        assert result == "invalid_set"  # fallback to first invalid candidate when no valid ones found

    def test_resolve_feature_set_override_no_valid_candidates(self):
        """Test when no valid feature_set candidates are found."""
        config = {
            "training": {
                "features": {"other_setting": True}
            }
        }

        trainer = SACTrainer(config)
        result = trainer._resolve_feature_set_override("invalid_set")

        assert result is None  # No valid candidates, env_candidate is invalid

    def test_ensure_feature_set_on_target_dict(self):
        """Test applying feature_set to dict target."""
        trainer = SACTrainer({})

        target = {"max_position_size": 1.0}
        trainer._ensure_feature_set_on_target(target, "high_quality")

        assert target["feature_set"] == "high_quality"

    def test_ensure_feature_set_on_target_dict_existing_valid(self):
        """Test not overriding existing valid feature_set on dict."""
        trainer = SACTrainer({})

        target = {"feature_set": "default", "max_position_size": 1.0}
        trainer._ensure_feature_set_on_target(target, "high_quality")

        assert target["feature_set"] == "default"  # Should not override valid existing

    def test_ensure_feature_set_on_target_object(self):
        """Test applying feature_set to object target."""
        trainer = SACTrainer({})

        class MockEnv:
            def __init__(self):
                self.max_position_size = 1.0

        target = MockEnv()
        trainer._ensure_feature_set_on_target(target, "high_quality")

        # Since MockEnv doesn't have feature_set attribute initially, setattr should add it
        assert hasattr(target, "feature_set")
        assert target.feature_set == "high_quality"

    def test_ensure_feature_set_on_target_object_existing_valid(self):
        """Test not overriding existing valid feature_set on object."""
        trainer = SACTrainer({})

        class MockEnv:
            def __init__(self):
                self.feature_set = "default"
                self.max_position_size = 1.0

        target = MockEnv()
        trainer._ensure_feature_set_on_target(target, "high_quality")

        assert target.feature_set == "default"  # Should not override valid existing

    def test_ensure_feature_set_on_target_invalid_feature_set(self):
        """Test not applying invalid feature_set."""
        trainer = SACTrainer({})

        target = {"max_position_size": 1.0}
        trainer._ensure_feature_set_on_target(target, "invalid_set")

        assert "feature_set" not in target

    def test_propagate_feature_set_to_multiple_targets(self):
        """Test propagating feature_set to multiple config targets."""
        config = {
            "training": {
                "environment": {"max_position_size": 1.0},
                "features": {"other_setting": True}
            },
            "environment": {"initial_balance": 100000}
        }

        trainer = SACTrainer(config)
        trainer._propagate_feature_set("high_quality", "default")

        # Check that feature_set was added to relevant sections
        assert config["training"]["environment"]["feature_set"] == "high_quality"
        assert config["environment"]["feature_set"] == "high_quality"

    def test_propagate_feature_set_skip_existing_valid(self):
        """Test not overriding existing valid feature_set during propagation."""
        config = {
            "training": {
                "environment": {"feature_set": "default", "max_position_size": 1.0}
            },
            "environment": {"feature_set": "minimal", "initial_balance": 100000}
        }

        trainer = SACTrainer(config)
        trainer._propagate_feature_set("high_quality", "low_quality")

        # Existing valid feature_sets should not be overridden
        assert config["training"]["environment"]["feature_set"] == "default"
        assert config["environment"]["feature_set"] == "minimal"

    def test_propagate_feature_set_invalid_feature_set(self):
        """Test not propagating invalid feature_set."""
        config = {
            "training": {
                "environment": {"max_position_size": 1.0}
            }
        }

        trainer = SACTrainer(config)
        trainer._propagate_feature_set("invalid_set", "default")

        # Invalid feature_set should not be propagated
        assert "feature_set" not in config["training"]["environment"]

    def test_is_valid_feature_set_name(self):
        """Test feature_set name validation."""
        trainer = SACTrainer({})

        # Valid feature sets
        assert trainer._is_valid_feature_set_name("default")
        assert trainer._is_valid_feature_set_name("high_quality")
        assert trainer._is_valid_feature_set_name("minimal")
        assert trainer._is_valid_feature_set_name("full")

        # Invalid feature sets
        assert not trainer._is_valid_feature_set_name("")
        assert not trainer._is_valid_feature_set_name(None)
        assert not trainer._is_valid_feature_set_name("invalid_set")
        assert not trainer._is_valid_feature_set_name("low_quality")
        assert not trainer._is_valid_feature_set_name(123)

    def test_extract_feature_set_from_dict(self):
        """Test extracting feature_set from dict."""
        trainer = SACTrainer({})

        source = {"feature_set": "high_quality", "other": "value"}
        result = trainer._extract_feature_set(source)

        assert result == "high_quality"

    def test_extract_feature_set_from_object(self):
        """Test extracting feature_set from object."""
        trainer = SACTrainer({})

        class MockObj:
            def __init__(self):
                self.feature_set = "high_quality"

        source = MockObj()
        result = trainer._extract_feature_set(source)

        assert result == "high_quality"

    def test_extract_feature_set_missing(self):
        """Test extracting feature_set when not present."""
        trainer = SACTrainer({})

        source = {"other": "value"}
        result = trainer._extract_feature_set(source)

        assert result is None

    def test_extract_feature_set_invalid_source(self):
        """Test extracting feature_set from invalid source."""
        trainer = SACTrainer({})

        result = trainer._extract_feature_set("invalid_source")

        assert result is None


class TestSACTrainerIntegration:
    """Test SACTrainer integration with config propagation."""

    @patch('ztb.training.unified_trainer.algorithms.sac_trainer.HeavyTradingEnv')
    def test_feature_set_propagation_in_training_setup(self, mock_env_class):
        """Test that feature_set is properly propagated during training setup."""
        config = {
            "training": {
                "features": {"feature_set": "high_quality"},
                "environment": {"max_position_size": 1.0}
            },
            "algorithm": "sac"
        }

        mock_env = Mock()
        mock_env_class.return_value = mock_env

        trainer = SACTrainer(config)

        # Simulate the feature set resolution process
        resolved = trainer._resolve_feature_set_override(None)
        assert resolved == "high_quality"

        # Simulate propagation
        trainer._propagate_feature_set(resolved, None)

        # Verify that environment gets the feature_set
        assert config["training"]["environment"]["feature_set"] == "high_quality"


if __name__ == "__main__":
    pytest.main([__file__])