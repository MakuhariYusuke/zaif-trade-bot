#!/usr/bin/env python3
"""
Unit tests for config_manager.py

Tests for ConfigManager class and configuration validation utilities.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ztb.utils.config_manager import ConfigManager, validate_config
from ztb.utils.exceptions.custom_exceptions import ConfigurationError


class TestConfigManager:
    """Test ConfigManager class."""

    def test_initialization_default(self):
        """Test ConfigManager initialization with default config directory."""
        manager = ConfigManager()

        assert manager.config_dir.exists()
        assert isinstance(manager._cache, dict)
        assert manager._cache == {}

    def test_initialization_custom_dir(self):
        """Test ConfigManager initialization with custom config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "custom_config"
            manager = ConfigManager(custom_dir)

            assert manager.config_dir == custom_dir
            assert manager.config_dir.exists()

    def test_load_config_json(self):
        """Test loading JSON configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)

            # Create a test JSON config file
            config_file = Path(temp_dir) / "test.json"
            config_data = {"key": "value", "number": 42}
            config_file.write_text('{"key": "value", "number": 42}')

            result = manager.load_config("test", config_type="general")

            assert result == config_data
            assert "general_test" in manager._cache

    def test_load_config_yaml(self):
        """Test loading YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)

            # Create a test YAML config file
            config_file = Path(temp_dir) / "test.yaml"
            config_content = "key: value\nnumber: 42\n"
            config_file.write_text(config_content)

            result = manager.load_config("test", config_type="general")

            assert result == {"key": "value", "number": 42}

    def test_save_config(self):
        """Test saving configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)

            config_data = {"key": "value", "number": 42}
            manager.save_config(config_data, "test_save", format="json")

            # Check that file was created
            config_file = Path(temp_dir) / "test_save.json"
            assert config_file.exists()

            # Check content
            import json
            with open(config_file, 'r') as f:
                saved_data = json.load(f)
            assert saved_data == config_data

    def test_clear_cache(self):
        """Test clearing configuration cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)

            # Add something to cache
            manager._cache["test"] = {"cached": True}

            assert len(manager._cache) > 0

            manager.clear_cache()

            assert len(manager._cache) == 0

    def test_get_cached_configs(self):
        """Test getting cached configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)

            test_config = {"test": "data"}
            manager._cache["test_key"] = test_config

            cached = manager.get_cached_configs()

            assert "test_key" in cached
            assert cached["test_key"] == test_config

            # Ensure it's a copy, not reference
            cached["test_key"]["modified"] = True
            assert "modified" not in manager._cache["test_key"]


class TestConfigValidation:
    """Test configuration validation functions."""

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        class MockConfig:
            def __init__(self):
                self.required_field1 = "value1"
                self.required_field2 = 42

        config = MockConfig()
        result = validate_config(config, ["required_field1", "required_field2"])

        assert result is True

    def test_validate_config_missing_field(self):
        """Test configuration validation with missing field."""
        class MockConfig:
            def __init__(self):
                self.required_field1 = "value1"
                # required_field2 is missing

        config = MockConfig()
        result = validate_config(config, ["required_field1", "required_field2"])

        assert result is False

    def test_validate_config_empty_required_fields(self):
        """Test configuration validation with empty required fields list."""
        class MockConfig:
            def __init__(self):
                self.some_field = "value"

        config = MockConfig()
        result = validate_config(config, [])

        assert result is True

    def test_validate_config_no_matching_attributes(self):
        """Test configuration validation when config has no matching attributes."""
        class MockConfig:
            pass

        config = MockConfig()
        result = validate_config(config, ["field1", "field2"])

        assert result is False


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""

    def test_load_save_roundtrip(self):
        """Test loading and saving configuration in a roundtrip."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)

            original_config = {
                "model": {
                    "type": "ppo",
                    "learning_rate": 0.001,
                    "batch_size": 64
                },
                "environment": {
                    "name": "trading_env",
                    "max_steps": 1000
                },
                "training": {
                    "epochs": 100,
                    "validation_freq": 10
                }
            }

            # Save config
            manager.save_config(original_config, "integration_test")

            # Load config
            loaded_config = manager.load_config("integration_test")

            assert loaded_config == original_config

    def test_config_file_not_found(self):
        """Test handling of non-existent configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)

            with pytest.raises(ConfigurationError):
                manager.load_config("nonexistent_config")

    @patch('ztb.utils.config_manager.yaml.safe_load')
    def test_yaml_parsing_error(self, mock_yaml_load):
        """Test handling of YAML parsing errors."""
        mock_yaml_load.side_effect = Exception("YAML parsing failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(temp_dir)

            # Create a malformed YAML file
            config_file = Path(temp_dir) / "malformed.yaml"
            config_file.write_text("invalid: yaml: content: [")

            with pytest.raises(Exception):
                manager.load_config("malformed")


if __name__ == "__main__":
    pytest.main([__file__])