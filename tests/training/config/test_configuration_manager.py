#!/usr/bin/env python3
"""
Tests for the unified configuration management system.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ztb.training.config.configuration_manager import (
    ConfigLoadError,
    ConfigurationManager,
    ValidationError,
    create_config_template,
    load_training_config,
    validate_config_file,
)


class TestConfigurationManager(unittest.TestCase):
    """Test cases for ConfigurationManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ConfigurationManager()
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create sample configuration
        self.sample_config = {
            "version": "1.0",
            "training": {
                "model_name": "test_model",
                "algorithm": "sac",
                "total_timesteps": 100000,
                "data_config": {"data_path": "data/test.csv", "use_real_data": True},
                "environment": {
                    "initial_balance": 10000.0,
                    "transaction_cost": 0.0015,
                    "max_position_size": 1.0,
                },
            },
        }

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        for file in self.temp_dir.glob("*"):
            file.unlink()
        self.temp_dir.rmdir()

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_path = self.temp_dir / "valid_config.json"
        with open(config_path, "w") as f:
            json.dump(self.sample_config, f)

        config = self.manager.load_config(config_path, "training")

        self.assertEqual(config["version"], "1.0")
        self.assertEqual(config["training"]["algorithm"], "sac")
        self.assertEqual(config["training"]["total_timesteps"], 100000)

    def test_load_invalid_config_file(self):
        """Test loading a non-existent configuration file."""
        config_path = self.temp_dir / "nonexistent.json"

        with self.assertRaises(ConfigLoadError):
            self.manager.load_config(config_path, "training")

    def test_validate_invalid_algorithm(self):
        """Test validation of invalid algorithm."""
        invalid_config = self.sample_config.copy()
        invalid_config["training"]["algorithm"] = "invalid_algorithm"

        config_path = self.temp_dir / "invalid_config.json"
        with open(config_path, "w") as f:
            json.dump(invalid_config, f)

        with self.assertRaises(ValidationError) as cm:
            self.manager.load_config(config_path, "training")

        self.assertIn("Algorithm must be one of", str(cm.exception))

    def test_validate_missing_required_field(self):
        """Test validation of missing required field."""
        invalid_config = self.sample_config.copy()
        del invalid_config["training"]["total_timesteps"]

        config_path = self.temp_dir / "missing_field.json"
        with open(config_path, "w") as f:
            json.dump(invalid_config, f)

        with self.assertRaises(ValidationError) as cm:
            self.manager.load_config(config_path, "training")

        self.assertIn(
            "Required field 'training.total_timesteps' is missing", str(cm.exception)
        )

    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        config_path = self.temp_dir / "env_config.json"
        with open(config_path, "w") as f:
            json.dump(self.sample_config, f)

        # Set environment variable
        env_value = "200000"
        with patch.dict(os.environ, {"TRAINING_TOTAL_TIMESTEPS": env_value}):
            config = self.manager.load_config(
                config_path, "training", env_prefix="TRAINING_"
            )

        self.assertEqual(config["training"]["total_timesteps"], int(env_value))

    def test_runtime_overrides(self):
        """Test runtime configuration overrides."""
        config_path = self.temp_dir / "override_config.json"
        with open(config_path, "w") as f:
            json.dump(self.sample_config, f)

        overrides = {
            "training": {"total_timesteps": 500000, "model_name": "overridden_model"}
        }

        config = self.manager.load_config(config_path, "training", overrides=overrides)

        self.assertEqual(config["training"]["total_timesteps"], 500000)
        self.assertEqual(config["training"]["model_name"], "overridden_model")

    def test_get_config_value(self):
        """Test getting configuration values using dot notation."""
        config = self.sample_config

        # Test existing path
        value = self.manager.get_config_value(config, "training.algorithm")
        self.assertEqual(value, "sac")

        # Test nested path
        value = self.manager.get_config_value(config, "training.data_config.data_path")
        self.assertEqual(value, "data/test.csv")

        # Test non-existent path
        value = self.manager.get_config_value(
            config, "nonexistent.path", default="default"
        )
        self.assertEqual(value, "default")

    def test_create_config_template(self):
        """Test creating configuration template."""
        template = self.manager.create_config_template("training")

        self.assertIn("version", template)
        self.assertIn("training", template)
        self.assertEqual(template["training"]["algorithm"], "sac")

    def test_validate_config_file(self):
        """Test configuration file validation."""
        # Valid config
        valid_path = self.temp_dir / "valid.json"
        with open(valid_path, "w") as f:
            json.dump(self.sample_config, f)

        errors = self.manager.validate_config_file(valid_path, "training")
        self.assertEqual(len(errors), 0)

        # Invalid config
        invalid_config = self.sample_config.copy()
        invalid_config["training"]["algorithm"] = "invalid"
        invalid_path = self.temp_dir / "invalid.json"
        with open(invalid_path, "w") as f:
            json.dump(invalid_config, f)

        errors = self.manager.validate_config_file(invalid_path, "training")
        self.assertGreater(len(errors), 0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_config = {
            "version": "1.0",
            "training": {
                "model_name": "test_model",
                "algorithm": "sac",
                "total_timesteps": 100000,
                "data_config": {"data_path": "data/test.csv", "use_real_data": True},
            },
        }

    def tearDown(self):
        """Clean up test fixtures."""
        for file in self.temp_dir.glob("*"):
            file.unlink()
        self.temp_dir.rmdir()

    def test_load_training_config(self):
        """Test load_training_config convenience function."""
        config_path = self.temp_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.sample_config, f)

        config = load_training_config(config_path)

        self.assertEqual(config["training"]["algorithm"], "sac")
        self.assertEqual(config["training"]["total_timesteps"], 100000)

    def test_validate_config_file_function(self):
        """Test validate_config_file convenience function."""
        config_path = self.temp_dir / "valid_config.json"
        with open(config_path, "w") as f:
            json.dump(self.sample_config, f)

        errors = validate_config_file(config_path)
        self.assertEqual(len(errors), 0)

    def test_create_config_template_function(self):
        """Test create_config_template convenience function."""
        template = create_config_template()

        self.assertIn("training", template)
        self.assertEqual(template["training"]["algorithm"], "sac")


if __name__ == "__main__":
    unittest.main()
