"""
Tests for configuration utilities.
"""

from pathlib import Path

import pytest

from ztb.utils.config import TypedConfig, ValidatedConfig, ZTBConfig
from ztb.utils.config_loader import ConfigLoader, load_config, load_toml_config


class TestTypedConfig:
    """Test TypedConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = TypedConfig(
            learning_rate=0.001, batch_size=32, gamma=0.99, total_timesteps=10000
        )

        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.gamma == 0.99
        assert config.total_timesteps == 10000

    def test_invalid_learning_rate(self):
        """Test invalid learning rate validation."""
        with pytest.raises(ValueError, match="Invalid value for learning_rate"):
            TypedConfig(learning_rate=1.5)  # Should be <= 1.0

    def test_invalid_batch_size(self):
        """Test invalid batch size validation."""
        with pytest.raises(ValueError, match="Invalid value for batch_size"):
            TypedConfig(batch_size=0)  # Should be > 0

    def test_invalid_gamma(self):
        """Test invalid gamma validation."""
        with pytest.raises(ValueError, match="Invalid value for gamma"):
            TypedConfig(gamma=1.5)  # Should be <= 1.0

    def test_invalid_total_timesteps(self):
        """Test invalid total timesteps validation."""
        with pytest.raises(ValueError, match="Invalid value for total_timesteps"):
            TypedConfig(total_timesteps=0)  # Should be > 0

    def test_to_dict(self):
        """Test configuration to dict conversion."""
        config = TypedConfig(learning_rate=0.01, batch_size=64)
        config_dict = config.to_dict()

        assert config_dict["learning_rate"] == 0.01
        assert config_dict["batch_size"] == 64

    def test_from_dict(self):
        """Test configuration from dict creation."""
        data = {"learning_rate": 0.001, "batch_size": 32, "gamma": 0.99}
        config = TypedConfig.from_dict(data)

        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.gamma == 0.99

    def test_get_models_default(self):
        """Test default model configuration."""
        config = TypedConfig()
        expected_config = ZTBConfig()
        models = config.get_models()

        assert len(models) == 1
        assert models[0]["path"] == expected_config.get_model_path(
            "trading_optimized_reward_v2_final.zip"
        )
        assert models[0]["weight"] == 1.0
        assert models[0]["feature_set"] == "full"


class TestValidatedConfig:
    """Test ValidatedConfig with JSON Schema."""

    def test_valid_validated_config(self):
        """Test valid validated configuration."""
        config = ValidatedConfig(
            learning_rate=0.001, batch_size=32, gamma=0.99, total_timesteps=10000
        )

        assert config.learning_rate == 0.001
        assert config.batch_size == 32

    def test_invalid_validated_config(self):
        """Test invalid validated configuration."""
        # ValidatedConfig inherits from TypedConfig, so validation still works
        with pytest.raises(ValueError, match="Invalid value for learning_rate"):
            ValidatedConfig(learning_rate=1.5)  # Should fail validation


class TestConfigLoader:
    """Test ConfigLoader functionality."""

    def test_load_yaml_config(self, tmp_path: Path):
        """Test loading YAML configuration."""
        temp_path = tmp_path / "config.yaml"
        temp_path.write_text(
            """
learning_rate: 0.001
batch_size: 32
gamma: 0.99
"""
        )
        config = load_config(temp_path)
        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 32
        assert config["gamma"] == 0.99

    def test_load_json_config(self, tmp_path: Path):
        """Test loading JSON configuration."""
        temp_path = tmp_path / "config.json"
        temp_path.write_text(
            """
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "gamma": 0.99
}
"""
        )
        config = load_config(temp_path)
        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 32
        assert config["gamma"] == 0.99

    def test_load_toml_config(self, tmp_path: Path):
        """Test loading TOML configuration."""
        temp_path = tmp_path / "config.toml"
        temp_path.write_text(
            """
learning_rate = 0.001
batch_size = 32
gamma = 0.99
"""
        )
        config = load_toml_config(temp_path)
        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 32
        assert config["gamma"] == 0.99

    def test_config_loader_class(self, tmp_path: Path):
        """Test ConfigLoader class methods."""
        temp_path = tmp_path / "config.yaml"
        temp_path.write_text(
            """
learning_rate: 0.001
batch_size: 32
"""
        )
        config = ConfigLoader.load(temp_path)
        assert config["learning_rate"] == 0.001
        assert config["batch_size"] == 32
        save_path = tmp_path / "config_saved.json"
        ConfigLoader.save(config, save_path, format="json")
        saved_config = ConfigLoader.load(save_path)
        assert saved_config["learning_rate"] == 0.001
        assert saved_config["batch_size"] == 32

    def test_unsupported_format(self):
        """Test unsupported file format."""
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config("test.txt")
