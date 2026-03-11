"""
Unit tests for ztb.config.loader module.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

try:
    from ztb.config.loader import ConfigLoader, initialize_risk_profiles, load_config
    from ztb.config.schema import GlobalConfig
except ImportError:
    pytest.skip("ztb.config modules not available", allow_module_level=True)


class TestConfigLoader:
    """Test cases for ConfigLoader class."""

    def test_init(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader.sources == {
            "defaults": {},
            "yaml": {},
            "env": {},
            "cli": {},
        }
        assert loader._file_cache == {}
        assert loader._file_mtimes == {}
        assert loader.environment in ["development", "production", "test"]

    def test_load_yaml_file_not_exists(self, tmp_path):
        """Test load_yaml when file doesn't exist."""
        loader = ConfigLoader()
        result = loader.load_yaml(str(tmp_path / "nonexistent.yaml"))
        assert result == {}

    def test_load_yaml_success(self, write_yaml_file):
        """Test successful YAML loading."""
        config_path = write_yaml_file("test.yaml", "key: value\n")
        loader = ConfigLoader()
        result = loader.load_yaml(str(config_path))

        assert result == {"key": "value"}
        assert str(config_path) in loader._file_cache
        assert loader._file_mtimes[str(config_path)] == config_path.stat().st_mtime

    def test_load_yaml_invalid_yaml(self, write_yaml_file):
        """Test load_yaml with invalid YAML."""
        config_path = write_yaml_file("invalid.yaml", "invalid: yaml: content:\n")
        loader = ConfigLoader()
        result = loader.load_yaml(str(config_path))

        # Should return empty dict on error
        assert result == {}

    def test_load_yaml_cached(self, write_yaml_file):
        """Test load_yaml returns cached result."""
        config_path = write_yaml_file("test.yaml", "cached: data\n")
        loader = ConfigLoader()
        loader._file_cache[str(config_path)] = {"cached": "data"}
        loader._file_mtimes[str(config_path)] = config_path.stat().st_mtime + 1.0

        result = loader.load_yaml(str(config_path))
        assert result == {"cached": "data"}

    def test_load_yaml_with_env_fallback_env_specific(self, write_yaml_file):
        """Test load_yaml_with_env_fallback with environment-specific file."""
        loader = ConfigLoader()
        loader.environment = "production"
        base_path = write_yaml_file("config.yaml", "base: config\n")
        env_path = write_yaml_file("config.production.yaml", "env: prod\n")

        result = loader.load_yaml_with_env_fallback(str(base_path))
        assert result == {"env": "prod"}

    def test_load_yaml_with_env_fallback_base(self, write_yaml_file):
        """Test load_yaml_with_env_fallback falls back to base file."""
        loader = ConfigLoader()
        loader.environment = "production"
        base_path = write_yaml_file("config.yaml", "base: config\n")

        result = loader.load_yaml_with_env_fallback(str(base_path))
        assert result == {"base": "config"}

    def test_validate_config_success(self):
        """Test successful config validation."""
        loader = ConfigLoader()
        config = {
            "training": {
                "model_name": "test_model",
                "algorithm": "sac",
                "total_timesteps": 1000,
            }
        }
        result = loader.validate_config(config)
        assert isinstance(result, dict)
        assert result["training"]["total_timesteps"] == 1000

    def test_validate_config_invalid(self):
        """Test config validation failure."""
        loader = ConfigLoader()
        config = {
            "training": {
                "model_name": "test_model",
                "algorithm": "sac",
                "total_timesteps": "invalid",
            }
        }
        with pytest.raises(ValueError, match="Configuration validation failed"):
            loader.validate_config(config)

    @patch.dict(
        os.environ,
        {"ZTB_TEST_KEY": "test_value", "OTHER_VAR": "ignore"},
        clear=True,
    )
    def test_load_env(self):
        """Test loading config from environment variables."""
        loader = ConfigLoader()
        result = loader.load_env("ZTB_")
        assert result == {"test": {"key": "test_value"}}

    @patch.dict(os.environ, {"ZTB_NESTED_KEY": "nested_value"}, clear=True)
    def test_load_env_nested(self):
        """Test loading nested config from environment variables."""
        loader = ConfigLoader()
        result = loader.load_env("ZTB_")
        assert result == {"nested": {"key": "nested_value"}}

    def test_load_cli(self):
        """Test loading config from CLI arguments."""
        loader = ConfigLoader()
        args = {"training.total_timesteps": 2000, "log_level": "DEBUG"}
        result = loader.load_cli(args)
        assert result == {"training": {"total_timesteps": 2000}, "log_level": "DEBUG"}

    def test_merge_configs(self):
        """Test merging configs with priority."""
        loader = ConfigLoader()
        loader.sources = {
            "defaults": {"key": "default", "shared": "default"},
            "yaml": {"key": "yaml", "yaml_only": "yaml"},
            "env": {"key": "env", "env_only": "env"},
            "cli": {"key": "cli", "cli_only": "cli"},
        }
        result = loader.merge_configs()
        assert result["key"] == "cli"  # CLI has highest priority
        assert result["shared"] == "default"
        assert result["yaml_only"] == "yaml"
        assert result["env_only"] == "env"
        assert result["cli_only"] == "cli"

    @patch("ztb.config.loaders.priority_loader.GlobalConfig")
    def test_get_config(self, mock_global_config):
        """Test get_config method."""
        # Mock for defaults
        mock_defaults = MagicMock()
        mock_defaults.model_dump.return_value = {"defaults": "data"}
        # Mock for final config
        mock_final = {"validated": "config"}

        def side_effect(*args, **kwargs):
            if not kwargs:  # GlobalConfig() for defaults
                return mock_defaults
            else:  # GlobalConfig(**merged)
                return mock_final

        mock_global_config.side_effect = side_effect

        loader = ConfigLoader()
        with patch.object(loader, "load_yaml"), patch.object(
            loader, "load_env"
        ), patch.object(loader, "load_cli"), patch.object(
            loader, "merge_configs", return_value={"merged": "config"}
        ):
            result = loader.get_config("config.yaml", {"cli": "args"})
            assert result == {"validated": "config"}

    def test_dump_schema(self, tmp_path):
        """Test dumping JSON schema."""
        loader = ConfigLoader()
        schema_path = tmp_path / "schema.json"
        loader.dump_schema(str(schema_path))

        assert schema_path.exists()
        with open(schema_path, "r") as f:
            schema = json.load(f)
        assert "type" in schema
        assert "properties" in schema

    @patch("ztb.config.loaders.priority_loader.initialize_risk_profiles")
    @patch("ztb.config.loaders.priority_loader.ConfigLoader")
    def test_load_config(self, mock_loader_class, mock_init_risk):
        """Test load_config convenience function."""
        mock_loader = MagicMock()
        mock_config = MagicMock()
        mock_loader.get_config.return_value = mock_config
        mock_loader_class.return_value = mock_loader

        result = load_config("config.yaml", {"cli": "args"})
        assert result == mock_config
        mock_init_risk.assert_called_once_with(mock_config)

    @patch("ztb.trading.live.risk.profiles.get_risk_manager")
    def test_initialize_risk_profiles(self, mock_get_manager):
        """Test initialize_risk_profiles function."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        config = SimpleNamespace(risk_profiles={"low": object(), "high": object()})
        initialize_risk_profiles(config)

        # Should call add_profile for each risk profile
        assert mock_manager.add_profile.call_count == len(config.risk_profiles)
