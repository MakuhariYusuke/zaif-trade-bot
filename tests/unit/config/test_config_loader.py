"""
Unit tests for ztb.config.loader module.
"""

import json
import os
from unittest.mock import MagicMock, mock_open, patch

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

    @patch("ztb.config.loader.Path")
    def test_load_yaml_file_not_exists(self, mock_path):
        """Test load_yaml when file doesn't exist."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        loader = ConfigLoader()
        result = loader.load_yaml("nonexistent.yaml")
        assert result == {}

    @patch("ztb.config.loader.Path")
    @patch("builtins.open", new_callable=mock_open, read_data="key: value\n")
    def test_load_yaml_success(self, mock_file, mock_path):
        """Test successful YAML loading."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_mtime = 1234567890
        mock_path.return_value = mock_path_instance

        loader = ConfigLoader()
        result = loader.load_yaml("test.yaml")

        assert result == {"key": "value"}
        assert "test.yaml" in loader._file_cache
        assert loader._file_mtimes["test.yaml"] == 1234567890

    @patch("ztb.config.loader.Path")
    @patch(
        "builtins.open", new_callable=mock_open, read_data="invalid: yaml: content:\n"
    )
    def test_load_yaml_invalid_yaml(self, mock_file, mock_path):
        """Test load_yaml with invalid YAML."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_mtime = 1234567890
        mock_path.return_value = mock_path_instance

        loader = ConfigLoader()
        result = loader.load_yaml("invalid.yaml")

        # Should return empty dict on error
        assert result == {}

    @patch("ztb.config.loader.Path")
    def test_load_yaml_cached(self, mock_path):
        """Test load_yaml returns cached result."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_mtime = 1234567890
        mock_path.return_value = mock_path_instance

        loader = ConfigLoader()
        loader._file_cache["test.yaml"] = {"cached": "data"}
        loader._file_mtimes["test.yaml"] = 1234567891  # newer than current

        result = loader.load_yaml("test.yaml")
        assert result == {"cached": "data"}

    @patch("ztb.config.loader.Path")
    def test_load_yaml_with_env_fallback_env_specific(self, mock_path):
        """Test load_yaml_with_env_fallback with environment-specific file."""
        loader = ConfigLoader()
        loader.environment = "production"

        # Mock paths
        mock_env_path = MagicMock()
        mock_env_path.exists.return_value = True
        mock_base_path = MagicMock()
        mock_base_path.exists.return_value = True

        def path_side_effect(path):
            if path == "config.production.yaml":
                return mock_env_path
            elif path == "config.yaml":
                return mock_base_path
            return MagicMock()

        mock_path.side_effect = path_side_effect

        with patch.object(loader, "load_yaml") as mock_load:
            mock_load.side_effect = (
                lambda p: {"env": "prod"} if "production" in p else {"base": "config"}
            )

            result = loader.load_yaml_with_env_fallback("config")
            mock_load.assert_called_once_with("config.production.yaml")
            assert result == {"env": "prod"}

    @patch("ztb.config.loader.Path")
    def test_load_yaml_with_env_fallback_base(self, mock_path):
        """Test load_yaml_with_env_fallback falls back to base file."""
        loader = ConfigLoader()
        loader.environment = "production"

        # Mock paths
        mock_env_path = MagicMock()
        mock_env_path.exists.return_value = False
        mock_base_path = MagicMock()
        mock_base_path.exists.return_value = True

        def path_side_effect(path):
            if path == "config.production.yaml":
                return mock_env_path
            elif path == "config.yaml":
                return mock_base_path
            return MagicMock()

        mock_path.side_effect = path_side_effect

        with patch.object(loader, "load_yaml") as mock_load:
            mock_load.return_value = {"base": "config"}

            result = loader.load_yaml_with_env_fallback("config")
            assert mock_load.call_count == 1
            assert result == {"base": "config"}

    def test_validate_config_success(self):
        """Test successful config validation."""
        loader = ConfigLoader()
        config = {"training": {"total_timesteps": 1000}}
        result = loader.validate_config(config)
        assert isinstance(result, dict)
        assert result["training"]["total_timesteps"] == 1000

    def test_validate_config_invalid(self):
        """Test config validation failure."""
        loader = ConfigLoader()
        config = {"training": {"total_timesteps": "invalid"}}
        with pytest.raises(ValueError, match="Configuration validation failed"):
            loader.validate_config(config)

    @patch.dict(os.environ, {"ZTB_TEST_KEY": "test_value", "OTHER_VAR": "ignore"})
    def test_load_env(self):
        """Test loading config from environment variables."""
        loader = ConfigLoader()
        result = loader.load_env("ZTB_")
        assert result == {"test": {"key": "test_value"}}

    @patch.dict(os.environ, {"ZTB_NESTED_KEY": "nested_value"})
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

    @patch("ztb.config.loader.GlobalConfig")
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

    @patch("ztb.config.loader.initialize_risk_profiles")
    @patch("ztb.config.loader.ConfigLoader")
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

        config = GlobalConfig()
        initialize_risk_profiles(config)

        # Should call add_profile for each risk profile
        assert mock_manager.add_profile.call_count == len(config.risk_profiles)
