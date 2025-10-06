"""
Unit tests for file_utils.py module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ztb.utils.file_utils import (
    load_config_file,
    safe_json_dump,
    safe_json_load,
    save_config_file,
)


class TestSafeJsonLoad:
    """Test cases for safe_json_load function."""

    def test_load_valid_json_file(self):
        """Test loading a valid JSON file."""
        test_data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            result = safe_json_load(temp_path)
            assert result == test_data
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file returns default."""
        result = safe_json_load(Path("nonexistent.json"), default={"default": True})
        assert result == {"default": True}

    def test_load_nonexistent_file_callable_default(self):
        """Test loading a nonexistent file with callable default."""

        def get_default():
            return {"callable": "default"}

        result = safe_json_load(Path("nonexistent.json"), default=get_default)
        assert result == {"callable": "default"}

    def test_load_invalid_json_file(self):
        """Test loading an invalid JSON file returns default."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = Path(f.name)

        try:
            result = safe_json_load(temp_path, default={"error": "fallback"})
            assert result == {"error": "fallback"}
        finally:
            temp_path.unlink()

    def test_load_json_with_exception(self):
        """Test loading JSON file that raises an exception."""
        with patch("builtins.open", side_effect=Exception("Test error")):
            result = safe_json_load(Path("test.json"), default="fallback")
            assert result == "fallback"


class TestSafeJsonDump:
    """Test cases for safe_json_dump function."""

    def test_dump_data_successfully(self):
        """Test successfully dumping data to JSON file."""
        test_data = {"test": "data", "array": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = safe_json_dump(test_data, temp_path)
            assert result is True
            assert temp_path.exists()

            # Verify content
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data
        finally:
            temp_path.unlink()

    def test_dump_data_with_string_path(self):
        """Test dumping data with string path."""
        test_data = {"string": "path"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = safe_json_dump(test_data, str(temp_path))
            assert result is True
            assert temp_path.exists()
        finally:
            temp_path.unlink()

    def test_dump_data_creates_parent_directories(self):
        """Test that parent directories are created when dumping."""
        test_data = {"nested": "dirs"}

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "deep" / "file.json"

            result = safe_json_dump(test_data, nested_path)
            assert result is True
            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_dump_data_with_custom_indent(self):
        """Test dumping data with custom indentation."""
        test_data = {"custom": "indent"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = safe_json_dump(test_data, temp_path, indent=4)
            assert result is True

            # Verify indentation
            with open(temp_path, "r") as f:
                content = f.read()
            assert "    " in content  # Should have 4-space indentation
        finally:
            temp_path.unlink()

    def test_dump_data_with_custom_encoder(self):
        """Test dumping data with custom JSON encoder."""

        class CustomObject:
            def __init__(self, value):
                self.value = value

        def custom_encoder(obj):
            if isinstance(obj, CustomObject):
                return {"custom_value": obj.value}
            return obj

        test_data = {"custom": CustomObject("test")}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = safe_json_dump(test_data, temp_path, default=custom_encoder)
            assert result is True

            with open(temp_path, "r") as f:
                loaded_data = json.load(f)
            assert loaded_data == {"custom": {"custom_value": "test"}}
        finally:
            temp_path.unlink()

    def test_dump_data_failure(self):
        """Test dumping data that fails."""
        test_data = {"test": "data"}

        with patch("pathlib.Path.mkdir", side_effect=Exception("Test error")):
            result = safe_json_dump(test_data, Path("test.json"))
            assert result is False


class TestLoadConfigFile:
    """Test cases for load_config_file function."""

    def test_load_valid_config_file(self):
        """Test loading a valid config file."""
        config_data = {"setting": "value", "enabled": True, "count": 10}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            result = load_config_file(temp_path)
            assert result == config_data
        finally:
            temp_path.unlink()

    def test_load_config_file_not_dict(self):
        """Test loading a config file that doesn't contain a dictionary."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump("not a dict", f)
            temp_path = Path(f.name)

        try:
            result = load_config_file(temp_path)
            assert result is None
        finally:
            temp_path.unlink()

    def test_load_config_file_nonexistent(self):
        """Test loading a nonexistent config file."""
        result = load_config_file(Path("nonexistent.json"))
        assert result is None


class TestSaveConfigFile:
    """Test cases for save_config_file function."""

    def test_save_valid_config_file(self):
        """Test saving a valid config file."""
        config_data = {"database": {"host": "localhost", "port": 5432}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = save_config_file(config_data, temp_path)
            assert result is True
            assert temp_path.exists()

            # Verify content
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)
            assert loaded_data == config_data
        finally:
            temp_path.unlink()

    def test_save_config_file_creates_directories(self):
        """Test that save_config_file creates parent directories."""
        config_data = {"test": "config"}

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "config" / "subdir" / "settings.json"

            result = save_config_file(config_data, nested_path)
            assert result is True
            assert nested_path.exists()

    def test_save_config_file_failure(self):
        """Test saving config file that fails."""
        config_data = {"test": "data"}

        with patch("ztb.utils.file_utils.safe_json_dump", return_value=False):
            result = save_config_file(config_data, Path("test.json"))
            assert result is False
