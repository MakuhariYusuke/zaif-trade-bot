"""Unit tests for file utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from ztb.utils.file_utils import safe_json_dump, safe_json_load


class TestSafeJsonLoad:
    """Test safe_json_load function."""

    def test_safe_json_load_valid_file(self):
        """Test loading valid JSON file."""
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            result = safe_json_load(Path(temp_file))
            assert result == test_data
        finally:
            Path(temp_file).unlink()

    def test_safe_json_load_invalid_json(self):
        """Test loading invalid JSON file returns default."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            result = safe_json_load(Path(temp_file), default={"fallback": True})
            assert result == {"fallback": True}
        finally:
            Path(temp_file).unlink()

    def test_safe_json_load_missing_file(self):
        """Test loading missing file returns default."""
        result = safe_json_load(Path("nonexistent_file.json"), default="default_value")
        assert result == "default_value"

    def test_safe_json_load_missing_file_no_default(self):
        """Test loading missing file without default returns None."""
        result = safe_json_load(Path("nonexistent_file.json"))
        assert result is None

    def test_safe_json_load_with_custom_default_factory(self):
        """Test loading with custom default factory function."""
        def default_factory():
            return {"custom": "default"}

        result = safe_json_load(Path("nonexistent_file.json"), default=default_factory)
        assert result == {"custom": "default"}

    def test_safe_json_load_file_with_extra_data(self):
        """Test loading JSON file with extra data after valid JSON."""
        test_data = {"key": "value"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"key": "value"}\nextra data')
            temp_file = f.name

        try:
            result = safe_json_load(Path(temp_file), default="fallback")
            assert result == "fallback"  # Should fail due to extra data
        finally:
            Path(temp_file).unlink()

    def test_safe_json_load_empty_file(self):
        """Test loading empty file returns default."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            result = safe_json_load(Path(temp_file), default=[])
            assert result == []
        finally:
            Path(temp_file).unlink()


class TestSafeJsonDump:
    """Test safe_json_dump function."""

    def test_safe_json_dump_basic(self):
        """Test basic JSON dump functionality."""
        test_data = {"key": "value", "number": 42}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            safe_json_dump(test_data, Path(temp_file))
            # Verify file was written correctly
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data
        finally:
            Path(temp_file).unlink()

    def test_safe_json_dump_creates_directory(self):
        """Test that safe_json_dump creates necessary directories."""
        test_data = {"test": "data"}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nested_file = temp_path / "nested" / "dir" / "file.json"

            # Directory shouldn't exist
            assert not nested_file.parent.exists()

            safe_json_dump(test_data, nested_file)

            # Directory should now exist
            assert nested_file.parent.exists()
            assert nested_file.exists()

            # Verify content
            with open(nested_file, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data

    def test_safe_json_dump_with_indent(self):
        """Test JSON dump with custom formatting."""
        test_data = {"key": "value"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            safe_json_dump(test_data, Path(temp_file), indent=2)
            with open(temp_file, 'r') as f:
                content = f.read()
            # Should be pretty-printed
            assert "\n" in content
            assert "  " in content
        finally:
            Path(temp_file).unlink()

    def test_safe_json_dump_overwrites_existing(self):
        """Test that safe_json_dump overwrites existing files."""
        initial_data = {"old": "data"}
        new_data = {"new": "data"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(initial_data, f)
            temp_file = f.name

        try:
            safe_json_dump(new_data, Path(temp_file))
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            assert loaded_data == new_data
        finally:
            Path(temp_file).unlink()

    def test_safe_json_dump_handles_exceptions(self):
        """Test that safe_json_dump handles exceptions gracefully."""
        # This is mainly to ensure the function doesn't crash
        # In a real scenario, we'd test specific exception cases
        test_data = {"test": "data"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.json"

            # Should work normally
            safe_json_dump(test_data, test_file)
            assert test_file.exists()