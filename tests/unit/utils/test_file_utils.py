"""Unit tests for file utilities."""

import json
from pathlib import Path

from ztb.utils.file_utils import safe_json_dump, safe_json_load


class TestSafeJsonLoad:
    """Test safe_json_load function."""

    def test_safe_json_load_valid_file(self, tmp_path: Path):
        """Test loading valid JSON file."""
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        temp_file = tmp_path / "valid.json"
        temp_file.write_text(json.dumps(test_data), encoding="utf-8")

        result = safe_json_load(temp_file)
        assert result == test_data

    def test_safe_json_load_invalid_json(self, tmp_path: Path):
        """Test loading invalid JSON file returns default."""
        temp_file = tmp_path / "invalid.json"
        temp_file.write_text("invalid json content", encoding="utf-8")

        result = safe_json_load(temp_file, default={"fallback": True})
        assert result == {"fallback": True}

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

    def test_safe_json_load_file_with_extra_data(self, tmp_path: Path):
        """Test loading JSON file with extra data after valid JSON."""
        temp_file = tmp_path / "extra.json"
        temp_file.write_text('{"key": "value"}\nextra data', encoding="utf-8")

        result = safe_json_load(temp_file, default="fallback")
        assert result == "fallback"  # Should fail due to extra data

    def test_safe_json_load_empty_file(self, tmp_path: Path):
        """Test loading empty file returns default."""
        temp_file = tmp_path / "empty.json"
        temp_file.write_text("", encoding="utf-8")

        result = safe_json_load(temp_file, default=[])
        assert result == []


class TestSafeJsonDump:
    """Test safe_json_dump function."""

    def test_safe_json_dump_basic(self, tmp_path: Path):
        """Test basic JSON dump functionality."""
        test_data = {"key": "value", "number": 42}
        temp_file = tmp_path / "basic.json"

        safe_json_dump(test_data, temp_file)
        loaded_data = json.loads(temp_file.read_text(encoding="utf-8"))
        assert loaded_data == test_data

    def test_safe_json_dump_creates_directory(self, tmp_path: Path):
        """Test that safe_json_dump creates necessary directories."""
        test_data = {"test": "data"}
        nested_file = tmp_path / "nested" / "dir" / "file.json"

        assert not nested_file.parent.exists()
        safe_json_dump(test_data, nested_file)
        assert nested_file.parent.exists()
        assert nested_file.exists()
        loaded_data = json.loads(nested_file.read_text(encoding="utf-8"))
        assert loaded_data == test_data

    def test_safe_json_dump_with_indent(self, tmp_path: Path):
        """Test JSON dump with custom formatting."""
        test_data = {"key": "value"}
        temp_file = tmp_path / "pretty.json"

        safe_json_dump(test_data, temp_file, indent=2)
        content = temp_file.read_text(encoding="utf-8")
        assert "\n" in content
        assert "  " in content

    def test_safe_json_dump_overwrites_existing(self, tmp_path: Path):
        """Test that safe_json_dump overwrites existing files."""
        initial_data = {"old": "data"}
        new_data = {"new": "data"}
        temp_file = tmp_path / "overwrite.json"
        temp_file.write_text(json.dumps(initial_data), encoding="utf-8")

        safe_json_dump(new_data, temp_file)
        loaded_data = json.loads(temp_file.read_text(encoding="utf-8"))
        assert loaded_data == new_data

    def test_safe_json_dump_handles_exceptions(self, tmp_path: Path):
        """Test that safe_json_dump handles exceptions gracefully."""
        # This is mainly to ensure the function doesn't crash
        # In a real scenario, we'd test specific exception cases
        test_data = {"test": "data"}
        test_file = tmp_path / "test.json"

        safe_json_dump(test_data, test_file)
        assert test_file.exists()
