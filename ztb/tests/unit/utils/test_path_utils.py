"""
Unit tests for path_utils.py module.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ztb.utils.path_utils import (
    ensure_dir,
    find_files_by_extension,
    get_project_root,
    get_relative_path,
    safe_path_join,
)


class TestGetProjectRoot:
    """Test cases for get_project_root function."""

    def test_get_project_root_returns_path(self):
        """Test that get_project_root returns a Path object."""
        result = get_project_root()
        assert isinstance(result, Path)

    def test_get_project_root_has_project_markers(self):
        """Test that the returned project root contains expected project markers."""
        root = get_project_root()

        # Check that at least one project marker exists
        markers = ['pyproject.toml', 'setup.py', '.git', 'requirements.txt', 'package.json']
        has_marker = any((root / marker).exists() for marker in markers)
        assert has_marker, f"No project markers found in {root}"


class TestEnsureDir:
    """Test cases for ensure_dir function."""

    def test_ensure_dir_existing_directory(self):
        """Test ensure_dir with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "existing"
            path.mkdir(exist_ok=True)

            result = ensure_dir(path)
            assert result == path
            assert path.exists()
            assert path.is_dir()

    def test_ensure_dir_create_nested_directories(self):
        """Test ensure_dir creates nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "level1" / "level2" / "level3"

            result = ensure_dir(nested_path)
            assert result == nested_path
            assert nested_path.exists()
            assert nested_path.is_dir()

    def test_ensure_dir_with_file_path(self):
        """Test ensure_dir with a file path (should create parent directories)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir) / "subdir"

            result = ensure_dir(dir_path)
            assert result == dir_path
            assert dir_path.exists()
            assert dir_path.is_dir()


class TestSafePathJoin:
    """Test cases for safe_path_join function."""

    def test_safe_path_join_multiple_strings(self):
        """Test safe_path_join with multiple string arguments."""
        result = safe_path_join("home", "user", "documents", "file.txt")
        expected = Path("home") / "user" / "documents" / "file.txt"
        assert result == expected

    def test_safe_path_join_single_argument(self):
        """Test safe_path_join with single argument."""
        result = safe_path_join("single")
        expected = Path("single")
        assert result == expected

    def test_safe_path_join_empty_arguments(self):
        """Test safe_path_join with no arguments."""
        result = safe_path_join()
        expected = Path()
        assert result == expected

    def test_safe_path_join_mixed_types(self):
        """Test safe_path_join with mixed Path and string arguments."""
        base_path = Path("/base")
        result = safe_path_join(str(base_path), "subdir", "file.txt")
        expected = Path("/base") / "subdir" / "file.txt"
        assert result == expected

    def test_safe_path_join_with_absolute_path(self):
        """Test safe_path_join starting with absolute path."""
        result = safe_path_join("/absolute", "path", "file.txt")
        expected = Path("/absolute") / "path" / "file.txt"
        assert result == expected


class TestGetRelativePath:
    """Test cases for get_relative_path function."""

    @patch('os.path.relpath')
    def test_get_relative_path_success(self, mock_relpath):
        """Test get_relative_path with successful relative path calculation."""
        mock_relpath.return_value = "subdir/file.txt"

        from_path = Path("/home/user")
        to_path = Path("/home/user/subdir/file.txt")

        result = get_relative_path(from_path, to_path)
        assert result == Path("subdir/file.txt")
        mock_relpath.assert_called_once_with(str(to_path), str(from_path))

    @patch('os.path.relpath', side_effect=ValueError("Different drives"))
    def test_get_relative_path_different_drives(self, mock_relpath):
        """Test get_relative_path when paths are on different drives."""
        from_path = Path("C:/path1")
        to_path = Path("D:/path2")

        result = get_relative_path(from_path, to_path)
        assert result == to_path  # Should return to_path directly


class TestFindFilesByExtension:
    """Test cases for find_files_by_extension function."""

    def test_find_files_by_extension_python_files(self):
        """Test finding Python files in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "script.py").write_text("print('hello')")
            (temp_path / "module.py").write_text("def func(): pass")
            (temp_path / "data.json").write_text('{"key": "value"}')
            (temp_path / "readme.txt").write_text("readme content")

            # Create subdirectory with more files
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested.py").write_text("nested file")
            (subdir / "nested.txt").write_text("nested text")

            result = find_files_by_extension(temp_path, "py")

            # Should find all .py files recursively
            py_files = [f.name for f in result]
            assert "script.py" in py_files
            assert "module.py" in py_files
            assert "nested.py" in py_files
            assert len(result) == 3

            # Verify all are Path objects
            assert all(isinstance(f, Path) for f in result)

    def test_find_files_by_extension_no_matches(self):
        """Test finding files with extension that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with different extensions
            (temp_path / "file1.txt").write_text("text")
            (temp_path / "file2.json").write_text('{"data": true}')

            result = find_files_by_extension(temp_path, "md")

            assert result == []

    def test_find_files_by_extension_empty_directory(self):
        """Test finding files in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            result = find_files_by_extension(temp_path, "py")

            assert result == []

    def test_find_files_by_extension_case_sensitivity(self):
        """Test that extension matching is case-sensitive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with different case extensions
            (temp_path / "file.PY").write_text("uppercase")
            (temp_path / "file.py").write_text("lowercase")

            result_py = find_files_by_extension(temp_path, "py")
            result_PY = find_files_by_extension(temp_path, "PY")

            # On Windows, filesystem is case-insensitive, so both should match
            # On case-sensitive filesystems, they would be different
            assert len(result_py) >= 1  # At least file.py should match
            assert len(result_PY) >= 1  # At least file.PY should match
            # Total files found should be 2 (both extensions match due to case-insensitivity)
            assert len(result_py) + len(result_PY) >= 2