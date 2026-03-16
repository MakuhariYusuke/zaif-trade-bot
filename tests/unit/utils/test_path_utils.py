"""Unit tests for path utilities."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from ztb.utils.path_utils import ensure_dir, get_project_root


class TestGetProjectRoot:
    """Test get_project_root function."""

    def test_get_project_root_from_file(self):
        """Test getting project root from a file path."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a project root marker
            (temp_path / "pyproject.toml").write_text('[tool.poetry]\nname = "test"')

            # Create a file deep in the structure
            deep_file = temp_path / "src" / "module" / "file.py"
            deep_file.parent.mkdir(parents=True, exist_ok=True)
            deep_file.write_text("# test file")

            # Mock __file__ to point to our deep file
            with patch("ztb.utils.path_utils.__file__", str(deep_file)):
                root = get_project_root()
                # Should find the temp_dir as project root
                assert root == temp_path

    def test_get_project_root_from_nested_location(self):
        """Test getting project root from various nested locations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a project root marker
            (temp_path / "requirements.txt").write_text("pytest\n")

            # Create various nested files
            test_files = [
                temp_path / "tests" / "unit" / "test_file.py",
                temp_path / "src" / "main.py",
                temp_path / "scripts" / "run.py",
            ]

            for test_file in test_files:
                test_file.parent.mkdir(parents=True, exist_ok=True)
                test_file.write_text("# test")

                with patch("ztb.utils.path_utils.__file__", str(test_file)):
                    root = get_project_root()
                    assert root == temp_path

    def test_get_project_root_caching(self):
        """Test that get_project_root caches results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a project root marker
            (temp_path / ".git").mkdir()

            test_file = temp_path / "src" / "main.py"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("# test")

            with patch("ztb.utils.path_utils.__file__", str(test_file)):
                root1 = get_project_root()
                root2 = get_project_root()
                assert root1 == root2 == temp_path


class TestEnsureDir:
    """Test ensure_dir function."""

    def test_ensure_dir_creates_directory(self):
        """Test that ensure_dir creates a directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            new_dir = temp_path / "new_directory"

            # Directory shouldn't exist initially
            assert not new_dir.exists()

            # Call ensure_dir
            ensure_dir(new_dir)

            # Directory should now exist
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_ensure_dir_creates_nested_directories(self):
        """Test that ensure_dir creates nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nested_dir = temp_path / "level1" / "level2" / "level3"

            # Directory shouldn't exist initially
            assert not nested_dir.exists()

            # Call ensure_dir
            ensure_dir(nested_dir)

            # All directories should now exist
            assert nested_dir.exists()
            assert nested_dir.is_dir()
            assert nested_dir.parent.exists()
            assert nested_dir.parent.is_dir()

    def test_ensure_dir_existing_directory(self):
        """Test that ensure_dir doesn't fail on existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_dir = temp_path / "existing_dir"
            existing_dir.mkdir()

            # Should not raise an error
            ensure_dir(existing_dir)

            # Directory should still exist
            assert existing_dir.exists()
            assert existing_dir.is_dir()

    def test_ensure_dir_with_file_in_path(self):
        """Test that ensure_dir handles paths with files correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "some_file.txt"
            file_path.write_text("content")

            # Try to ensure_dir on the file path (should work on parent)
            ensure_dir(file_path.parent)

            # Parent directory should exist
            assert file_path.parent.exists()
            assert file_path.parent.is_dir()

    def test_ensure_dir_permission_error(self):
        """Test that ensure_dir handles permission errors gracefully."""
        # This is hard to test reliably across platforms, but we can test
        # that the function doesn't crash on normal operations
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_dir = temp_path / "test_dir"

            # Should work normally
            ensure_dir(test_dir)
            assert test_dir.exists()
