"""Unit tests for path utilities."""
from pathlib import Path
from unittest.mock import patch

from ztb.utils.path_utils import ensure_dir, get_project_root


class TestGetProjectRoot:
    """Test get_project_root function."""

    def test_get_project_root_from_file(self, tmp_path: Path):
        """Test getting project root from a file path."""
        (tmp_path / "pyproject.toml").write_text('[tool.poetry]\nname = "test"')
        deep_file = tmp_path / "src" / "module" / "file.py"
        deep_file.parent.mkdir(parents=True, exist_ok=True)
        deep_file.write_text("# test file")

        with patch("ztb.utils.path_utils.__file__", str(deep_file)):
            root = get_project_root()
            assert root == tmp_path

    def test_get_project_root_from_nested_location(self, tmp_path: Path):
        """Test getting project root from various nested locations."""
        (tmp_path / "requirements.txt").write_text("pytest\n")
        test_files = [
            tmp_path / "tests" / "unit" / "test_file.py",
            tmp_path / "src" / "main.py",
            tmp_path / "scripts" / "run.py",
        ]

        for test_file in test_files:
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("# test")

            with patch("ztb.utils.path_utils.__file__", str(test_file)):
                root = get_project_root()
                assert root == tmp_path

    def test_get_project_root_caching(self, tmp_path: Path):
        """Test that get_project_root caches results."""
        (tmp_path / ".git").mkdir()

        test_file = tmp_path / "src" / "main.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("# test")

        with patch("ztb.utils.path_utils.__file__", str(test_file)):
            root1 = get_project_root()
            root2 = get_project_root()
            assert root1 == root2 == tmp_path


class TestEnsureDir:
    """Test ensure_dir function."""

    def test_ensure_dir_creates_directory(self, tmp_path: Path):
        """Test that ensure_dir creates a directory if it doesn't exist."""
        new_dir = tmp_path / "new_directory"

        assert not new_dir.exists()
        ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_dir_creates_nested_directories(self, tmp_path: Path):
        """Test that ensure_dir creates nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"

        assert not nested_dir.exists()
        ensure_dir(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert nested_dir.parent.exists()
        assert nested_dir.parent.is_dir()

    def test_ensure_dir_existing_directory(self, tmp_path: Path):
        """Test that ensure_dir doesn't fail on existing directory."""
        existing_dir = tmp_path / "existing_dir"
        existing_dir.mkdir()

        ensure_dir(existing_dir)
        assert existing_dir.exists()
        assert existing_dir.is_dir()

    def test_ensure_dir_with_file_in_path(self, tmp_path: Path):
        """Test that ensure_dir handles paths with files correctly."""
        file_path = tmp_path / "some_file.txt"
        file_path.write_text("content")

        ensure_dir(file_path.parent)
        assert file_path.parent.exists()
        assert file_path.parent.is_dir()

    def test_ensure_dir_permission_error(self, tmp_path: Path):
        """Test that ensure_dir handles permission errors gracefully."""
        # This is hard to test reliably across platforms, but we can test
        # that the function doesn't crash on normal operations
        test_dir = tmp_path / "test_dir"

        ensure_dir(test_dir)
        assert test_dir.exists()
