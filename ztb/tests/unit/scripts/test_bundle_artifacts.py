import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

from ztb.ops.rollup.bundle_artifacts import create_bundle, should_exclude_file


def test_should_exclude_file():
    test_path = Path("test.log")
    assert should_exclude_file(test_path, True) == True
    assert should_exclude_file(test_path, False) == False

    test_path2 = Path("test.txt")
    assert should_exclude_file(test_path2, True) == False


def test_create_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp) / "artifacts" / "test123"
        artifacts_dir.mkdir(parents=True)

        # Create test files
        (artifacts_dir / "metrics.json").write_text('{"test": "data"}')
        (artifacts_dir / "summary.json").write_text('{"summary": "data"}')
        (artifacts_dir / "train.log").write_text("log data")

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            create_bundle("test123", False)
        finally:
            os.chdir(old_cwd)

        bundle_path = artifacts_dir / "bundle.zip"
        hash_path = artifacts_dir / "bundle.sha256"

        assert bundle_path.exists()
        assert hash_path.exists()

        # Check ZIP contents
        with zipfile.ZipFile(bundle_path, "r") as zipf:
            files = zipf.namelist()
            assert "metrics.json" in files
            assert "summary.json" in files
            assert "train.log" in files

        # Check SHA256
        with open(hash_path, "r") as f:
            hash_line = f.read().strip()
            expected_hash = hash_line.split()[0]

        sha256 = hashlib.sha256()
        with open(bundle_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        assert sha256.hexdigest() == expected_hash


def test_create_bundle_exclude_logs():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp) / "artifacts" / "test123"
        artifacts_dir.mkdir(parents=True)

        # Create test files
        (artifacts_dir / "metrics.json").write_text('{"test": "data"}')
        (artifacts_dir / "train.log").write_text("log data")

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            create_bundle("test123", True)
        finally:
            os.chdir(old_cwd)

        bundle_path = artifacts_dir / "bundle.zip"

        # Check ZIP contents
        with zipfile.ZipFile(bundle_path, "r") as zipf:
            files = zipf.namelist()
            assert "metrics.json" in files
            assert "train.log" not in files


def test_create_bundle_missing_dir():
    with patch("sys.exit") as mock_exit, patch("builtins.print") as mock_print:
        # Since artifacts/nonexistent doesn't exist, it should exit
        try:
            create_bundle("nonexistent", False)
        except SystemExit:
            pass  # Expected
        mock_exit.assert_called_with(1)


def test_main():
    with patch("ztb.scripts.bundle_artifacts.create_bundle") as mock_create:
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(correlation_id="test", exclude_logs=True),
        ):
            from ztb.ops.rollup.bundle_artifacts import main

            main()
            mock_create.assert_called_with("test", True)
