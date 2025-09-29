import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

from ztb.scripts.validate_artifacts import (
    load_expectations,
    validate_artifacts,
    validate_file_presence,
    validate_file_sizes,
)


def test_load_expectations():
    expectations = load_expectations()
    assert "required_files" in expectations
    assert "summary.json" in expectations["required_files"]


def test_validate_file_presence():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp)

        # Create required files
        (artifacts_dir / "summary.json").write_text("{}")
        (artifacts_dir / "metrics.json").write_text("{}")
        (artifacts_dir / "reports").mkdir()

        expectations = {"required_files": ["summary.json", "metrics.json", "reports/"]}
        errors = validate_file_presence(artifacts_dir, expectations)
        assert len(errors) == 0


def test_validate_file_presence_missing():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp)

        expectations = {"required_files": ["summary.json"]}
        errors = validate_file_presence(artifacts_dir, expectations)
        assert len(errors) == 1
        assert "Required file missing" in errors[0]


def test_validate_file_sizes():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp)

        # Small files
        (artifacts_dir / "summary.json").write_text("x")
        (artifacts_dir / "metrics.json").write_text("x")

        warnings = validate_file_sizes(artifacts_dir)
        assert len(warnings) == 2
        assert "very small" in warnings[0]


def test_validate_artifacts_valid():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp) / "artifacts" / "test123"
        artifacts_dir.mkdir(parents=True)

        # Create required files
        (artifacts_dir / "summary.json").write_text(
            '{"correlation_id": "test123", "status": "completed"}'
        )
        (artifacts_dir / "metrics.json").write_text('{"steps": 1000}')
        (artifacts_dir / "reports").mkdir()

        with patch(
            "ztb.scripts.validate_artifacts.load_expectations",
            return_value={
                "required_files": ["summary.json", "metrics.json", "reports/"],
                "summary_schema": {},
                "metrics_schema": {},
            },
        ):
            import os

            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                result = validate_artifacts("test123", False)
            finally:
                os.chdir(old_cwd)

        assert result["valid"] == True
        assert len(result["errors"]) == 0


def test_validate_artifacts_missing():
    with tempfile.TemporaryDirectory() as tmp:
        # No artifacts dir
        result = validate_artifacts("nonexistent", False)

        assert result["valid"] == False
        assert len(result["errors"]) == 3
        assert "Required file missing" in result["errors"][0]


def test_validate_artifacts_invalid_schema():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp) / "artifacts" / "test123"
        artifacts_dir.mkdir(parents=True)

        # Create files with invalid schema
        (artifacts_dir / "summary.json").write_text('{"invalid": "data"}')
        (artifacts_dir / "metrics.json").write_text('{"steps": 1000}')
        (artifacts_dir / "reports").mkdir()

        with patch(
            "ztb.scripts.validate_artifacts.load_expectations",
            return_value={
                "required_files": ["summary.json", "metrics.json", "reports/"],
                "summary_schema": {"type": "object", "required": ["correlation_id"]},
                "metrics_schema": {},
            },
        ):
            result = validate_artifacts("test123", True)

        assert result["valid"] == False
        assert len(result["errors"]) > 0


def test_main_valid():
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp) / "artifacts" / "test123"
        artifacts_dir.mkdir(parents=True)

        (artifacts_dir / "summary.json").write_text('{"correlation_id": "test123"}')
        (artifacts_dir / "metrics.json").write_text('{"steps": 100}')
        (artifacts_dir / "reports").mkdir()

        with (
            patch(
                "ztb.scripts.validate_artifacts.load_expectations",
                return_value={
                    "required_files": ["summary.json", "metrics.json", "reports/"],
                    "summary_schema": {},
                    "metrics_schema": {},
                },
            ),
            patch(
                "argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(correlation_id="test123", strict=False),
            ),
            patch("builtins.print") as mock_print,
            patch("sys.exit") as mock_exit,
        ):
            import os

            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                from ztb.scripts.validate_artifacts import main

                main()
            finally:
                os.chdir(old_cwd)

            mock_print.assert_any_call("✓ Artifacts valid for test123")
            mock_exit.assert_not_called()


def test_main_invalid():
    with (
        patch(
            "ztb.scripts.validate_artifacts.validate_artifacts",
            return_value={
                "valid": False,
                "errors": ["missing file"],
                "warnings": [],
                "correlation_id": "test123",
            },
        ),
        patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(correlation_id="test123", strict=False),
        ),
        patch("builtins.print") as mock_print,
        patch("sys.exit") as mock_exit,
    ):
        from ztb.scripts.validate_artifacts import main

        main()

        mock_print.assert_any_call("✗ Artifacts invalid for test123")
        mock_exit.assert_called_with(1)
