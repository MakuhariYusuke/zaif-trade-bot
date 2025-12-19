#!/usr/bin/env python3
"""
SAC Utilities Suite - Comprehensive utility tools for SAC development

This script provides unified utility capabilities for SAC trading models including:
- Configuration validation and consistency checking
- Data validation and cleaning
- File operations and maintenance
- Code quality checks
- Project health monitoring
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ztb.utils.file_utils import get_project_root

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACUtilities:
    """Comprehensive SAC utility toolkit."""


    def check_config_consistency(self, config_dir: str = "configs") -> Dict[str, Any]:
        """
        Check configuration file consistency across all config files.
        Validate data files for consistency and quality.

        Args:
            data_dir: Directory containing data files

        Returns:
            Data validation results
        """
        data_path = self.project_root / data_dir
        if not data_path.exists():
            return {"error": f"Data directory not found: {data_path}"}

        logger.info(f"Validating data files in: {data_path}")

        validation_results = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "file_details": [],
        }

        for data_file in data_path.glob("*.csv"):
            validation_results["total_files"] += 1

            try:
                import pandas as pd

                df = pd.read_csv(data_file)

                file_info = {
                    "filename": data_file.name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "columns_list": list(df.columns),
                    "missing_values": df.isnull().sum().sum(),
                    "valid": True,
                }

                # Check for required columns (basic check)
                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    file_info["missing_columns"] = missing_cols
                    file_info["valid"] = False

                if file_info["valid"]:
                    validation_results["valid_files"] += 1
                else:
                    validation_results["invalid_files"] += 1

                validation_results["file_details"].append(file_info)

            except Exception as e:
                validation_results["invalid_files"] += 1
                validation_results["file_details"].append(
                    {"filename": data_file.name, "error": str(e), "valid": False}
                )

        logger.info(
            f"Data validation completed: {validation_results['valid_files']}/{validation_results['total_files']} files valid"
        )
        return validation_results

    def clean_project_files(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up temporary and unnecessary files in the project.

        Args:
            dry_run: If True, only report what would be cleaned

        results = {
            "mypy": {"status": "not_run", "errors": 0},
            "flake8": {"status": "not_run", "errors": 0},
            "tests": {"status": "not_run", "passed": 0, "failed": 0},
        }

        # Run mypy
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "ztb/", "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            results["mypy"]["status"] = "completed"
            results["mypy"]["errors"] = len(
                [line for line in result.stdout.split("\n") if "error:" in line]
            )
        except Exception as e:
            results["mypy"]["status"] = f"failed: {e}"

        # Run flake8
        try:
            result = subprocess.run(
                [sys.executable, "-m", "flake8", "ztb/", "--max-line-length=120"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            results["flake8"]["status"] = "completed"
            results["flake8"]["errors"] = (
                len(result.stdout.split("\n")) - 1
            )  # Subtract empty line
        except Exception as e:
            results["flake8"]["status"] = f"failed: {e}"

        # Run tests
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "--tb=no", "-q"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )
            results["tests"]["status"] = "completed"
            # Parse pytest output
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "passed" in line and "failed" in line:
                    parts = line.split(",")
                    for part in parts:
                        if "passed" in part:
                            results["tests"]["passed"] = int(part.strip().split()[0])
                        elif "failed" in part:
                            results["tests"]["failed"] = int(part.strip().split()[0])
        except Exception as e:
            results["tests"]["status"] = f"failed: {e}"

        logger.info("Code quality checks completed")
        return results

    def fix_common_issues(self) -> Dict[str, Any]:
        """
        Fix common code and configuration issues.

        Returns:
            Fix results
        """
        logger.info("Running common issue fixes")

        fixes_applied = []

        # Fix trailing whitespace in Python files
        python_files = list(self.project_root.glob("**/*.py"))
        whitespace_fixed = 0

        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                len(lines)
                fixed_lines = [line.rstrip() + "\n" for line in lines]

                if fixed_lines != lines:
                    with open(py_file, "w", encoding="utf-8") as f:
                        f.writelines(fixed_lines)
                    whitespace_fixed += 1

            except Exception as e:
                logger.warning(f"Failed to fix whitespace in {py_file}: {e}")

        if whitespace_fixed > 0:
            fixes_applied.append(
                f"Fixed trailing whitespace in {whitespace_fixed} files"
            )

        # Fix JSON formatting
        json_files = list(self.project_root.glob("**/*.json"))
        json_fixed = 0

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Rewrite with proper formatting
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                json_fixed += 1

            except Exception as e:
                logger.warning(f"Failed to fix JSON formatting in {json_file}: {e}")

        if json_fixed > 0:
            fixes_applied.append(f"Fixed JSON formatting in {json_fixed} files")

        results = {
            "fixes_applied": fixes_applied,
            "files_processed": whitespace_fixed + json_fixed,
        }

        logger.info(f"Common fixes applied: {len(fixes_applied)} fixes")
        return results




if __name__ == "__main__":
    main()
