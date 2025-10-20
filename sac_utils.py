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

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.utils.path_utils import get_project_root

# Get project root using utility
project_root = get_project_root()

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACUtilities:
    """Comprehensive SAC utility toolkit."""

    def __init__(self):
        """Initialize utilities."""
        self.project_root = project_root

    def check_config_consistency(self, config_dir: str = "configs") -> Dict[str, Any]:
        """
        Check configuration file consistency across all config files.

        Args:
            config_dir: Directory containing config files

        Returns:
            Consistency analysis results
        """
        config_path = self.project_root / config_dir
        if not config_path.exists():
            return {"error": f"Config directory not found: {config_path}"}

        logger.info(f"Checking config consistency in: {config_path}")

        all_keys: Dict[str, List[str]] = {}
        all_values: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Read all config files
        for config_file in config_path.glob("*.json"):
            try:
                with open(config_file, encoding="utf-8") as f:
                    data = json.load(f)

                all_keys[config_file.name] = list(data.keys())

                # Record types for each key
                for key, value in data.items():
                    all_values[key][config_file.name] = str(type(value))

            except Exception as e:
                logger.warning(f"Error reading {config_file.name}: {e}")
                continue

        if not all_keys:
            return {"error": "No config files found"}

        # Calculate common and unique keys
        key_sets = [set(keys) for keys in all_keys.values()]
        common_keys = list(set.intersection(*key_sets)) if key_sets else []

        unique_keys = {}
        for name, keys in all_keys.items():
            unique = set(keys) - set(common_keys)
            unique_keys[name] = list(unique)

        # Check type inconsistencies
        type_inconsistencies = {}
        for key, type_dict in all_values.items():
            types = set(type_dict.values())
            if len(types) > 1:
                type_inconsistencies[key] = {
                    "types_found": list(types),
                    "files": type_dict,
                }

        results = {
            "total_files": len(all_keys),
            "common_keys": common_keys,
            "unique_keys": unique_keys,
            "type_inconsistencies": type_inconsistencies,
            "consistency_score": len(common_keys) / max(len(all_keys), 1),
        }

        logger.info(
            f"Config consistency check completed: {len(common_keys)} common keys found"
        )
        return results

    def validate_data_files(self, data_dir: str = "data") -> Dict[str, Any]:
        """
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

        Returns:
            Cleanup results
        """
        logger.info("Starting project cleanup" + (" (dry run)" if dry_run else ""))

        cleanup_targets = [
            "**/*.pyc",
            "**/__pycache__/",
            "**/.pytest_cache/",
            "**/*.tmp",
            "**/*.log",
            "**/node_modules/",  # If any
        ]

        total_removed = 0
        total_size = 0
        removed_files = []

        for pattern in cleanup_targets:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    size = path.stat().st_size
                    total_size += size
                    removed_files.append(
                        {"path": str(path.relative_to(self.project_root)), "size": size}
                    )

                    if not dry_run:
                        try:
                            path.unlink()
                            total_removed += 1
                        except Exception as e:
                            logger.warning(f"Failed to remove {path}: {e}")

                elif path.is_dir() and not dry_run:
                    try:
                        shutil.rmtree(path)
                        total_removed += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove directory {path}: {e}")

        results = {
            "dry_run": dry_run,
            "files_found": len(removed_files),
            "files_removed": total_removed if not dry_run else 0,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "removed_files": removed_files[:100],  # Limit output
        }

        logger.info(
            f"Cleanup completed: {len(removed_files)} files found, {total_removed if not dry_run else 0} removed"
        )
        return results

    def check_code_quality(self) -> Dict[str, Any]:
        """
        Run code quality checks (mypy, flake8, etc.).

        Returns:
            Code quality check results
        """
        logger.info("Running code quality checks")

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

                original_lines = len(lines)
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SAC Utilities Suite")
    parser.add_argument(
        "command",
        choices=["config", "data", "clean", "quality", "fix"],
        help="Utility command to run",
    )
    parser.add_argument(
        "--config-dir", default="config", help="Config directory for consistency check"
    )
    parser.add_argument(
        "--data-dir", default="data", help="Data directory for validation"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (for clean and fix commands)",
    )

    args = parser.parse_args()

    utilities = SACUtilities()

    if args.command == "config":
        results = utilities.check_config_consistency(args.config_dir)

        print("\n" + "=" * 60)
        print("CONFIG CONSISTENCY CHECK")
        print("=" * 60)

        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"📊 Total Files: {results['total_files']}")
            print(f"🔑 Common Keys: {len(results['common_keys'])}")
            print(f"📈 Consistency Score: {results['consistency_score']:.2%}")

            if results["type_inconsistencies"]:
                print(
                    f"\n⚠️  Type Inconsistencies: {len(results['type_inconsistencies'])}"
                )
                for key, info in list(results["type_inconsistencies"].items())[
                    :5
                ]:  # Show first 5
                    print(f"  • {key}: {info['types_found']}")

    elif args.command == "data":
        results = utilities.validate_data_files(args.data_dir)

        print("\n" + "=" * 60)
        print("DATA VALIDATION")
        print("=" * 60)

        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"📊 Total Files: {results['total_files']}")
            print(f"✅ Valid Files: {results['valid_files']}")
            print(f"❌ Invalid Files: {results['invalid_files']}")

            if results["invalid_files"] > 0:
                print("\n⚠️  Invalid Files:")
                for file_info in results["file_details"]:
                    if not file_info["valid"]:
                        print(
                            f"  • {file_info['filename']}: {file_info.get('error', 'Missing required columns')}"
                        )

    elif args.command == "clean":
        results = utilities.clean_project_files(dry_run=not args.apply)

        print("\n" + "=" * 60)
        print("PROJECT CLEANUP")
        print("=" * 60)

        print(f"🔍 Files Found: {results['files_found']}")
        print(".2f")
        print(f"🗑️  Files Removed: {results['files_removed']}")

        if not args.apply:
            print("\n💡 Use --apply to actually remove files")

    elif args.command == "quality":
        results = utilities.check_code_quality()

        print("\n" + "=" * 60)
        print("CODE QUALITY CHECK")
        print("=" * 60)

        for check, info in results.items():
            status = info["status"]
            if status == "completed":
                if check == "mypy":
                    print(f"🔍 MyPy: ✅ {info['errors']} errors")
                elif check == "flake8":
                    print(f"🔍 Flake8: ✅ {info['errors']} issues")
                elif check == "tests":
                    print(
                        f"🧪 Tests: ✅ {info['passed']} passed, ❌ {info['failed']} failed"
                    )
            else:
                print(f"🔍 {check.title()}: ❌ {status}")

    elif args.command == "fix":
        results = utilities.fix_common_issues()

        print("\n" + "=" * 60)
        print("COMMON ISSUE FIXES")
        print("=" * 60)

        print(f"🔧 Files Processed: {results['files_processed']}")

        if results["fixes_applied"]:
            print("\n✅ Fixes Applied:")
            for fix in results["fixes_applied"]:
                print(f"  • {fix}")
        else:
            print("\n✅ No fixes needed")


if __name__ == "__main__":
    main()
