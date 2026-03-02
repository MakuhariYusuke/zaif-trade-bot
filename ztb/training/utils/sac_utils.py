#!/usr/bin/env python3
"""
SAC utility tooling for configuration/data validation and maintenance tasks.

This module is intentionally lightweight and script-friendly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

# Allow direct script execution: `python ztb/training/utils/sac_utils.py ...`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ztb.io.json_io import read_json, read_json_object, write_json

logger = logging.getLogger(__name__)

class CommandResult(TypedDict):
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

class SACUtilities:
    """Comprehensive SAC utility toolkit."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self._ignored_roots = {".git", ".venv", "venv", "node_modules"}

    def _is_ignored(self, path: Path) -> bool:
        return any(part in self._ignored_roots for part in path.parts)

    def _scan_roots(self) -> list[Path]:
        roots: list[Path] = []
        for name in ("ztb", "tests", "scripts", "configs"):
            candidate = self.project_root / name
            if candidate.exists():
                roots.append(candidate)
        if not roots:
            roots.append(self.project_root)
        return roots

    def _relative_path_text(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def _run_command(
        self, cmd: list[str], timeout: int = 300, cwd: Path | None = None
    ) -> CommandResult:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or self.project_root,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -1,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or f"Command timed out after {timeout}s",
                "timed_out": True,
            }
        except Exception as exc:  # pragma: no cover - defensive path
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
            }

    def check_config_consistency(
        self, config_dir: str = "configs", max_details: int = 300
    ) -> dict[str, object]:
        """
        Check JSON config files for parseability and basic contract consistency.
        """
        config_path = self.project_root / config_dir
        if not config_path.exists():
            return {"error": f"Config directory not found: {config_path}"}

        result: dict[str, object] = {
            "config_dir": str(config_path),
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "warning_files": 0,
            "file_details": [],
        }

        expected_key_groups = ("training", "environment", "sac_hyperparameters", "ppo_hyperparameters")
        details: list[dict[str, object]] = []
        omitted_details = 0

        for config_file in sorted(config_path.rglob("*.json")):
            if self._is_ignored(config_file):
                continue

            result["total_files"] = int(result["total_files"]) + 1
            item: dict[str, object] = {"file": str(config_file.relative_to(self.project_root))}
            try:
                payload = read_json_object(config_file)
                keys = sorted(payload.keys())
                item["top_level_keys"] = keys
                item["valid"] = True

                has_expected_group = any(key in payload for key in expected_key_groups)
                if not has_expected_group:
                    item["warning"] = "No common training/environment key group found"
                    result["warning_files"] = int(result["warning_files"]) + 1

                result["valid_files"] = int(result["valid_files"]) + 1
            except Exception as exc:
                item["valid"] = False
                item["error"] = str(exc)
                result["invalid_files"] = int(result["invalid_files"]) + 1

            if len(details) < max_details:
                details.append(item)
            else:
                omitted_details += 1

        result["file_details"] = details
        result["file_details_omitted"] = omitted_details
        result["max_details"] = max_details
        logger.info(
            "Config consistency check completed: %s valid / %s total",
            result["valid_files"],
            result["total_files"],
        )
        return result

    def validate_data_files(
        self, data_dir: str = "data", max_files: int = 200
    ) -> dict[str, object]:
        """
        Validate CSV data files for basic schema/quality checks.
        """
        try:
            from ztb.io.data_loader import DataLoader
        except ModuleNotFoundError as exc:
            return {
                "error": "Data validation requires optional dependencies",
                "missing_dependency": str(exc),
            }

        data_path = self.project_root / data_dir
        if not data_path.exists():
            return {"error": f"Data directory not found: {data_path}"}

        required_cols = ("timestamp", "open", "high", "low", "close", "volume")
        validation_results: dict[str, object] = {
            "data_dir": str(data_path),
            "max_files": max_files,
            "total_files": 0,
            "validated_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "file_details": [],
        }

        details: list[dict[str, object]] = []
        csv_files = [p for p in sorted(data_path.rglob("*.csv")) if not self._is_ignored(p)]
        validation_results["total_files"] = len(csv_files)

        for data_file in csv_files[:max_files]:
            info: dict[str, object] = {"file": str(data_file.relative_to(self.project_root))}
            validation_results["validated_files"] = int(validation_results["validated_files"]) + 1
            try:
                df = DataLoader.load_csv_strict(data_file)
                missing_cols = [col for col in required_cols if col not in df.columns]
                info.update(
                    {
                        "rows": int(len(df)),
                        "columns": int(len(df.columns)),
                        "missing_columns": missing_cols,
                        "null_cells": int(df.isnull().sum().sum()),
                        "valid": len(missing_cols) == 0,
                    }
                )
                if info["valid"]:
                    validation_results["valid_files"] = int(validation_results["valid_files"]) + 1
                else:
                    validation_results["invalid_files"] = int(validation_results["invalid_files"]) + 1
            except Exception as exc:
                info["valid"] = False
                info["error"] = str(exc)
                validation_results["invalid_files"] = int(validation_results["invalid_files"]) + 1
            details.append(info)

        validation_results["file_details"] = details
        logger.info(
            "Data validation completed: %s valid / %s checked",
            validation_results["valid_files"],
            validation_results["validated_files"],
        )
        return validation_results

    def run_code_quality_checks(self) -> dict[str, object]:
        """
        Run mypy/flake8/pytest checks and return normalized result payload.
        """
        checks: list[tuple[str, list[str], int]] = [
            ("mypy", [sys.executable, "-m", "mypy", "ztb/", "--ignore-missing-imports"], 900),
            ("flake8", [sys.executable, "-m", "flake8", "ztb/", "--max-line-length=120"], 900),
            ("pytest", [sys.executable, "-m", "pytest", "tests/", "--tb=no", "-q"], 1800),
        ]
        check_results: list[dict[str, object]] = []
        all_passed = True

        for name, cmd, timeout in checks:
            result = self._run_command(cmd, timeout=timeout)
            passed = result["returncode"] == 0
            all_passed = all_passed and passed

            entry: dict[str, object] = {
                "name": name,
                "passed": passed,
                "returncode": result["returncode"],
                "timed_out": result["timed_out"],
            }
            if name == "pytest":
                summary_line = ""
                for line in result["stdout"].splitlines():
                    if " passed" in line or " failed" in line:
                        summary_line = line.strip()
                entry["summary"] = summary_line
            else:
                entry["error_lines"] = sum(
                    1 for line in result["stdout"].splitlines() if "error" in line.lower()
                )
            if result["stderr"]:
                entry["stderr_excerpt"] = result["stderr"][:400]
            check_results.append(entry)

        return {"all_passed": all_passed, "checks": check_results}

    def clean_project_files(
        self, dry_run: bool = True, max_scan_seconds: float = 20.0
    ) -> dict[str, object]:
        """
        Clean temporary/cache artifacts under project root.
        """
        removable_dirs = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        removable_suffixes = {".pyc", ".pyo"}
        targets: list[Path] = []

        started_at = time.time()
        truncated = False
        for source_root in self._scan_roots():
            for root, dirs, files in os.walk(source_root):
                if max_scan_seconds > 0 and (time.time() - started_at) > max_scan_seconds:
                    truncated = True
                    break
                root_path = Path(root)
                # prune heavy/irrelevant roots early
                dirs[:] = [d for d in dirs if d not in self._ignored_roots]

                for dirname in dirs:
                    if dirname in removable_dirs:
                        targets.append(root_path / dirname)

                for filename in files:
                    suffix = Path(filename).suffix
                    if suffix in removable_suffixes:
                        targets.append(root_path / filename)
            if truncated:
                break

        unique_targets = sorted({path.resolve() for path in targets}, key=lambda p: str(p))
        sample_limit = 200
        cleaned_count = 0
        removed_sample: list[str] = []
        failed: list[dict[str, str]] = []
        failed_count = 0

        for target in unique_targets:
            rel = self._relative_path_text(target)
            if dry_run:
                cleaned_count += 1
                if len(removed_sample) < sample_limit:
                    removed_sample.append(rel)
                continue
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                cleaned_count += 1
                if len(removed_sample) < sample_limit:
                    removed_sample.append(rel)
            except Exception as exc:
                failed_count += 1
                if len(failed) < sample_limit:
                    failed.append({"path": rel, "error": str(exc)})

        return {
            "dry_run": dry_run,
            "truncated": truncated,
            "scan_seconds": round(time.time() - started_at, 3),
            "max_scan_seconds": max_scan_seconds,
            "candidates": len(unique_targets),
            "cleaned": cleaned_count,
            "failed": failed_count,
            "cleaned_paths_sample": removed_sample,
            "cleaned_paths_omitted": max(cleaned_count - len(removed_sample), 0),
            "failed_paths": failed,
            "failed_paths_omitted": max(failed_count - len(failed), 0),
        }

    def fix_common_issues(
        self, dry_run: bool = True, max_files: int = 3000
    ) -> dict[str, object]:
        """
        Fix trailing whitespace in Python files and JSON formatting issues.
        """
        py_changed = 0
        json_changed = 0
        py_examined = 0
        json_examined = 0
        examined_total = 0
        truncated = False
        failed: list[dict[str, str]] = []

        for source_root in self._scan_roots():
            for py_file in source_root.rglob("*.py"):
                if self._is_ignored(py_file):
                    continue
                if max_files > 0 and examined_total >= max_files:
                    truncated = True
                    break
                examined_total += 1
                py_examined += 1
                try:
                    original = py_file.read_text(encoding="utf-8")
                    normalized = "".join(line.rstrip() + "\n" for line in original.splitlines())
                    if normalized == original:
                        continue
                    py_changed += 1
                    if not dry_run:
                        py_file.write_text(normalized, encoding="utf-8")
                except Exception as exc:
                    failed.append({"path": self._relative_path_text(py_file), "error": str(exc)})
            if truncated:
                break

        if not truncated:
            for source_root in self._scan_roots():
                for json_file in source_root.rglob("*.json"):
                    if self._is_ignored(json_file):
                        continue
                    if max_files > 0 and examined_total >= max_files:
                        truncated = True
                        break
                    examined_total += 1
                    json_examined += 1
                    try:
                        payload = read_json(json_file)
                        formatted = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
                        current = json_file.read_text(encoding="utf-8")
                        if formatted == current:
                            continue
                        json_changed += 1
                        if not dry_run:
                            write_json(
                                json_file, payload, indent=2, ensure_ascii=False, default=str
                            )
                    except Exception as exc:
                        failed.append(
                            {"path": self._relative_path_text(json_file), "error": str(exc)}
                        )
                if truncated:
                    break

        failed_sample_limit = 200

        return {
            "dry_run": dry_run,
            "truncated": truncated,
            "max_files": max_files,
            "python_files_examined": py_examined,
            "json_files_examined": json_examined,
            "files_examined_total": examined_total,
            "python_files_changed": py_changed,
            "json_files_changed": json_changed,
            "failed": failed[:failed_sample_limit],
            "failed_omitted": max(len(failed) - failed_sample_limit, 0),
        }

def _emit_result(result: dict[str, object], json_out: str | None) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if json_out:
        write_json(json_out, result, indent=2, ensure_ascii=False, default=str)
        logger.info("Wrote report to %s", json_out)

def main() -> int:
    parser = argparse.ArgumentParser(description="SAC utility toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_cfg = subparsers.add_parser("check-config", help="Validate config files")
    check_cfg.add_argument("--config-dir", default="configs")
    check_cfg.add_argument("--max-details", type=int, default=300)
    check_cfg.add_argument("--json-out")

    check_data = subparsers.add_parser("validate-data", help="Validate CSV data files")
    check_data.add_argument("--data-dir", default="data")
    check_data.add_argument("--max-files", type=int, default=200)
    check_data.add_argument("--json-out")

    quality = subparsers.add_parser("quality-checks", help="Run mypy/flake8/pytest")
    quality.add_argument("--json-out")

    clean = subparsers.add_parser("clean", help="Clean temp/cache files")
    clean.add_argument("--apply", action="store_true")
    clean.add_argument("--max-scan-seconds", type=float, default=20.0)
    clean.add_argument("--json-out")

    fix = subparsers.add_parser("fix-common", help="Fix common formatting issues")
    fix.add_argument("--apply", action="store_true")
    fix.add_argument("--max-files", type=int, default=3000)
    fix.add_argument("--json-out")

    args = parser.parse_args()
    util = SACUtilities()

    if args.command == "check-config":
        result = util.check_config_consistency(
            config_dir=args.config_dir, max_details=args.max_details
        )
        _emit_result(result, args.json_out)
        return 0 if "error" not in result else 1

    if args.command == "validate-data":
        result = util.validate_data_files(data_dir=args.data_dir, max_files=args.max_files)
        _emit_result(result, args.json_out)
        return 0 if "error" not in result else 1

    if args.command == "quality-checks":
        result = util.run_code_quality_checks()
        _emit_result(result, args.json_out)
        return 0 if bool(result.get("all_passed")) else 1

    if args.command == "clean":
        result = util.clean_project_files(
            dry_run=not args.apply, max_scan_seconds=args.max_scan_seconds
        )
        _emit_result(result, args.json_out)
        return 0

    if args.command == "fix-common":
        result = util.fix_common_issues(
            dry_run=not args.apply, max_files=args.max_files
        )
        _emit_result(result, args.json_out)
        return 0

    return 1

if __name__ == "__main__":
    sys.exit(main())
