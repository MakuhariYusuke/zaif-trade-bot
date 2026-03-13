#!/usr/bin/env python3
"""
Run metadata capture for trading bot executions.

Captures environment and system information for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import Distribution, distributions
from pathlib import Path
from typing import TypedDict

# Allow direct script execution: `python ztb/utils/run_metadata.py ...`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ztb.io.json_io import read_json_object, write_json
from ztb.utils.git_utils import (
    get_git_branch,
    get_git_remote_url,
    get_git_sha,
    get_git_status_lines,
)

logger = logging.getLogger(__name__)

class PackageInfo(TypedDict):
    version: str
    hash: str | None

class GitInfo(TypedDict):
    sha: str
    branch: str
    status: str
    remote_url: str
    is_dirty: str

class RunMetadata:
    """Captures and manages run metadata."""

    _HASHABLE_SUFFIXES = {".py", ".pyi", ".so", ".pyd", ".dll", ".dylib"}

    def __init__(
        self,
        random_seed: int = 42,
        include_package_hashes: bool = False,
        package_hash_file_limit: int = 200,
    ):
        self.random_seed = random_seed
        self.include_package_hashes = include_package_hashes
        self.package_hash_file_limit = max(int(package_hash_file_limit), 1)
        self.metadata: dict[str, object] = {}

    def capture_system_info(self) -> dict[str, object]:
        """Capture system and environment information."""
        now = datetime.now().astimezone()
        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "cpu_model": self._get_cpu_model(),
            "hostname": platform.node(),
            "random_seed": self.random_seed,
            "timestamp": now.isoformat(),
            "timezone": str(now.tzinfo),
            "working_directory": os.getcwd(),
            "environment_variables": self._get_relevant_env_vars(),
        }

    def _get_cpu_model(self) -> str:
        """Get CPU model information."""
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", maxsplit=1)[1].strip()
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            elif platform.system() == "Windows":
                return "Windows CPU"
        except Exception as e:
            logger.debug("GPU info detection failed: %s", e)

        return "Unknown"

    def _get_relevant_env_vars(self) -> dict[str, str]:
        """Get relevant environment variables (without sensitive data)."""
        relevant_vars = (
            "PYTHONPATH",
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "LANG",
            "LC_ALL",
            "TZ",
        )

        env_vars: dict[str, str] = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if not value:
                continue
            if len(value) > 200:
                value = value[:197] + "..."
            env_vars[var] = value
        return env_vars

    def _distribution_name(self, dist: Distribution) -> str:
        name = dist.metadata.get("Name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        fallback = getattr(dist, "name", "")
        return str(fallback).strip() if fallback else ""

    def _distribution_sort_key(self, dist: Distribution) -> str:
        return self._distribution_name(dist).lower()

    def _distribution_paths(self, dist: Distribution, package_name: str) -> list[Path]:
        base_path = Path(dist.locate_file(""))

        top_level_names: list[str] = []
        top_level = dist.read_text("top_level.txt")
        if top_level:
            top_level_names.extend(
                line.strip() for line in top_level.splitlines() if line.strip()
            )

        if not top_level_names:
            top_level_names.append(package_name.replace("-", "_"))

        paths: list[Path] = []
        seen: set[Path] = set()
        for name in top_level_names:
            for candidate in (base_path / name, base_path / f"{name}.py"):
                if candidate.exists() and candidate not in seen:
                    paths.append(candidate)
                    seen.add(candidate)

        return paths

    def _iter_hash_files(self, path: Path):
        if path.is_file():
            yield path
            return
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in self._HASHABLE_SUFFIXES:
                continue
            yield file_path

    def _get_package_hash(self, dist: Distribution, package_name: str) -> str | None:
        """Generate a lightweight package hash from path + file metadata."""
        paths = self._distribution_paths(dist, package_name)
        if not paths:
            return None

        hasher = hashlib.sha256()
        files_hashed = 0

        for root_path in paths:
            for file_path in self._iter_hash_files(root_path):
                try:
                    stat = file_path.stat()
                except OSError:
                    continue

                try:
                    rel_path = str(file_path.relative_to(root_path.parent))
                except ValueError:
                    rel_path = file_path.name
                hasher.update(rel_path.encode("utf-8", errors="ignore"))
                hasher.update(str(stat.st_size).encode("ascii", errors="ignore"))
                hasher.update(str(stat.st_mtime_ns).encode("ascii", errors="ignore"))

                files_hashed += 1
                if files_hashed >= self.package_hash_file_limit:
                    return hasher.hexdigest()[:16]

        if files_hashed == 0:
            return None
        return hasher.hexdigest()[:16]

    def _capture_package_info_via_pip(self) -> dict[str, PackageInfo]:
        packages: dict[str, PackageInfo] = {}
        try:
            # 169# subprocess popup 抑制
            extra_kwargs: dict[str, int] = {}
            if sys.platform == "win32":
                extra_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
                **extra_kwargs,
            )
            if result.returncode != 0:
                return packages
            payload = json.loads(result.stdout)
            if not isinstance(payload, list):
                return packages
            for item in payload:
                if not isinstance(item, dict):
                    continue
                name_obj = item.get("name")
                version_obj = item.get("version")
                if not isinstance(name_obj, str) or not name_obj:
                    continue
                version = version_obj if isinstance(version_obj, str) else "unknown"
                packages[name_obj] = {"version": version, "hash": None}
        except Exception as e:
            logger.debug("pip package info fallback failed: %s", e)
        return packages

    def capture_package_info(self) -> dict[str, PackageInfo]:
        """Capture installed package versions and optional hashes."""
        packages: dict[str, PackageInfo] = {}

        try:
            for dist in sorted(distributions(), key=self._distribution_sort_key):
                package_name = self._distribution_name(dist)
                if not package_name:
                    continue

                info: PackageInfo = {
                    "version": dist.version or "unknown",
                    "hash": None,
                }
                if self.include_package_hashes:
                    info["hash"] = self._get_package_hash(dist, package_name)
                packages[package_name] = info
        except Exception as e:
            logger.debug("package info capture failed, falling back to pip: %s", e)
            return self._capture_package_info_via_pip()

        return packages

    def _sha256_short(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 64), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def capture_config_hashes(
        self, config_files: list[str] | None = None
    ) -> dict[str, str]:
        """Capture hashes of configuration files."""
        if config_files is None:
            config_files = [
                "trade-config.json",
                "config/trade-config.json",
                "venues/zaif.yaml",
                "venues/coincheck.yaml",
            ]

        config_hashes: dict[str, str] = {}
        for config_file in config_files:
            path = Path(config_file)
            if not path.exists():
                continue
            try:
                config_hashes[config_file] = self._sha256_short(path)
            except Exception as e:
                logger.debug("config hash failed for %s: %s", config_file, e)
                config_hashes[config_file] = "error"

        return config_hashes

    def capture_git_info(self) -> GitInfo:
        """Capture git repository information."""
        cwd = Path.cwd()
        # Keep metadata capture fast on large repos: tracked changes only.
        status_lines = get_git_status_lines(
            cwd=cwd, include_untracked=False, max_lines=200
        )
        status_summary = "\n".join(status_lines)
        if len(status_summary) > 200:
            status_summary = status_summary[:200]

        return {
            "sha": get_git_sha(cwd=cwd),
            "branch": get_git_branch(cwd=cwd),
            "status": status_summary,
            "remote_url": get_git_remote_url(cwd=cwd),
            "is_dirty": "true" if status_lines else "false",
        }

    def capture_all_metadata(self) -> dict[str, object]:
        """Capture all metadata."""
        metadata: dict[str, object] = {
            "system": self.capture_system_info(),
            "packages": self.capture_package_info(),
            "git": self.capture_git_info(),
            "run_config": {
                "random_seed": self.random_seed,
                "captured_at": datetime.now().isoformat(),
                "include_package_hashes": self.include_package_hashes,
            },
        }

        self.metadata = metadata
        return metadata

    def save_to_file(self, file_path: str) -> None:
        """Save metadata to JSON file."""
        write_json(file_path, self.metadata, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> "RunMetadata":
        """Load metadata from JSON file."""
        instance = cls()
        try:
            instance.metadata = read_json_object(Path(file_path))
        except Exception as e:
            logger.debug("metadata load failed: %s", e)
            instance.metadata = {}
        return instance

    def get_summary(self) -> str:
        """Get a human-readable summary of the metadata."""
        if not self.metadata:
            return "No metadata captured"

        system_obj = self.metadata.get("system")
        git_obj = self.metadata.get("git")
        packages_obj = self.metadata.get("packages")

        system = system_obj if isinstance(system_obj, dict) else {}
        git = git_obj if isinstance(git_obj, dict) else {}
        packages = packages_obj if isinstance(packages_obj, dict) else {}

        git_sha_obj = git.get("sha")
        git_sha = git_sha_obj[:8] if isinstance(git_sha_obj, str) else "Unknown"

        summary = [
            f"Python: {system.get('python_version', 'Unknown')}",
            f"OS: {system.get('os', 'Unknown')} {system.get('os_version', 'Unknown')}",
            f"CPU: {system.get('cpu_model', 'Unknown')}",
            f"Git SHA: {git_sha}",
            f"Branch: {git.get('branch', 'Unknown')}",
            f"Packages: {len(packages)} installed",
            f"Random Seed: {system.get('random_seed', 'Unknown')}",
            f"Timestamp: {system.get('timestamp', 'Unknown')}",
        ]

        return "\n".join(summary)

    def to_dict(self) -> dict[str, object]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "random_seed": self.random_seed,
            "metadata": self.metadata,
        }

def capture_run_metadata(
    output_path: str,
    random_seed: int = 42,
    include_package_hashes: bool = False,
    package_hash_file_limit: int = 200,
) -> RunMetadata:
    """Convenience function to capture and save run metadata."""
    metadata = RunMetadata(
        random_seed=random_seed,
        include_package_hashes=include_package_hashes,
        package_hash_file_limit=package_hash_file_limit,
    )
    metadata.capture_all_metadata()
    metadata.save_to_file(output_path)
    return metadata

if __name__ == "__main__":
    # CLI usage

    from ztb.utils.cli_common import CLIFormatter, CLIValidator, create_standard_parser

    parser = create_standard_parser("Capture run metadata")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument(
        "--seed",
        type=lambda x: CLIValidator.validate_positive_int(x, "seed"),
        default=42,
        help=CLIFormatter.format_help("Random seed", 42),
    )
    parser.add_argument(
        "--include-package-hashes",
        action="store_true",
        help="Enable package hash capture (disabled by default for performance)",
    )
    parser.add_argument(
        "--package-hash-file-limit",
        type=lambda x: CLIValidator.validate_positive_int(x, "package_hash_file_limit"),
        default=200,
        help=CLIFormatter.format_help("Max files to sample per package hash", 200),
    )

    args = parser.parse_args()

    metadata = capture_run_metadata(
        args.output,
        args.seed,
        include_package_hashes=args.include_package_hashes,
        package_hash_file_limit=args.package_hash_file_limit,
    )
    print("Run metadata captured:")
    print(metadata.get_summary())
