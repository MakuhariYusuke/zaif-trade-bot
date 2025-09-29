#!/usr/bin/env python3
"""
Run metadata capture for trading bot executions.

Captures environment and system information for reproducibility.
"""

import platform
import sys
import hashlib
import subprocess
import os
from typing import Dict, Any
from datetime import datetime
import json
import pkg_resources

from .observability import generate_correlation_id


class RunMetadata:
    """Captures and manages run metadata."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.metadata = {}

    def capture_system_info(self) -> Dict[str, Any]:
        """Capture system and environment information."""
        info = {
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
            "timestamp": datetime.now().isoformat(),
            "timezone": str(datetime.now().astimezone().tzinfo),
            "working_directory": os.getcwd(),
            "environment_variables": self._get_relevant_env_vars()
        }

        return info

    def _get_cpu_model(self) -> str:
        """Get CPU model information."""
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":")[1].strip()
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return result.stdout.strip()
            elif platform.system() == "Windows":
                # Skip slow wmic command
                return "Windows CPU"
        except Exception:
            pass

        return "Unknown"

    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables (without sensitive data)."""
        relevant_vars = [
            "PYTHONPATH",
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "LANG",
            "LC_ALL",
            "TZ"
        ]

        env_vars = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value:
                # Truncate long paths
                if len(value) > 200:
                    value = value[:197] + "..."
                env_vars[var] = value

        return env_vars

    def capture_git_info(self) -> Dict[str, str]:
        """Capture git repository information."""
        git_info = {
            "sha": "unknown",
            "branch": "unknown",
            "status": "unknown",
            "remote_url": "unknown"
        }

        try:
            # Get current commit SHA
            result = subprocess.run(["git", "rev-parse", "HEAD"],
                                  capture_output=True, text=True, cwd=os.getcwd(), timeout=10)
            if result.returncode == 0:
                git_info["sha"] = result.stdout.strip()

            # Get current branch
            result = subprocess.run(["git", "branch", "--show-current"],
                                  capture_output=True, text=True, cwd=os.getcwd(), timeout=10)
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()

            # Get remote URL
            result = subprocess.run(["git", "remote", "get-url", "origin"],
                                  capture_output=True, text=True, cwd=os.getcwd(), timeout=10)
            if result.returncode == 0:
                git_info["remote_url"] = result.stdout.strip()

        except Exception:
            pass

        return git_info

    def capture_package_info(self) -> Dict[str, str]:
        """Capture installed package versions and hashes."""
        packages = {}

        try:
            # Get all installed packages
            for dist in pkg_resources.working_set:
                package_name = dist.project_name
                version = dist.version

                # Create a hash of the package files for change detection
                try:
                    package_hash = self._get_package_hash(dist)
                    packages[package_name] = {
                        "version": version,
                        "hash": package_hash
                    }
                except Exception:
                    # Fallback to just version
                    packages[package_name] = {
                        "version": version,
                        "hash": None
                    }
        except Exception:
            # If pkg_resources fails, try pip
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    pip_packages = json.loads(result.stdout)
                    for pkg in pip_packages:
                        packages[pkg["name"]] = {
                            "version": pkg["version"],
                            "hash": None
                        }
            except Exception:
                pass

        return packages

    def capture_config_hashes(self, config_files: list = None) -> Dict[str, str]:
        """Capture hashes of configuration files."""
        if config_files is None:
            config_files = [
                "trade-config.json",
                "config/trade-config.json",
                "venues/zaif.yaml",
                "venues/coincheck.yaml"
            ]

        config_hashes = {}
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'rb') as f:
                        hasher = hashlib.sha256()
                        hasher.update(f.read())
                        config_hashes[config_file] = hasher.hexdigest()[:16]
                except Exception:
                    config_hashes[config_file] = "error"

        return config_hashes

    def capture_all_metadata(self) -> Dict[str, Any]:
        """Capture all metadata in one call."""
        metadata = {
            "correlation_id": generate_correlation_id(),
            "system": self.capture_system_info(),
            "git": self.capture_git_info(),
            # "packages": self.capture_package_info(),  # Skip slow package capture
            "config_hashes": self.capture_config_hashes(),
            "run_config": {
                "random_seed": self.random_seed,
                "captured_at": datetime.now().isoformat()
            }
        }

        self.metadata = metadata
        return metadata

    def _get_package_hash(self, dist) -> str:
        """Generate hash of package files."""
        hasher = hashlib.sha256()

        try:
            # Get package location
            location = dist.location
            if location and os.path.isdir(location):
                # Walk through package files
                for root, dirs, files in os.walk(location):
                    for file in files:
                        if file.endswith(('.py', '.pyc', '.pyo')):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'rb') as f:
                                    hasher.update(f.read())
                            except Exception:
                                continue
        except Exception:
            pass

        return hasher.hexdigest()[:16]  # Short hash

    def capture_git_info(self) -> Dict[str, str]:
        """Capture git repository information."""
        git_info = {
            "sha": None,
            "branch": None,
            "status": None,
            "remote_url": None,
            "is_dirty": None
        }

        try:
            # Get current commit SHA
            result = subprocess.run(["git", "rev-parse", "HEAD"],
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                git_info["sha"] = result.stdout.strip()

            # Get current branch
            result = subprocess.run(["git", "branch", "--show-current"],
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()

            # Check if working directory is dirty
            result = subprocess.run(["git", "status", "--porcelain"],
                                  capture_output=True, text=True, cwd=os.getcwd())
            git_info["is_dirty"] = len(result.stdout.strip()) > 0

            # Get remote URL
            result = subprocess.run(["git", "remote", "get-url", "origin"],
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                git_info["remote_url"] = result.stdout.strip()

            # Get status summary
            result = subprocess.run(["git", "status", "--short"],
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                git_info["status"] = result.stdout.strip()[:200]  # Truncate

        except Exception:
            pass

        return git_info

    def capture_all_metadata(self) -> Dict[str, Any]:
        """Capture all metadata."""
        metadata = {
            "system": self.capture_system_info(),
            "packages": self.capture_package_info(),
            "git": self.capture_git_info(),
            "run_config": {
                "random_seed": self.random_seed,
                "captured_at": datetime.now().isoformat()
            }
        }

        self.metadata = metadata
        return metadata

    def save_to_file(self, file_path: str):
        """Save metadata to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'RunMetadata':
        """Load metadata from JSON file."""
        instance = cls()
        with open(file_path, 'r', encoding='utf-8') as f:
            instance.metadata = json.load(f)
        return instance

    def get_summary(self) -> str:
        """Get a human-readable summary of the metadata."""
        if not self.metadata:
            return "No metadata captured"

        system = self.metadata.get("system", {})
        git = self.metadata.get("git", {})
        packages = self.metadata.get("packages", {})

        summary = []
        summary.append(f"Python: {system.get('python_version', 'Unknown')}")
        summary.append(f"OS: {system.get('os', 'Unknown')} {system.get('os_version', 'Unknown')}")
        summary.append(f"CPU: {system.get('cpu_model', 'Unknown')}")
        summary.append(f"Git SHA: {git.get('sha', 'Unknown')[:8] if git.get('sha') else 'Unknown'}")
        summary.append(f"Branch: {git.get('branch', 'Unknown')}")
        summary.append(f"Packages: {len(packages)} installed")
        summary.append(f"Random Seed: {system.get('random_seed', 'Unknown')}")
        summary.append(f"Timestamp: {system.get('timestamp', 'Unknown')}")

        return "\n".join(summary)


def capture_run_metadata(output_path: str, random_seed: int = 42) -> RunMetadata:
    """Convenience function to capture and save run metadata."""
    metadata = RunMetadata(random_seed=random_seed)
    metadata.capture_all_metadata()
    metadata.save_to_file(output_path)
    return metadata


if __name__ == '__main__':
    # CLI usage
    import argparse

    parser = argparse.ArgumentParser(description='Capture run metadata')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    metadata = capture_run_metadata(args.output, args.seed)
    print("Run metadata captured:")
    print(metadata.get_summary())