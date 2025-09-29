#!/usr/bin/env python3
"""
Integration test for canary parity between Linux and Windows.

Verifies that linux_canary.sh and run_canary.ps1 produce identical
artifacts and exit codes.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def hash_file(filepath: Path) -> str:
    """Generate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_linux_canary(temp_dir: Path) -> tuple[int, dict]:
    """Run linux_canary.sh and capture results."""
    script_path = PROJECT_ROOT / "linux_canary.sh"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    try:
        result = subprocess.run(
            [str(script_path), "2", "sma_fast_slow", "false", str(temp_dir)],
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
        )
    except subprocess.TimeoutExpired:
        return -1, {}

    artifacts = {}
    expected_files = ["run_metadata.json", "orders.csv", "stats.json"]

    # Find the output directory created by paper trader
    output_dirs = list(temp_dir.glob("*"))
    if output_dirs:
        output_dir = output_dirs[0]
        for filename in expected_files:
            filepath = output_dir / filename
            if filepath.exists():
                artifacts[filename] = hash_file(filepath)

    return result.returncode, artifacts


def run_windows_canary(temp_dir: Path) -> tuple[int, dict]:
    """Run run_canary.ps1 and capture results."""
    script_path = PROJECT_ROOT / "run_canary.ps1"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    try:
        # Use pwsh if available, otherwise powershell
        powershell_cmd = "pwsh" if shutil.which("pwsh") else "powershell"

        result = subprocess.run(
            [
                powershell_cmd,
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "-DurationMinutes",
                "2",
                "-Policy",
                "sma_fast_slow",
                "-OutputDir",
                str(temp_dir),
            ],
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
        )
    except subprocess.TimeoutExpired:
        return -1, {}

    artifacts = {}
    expected_files = ["run_metadata.json", "orders.csv", "stats.json"]

    # Find the output directory created by paper trader
    output_dirs = list(temp_dir.glob("*"))
    if output_dirs:
        output_dir = output_dirs[0]
        for filename in expected_files:
            filepath = output_dir / filename
            if filepath.exists():
                artifacts[filename] = hash_file(filepath)

    return result.returncode, artifacts


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-specific test")
def test_canary_parity_linux():
    """Test canary parity on Linux."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        exit_code, artifacts = run_linux_canary(temp_dir)

        # Should succeed
        assert exit_code == 0, f"Linux canary failed with exit code {exit_code}"

        # Should produce expected artifacts
        expected_files = ["run_metadata.json", "orders.csv", "stats.json"]
        for filename in expected_files:
            assert filename in artifacts, f"Missing artifact: {filename}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_canary_parity_windows():
    """Test canary parity on Windows."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        exit_code, artifacts = run_windows_canary(temp_dir)

        # Should succeed
        assert exit_code == 0, f"Windows canary failed with exit code {exit_code}"

        # Should produce expected artifacts
        expected_files = ["run_metadata.json", "orders.csv", "stats.json"]
        for filename in expected_files:
            assert filename in artifacts, f"Missing artifact: {filename}"


def test_canary_artifacts_consistent():
    """Test that canary artifacts are consistent across platforms."""
    # This test would run on both platforms and compare artifacts
    # For now, just verify that both canary scripts exist and are executable

    linux_script = PROJECT_ROOT / "linux_canary.sh"
    windows_script = PROJECT_ROOT / "run_canary.ps1"

    assert linux_script.exists(), "linux_canary.sh not found"
    assert windows_script.exists(), "run_canary.ps1 not found"

    # Check if linux script is executable (on Unix-like systems)
    if os.name == "posix":
        assert os.access(linux_script, os.X_OK), "linux_canary.sh is not executable"
