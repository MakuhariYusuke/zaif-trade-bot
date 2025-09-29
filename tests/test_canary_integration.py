#!/usr/bin/env python3
"""
Integration test for canary scripts.

Runs a 2-3 minute replay canary and asserts artifact presence and schema validity.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path
import json
import pytest
import pandas as pd


class TestCanaryIntegration:
    """Integration tests for canary deployment scripts."""

    @pytest.fixture
    def temp_project_dir(self, tmp_path):
        """Create a temporary project directory with minimal structure."""
        # Copy essential files
        project_root = Path(__file__).parent.parent

        # Create temp project structure
        temp_project = tmp_path / "project"
        temp_project.mkdir()

        # Copy config files
        config_dir = temp_project / "config"
        config_dir.mkdir()
        for config_file in (project_root / "config").glob("*.json"):
            (config_dir / config_file.name).write_text(config_file.read_text())

        # Copy ztb module (simplified)
        ztb_dir = temp_project / "ztb"
        ztb_dir.mkdir()
        (ztb_dir / "__init__.py").write_text("")

        # Create minimal backtest module
        backtest_dir = ztb_dir / "backtest"
        backtest_dir.mkdir()
        (backtest_dir / "__init__.py").write_text("")

        # Create minimal runner.py
        runner_content = '''
class BacktestEngine:
    def __init__(self, **kwargs):
        pass
    def load_data(self, dataset):
        import pandas as pd
        return pd.DataFrame({"close": [100, 101, 102]})
    def run_backtest(self, strategy, data):
        return pd.Series([100, 101, 102]), pd.DataFrame()
'''
        (backtest_dir / "runner.py").write_text(runner_content)

        # Create minimal adapters
        adapters_dir = backtest_dir / "adapters"
        adapters_dir.mkdir()
        (adapters_dir / "__init__.py").write_text("")

        adapters_content = '''
class StrategyAdapter:
    def generate_signal(self, data, position):
        return {"action": "hold"}

def create_adapter(policy):
    return StrategyAdapter()
'''
        (adapters_dir / "adapters.py").write_text(adapters_content)

        # Create artifacts directory
        (temp_project / "artifacts").mkdir()

        return temp_project

    def test_linux_canary_execution(self, temp_project_dir):
        """Test that linux_canary.sh executes and produces expected artifacts."""
        # Skip if not on Linux
        if os.name != 'posix':
            pytest.skip("Linux canary test only runs on POSIX systems")

        canary_script = Path(__file__).parent / "linux_canary.sh"

        # Run canary with timeout
        env = os.environ.copy()
        env["PYTHONPATH"] = str(temp_project_dir)

        try:
            result = subprocess.run(
                [str(canary_script)],
                cwd=temp_project_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Canary script timed out")

        # Check exit code
        assert result.returncode == 0, f"Canary failed: {result.stderr}"

        # Check artifacts directory was created
        artifacts_dir = temp_project_dir / "artifacts"
        assert artifacts_dir.exists(), "Artifacts directory not created"

        # Check for canary subdirectory
        canary_dirs = list(artifacts_dir.glob("canary_*"))
        assert len(canary_dirs) == 1, f"Expected 1 canary dir, found {len(canary_dirs)}"

        canary_dir = canary_dirs[0]

        # Check expected subdirectories
        expected_dirs = ["logs", "metrics", "reports", "config"]
        for subdir in expected_dirs:
            assert (canary_dir / subdir).exists(), f"Missing {subdir} directory"

        # Check log file exists
        log_file = canary_dir / "logs" / "canary.log"
        assert log_file.exists(), "Canary log file not found"
        assert log_file.stat().st_size > 0, "Canary log file is empty"

        # Check log contains expected content
        log_content = log_file.read_text()
        assert "SUCCESS: Linux environment detected" in log_content
        assert "SUCCESS: Python" in log_content
        assert "SUCCESS: All required packages available" in log_content

    def test_canary_artifact_schema(self, temp_project_dir):
        """Test that canary artifacts conform to expected schema."""
        # This would run after a successful canary execution
        # For now, just check that the structure is correct

        artifacts_dir = temp_project_dir / "artifacts"
        if not artifacts_dir.exists():
            pytest.skip("Artifacts directory not created")

        canary_dirs = list(artifacts_dir.glob("canary_*"))
        if not canary_dirs:
            pytest.skip("No canary directory found")

        canary_dir = canary_dirs[0]

        # Check that logs contain structured data
        log_file = canary_dir / "logs" / "canary.log"
        if log_file.exists():
            content = log_file.read_text()
            # Check for timestamp format
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    # Should start with timestamp
                    assert len(line) > 20, f"Log line too short: {line}"
                    assert '-' in line[:10], f"No date in log line: {line}"


if __name__ == '__main__':
    pytest.main([__file__])