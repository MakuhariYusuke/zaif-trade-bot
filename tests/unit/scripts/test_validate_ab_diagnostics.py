"""
Tests for validate_ab_diagnostics.py script.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import pytest

from ztb.utils.config import ZTBConfig


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create temporary directory for test results."""
    return tmp_path / "diagnostics"


def create_mock_results(
    data_name: str,
    prob_std_positive: bool,
    legal_sell_rate_ok: bool,
) -> Dict[str, Any]:
    """Create mock diagnostic result."""
    return {
        "data_name": data_name,
        "total_steps": 300,
        "action_distribution": {"HOLD": 0.5, "BUY": 0.3, "SELL": 0.2},
        "legal_sell_stats": {
            "legal_sell_rate": 0.18 if legal_sell_rate_ok else 0.10,
            "total_legal_opportunities": 250,
            "legal_sells": 45 if legal_sell_rate_ok else 25,
        },
        "probability_variance": {
            "HOLD_std": 0.15 if prob_std_positive else 0.0,
            "BUY_std": 0.12 if prob_std_positive else 0.0,
            "SELL_std": 0.10 if prob_std_positive else 0.0,
            "mean_std": 0.12 if prob_std_positive else 0.0,
        },
        "acceptance_criteria": {
            "prob_std_positive": prob_std_positive,
            "legal_sell_rate_ok": legal_sell_rate_ok,
        },
    }


def create_results_file(
    output_path: Path,
    results: list,
    temperature: float = 0.7,
) -> None:
    """Create mock diagnostics results JSON file."""
    config = ZTBConfig()
    data = {
        "config": {
            "model_path": config.get_model_path("test_model"),
            "temperature": temperature,
            "tiebreaker_tau": 0.05,
            "enable_tiebreaker": True,
            "steps": 300,
        },
        "results": results,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def test_all_pass(temp_results_dir):
    """Test validation with all criteria passing."""
    results = [
        create_mock_results("Dataset1", prob_std_positive=True, legal_sell_rate_ok=True),
        create_mock_results("Dataset2", prob_std_positive=True, legal_sell_rate_ok=True),
    ]
    
    results_file = temp_results_dir / "all_pass.json"
    create_results_file(results_file, results)
    
    # Run validation (should exit 0)
    result = subprocess.run(
        [sys.executable, "scripts/validate_ab_diagnostics.py", "--input", str(results_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "AT LEAST ONE TEST PASSED" in result.stdout or "ALL TESTS PASSED" in result.stdout


def test_all_fail(temp_results_dir):
    """Test validation with all criteria failing."""
    results = [
        create_mock_results("Dataset1", prob_std_positive=False, legal_sell_rate_ok=False),
        create_mock_results("Dataset2", prob_std_positive=False, legal_sell_rate_ok=False),
    ]
    
    results_file = temp_results_dir / "all_fail.json"
    create_results_file(results_file, results)
    
    # Run validation (should exit 1)
    result = subprocess.run(
        [sys.executable, "scripts/validate_ab_diagnostics.py", "--input", str(results_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "ALL TESTS FAILED" in result.stdout


def test_mixed_results_default_mode(temp_results_dir):
    """Test validation with mixed results (default mode: at least one pass)."""
    results = [
        create_mock_results("Dataset1", prob_std_positive=True, legal_sell_rate_ok=True),
        create_mock_results("Dataset2", prob_std_positive=False, legal_sell_rate_ok=False),
    ]
    
    results_file = temp_results_dir / "mixed.json"
    create_results_file(results_file, results)
    
    # Run validation without --strict (should exit 0 if at least one passes)
    result = subprocess.run(
        [sys.executable, "scripts/validate_ab_diagnostics.py", "--input", str(results_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "AT LEAST ONE TEST PASSED" in result.stdout


def test_mixed_results_strict_mode(temp_results_dir):
    """Test validation with mixed results (strict mode: all must pass)."""
    results = [
        create_mock_results("Dataset1", prob_std_positive=True, legal_sell_rate_ok=True),
        create_mock_results("Dataset2", prob_std_positive=False, legal_sell_rate_ok=False),
    ]
    
    results_file = temp_results_dir / "mixed_strict.json"
    create_results_file(results_file, results)
    
    # Run validation with --strict (should exit 1 if not all pass)
    result = subprocess.run(
        [sys.executable, "scripts/validate_ab_diagnostics.py", "--input", str(results_file), "--strict"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "SOME TESTS FAILED" in result.stdout


def test_file_not_found():
    """Test validation with non-existent file."""
    result = subprocess.run(
        [sys.executable, "scripts/validate_ab_diagnostics.py", "--input", "nonexistent.json"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Input file not found" in result.stderr


def test_invalid_json(temp_results_dir):
    """Test validation with invalid JSON."""
    invalid_file = temp_results_dir / "invalid.json"
    invalid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(invalid_file, "w") as f:
        f.write("{ invalid json }")
    
    result = subprocess.run(
        [sys.executable, "scripts/validate_ab_diagnostics.py", "--input", str(invalid_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "Invalid JSON" in result.stderr


def test_empty_results(temp_results_dir):
    """Test validation with empty results list."""
    results_file = temp_results_dir / "empty.json"
    create_results_file(results_file, [])
    
    result = subprocess.run(
        [sys.executable, "scripts/validate_ab_diagnostics.py", "--input", str(results_file)],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 1
    assert "No results found" in result.stderr


def test_verbose_output(temp_results_dir):
    """Test validation with verbose output."""
    results = [
        create_mock_results("Dataset1", prob_std_positive=True, legal_sell_rate_ok=True),
    ]
    
    results_file = temp_results_dir / "verbose.json"
    create_results_file(results_file, results)
    
    result = subprocess.run(
        [sys.executable, "scripts/validate_ab_diagnostics.py", "--input", str(results_file), "--verbose"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Configuration" in result.stdout
    assert "Dataset1" in result.stdout
    assert "Probability std" in result.stdout
    assert "Legal SELL rate" in result.stdout
