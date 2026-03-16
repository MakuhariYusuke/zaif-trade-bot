#!/usr/bin/env python3
"""Quick test to verify reward_components are now saved correctly."""

import subprocess
import sys
from pathlib import Path

from ztb.reporting.services.catalog import (
    extract_reward_components_from_payload,
    get_recent_training_reports,
    load_training_report,
)
from ztb.utils.safety import ensure_dict


def test_quick_training() -> bool:
    """Run a quick 500-step training to verify reward_components."""
    # Test with minimal config
    config_path = "config/v447/baseline.json"
    
    # Run quick test using AB test runner
    print(f"Running quick test with {config_path}...")
    
    cmd = [
        sys.executable,
        "tools/ab_test_runner.py",
        "--configs",
        config_path,
        "--timesteps", "500",
        "--seeds",
        "1",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Training failed with exit code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print("✓ Training completed")
    
    # Check most recent training report
    reports_dir = Path("training_results/reports")
    reports = get_recent_training_reports(limit=1, reports_dir=reports_dir)
    
    if not reports:
        print("❌ No training reports found")
        return False
    
    latest_report = reports[0]
    print(f"Checking report: {latest_report.name}")

    report = load_training_report(latest_report)
    if report is None:
        print("❌ Could not load latest report JSON")
        return False

    training_stats = ensure_dict(report.get("training_stats"))
    reward_components = extract_reward_components_from_payload(report)
    
    if not reward_components:
        print("❌ reward_components NOT FOUND in training_stats")
        print(f"Available keys: {list(training_stats.keys())}")
        return False
    
    print(f"✓ reward_components FOUND: {list(reward_components.keys())}")
    print(f"  stage: {reward_components.get('stage')}")
    print(f"  pnl: {reward_components.get('pnl')}")
    print(f"  final_reward: {reward_components.get('final_reward')}")
    
    return True


if __name__ == "__main__":
    success = test_quick_training()
    raise SystemExit(0 if success else 1)
