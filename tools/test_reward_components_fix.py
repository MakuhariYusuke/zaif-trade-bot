#!/usr/bin/env python3
"""Quick test to verify reward_components are now saved correctly."""

import json
import subprocess
from pathlib import Path


def test_quick_training():
    """Run a quick 500-step training to verify reward_components."""
    # Test with minimal config
    config_path = "config/v447/baseline.json"
    
    # Run quick test using AB test runner
    print(f"Running quick test with {config_path}...")
    
    cmd = [
        "python",
        "tools/ab_test_runner.py",
        "--config", config_path,
        "--timesteps", "500",
        "--name", "reward_components_test"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Training failed with exit code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print("✓ Training completed")
    
    # Check most recent training report
    reports_dir = Path("training_results/reports")
    reports = sorted(reports_dir.glob("training_report_*.json"))
    
    if not reports:
        print("❌ No training reports found")
        return False
    
    latest_report = reports[-1]
    print(f"Checking report: {latest_report.name}")
    
    with open(latest_report) as f:
        report = json.load(f)
    
    # Check for reward_components
    training_stats = report.get("training_stats", {})
    reward_components = training_stats.get("reward_components", {})
    
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
    exit(0 if success else 1)
