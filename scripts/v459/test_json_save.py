#!/usr/bin/env python3
"""Test JSON serialization of ExperimentResult"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Project root setup
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.parallel_experiments import ExperimentResult

# Create a mock ExperimentResult
result = ExperimentResult(
    experiment_name="test",
    timestamp=datetime.now().isoformat(),
    status="completed",
    config={"test": "value"},
    metrics={"test_roi": 0.05, "test_sharpe": 1.5},
    artifacts={"report": {"key": "value"}}
)

print("ExperimentResult created successfully")
print(f"Type: {type(result)}")
print(f"Has __dict__: {hasattr(result, '__dict__')}")

# Try to serialize
results_data = [{
    "experiment_name": result.experiment_name,
    "timestamp": result.timestamp,
    "status": result.status,
    "config": result.config,
    "metrics": result.metrics,
    "artifacts": result.artifacts
}]

print("\nAttempting JSON serialization...")

try:
    json_str = json.dumps(results_data, indent=2)
    print("✅ JSON serialization successful!")
    print(f"Length: {len(json_str)} characters")
except Exception as e:
    print(f"❌ JSON serialization failed: {type(e).__name__}: {e}")
    
    # Debug each field
    print("\nDebugging each field:")
    for key, value in results_data[0].items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        try:
            json.dumps(value)
            print(f"  ✅ Serializable")
        except Exception as ex:
            print(f"  ❌ Not serializable: {ex}")
