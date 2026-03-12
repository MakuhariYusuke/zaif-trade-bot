"""
Training report IO utilities.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ztb.io.json_io import write_json

def save_training_report(report: dict[str, Any], output_dir: str = "reports") -> str:
    """Save a training report using the standard filename pattern."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm = report.get("metadata", {}).get("algorithm", "unknown")
    model_name = report.get("metadata", {}).get("model_name", "unknown")
    filename = f"training_report_{algorithm}_{model_name}_{timestamp}.json"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report, indent=2, ensure_ascii=False, default=str)
    return str(output_path)

def save_ensemble_report(report: dict[str, Any], output_dir: str = "reports") -> str:
    """Save an ensemble report using the standard filename pattern."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ensemble_analysis_report_{timestamp}.json"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report, indent=2, ensure_ascii=False, default=str)
    return str(output_path)
