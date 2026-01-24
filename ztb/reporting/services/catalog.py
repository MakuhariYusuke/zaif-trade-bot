"""
Report catalog utilities.
"""

from pathlib import Path
from typing import Dict, List, Optional

from ztb.io.json_io import read_json

def find_reports_for_model(
    model_name: str, reports_dir: Optional[Path] = None
) -> List[Path]:
    """
    Find training report files in the project 'reports' directory that match the given model_name.
    """
    reports_root = reports_dir or Path("reports")
    matches: List[Path] = []

    for path in reports_root.glob("training_report_*.json"):
        try:
            obj = read_json(path)
        except Exception:
            continue
        try:
            name = obj.get("configuration", {}).get("training", {}).get("model_name")
        except Exception:
            name = None
        if name == model_name:
            matches.append(path)

    return matches


def extract_action_distribution(report_path: Path) -> Dict[str, float]:
    """
    Extract action_distribution dictionary from a training report file if present.
    """
    obj = read_json(report_path)
    return obj.get("training_stats", {}).get("action_distribution", {})


def get_latest_report_for_model(
    model_name: str, reports_dir: Optional[Path] = None
) -> Optional[Path]:
    reports = find_reports_for_model(model_name, reports_dir=reports_dir)
    if not reports:
        return None
    return sorted(reports)[-1]
