import json
from pathlib import Path
from typing import Dict, List, Optional


def find_reports_for_model(
    model_name: str, reports_dir: Optional[Path] = None
) -> List[Path]:
    """
    Find training report files in the project 'reports' directory that match the given model_name.

    Args:
        model_name: The model name to search for in report files (from report.configuration.training.model_name)
        reports_dir: Optional Path to override the 'reports' directory. Defaults to './reports'

    Returns:
        List[Path] matching report files (possibly empty)
    """
    if reports_dir is None:
        r = Path("reports")
    else:
        r = reports_dir
    matches = []
    for p in r.glob("training_report_*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            name = obj.get("configuration", {}).get("training", {}).get("model_name")
        except Exception:
            name = None
        if name == model_name:
            matches.append(p)
    return matches


def extract_action_distribution(report_path: Path) -> Dict[str, float]:
    """
    Extract `action_distribution` dictionary from a training report file if present.
    Returns an empty dict when not found.
    """
    obj = json.loads(report_path.read_text(encoding="utf-8"))
    return obj.get("training_stats", {}).get("action_distribution", {})


def get_latest_report_for_model(
    model_name: str, reports_dir: Optional[Path] = None
) -> Optional[Path]:
    reports = find_reports_for_model(model_name, reports_dir=reports_dir)
    if not reports:
        return None
    return sorted(reports)[-1]
