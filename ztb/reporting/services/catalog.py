"""
Report catalog utilities.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from ztb.io.json_io import read_json
from ztb.utils.safety import ensure_dict, safe_to_float


class _ReportFileState(TypedDict):
    mtime_ns: int
    size: int


ReportCacheKey = tuple[str, int, int]
REPORT_MODEL_NAME_CACHE_MAX_SIZE = 2048
_REPORT_MODEL_NAME_CACHE: "OrderedDict[ReportCacheKey, str | None]" = OrderedDict()


def clear_report_cache() -> None:
    """Clear report parsing cache to release memory in long-running processes."""
    _REPORT_MODEL_NAME_CACHE.clear()


def _get_report_file_state(report_path: Path) -> _ReportFileState | None:
    try:
        stat = report_path.stat()
    except OSError:
        return None
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _make_report_cache_key(report_path: Path, state: _ReportFileState) -> ReportCacheKey:
    return (str(report_path.resolve()), state["mtime_ns"], state["size"])


def _extract_report_model_name(report_path: Path) -> str | None:
    state = _get_report_file_state(report_path)
    if state is None:
        return None

    cache_key = _make_report_cache_key(report_path, state)
    cached_model_name = _REPORT_MODEL_NAME_CACHE.get(cache_key)
    if cached_model_name is not None or cache_key in _REPORT_MODEL_NAME_CACHE:
        _REPORT_MODEL_NAME_CACHE.move_to_end(cache_key)
        return cached_model_name

    try:
        payload = ensure_dict(read_json(report_path))
    except Exception:
        model_name = None
    else:
        configuration = ensure_dict(payload.get("configuration"))
        training = ensure_dict(configuration.get("training"))
        raw_model_name = training.get("model_name")
        model_name = raw_model_name if isinstance(raw_model_name, str) else None

    _REPORT_MODEL_NAME_CACHE[cache_key] = model_name
    _REPORT_MODEL_NAME_CACHE.move_to_end(cache_key)
    while len(_REPORT_MODEL_NAME_CACHE) > REPORT_MODEL_NAME_CACHE_MAX_SIZE:
        _REPORT_MODEL_NAME_CACHE.popitem(last=False)

    return model_name

def find_reports_for_model(
    model_name: str, reports_dir: Optional[Path] = None
) -> List[Path]:
    """
    Find training report files in the project 'reports' directory that match the given model_name.
    """
    reports_root = reports_dir or Path("reports")
    matches: List[Path] = []

    for path in sorted(reports_root.glob("training_report_*.json")):
        if _extract_report_model_name(path) == model_name:
            matches.append(path)

    return matches


def extract_action_distribution(report_path: Path) -> Dict[str, float]:
    """
    Extract action_distribution dictionary from a training report file if present.
    """
    payload = ensure_dict(read_json(report_path))
    training_stats = ensure_dict(payload.get("training_stats"))
    action_distribution = ensure_dict(training_stats.get("action_distribution"))
    return {str(k): safe_to_float(v, 0.0) for k, v in action_distribution.items()}


def get_latest_report_for_model(
    model_name: str, reports_dir: Optional[Path] = None
) -> Optional[Path]:
    reports = find_reports_for_model(model_name, reports_dir=reports_dir)
    if not reports:
        return None
    def report_mtime_ns(path: Path) -> int:
        state = _get_report_file_state(path)
        return state["mtime_ns"] if state is not None else 0

    latest_report = max(reports, key=report_mtime_ns)
    return latest_report
