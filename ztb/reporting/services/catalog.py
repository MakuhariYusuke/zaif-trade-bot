"""
Report catalog utilities.
"""

import heapq
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from ztb.io.json_io import read_json
from ztb.utils.safety import ensure_dict, safe_to_float


class _ReportFileState(TypedDict):
    mtime_ns: int
    size: int


ReportCacheKey = tuple[str, int, int]
TRAINING_REPORT_PATTERN = "training_report_*.json"
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


def _drop_stale_cache_entries_for_path(resolved_path: str) -> None:
    stale_keys = [key for key in _REPORT_MODEL_NAME_CACHE if key[0] == resolved_path]
    for key in stale_keys:
        _REPORT_MODEL_NAME_CACHE.pop(key, None)


def load_training_report(report_path: Path) -> dict[str, object] | None:
    """Load a training report safely as an object map."""
    try:
        return ensure_dict(read_json(report_path))
    except Exception:
        return None


def _extract_model_name_from_payload(payload: Mapping[str, object]) -> str | None:
    configuration = ensure_dict(payload.get("configuration"))
    training = ensure_dict(configuration.get("training"))
    raw_model_name = training.get("model_name")
    if isinstance(raw_model_name, str) and raw_model_name:
        return raw_model_name
    return None


def _extract_report_model_name(report_path: Path) -> str | None:
    state = _get_report_file_state(report_path)
    if state is None:
        return None

    cache_key = _make_report_cache_key(report_path, state)
    resolved_path = cache_key[0]
    cached_model_name = _REPORT_MODEL_NAME_CACHE.get(cache_key)
    if cached_model_name is not None or cache_key in _REPORT_MODEL_NAME_CACHE:
        _REPORT_MODEL_NAME_CACHE.move_to_end(cache_key)
        return cached_model_name

    payload = load_training_report(report_path)
    model_name = _extract_model_name_from_payload(payload) if payload is not None else None

    # Keep only the latest cache entry for a report path to avoid stale-key accumulation.
    _drop_stale_cache_entries_for_path(resolved_path)
    _REPORT_MODEL_NAME_CACHE[cache_key] = model_name
    _REPORT_MODEL_NAME_CACHE.move_to_end(cache_key)
    while len(_REPORT_MODEL_NAME_CACHE) > REPORT_MODEL_NAME_CACHE_MAX_SIZE:
        _REPORT_MODEL_NAME_CACHE.popitem(last=False)

    return model_name


def list_training_reports(reports_dir: Optional[Path] = None) -> List[Path]:
    """List training report JSON files under reports directory."""
    reports_root = reports_dir or Path("reports")
    if not reports_root.exists():
        return []
    return list(reports_root.glob(TRAINING_REPORT_PATTERN))


def get_recent_training_reports(
    limit: int, reports_dir: Optional[Path] = None
) -> List[Path]:
    """Return latest training reports by mtime_ns, newest first."""
    if limit <= 0:
        return []
    latest_entries = heapq.nlargest(
        limit,
        (
            (state["mtime_ns"], str(path), path)
            for path in list_training_reports(reports_dir=reports_dir)
            if (state := _get_report_file_state(path)) is not None
        ),
    )
    return [path for _, _, path in latest_entries]


def find_reports_for_model(
    model_name: str, reports_dir: Optional[Path] = None
) -> List[Path]:
    """
    Find training report files in the project 'reports' directory that match the given model_name.
    """
    matches: List[Path] = []

    for path in list_training_reports(reports_dir=reports_dir):
        if _extract_report_model_name(path) == model_name:
            matches.append(path)

    return matches


def extract_action_distribution_from_payload(
    payload: Mapping[str, object],
) -> Dict[str, float]:
    """Extract action_distribution dictionary from a training report payload."""
    training_stats = ensure_dict(payload.get("training_stats"))
    action_distribution = ensure_dict(training_stats.get("action_distribution"))
    return {str(k): safe_to_float(v, 0.0) for k, v in action_distribution.items()}


def extract_action_distribution(report_path: Path) -> Dict[str, float]:
    """
    Extract action_distribution dictionary from a training report file if present.
    """
    payload = load_training_report(report_path)
    if payload is None:
        return {}
    return extract_action_distribution_from_payload(payload)


def extract_reward_components_from_payload(
    payload: Mapping[str, object],
) -> Dict[str, float]:
    """Extract reward_components from report payload (root or training_stats)."""
    root_components = ensure_dict(payload.get("reward_components"))
    if root_components:
        source_components = root_components
    else:
        training_stats = ensure_dict(payload.get("training_stats"))
        source_components = ensure_dict(training_stats.get("reward_components"))
    return {str(k): safe_to_float(v, 0.0) for k, v in source_components.items()}


def extract_reward_components(report_path: Path) -> Dict[str, float]:
    """Extract reward_components dictionary from a training report file if present."""
    payload = load_training_report(report_path)
    if payload is None:
        return {}
    return extract_reward_components_from_payload(payload)


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
