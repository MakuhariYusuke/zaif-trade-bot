import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ztb.ops.alerts.alert_notifier import load_alerts as load_alerts_ops
from ztb.ops.monitoring.alert_notifier import load_alerts as load_alerts_monitoring


def _ts(seconds_offset: int, *, with_z: bool = False) -> str:
    value = datetime.now(timezone.utc) + timedelta(seconds=seconds_offset)
    iso = value.isoformat()
    if with_z:
        return iso.replace("+00:00", "Z")
    return iso


def test_load_alerts_filters_by_level_and_time_and_skips_malformed(tmp_path: Path) -> None:
    path = tmp_path / "watch_log.jsonl"
    lines = [
        "{not-json}",
        json.dumps({"timestamp": _ts(-3600), "level": "ERROR", "message": "too old"}),
        json.dumps({"timestamp": _ts(-30), "level": "INFO", "message": "info"}),
        json.dumps({"timestamp": _ts(-20, with_z=True), "level": "WARN", "message": "warn"}),
        json.dumps({"timestamp": _ts(-10), "level": "CRITICAL", "message": "critical"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    expected = ["warn", "critical"]
    loaded_ops = load_alerts_ops(path, since_seconds=300, min_level="WARN")
    loaded_monitoring = load_alerts_monitoring(path, since_seconds=300, min_level="WARN")

    assert [str(item.get("message", "")) for item in loaded_ops] == expected
    assert [str(item.get("message", "")) for item in loaded_monitoring] == expected


def test_load_alerts_missing_file_returns_empty_list(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    assert load_alerts_ops(missing, since_seconds=60, min_level="WARN") == []
    assert load_alerts_monitoring(missing, since_seconds=60, min_level="WARN") == []
