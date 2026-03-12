import json
from pathlib import Path

from ztb.ops.monitoring.disk_health import measure_io_latency, write_alerts


def test_measure_io_latency_writes_and_cleans_temp_file(tmp_path: Path) -> None:
    latency_ms = measure_io_latency(tmp_path)

    assert latency_ms is not None
    assert latency_ms >= 0.0
    assert not (tmp_path / ".disk_health_test").exists()


def test_write_alerts_appends_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "ops_alerts.jsonl"
    alerts = [
        {"level": "WARN", "message": "low disk", "path": str(tmp_path)},
        {"level": "FAIL", "message": "very low disk", "path": str(tmp_path)},
    ]

    write_alerts(alerts, log_path)

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert rows == alerts
