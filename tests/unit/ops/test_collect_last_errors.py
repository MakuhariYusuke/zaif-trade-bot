import json
from datetime import datetime, timezone
from pathlib import Path

from ztb.ops.monitoring.collect_last_errors import collect_last_errors


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def test_collect_last_errors_prefers_artifacts_watch_log(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cid = "cid-pref"

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "app.log").write_text("INFO ok\nERROR app failure\n", encoding="utf-8")

    watch_log = tmp_path / "artifacts" / cid / "logs" / "watch_log.jsonl"
    _append_jsonl(
        watch_log,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "ERROR",
            "message": "watch failure",
        },
    )

    errors = collect_last_errors(cid)

    assert any("app.log:" in item and "app failure" in item for item in errors)
    assert any("watch_log:" in item and "watch failure" in item for item in errors)


def test_collect_last_errors_falls_back_to_legacy_watch_log(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cid = "cid-legacy"

    legacy_watch_log = tmp_path / "watch_log.jsonl"
    _append_jsonl(
        legacy_watch_log,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "CRITICAL",
            "message": "legacy failure",
        },
    )

    errors = collect_last_errors(cid)

    assert any("watch_log:" in item and "legacy failure" in item for item in errors)


def test_collect_last_errors_reads_only_recent_log_tail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cid = "cid-tail"

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    lines = ["ERROR very old failure\n"] + [f"INFO line {i}\n" for i in range(1003)] + ["ERROR new failure\n"]
    (logs_dir / "tail.log").write_text("".join(lines), encoding="utf-8")

    errors = collect_last_errors(cid)

    assert not any("very old failure" in item for item in errors)
    assert any("new failure" in item for item in errors)
