from pathlib import Path

from ztb.ops.monitoring.watch_1m import TrainingWatcher


def test_get_current_step_reads_recent_lines_from_log(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    lines = [f"info line {i}\n" for i in range(200)]
    lines[-3] = "2026-02-22T00:00:00 global_step=12345 done\n"
    log_file.write_text("".join(lines), encoding="utf-8")

    watcher = TrainingWatcher(correlation_id="test-cid", log_dir=log_dir)

    assert watcher.get_current_step() == 12345
