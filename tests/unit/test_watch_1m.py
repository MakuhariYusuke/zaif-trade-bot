import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.watch_1m import TrainingWatcher


@pytest.fixture
def temp_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_get_current_step_from_metrics(temp_artifacts):
    # Create artifacts structure
    corr_id = "test123"
    artifacts_dir = temp_artifacts / "artifacts" / corr_id
    artifacts_dir.mkdir(parents=True)

    metrics = {"global_step": 1000}
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    watcher = TrainingWatcher(corr_id, temp_artifacts / "logs")
    step = watcher.get_current_step()
    assert step == 1000


def test_get_current_step_from_log(temp_artifacts):
    # Create log file
    log_dir = temp_artifacts / "logs"
    log_dir.mkdir()
    log_file = log_dir / "training.log"
    with open(log_file, "w") as f:
        f.write("some log line\nglobal_step=500\nmore logs\n")

    watcher = TrainingWatcher("test123", log_dir)
    step = watcher.get_current_step()
    assert step == 500


def test_check_step_progress_stall(temp_artifacts):
    watcher = TrainingWatcher("test123", temp_artifacts / "logs")
    watcher.last_step = 1000
    watcher.last_step_time = time.time() - 700  # 11.5 min ago

    with patch.object(watcher, "get_current_step", return_value=1000):
        stalled = watcher.check_step_progress()
        assert stalled  # Should detect stall


def test_check_memory_usage_rss_breach(temp_artifacts):
    watcher = TrainingWatcher("test123", temp_artifacts / "logs")

    with patch("scripts.watch_1m.psutil") as mock_psutil:
        mock_process = mock_psutil.Process.return_value
        mock_process.memory_info.return_value.rss = 3 * 1024 * 1024 * 1024  # 3GB

        rss_breach, vram_breach = watcher.check_memory_usage()
        assert rss_breach
        assert not vram_breach


def test_check_kill_file(temp_artifacts):
    kill_file = temp_artifacts / "ztb.stop"
    kill_file.write_text("stop training")

    watcher = TrainingWatcher("test123", temp_artifacts / "logs")
    killed = watcher.check_kill_file()
    assert killed


def test_run_once_no_breaches(temp_artifacts):
    watcher = TrainingWatcher("test123", temp_artifacts / "logs")

    with (
        patch.object(watcher, "check_kill_file", return_value=False),
        patch.object(watcher, "check_memory_usage", return_value=(False, False)),
        patch.object(watcher, "check_step_progress", return_value=False),
        patch.object(watcher, "check_checkpoint_backlog", return_value=False),
        patch.object(watcher, "check_error_rate", return_value=False),
        patch.object(watcher, "check_reward_drop", return_value=False),
    ):
        exit_code = watcher.run_once()
        assert exit_code == 0
