import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.supervise_1m import acquire_lock, check_kill_file, get_training_command


def test_acquire_lock_success():
    with tempfile.TemporaryDirectory() as tmp:
        lock_file = Path(tmp) / "lock"
        locked = acquire_lock(lock_file)
        assert locked


def test_acquire_lock_fail():
    with tempfile.TemporaryDirectory() as tmp:
        lock_file = Path(tmp) / "lock"
        # Acquire first
        locked1 = acquire_lock(lock_file)
        assert locked1
        # Second should fail
        locked2 = acquire_lock(lock_file)
        assert not locked2


def test_check_kill_file_exists():
    with tempfile.TemporaryDirectory() as tmp:
        kill_file = Path(tmp) / "ztb.stop"
        kill_file.write_text("")
        with patch("scripts.supervise_1m.Path", return_value=kill_file):
            assert check_kill_file()


def test_check_kill_file_not_exists():
    with patch("scripts.supervise_1m.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        assert not check_kill_file()


def test_get_training_command_with_run_1m():
    with patch("scripts.supervise_1m.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        cmd = get_training_command("test123")
        assert "scripts/run_1m.py" in cmd


def test_get_training_command_canonical():
    with patch("scripts.supervise_1m.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        cmd = get_training_command("test123", "--extra arg")
        assert "-m ztb.training.ppo_trainer" in " ".join(cmd)
        assert "--extra" in cmd
        assert "arg" in cmd


# Integration test for backoff would require mocking time and subprocess, omitted for brevity
