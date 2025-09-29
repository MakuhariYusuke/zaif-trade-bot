import gzip
import os
import tempfile
import time
from pathlib import Path

from ztb.ops.artifacts.artifacts_janitor import delete_old_runs, rotate_log


def test_rotate_log():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "test.log"
        with open(log_path, "w") as f:
            f.write("x" * (11 * 1024 * 1024))  # 11MB

        rotate_log(log_path, 10)

        assert not log_path.exists()
        rotated = list(Path(tmp).glob("test.*.gz"))
        assert len(rotated) == 1

        with gzip.open(rotated[0], "rt") as f:
            content = f.read()
            assert len(content) == 11 * 1024 * 1024


def test_delete_old_runs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_dir = root / "20240101T000000Z"
        old_dir.mkdir()
        # Set mtime to old
        os.utime(old_dir, (time.time() - 86400 * 40, time.time() - 86400 * 40))

        delete_old_runs(root, 30, dry_run=True)

        assert old_dir.exists()  # Dry run

        delete_old_runs(root, 30, dry_run=False)

        assert not old_dir.exists()
