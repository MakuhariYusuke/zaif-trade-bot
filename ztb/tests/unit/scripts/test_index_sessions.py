import json
import tempfile
from pathlib import Path

from ztb.ops.indexing.index_sessions import get_session_status, index_sessions


def test_get_session_status_stopped():
    with tempfile.TemporaryDirectory() as tmp:
        corr_dir = Path(tmp)
        stop_file = Path(tmp) / "ztb.stop"
        stop_file.touch()

        status = get_session_status(corr_dir)
        assert status == "stopped"


def test_index_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sess_dir = root / "20250929T120000Z"
        sess_dir.mkdir()

        summary = {"summary": {"global_step": 1000}}
        with open(sess_dir / "summary.json", "w") as f:
            json.dump(summary, f)

        index = index_sessions(root)
        assert len(index["sessions"]) == 1
        assert index["sessions"][0]["correlation_id"] == "20250929T120000Z"
        assert index["sessions"][0]["latest_step"] == 1000
