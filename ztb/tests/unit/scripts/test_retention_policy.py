import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from ztb.scripts.retention_policy import (
    apply_cleanup,
    find_candidates,
    get_session_info,
)


class TestRetentionPolicy(unittest.TestCase):
    def test_get_session_info(self):
        """Test getting session info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sess_dir = Path(tmpdir) / "test_sess"
            sess_dir.mkdir()
            (sess_dir / "dummy.txt").write_text("x" * 1000)
            (sess_dir / "best.marker").write_text("marked")

            sess_id, mtime, size_mb, is_best = get_session_info(sess_dir)
            self.assertEqual(sess_id, "test_sess")
            self.assertTrue(isinstance(mtime, datetime))
            self.assertAlmostEqual(size_mb, 1000 / (1024 * 1024), places=3)
            self.assertTrue(is_best)

    def test_find_candidates_age_based(self):
        """Test finding candidates based on age."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create old session
            old_dir = root / "old_sess"
            old_dir.mkdir()
            # Set mtime to 20 days ago
            old_time = datetime.now() - timedelta(days=20)
            old_dir.touch()
            import os

            os.utime(old_dir, (old_time.timestamp(), old_time.timestamp()))

            # Create new session
            new_dir = root / "new_sess"
            new_dir.mkdir()

            candidates = find_candidates(
                root, keep_days=14, keep_best=0, max_size_gb=100
            )

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0][0], "old_sess")
            self.assertIn("days > 14", candidates[0][1])

    def test_find_candidates_best_protection(self):
        """Test that best sessions are protected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create old best session
            best_dir = root / "best_sess"
            best_dir.mkdir()
            (best_dir / "best.marker").write_text("marked")
            old_time = datetime.now() - timedelta(days=20)
            import os

            os.utime(best_dir, (old_time.timestamp(), old_time.timestamp()))

            candidates = find_candidates(
                root, keep_days=14, keep_best=1, max_size_gb=100
            )

            self.assertEqual(len(candidates), 0)  # Best session protected

    @patch("shutil.rmtree")
    def test_apply_cleanup(self, mock_rmtree):
        """Test applying cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidates = [("old_sess", "Age: 20 days > 14")]

            apply_cleanup(root, candidates)
            mock_rmtree.assert_called_once()


if __name__ == "__main__":
    unittest.main()
