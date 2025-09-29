import tempfile
import unittest
from pathlib import Path

from ztb.ops.indexing.index_sessions import index_sessions
from ztb.ops.indexing.mark_best import mark_best


class TestIndexMark(unittest.TestCase):
    def test_index_sessions_with_metrics(self) -> None:
        """Test indexing with metrics and best markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create dummy session 1 (newer)
            sess1 = root / "sess1"
            sess1.mkdir()
            (sess1 / "run_metadata.json").write_text('{"model": "test"}')
            (sess1 / "summary.json").write_text(
                '{"summary": {"global_step": 100, "loss": 0.5, "accuracy": 0.9}}'
            )
            (sess1 / "best.marker").write_text("marked")

            # Create dummy session 2 (older)
            sess2 = root / "sess2"
            sess2.mkdir()
            (sess2 / "summary.json").write_text(
                '{"summary": {"global_step": 50, "loss": 0.7}}'
            )

            # Create a file in sess1 to test size
            (sess1 / "dummy.txt").write_text("x" * 1000)

            # Ensure sess1 is newer
            import time

            time.sleep(0.01)  # Small delay
            sess1.touch()  # Update mtime

            index = index_sessions(root)

            self.assertEqual(len(index["sessions"]), 2)
            self.assertEqual(index["latest"], "sess1")
            self.assertEqual(index["best"], ["sess1"])

            sess1_data = next(
                s for s in index["sessions"] if s["correlation_id"] == "sess1"
            )
            self.assertTrue(sess1_data["is_best"])
            self.assertAlmostEqual(
                sess1_data["size_mb"], 1000 / (1024 * 1024), places=3
            )
            self.assertEqual(sess1_data["latest_step"], 100)
            self.assertEqual(sess1_data["metrics"]["loss"], 0.5)
            self.assertEqual(sess1_data["metrics"]["accuracy"], 0.9)

    def test_mark_best_success(self) -> None:
        """Test marking session as best."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sess = root / "test_sess"
            sess.mkdir()

            result = mark_best("test_sess", root)
            self.assertEqual(result, 0)
            self.assertTrue((sess / "best.marker").exists())

    def test_mark_best_not_found(self) -> None:
        """Test marking non-existent session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = mark_best("nonexistent", root)
            self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
