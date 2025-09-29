import csv
import json
import tempfile
import unittest
from pathlib import Path

from ztb.scripts.aggregate_trends import load_json


class TestAggregateTrends(unittest.TestCase):
    def test_load_json_success(self):
        """Test loading valid JSON."""
        data = {"test": "value"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_json(f.name)

        Path(f.name).unlink()
        self.assertEqual(result, data)

    def test_load_json_missing(self):
        """Test loading missing file."""
        result = load_json("nonexistent.json")
        self.assertIsNone(result)

    def test_aggregate_trends(self):
        """Test trend aggregation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir) / "artifacts"
            artifacts_dir.mkdir()

            # Create dummy session directories with summary.json
            session1_dir = artifacts_dir / "session1"
            session1_dir.mkdir()
            with open(session1_dir / "summary.json", "w") as f:
                json.dump(
                    {
                        "global_step": 1000,
                        "rl_sharpe": 1.5,
                        "dsr": 0.8,
                        "p_value": 0.05,
                    },
                    f,
                )

            session2_dir = artifacts_dir / "session2"
            session2_dir.mkdir()
            with open(session2_dir / "summary.json", "w") as f:
                json.dump(
                    {
                        "global_step": 2000,
                        "rl_sharpe": 2.0,
                        "dsr": 0.9,
                        # p_value missing
                    },
                    f,
                )

            # Change to temp dir to run script
            import os

            old_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                from unittest.mock import patch

                from ztb.scripts.aggregate_trends import main

                with patch("sys.argv", ["aggregate_trends.py", "--out", "trends.csv"]):
                    main()

                # Check output CSV
                csv_path = Path(temp_dir) / "trends.csv"
                self.assertTrue(csv_path.exists())

                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                self.assertEqual(len(rows), 2)

                # Check sorted by session_id
                self.assertEqual(rows[0]["session_id"], "session1")
                self.assertEqual(rows[0]["global_step"], "1000")
                self.assertEqual(rows[0]["rl_sharpe"], "1.5")
                self.assertEqual(rows[0]["dsr"], "0.8")
                self.assertEqual(rows[0]["p_value"], "0.05")

                self.assertEqual(rows[1]["session_id"], "session2")
                self.assertEqual(rows[1]["global_step"], "2000")
                self.assertEqual(rows[1]["rl_sharpe"], "2.0")
                self.assertEqual(rows[1]["dsr"], "0.9")
                self.assertEqual(rows[1]["p_value"], "")  # missing

            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
