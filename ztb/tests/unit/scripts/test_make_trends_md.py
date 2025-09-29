import csv
import tempfile
import unittest
from pathlib import Path

from ztb.ops.reports.make_trends_md import (
    generate_markdown,
    generate_sparkline,
    load_trends,
)


class TestMakeTrendsMd(unittest.TestCase):
    def test_load_trends_success(self):
        """Test loading trends from CSV."""
        csv_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["session_id", "loss", "accuracy"])
                writer.writerow(["sess1", "0.5", "0.9"])
                writer.writerow(["sess2", "0.3", "0.95"])
                f.flush()
                csv_path = Path(f.name)

            trends = load_trends(csv_path)

            self.assertEqual(len(trends), 2)
            self.assertEqual(trends[0]["session_id"], "sess1")
            self.assertEqual(trends[0]["loss"], 0.5)
            self.assertEqual(trends[0]["accuracy"], 0.9)
        finally:
            if csv_path and csv_path.exists():
                csv_path.unlink()

    def test_generate_sparkline(self):
        """Test ASCII sparkline generation."""
        values = [1.0, 2.0, 3.0, 2.0, 1.0]
        spark = generate_sparkline(values)
        self.assertEqual(len(spark), 5)
        # Should have varying heights
        self.assertIn("▁", spark)
        self.assertIn("█", spark)

    def test_generate_sparkline_empty(self):
        """Test sparkline with empty data."""
        spark = generate_sparkline([])
        self.assertEqual(spark, "")

    def test_generate_markdown(self):
        """Test Markdown generation."""
        trends = [
            {"session_id": "sess1", "loss": 0.5, "accuracy": 0.9},
            {"session_id": "sess2", "loss": 0.3, "accuracy": 0.95},
        ]
        md = generate_markdown(trends)

        self.assertIn("# Trend Report", md)
        self.assertIn("## loss", md)
        self.assertIn("## accuracy", md)
        self.assertIn("| sess1 |", md)
        self.assertIn("▁", md)  # Sparkline chars


if __name__ == "__main__":
    unittest.main()
