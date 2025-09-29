#!/usr/bin/env python3
"""
Unit tests for tb_scrape_summary.py
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ztb.scripts import tb_scrape_summary


class TestTBScrapeSummary(unittest.TestCase):
    def test_find_tb_dirs_empty(self):
        """Test finding TB dirs in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            tb_dirs = tb_scrape_summary.find_tb_dirs(base_dir)
            self.assertEqual(len(tb_dirs), 0)

    def test_find_tb_dirs_with_tb_files(self):
        """Test finding TB dirs with event files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            # Create a TB dir with event file
            tb_dir = base_dir / "experiment1"
            tb_dir.mkdir()
            (tb_dir / "events.out.tfevents.123").touch()

            tb_dirs = tb_scrape_summary.find_tb_dirs(base_dir)
            self.assertEqual(len(tb_dirs), 1)
            self.assertEqual(tb_dirs[0], tb_dir)

    def test_find_tb_dirs_nested(self):
        """Test finding TB dirs in nested structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            # Create nested TB dirs
            tb_dir1 = base_dir / "exp1" / "run1"
            tb_dir1.mkdir(parents=True)
            (tb_dir1 / "tfevents").touch()

            tb_dir2 = base_dir / "exp2"
            tb_dir2.mkdir()
            (tb_dir2 / "events.out.tfevents.456").touch()

            tb_dirs = tb_scrape_summary.find_tb_dirs(base_dir)
            self.assertEqual(len(tb_dirs), 2)
            self.assertIn(tb_dir1, tb_dirs)
            self.assertIn(tb_dir2, tb_dirs)

    def test_extract_scalars_from_file(self):
        """Test extracting scalars from existing summary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tb_dir = Path(tmpdir)
            summary_file = tb_dir / "scalars_summary.json"
            test_data = {"loss": 0.5, "accuracy": 0.9}
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            scalars = tb_scrape_summary.extract_scalars(tb_dir)
            self.assertEqual(scalars, test_data)

    def test_extract_scalars_simulated(self):
        """Test simulated scalar extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tb_dir = Path(tmpdir)
            scalars = tb_scrape_summary.extract_scalars(tb_dir)
            # Should have simulated data
            self.assertIn("loss", scalars)
            self.assertIn("accuracy", scalars)
            self.assertIn("step", scalars)
            self.assertIn("timestamp", scalars)

    def test_scrape_summaries_no_dirs(self):
        """Test scraping when no TB dirs found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summaries = tb_scrape_summary.scrape_summaries(run_dir)
            self.assertEqual(summaries, {})

    def test_scrape_summaries_with_dirs(self):
        """Test scraping with TB directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # Create TB dir with summary file
            tb_dir = run_dir / "exp1"
            tb_dir.mkdir()
            (tb_dir / "events.out.tfevents.123").touch()
            summary_file = tb_dir / "scalars_summary.json"
            test_data = {"loss": 0.3, "accuracy": 0.95}
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            summaries = tb_scrape_summary.scrape_summaries(run_dir)
            self.assertIn("exp1", summaries)
            self.assertEqual(summaries["exp1"], test_data)

    def test_scrape_summaries_with_output(self):
        """Test scraping with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_file = Path(tmpdir) / "output.json"

            # Create TB dir
            tb_dir = run_dir / "exp1"
            tb_dir.mkdir()
            (tb_dir / "tfevents").touch()

            summaries = tb_scrape_summary.scrape_summaries(run_dir, output_file)
            self.assertTrue(output_file.exists())

            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data, summaries)

    def test_main_default_args(self):
        """Test main with default arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # Create a TB dir
            tb_dir = run_dir / "exp1"
            tb_dir.mkdir()
            (tb_dir / "events.out.tfevents.123").touch()

            with patch("sys.argv", ["tb_scrape_summary.py"]):
                with patch("builtins.print") as mock_print:
                    with patch("pathlib.Path.cwd", return_value=run_dir):
                        tb_scrape_summary.main()
            # Should print JSON
            output = mock_print.call_args[0][0]
            data = json.loads(output)
            self.assertIn("exp1", data)

    def test_main_with_output(self):
        """Test main with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_file = run_dir / "summary.json"

            # Create TB dir
            tb_dir = run_dir / "exp1"
            tb_dir.mkdir()
            (tb_dir / "tfevents").touch()

            with patch(
                "sys.argv", ["tb_scrape_summary.py", "--output", str(output_file)]
            ):
                with patch("builtins.print") as mock_print:
                    tb_scrape_summary.main()

            self.assertTrue(output_file.exists())
            mock_print.assert_called_with(f"Summaries written to {output_file}")


if __name__ == "__main__":
    unittest.main()
