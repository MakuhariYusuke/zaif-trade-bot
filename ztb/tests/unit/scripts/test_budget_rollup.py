#!/usr/bin/env python3
"""
Unit tests for budget_rollup.py
"""

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

import budget_rollup


class TestBudgetRollup(unittest.TestCase):
    def test_load_run_metadata_empty(self):
        """Test loading metadata from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            metadata = budget_rollup.load_run_metadata(runs_dir)
            self.assertEqual(len(metadata), 0)

    def test_load_run_metadata_with_files(self):
        """Test loading metadata with valid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            # Create run directory with metadata
            run_dir = runs_dir / "run1"
            run_dir.mkdir()
            metadata_file = run_dir / "run_metadata.json"
            test_metadata = {"start_time": "2024-01-15T10:00:00", "model": "test"}
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(test_metadata, f)

            metadata = budget_rollup.load_run_metadata(runs_dir)
            self.assertEqual(len(metadata), 1)
            self.assertEqual(metadata[0]["run_dir"], str(run_dir))
            self.assertEqual(metadata[0]["model"], "test")

    def test_load_cost_estimates_empty(self):
        """Test loading cost estimates from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            costs = budget_rollup.load_cost_estimates(runs_dir)
            self.assertEqual(len(costs), 0)

    def test_load_cost_estimates_with_files(self):
        """Test loading cost estimates with valid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            # Create run directory with cost estimate
            run_dir = runs_dir / "run1"
            run_dir.mkdir()
            cost_file = run_dir / "cost_estimate.json"
            test_cost = {"total_cost_jpy": 1000.0, "gpu_hours": 5.0}
            with open(cost_file, "w", encoding="utf-8") as f:
                json.dump(test_cost, f)

            costs = budget_rollup.load_cost_estimates(runs_dir)
            self.assertEqual(len(costs), 1)
            self.assertEqual(costs[str(run_dir)], test_cost)

    def test_aggregate_by_date(self):
        """Test aggregating costs by date."""
        metadata_list = [
            {"run_dir": "runs/run1", "start_time": "2024-01-15T10:00:00"},
            {"run_dir": "runs/run2", "start_time": "2024-01-15T12:00:00"},
            {"run_dir": "runs/run3", "start_time": "2024-01-16T10:00:00"},
        ]

        cost_estimates = {
            "runs/run1": {"total_cost_jpy": 500.0, "gpu_hours": 2.0},
            "runs/run2": {"total_cost_jpy": 750.0, "gpu_hours": 3.0},
            "runs/run3": {"total_cost_jpy": 1000.0, "gpu_hours": 4.0},
        }

        daily_totals = budget_rollup.aggregate_by_date(metadata_list, cost_estimates)

        self.assertIn("2024-01-15", daily_totals)
        self.assertIn("2024-01-16", daily_totals)

        day1 = daily_totals["2024-01-15"]
        self.assertEqual(day1["total_cost_jpy"], 1250.0)
        self.assertEqual(day1["gpu_hours"], 5.0)
        self.assertEqual(day1["run_count"], 2)

        day2 = daily_totals["2024-01-16"]
        self.assertEqual(day2["total_cost_jpy"], 1000.0)
        self.assertEqual(day2["gpu_hours"], 4.0)
        self.assertEqual(day2["run_count"], 1)

    def test_aggregate_by_date_filtered(self):
        """Test aggregating with date filter."""
        metadata_list = [
            {"run_dir": "runs/run1", "start_time": "2024-01-15T10:00:00"},
            {"run_dir": "runs/run2", "start_time": "2024-01-16T10:00:00"},
        ]

        cost_estimates = {
            "runs/run1": {"total_cost_jpy": 500.0, "gpu_hours": 2.0},
            "runs/run2": {"total_cost_jpy": 750.0, "gpu_hours": 3.0},
        }

        target_date = date(2024, 1, 15)
        daily_totals = budget_rollup.aggregate_by_date(
            metadata_list, cost_estimates, target_date
        )

        self.assertIn("2024-01-15", daily_totals)
        self.assertNotIn("2024-01-16", daily_totals)

    def test_generate_markdown_report_empty(self):
        """Test generating markdown with no data."""
        report = budget_rollup.generate_markdown_report({})
        self.assertIn("# Daily Budget Report", report)
        self.assertIn("No cost data found.", report)

    def test_generate_markdown_report_with_data(self):
        """Test generating markdown with data."""
        daily_totals = {
            "2024-01-15": {
                "total_cost_jpy": 1250.0,
                "gpu_hours": 5.0,
                "run_count": 2,
                "runs": [
                    {
                        "run_dir": "runs/run1",
                        "cost_jpy": 500.0,
                        "gpu_hours": 2.0,
                        "start_time": "2024-01-15T10:00:00",
                    },
                    {
                        "run_dir": "runs/run2",
                        "cost_jpy": 750.0,
                        "gpu_hours": 3.0,
                        "start_time": "2024-01-15T12:00:00",
                    },
                ],
            }
        }

        report = budget_rollup.generate_markdown_report(daily_totals)

        self.assertIn("# Daily Budget Report", report)
        self.assertIn("## 2024-01-15", report)
        self.assertIn("¥1,250", report)
        self.assertIn("5.0", report)
        self.assertIn("Runs: 2", report)
        self.assertIn("runs/run1", report)
        self.assertIn("runs/run2", report)
        self.assertIn("## Summary", report)
        self.assertIn("¥1,250", report)  # Total cost

    def test_main_with_output(self):
        """Test main with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            output_file = runs_dir / "budget.md"

            # Create test data
            run_dir = runs_dir / "run1"
            run_dir.mkdir()
            metadata_file = run_dir / "run_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump({"start_time": "2024-01-15T10:00:00"}, f)

            cost_file = run_dir / "cost_estimate.json"
            with open(cost_file, "w", encoding="utf-8") as f:
                json.dump({"total_cost_jpy": 1000.0, "gpu_hours": 5.0}, f)

            with patch("sys.argv", ["budget_rollup.py", "--output", str(output_file)]):
                with patch("builtins.print") as mock_print:
                    budget_rollup.main()

            self.assertTrue(output_file.exists())
            mock_print.assert_called_with(f"Budget report written to {output_file}")

            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# Daily Budget Report", content)
            self.assertIn("¥1,000", content)


if __name__ == "__main__":
    unittest.main()
