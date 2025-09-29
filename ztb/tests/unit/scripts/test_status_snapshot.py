#!/usr/bin/env python3
"""
Unit tests for status_snapshot.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from status_snapshot import (
    collect_status_data,
    generate_markdown_table,
    generate_summary,
    load_json_file,
)


class TestStatusSnapshot(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.temp_dir / "artifacts"
        self.artifacts_dir.mkdir()
        self.correlation_id = "test123"
        self.session_dir = self.artifacts_dir / self.correlation_id
        self.reports_dir = self.session_dir / "reports"
        self.reports_dir.mkdir(parents=True)

    def tearDown(self):
        # Clean up
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_json_file_exists(self):
        """Test loading existing JSON file."""
        test_file = self.temp_dir / "test.json"
        test_data = {"key": "value"}
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        result = load_json_file(test_file)
        self.assertEqual(result, test_data)

    def test_load_json_file_not_exists(self):
        """Test loading non-existent JSON file."""
        test_file = self.temp_dir / "nonexistent.json"
        result = load_json_file(test_file)
        self.assertIsNone(result)

    def test_load_json_file_invalid(self):
        """Test loading invalid JSON file."""
        test_file = self.temp_dir / "invalid.json"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("invalid json")

        result = load_json_file(test_file)
        self.assertIsNone(result)

    def test_collect_status_data_all_present(self):
        """Test collecting status data when all files are present."""
        # Create index.json
        index_file = self.artifacts_dir / "index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump([{"id": "test123"}], f)

        # Create gates.json
        gates_file = self.reports_dir / "gates.json"
        with open(gates_file, "w", encoding="utf-8") as f:
            json.dump({"gates": [{"name": "gate1", "status": "pass"}]}, f)

        # Create tb_summary.json
        tb_file = self.reports_dir / "tb_summary.json"
        with open(tb_file, "w", encoding="utf-8") as f:
            json.dump({"run1": {"loss": 0.5}}, f)

        # Create disk_health.json
        disk_file = self.reports_dir / "disk_health.json"
        with open(disk_file, "w", encoding="utf-8") as f:
            json.dump({"free_gb": 100}, f)

        # Create last_errors.json
        errors_file = self.reports_dir / "last_errors.json"
        with open(errors_file, "w", encoding="utf-8") as f:
            json.dump([{"error": "test error"}], f)

        status = collect_status_data(self.correlation_id, self.artifacts_dir)

        self.assertEqual(status["correlation_id"], self.correlation_id)
        self.assertIn("timestamp", status)
        self.assertEqual(len(status["sources"]), 5)

        # Check all sources are OK
        for source_name in [
            "index",
            "gates",
            "tb_summary",
            "disk_health",
            "last_errors",
        ]:
            self.assertEqual(status["sources"][source_name]["status"], "OK")
            self.assertIsNotNone(status["sources"][source_name]["data"])

    def test_collect_status_data_some_missing(self):
        """Test collecting status data when some files are missing."""
        # Only create gates.json
        gates_file = self.reports_dir / "gates.json"
        with open(gates_file, "w", encoding="utf-8") as f:
            json.dump({"gates": []}, f)

        status = collect_status_data(self.correlation_id, self.artifacts_dir)

        # gates should be OK, others SKIP
        self.assertEqual(status["sources"]["gates"]["status"], "OK")
        for source_name in ["index", "tb_summary", "disk_health", "last_errors"]:
            self.assertEqual(status["sources"][source_name]["status"], "SKIP")
            self.assertIsNone(status["sources"][source_name]["data"])

    def test_generate_summary(self):
        """Test generating summary string."""
        status = {
            "correlation_id": "test123",
            "timestamp": "2024-01-01T00:00:00",
            "sources": {
                "index": {"status": "OK", "data": []},
                "gates": {"status": "OK", "data": {}},
                "tb_summary": {"status": "SKIP", "data": None},
                "disk_health": {"status": "SKIP", "data": None},
                "last_errors": {"status": "OK", "data": []},
            },
        }

        summary = generate_summary(status)
        self.assertIn("Status: 3/5 sources OK", summary)
        self.assertIn("2 skipped", summary)

    def test_generate_markdown_table(self):
        """Test generating markdown table."""
        status = {
            "correlation_id": "test123",
            "timestamp": "2024-01-01T00:00:00",
            "sources": {
                "index": {"status": "OK", "data": [{"id": "run1"}]},
                "gates": {
                    "status": "OK",
                    "data": {"gates": [{"status": "pass"}, {"status": "fail"}]},
                },
                "tb_summary": {"status": "SKIP", "data": None},
            },
        }

        md = generate_markdown_table(status)

        self.assertIn("# Status Snapshot", md)
        self.assertIn("test123", md)
        self.assertIn("| Source | Status | Details |", md)
        self.assertIn("| index | OK | 1 sessions |", md)
        self.assertIn("| gates | OK | 1/2 gates passed |", md)
        self.assertIn("| tb_summary | SKIP | No data |", md)


if __name__ == "__main__":
    unittest.main()
