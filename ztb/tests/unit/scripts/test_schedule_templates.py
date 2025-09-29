#!/usr/bin/env python3
"""
Unit tests for schedule_templates.py
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ztb.scripts.schedule_templates import get_template, list_templates, main


class TestScheduleTemplates(unittest.TestCase):
    def test_get_template_known(self):
        """Test getting a known template."""
        template = get_template("daily")
        self.assertEqual(template["description"], "Run daily at 9:00 AM")
        self.assertEqual(template["windows"], "0 9 * * *")
        self.assertEqual(template["linux"], "0 9 * * *")
        self.assertEqual(template["timezone"], "Asia/Tokyo")

    def test_get_template_unknown(self):
        """Test getting an unknown template raises ValueError."""
        with self.assertRaises(ValueError):
            get_template("unknown")

    def test_list_templates(self):
        """Test listing templates (captures output)."""
        with patch("builtins.print") as mock_print:
            list_templates()
            # Check that print was called multiple times
            self.assertGreater(mock_print.call_count, 1)
            # Check first call contains "Available templates:"
            self.assertIn("Available templates:", mock_print.call_args_list[0][0][0])

    def test_main_list(self):
        """Test main with --list option."""
        with patch("builtins.print") as mock_print:
            with patch("sys.argv", ["schedule_templates.py", "--list"]):
                main()
            self.assertIn("Available templates:", mock_print.call_args_list[0][0][0])

    def test_main_template_stdout(self):
        """Test main with template output to stdout."""
        with patch("builtins.print") as mock_print:
            with patch("sys.argv", ["schedule_templates.py", "--template", "daily"]):
                main()
            # Check that JSON was printed
            output = mock_print.call_args[0][0]
            data = json.loads(output)
            self.assertEqual(data["template"], "daily")
            self.assertIn("schedule", data)

    def test_main_template_file(self):
        """Test main with template output to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test.json"
            with patch(
                "sys.argv",
                [
                    "schedule_templates.py",
                    "--template",
                    "hourly",
                    "--output",
                    str(output_file),
                ],
            ):
                with patch("builtins.print") as mock_print:
                    main()
            # Check file was created
            self.assertTrue(output_file.exists())
            # Check content
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["template"], "hourly")
            self.assertIn("schedule", data)
            # Check print message
            mock_print.assert_called_with(f"Template written to {output_file}")

    def test_main_no_template(self):
        """Test main without --template (should error)."""
        with patch("sys.argv", ["schedule_templates.py"]):
            with patch("sys.stderr") as mock_stderr:
                with self.assertRaises(SystemExit):
                    main()
            # Check error message
            mock_stderr.write.assert_called()

    def test_main_unknown_template(self):
        """Test main with unknown template."""
        with patch("sys.argv", ["schedule_templates.py", "--template", "unknown"]):
            with patch("sys.stderr") as mock_stderr:
                with self.assertRaises(SystemExit):
                    main()
            mock_stderr.write.assert_called()


if __name__ == "__main__":
    unittest.main()
