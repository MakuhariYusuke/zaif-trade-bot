import json
import tempfile
import unittest
from pathlib import Path

from ztb.scripts.make_dashboard import generate_html, load_json


class TestMakeDashboard(unittest.TestCase):
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

    def test_load_json_invalid(self):
        """Test loading invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json")
            f.flush()
            result = load_json(f.name)

        Path(f.name).unlink()
        self.assertIsNone(result)

    def test_generate_html(self):
        """Test HTML generation."""
        index_data = {
            "sessions": [
                {
                    "id": "test123",
                    "start_time": "2025-09-29T10:00:00Z",
                    "latest_step": 1000,
                    "eta": "2025-09-30T10:00:00Z",
                    "status": "running",
                }
            ]
        }

        summaries = {
            "test123": {
                "rl_sharpe": 1.5,
                "dsr": 0.8,
                "p_value": 0.05,
                "memory_peak": 1024,
                "rss_peak": 512,
            }
        }

        html = generate_html(index_data, summaries)

        self.assertIn("Zaif Trade Bot Dashboard", html)
        self.assertIn("test123", html)
        self.assertIn("running", html)
        self.assertIn("1.5", html)
        self.assertIn("1024", html)

    def test_generate_html_empty(self):
        """Test HTML generation with empty data."""
        html = generate_html({"sessions": []}, {})

        self.assertIn("Zaif Trade Bot Dashboard", html)
        self.assertIn("<table>", html)


if __name__ == "__main__":
    unittest.main()
