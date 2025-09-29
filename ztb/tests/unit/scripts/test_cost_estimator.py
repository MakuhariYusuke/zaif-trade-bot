import json
import tempfile
import unittest
from pathlib import Path

from ztb.ops.artifacts.cost_estimator import (
    estimate_cost,
    generate_markdown,
    load_metadata,
)


class TestCostEstimator(unittest.TestCase):
    def test_load_metadata_success(self):
        """Test loading metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sess_dir = root / "test_sess"
            sess_dir.mkdir()
            metadata = {
                "correlation_id": "test",
                "duration_seconds": 3600,
                "gpu_count": 2,
            }
            (sess_dir / "run_metadata.json").write_text(json.dumps(metadata))

            result = load_metadata("test_sess", root)
            self.assertEqual(result["correlation_id"], "test")

    def test_estimate_cost(self):
        """Test cost estimation."""
        metadata = {
            "correlation_id": "test",
            "duration_seconds": 3600,  # 1 hour
            "gpu_count": 2,
        }
        summary = {"summary": {"global_step": 1000}}

        estimate = estimate_cost(
            metadata, summary, gpu_rate=300, kwh_rate=35, cloud_rate=50
        )

        self.assertEqual(estimate["duration_hours"], 1.0)
        self.assertEqual(estimate["gpu_hours"], 2.0)  # 2 GPUs * 1 hour
        self.assertEqual(estimate["power_kwh"], 0.6)  # 0.3 kW * 2 GPUs * 1 hour
        self.assertEqual(estimate["steps_per_sec"], 1000 / 3600)
        self.assertEqual(estimate["costs"]["gpu_jpy"], 600)  # 2 * 300
        self.assertEqual(estimate["costs"]["power_jpy"], 21)  # 0.6 * 35
        self.assertEqual(estimate["costs"]["cloud_jpy"], 50)  # 1 * 50

    def test_estimate_cost_jp_tiered_edges(self):
        """Test JP residential tiered pricing at edge values."""
        metadata = {"correlation_id": "test", "duration_seconds": 3600, "gpu_count": 1}
        summary = {"summary": {"global_step": 1000}}

        # Test 119 kWh (tier 1: 119 * 29.70)
        estimate_119 = estimate_cost(
            metadata, summary, gpu_rate=0, kwh_rate=35, manual_kwh=119
        )
        expected_119 = 119 * 29.70 + 1246.96  # basic fee included
        self.assertAlmostEqual(
            estimate_119["costs"]["power_jpy"], expected_119, places=2
        )

        # Test 120 kWh (tier 2 start: 120 * 29.70)
        estimate_120 = estimate_cost(
            metadata, summary, gpu_rate=0, kwh_rate=35, manual_kwh=120
        )
        expected_120 = 120 * 29.70 + 1246.96
        self.assertAlmostEqual(
            estimate_120["costs"]["power_jpy"], expected_120, places=2
        )

        # Test 299 kWh (tier 2: 120*29.70 + 179*35.69)
        estimate_299 = estimate_cost(
            metadata, summary, gpu_rate=0, kwh_rate=35, manual_kwh=299
        )
        expected_299 = 120 * 29.70 + 179 * 35.69 + 1246.96
        self.assertAlmostEqual(
            estimate_299["costs"]["power_jpy"], expected_299, places=2
        )

        # Test 300 kWh (tier 3 start: 120*29.70 + 180*35.69 + 0*39.50)
        estimate_300 = estimate_cost(
            metadata, summary, gpu_rate=0, kwh_rate=35, manual_kwh=300
        )
        expected_300 = 120 * 29.70 + 180 * 35.69 + 1246.96
        self.assertAlmostEqual(
            estimate_300["costs"]["power_jpy"], expected_300, places=2
        )

    def test_generate_markdown(self):
        """Test Markdown generation."""
        estimate = {
            "correlation_id": "test",
            "duration_hours": 1.0,
            "gpu_count": 1,
            "gpu_hours": 1.0,
            "power_kwh": 0.3,
            "steps_per_sec": 0.5,
            "costs": {
                "gpu_jpy": 300,
                "power_jpy": 10.5,
                "cloud_jpy": 0,
                "total_jpy": 310.5,
            },
            "rates": {
                "gpu_rate_jpy_per_hour": 300,
                "kwh_rate_jpy": 35,
                "cloud_rate_jpy_per_hour": 0,
            },
        }

        md = generate_markdown(estimate)
        self.assertIn("# Cost Estimate: test", md)
        self.assertIn("¥310", md)  # Total cost


if __name__ == "__main__":
    unittest.main()
