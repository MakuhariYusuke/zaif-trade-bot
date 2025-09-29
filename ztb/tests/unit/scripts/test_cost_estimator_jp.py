#!/usr/bin/env python3
"""
Unit tests for Japanese residential electricity tariff in cost_estimator.py
"""

import sys
import unittest
from pathlib import Path

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from cost_estimator import calculate_jp_residential_tiered, estimate_cost


class TestCostEstimatorJP(unittest.TestCase):
    def test_calculate_jp_residential_tiered_tier1(self):
        """Test tier 1 (0-120 kWh)."""
        # 100 kWh
        cost = calculate_jp_residential_tiered(100)
        expected_power = 100 * 29.70
        expected_total = expected_power + 1246.96
        self.assertAlmostEqual(cost, expected_total, places=2)

    def test_calculate_jp_residential_tiered_tier2(self):
        """Test tier 2 (120-300 kWh)."""
        # 200 kWh
        cost = calculate_jp_residential_tiered(200)
        expected_power = 120 * 29.70 + 80 * 35.69
        expected_total = expected_power + 1246.96
        self.assertAlmostEqual(cost, expected_total, places=2)

    def test_calculate_jp_residential_tiered_tier3(self):
        """Test tier 3 (300+ kWh)."""
        # 400 kWh
        cost = calculate_jp_residential_tiered(400)
        expected_power = 120 * 29.70 + 180 * 35.69 + 100 * 39.50
        expected_total = expected_power + 1246.96
        self.assertAlmostEqual(cost, expected_total, places=2)

    def test_calculate_jp_residential_tiered_boundary_120(self):
        """Test boundary at 120 kWh."""
        cost = calculate_jp_residential_tiered(120)
        expected_power = 120 * 29.70
        expected_total = expected_power + 1246.96
        self.assertAlmostEqual(cost, expected_total, places=2)

    def test_calculate_jp_residential_tiered_boundary_300(self):
        """Test boundary at 300 kWh."""
        cost = calculate_jp_residential_tiered(300)
        expected_power = 120 * 29.70 + 180 * 35.69
        expected_total = expected_power + 1246.96
        self.assertAlmostEqual(cost, expected_total, places=2)

    def test_estimate_cost_jp_tariff(self):
        """Test estimate_cost with jp_residential_tiered tariff."""
        metadata = {
            "correlation_id": "test123",
            "duration_seconds": 3600,  # 1 hour
            "gpu_count": 1,
        }
        summary = {"summary": {"global_step": 1000}}

        estimate = estimate_cost(
            metadata, summary, gpu_rate=300, kwh_rate=35, tariff="jp_residential_tiered"
        )

        # Check basic fields
        self.assertEqual(estimate["correlation_id"], "test123")
        self.assertEqual(estimate["duration_hours"], 1.0)
        self.assertEqual(estimate["gpu_count"], 1)
        self.assertEqual(estimate["gpu_hours"], 1.0)
        self.assertEqual(estimate["power_kwh"], 0.3)  # 0.3 kW * 1 hour
        self.assertEqual(estimate["steps_per_sec"], 1000 / 3600)

        # Check costs
        self.assertEqual(estimate["costs"]["gpu_jpy"], 300.0)  # 1 hour * 300
        # Power cost should be tiered calculation for 0.3 kWh
        expected_power_cost = 0.3 * 29.70 + 1246.96
        self.assertAlmostEqual(
            estimate["costs"]["power_jpy"], expected_power_cost, places=2
        )
        self.assertEqual(estimate["costs"]["cloud_jpy"], 0.0)
        self.assertAlmostEqual(
            estimate["costs"]["total_jpy"], 300 + expected_power_cost, places=2
        )

        # Check rates
        self.assertEqual(estimate["rates"]["tariff"], "jp_residential_tiered")

    def test_estimate_cost_jp_tariff_manual_kwh(self):
        """Test estimate_cost with manual kWh override."""
        metadata = {
            "correlation_id": "test123",
            "duration_seconds": 3600,
            "gpu_count": 1,
        }
        summary = {"summary": {"global_step": 1000}}

        # Override with 200 kWh
        estimate = estimate_cost(
            metadata,
            summary,
            gpu_rate=300,
            kwh_rate=35,
            tariff="jp_residential_tiered",
            manual_kwh=200,
        )

        self.assertEqual(estimate["power_kwh"], 200)
        # Power cost for 200 kWh (tier 2)
        expected_power_cost = 120 * 29.70 + 80 * 35.69 + 1246.96
        self.assertAlmostEqual(
            estimate["costs"]["power_jpy"], expected_power_cost, places=2
        )

    def test_estimate_cost_simple_tariff_fallback(self):
        """Test estimate_cost falls back to simple tariff."""
        metadata = {
            "correlation_id": "test123",
            "duration_seconds": 3600,
            "gpu_count": 1,
        }
        summary = {"summary": {"global_step": 1000}}

        estimate = estimate_cost(
            metadata, summary, gpu_rate=300, kwh_rate=35, tariff="simple"
        )

        # Simple tariff: just kwh * rate
        expected_power_cost = 0.3 * 35
        self.assertEqual(estimate["costs"]["power_jpy"], expected_power_cost)
        self.assertEqual(estimate["rates"]["tariff"], "simple")


if __name__ == "__main__":
    unittest.main()
