#!/usr/bin/env python3
"""
Unit tests for coverage validation enhancements
"""
import unittest
import json
from unittest.mock import Mock, patch

from ztb.evaluation.status import CoverageValidator, validate_coverage_comprehensive, _get_feature_category


class TestCoverageValidation(unittest.TestCase):
    """Test cases for enhanced coverage validation"""

    def setUp(self):
        """Set up test fixtures"""
        self.validator = CoverageValidator()

        # Sample coverage data
        self.sample_coverage = {
            "verified": ["ema_12", "ema_26", "kama_5_20_10"],
            "pending": ["sma_20"],
            "failed": ["invalid_feature"],
            "business_rules": {
                "trend_features_min": 2,
                "oscillator_features_min": 1,
                "total_verified_min": 5
            }
        }

    def test_merge_coverage_data_strictness_priority(self):
        """Test merging with strictness priority (VERIFIED > PENDING > UNVERIFIED > FAILED)"""
        existing = {
            "verified": ["ema_12"],
            "pending": [{"name": "ema_26", "reason": "insufficient_data"}],
            "unverified": [{"name": "sma_20", "reason": "not_tested"}],
            "failed": [],
            "metadata": {"source_files": []}
        }

        new_data = {
            "verified": ["kama_5_20_10"],
            "pending": [{"name": "ema_12", "reason": "high_nan_rate"}],  # Lower priority - should not override VERIFIED
            "unverified": [{"name": "ema_26", "reason": "not_tested"}],   # Higher priority - should override PENDING
            "failed": [{"name": "sma_20", "reason": "computation_error"}] # Lower priority - should not override UNVERIFIED
        }

        CoverageValidator._merge_coverage_data(existing, new_data, "test_file.json")

        # VERIFIED should not be overridden by lower priority
        self.assertIn("ema_12", existing["verified"])

        # PENDING should NOT be overridden by UNVERIFIED (lower priority)
        ema_26_items = [item for item in existing["pending"] if item["name"] == "ema_26"]
        self.assertEqual(len(ema_26_items), 1)

        # UNVERIFIED should not be overridden by FAILED
        sma_20_items = [item for item in existing["unverified"] if item["name"] == "sma_20"]
        self.assertEqual(len(sma_20_items), 1)

        # New feature should be added
        self.assertIn("kama_5_20_10", existing["verified"])

    def test_validate_coverage_comprehensive_business_rules(self):
        """Test comprehensive business rule validation"""
        # Valid coverage
        valid_coverage = {
            "verified": ["ema_12", "ema_26", "kama_5_20_10", "rsi_14", "macd_12_26_9", "sma_20"],
            "pending": [{"name": "stoch_14_3_3", "reason": "insufficient_data"}],
            "failed": [{"name": "invalid_feature", "reason": "computation_error"}],
            "unverified": [],
            "business_rules": {
                "trend_features_min": 3,
                "oscillator_features_min": 2,
                "total_verified_min": 5
            }
        }

        is_valid, errors = validate_coverage_comprehensive(valid_coverage)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_coverage_comprehensive_insufficient_trend(self):
        """Test validation fails with insufficient trend features"""
        invalid_coverage = {
            "verified": ["rsi_14", "macd_12_26_9"],  # Only oscillators, no trend
            "pending": [],
            "failed": [],
            "unverified": [],
            "business_rules": {
                "trend_features_min": 1,
                "oscillator_features_min": 1,
                "total_verified_min": 1
            }
        }

        is_valid, errors = validate_coverage_comprehensive(invalid_coverage)
        self.assertFalse(is_valid)
        self.assertIn("trend", errors[0].lower())

    def test_validate_coverage_comprehensive_insufficient_total(self):
        """Test validation fails with insufficient total verified features"""
        invalid_coverage = {
            "verified": ["ema_12"],  # Only 1 verified feature
            "pending": [],
            "failed": [],
            "unverified": [],
            "business_rules": {
                "trend_features_min": 1,
                "oscillator_features_min": 0,
                "total_verified_min": 2
            }
        }

        is_valid, errors = validate_coverage_comprehensive(invalid_coverage)
        self.assertFalse(is_valid)
        self.assertIn("total", errors[0].lower())

    def test_validate_coverage_comprehensive_missing_rules(self):
        """Test validation with missing business rules"""
        coverage_without_rules = {
            "verified": ["ema_12", "ema_26"],
            "pending": [],
            "failed": [],
            "unverified": []
            # No business_rules key
        }

        is_valid, errors = validate_coverage_comprehensive(coverage_without_rules)
        self.assertFalse(is_valid)
        self.assertIn("business_rules", errors[0].lower())

    def test_get_feature_category_trend(self):
        """Test feature category detection for trend indicators"""
        trend_features = ["ema_12", "sma_20", "kama_5_20_10", "tema_20", "dema_15"]

        for feature in trend_features:
            with self.subTest(feature=feature):
                category = _get_feature_category(feature)
                self.assertEqual(category, "trend")

    def test_get_feature_category_oscillator(self):
        """Test feature category detection for oscillators"""
        oscillator_features = ["rsi_14", "stoch_14_3_3", "macd_12_26_9", "cci_20", "williams_r_14"]

        for feature in oscillator_features:
            with self.subTest(feature=feature):
                category = _get_feature_category(feature)
                self.assertEqual(category, "oscillator")

    def test_get_feature_category_other(self):
        """Test feature category detection for unknown categories"""
        other_features = ["unknown_feature", "custom_indicator", "random_calc"]

        for feature in other_features:
            with self.subTest(feature=feature):
                category = _get_feature_category(feature)
                self.assertEqual(category, "other")


if __name__ == '__main__':
    unittest.main()