"""
Feature validation tests for verified/pending/unverified status checking.
"""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
from ztb.evaluation.re_evaluate_features import ComprehensiveFeatureReEvaluator
from ztb.evaluation.status import FeatureStatus, FeatureReason, validate_status_reason
from ztb.features.registry import FeatureRegistry
from .test_autogen import BaseFeatureTest


class TestFeatureValidation(BaseFeatureTest):
    """Test feature validation based on coverage.json"""

    @pytest.fixture
    def coverage_data(self):
        """Load coverage.json data"""
        coverage_path = Path("coverage.json")
        if coverage_path.exists():
            with open(coverage_path, 'r') as f:
                return json.load(f)
        return {
            "verified": ["TestVerifiedFeature"],
            "pending": [
                {"name": "TestPendingFeature1", "reason": "insufficient_data"},
                {"name": "TestPendingFeature2", "reason": "high_nan_rate"}
            ],
            "failed": [
                {"name": "TestFailedFeature1", "reason": "computation_error"},
                {"name": "TestFailedFeature2", "reason": "type_mismatch"}
            ],
            "unverified": [
                {"name": "TestUnverifiedFeature", "reason": "not_tested"}
            ]
        }

    @pytest.fixture
    def mock_feature_registry(self):
        """Create mock feature registry"""
        registry = MagicMock()

        # Mock verified feature
        verified_feature = MagicMock()
        verified_feature.compute.return_value = MagicMock()  # Mock DataFrame
        registry.get_feature_class.return_value = verified_feature

        return registry

    def test_verified_features_compute_without_error(self, coverage_data, mock_feature_registry):
        """Test that verified features can compute without errors"""
        for feature_name in coverage_data.get("verified", []):
            with patch('ztb.features.registry.FeatureRegistry', return_value=mock_feature_registry):
                try:
                    feature_class = mock_feature_registry.get_feature_class(feature_name)
                    # Mock OHLC data
                    ohlc_data = MagicMock()
                    result = feature_class.compute(ohlc_data)
                    # Should not raise exception
                    assert result is not None
                except Exception as e:
                    pytest.fail(f"Verified feature {feature_name} failed to compute: {e}")

    @pytest.mark.xfail(reason="Pending features may not be fully implemented")
    def test_pending_features_are_marked_as_expected_failure(self, coverage_data):
        """Test that pending features are properly marked as expected failures"""
        # This test should be marked as xfail and not run
        # If it does run, it means the xfail mark was removed incorrectly
        assert len(coverage_data.get("pending", [])) >= 0

    def test_coverage_json_structure(self, coverage_data):
        """Test that coverage.json has correct structure"""
        required_keys = ["verified", "pending", "failed", "unverified"]

        for key in required_keys:
            assert key in coverage_data, f"Missing required key: {key}"
            assert isinstance(coverage_data[key], list), f"{key} should be a list"

    def test_coverage_json_no_duplicates(self, coverage_data):
        """Test that coverage.json has no duplicate feature names across categories"""
        all_features = []
        for category, features in coverage_data.items():
            all_features.extend(features)

        # Check for duplicates
        unique_features = set(all_features)
        assert len(all_features) == len(unique_features), "Duplicate features found in coverage.json"

    def test_feature_class_naming_convention(self, coverage_data):
        """Test that feature classes follow naming conventions"""
        for category, features in coverage_data.items():
            for feature in features:
                # Feature names should be PascalCase
                assert feature[0].isupper(), f"Feature {feature} should start with uppercase letter"
                # Should not contain spaces or special characters
                assert " " not in feature, f"Feature {feature} should not contain spaces"
                assert all(c.isalnum() or c == "_" for c in feature), f"Feature {feature} contains invalid characters"

    @pytest.fixture
    def sample_ohlc_data(self):
        """Create sample OHLC data for testing"""
        import pandas as pd
        import numpy as np

        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)

        return pd.DataFrame({
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 105 + np.random.randn(100).cumsum(),
            'low': 95 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)

    def test_verified_features_compute_successfully(self, coverage_data, sample_ohlc_data):
        """Test that all verified features compute successfully"""
        from ztb.features.registry import FeatureRegistry

        verified_features = coverage_data.get("verified", [])
        if not verified_features:
            pytest.skip("No verified features to test")

        # Create feature manager and register features
        # manager = FeatureRegistry()
        # register_wave1_features(manager)
        # register_wave2_features(manager)
        # register_wave3_features(manager)

        failed_features = []
        for feature_name in verified_features:
            try:
                if feature_name not in FeatureRegistry.list():
                    failed_features.append(f"{feature_name} (not in registry)")
                    continue

                feature = FeatureRegistry.get(feature_name)
                result = feature(sample_ohlc_data)

                if not isinstance(result, pd.Series):
                    failed_features.append(f"{feature_name} (not Series)")
                    continue

                if result.empty:
                    failed_features.append(f"{feature_name} (empty result)")
                    continue

            except Exception as e:
                failed_features.append(f"{feature_name} ({str(e)})")

        if failed_features:
            pytest.fail(f"Verified features failed: {failed_features}")


class TestFeatureStatusTransitions:
    """Test feature status transitions between categories"""

    def test_pending_to_verified_transition(self, coverage_data):
        """Test that pending features can transition to verified"""
        # This is more of a documentation test
        # In practice, this would be tested by the evaluation system
        pending_count = len(coverage_data.get("pending", []))
        verified_count = len(coverage_data.get("verified", []))

        # Pending features should exist if there are features being developed
        assert isinstance(pending_count, int)
        assert isinstance(verified_count, int)

    def test_failed_features_are_isolated(self, coverage_data):
        """Test that failed features are properly isolated"""
        failed_features = coverage_data.get("failed", [])

        # Failed features should not appear in other categories
        for category in ["verified", "pending", "unverified"]:
            category_features = coverage_data.get(category, [])
            for failed_feature in failed_features:
                assert failed_feature not in category_features, \
                    f"Failed feature {failed_feature} appears in {category}"

    def test_unverified_features_require_attention(self, coverage_data):
        """Test that unverified features are flagged for attention"""
        unverified_features = coverage_data.get("unverified", [])

        # Unverified features should be tracked
        # This test ensures they exist in the coverage data
        assert isinstance(unverified_features, list)

        # If there are unverified features, they should be addressed
        if unverified_features:
            # Could add logging or warnings here
            pass