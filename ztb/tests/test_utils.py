"""
Common test utilities for ztb testing.
"""

import sys
import types
from typing import Any

import pandas as pd


def create_mock_feature_engine():
    """Create a mock feature engine for testing."""
    fake_features = types.ModuleType("ztb.features")
    sys.modules["ztb.features"] = fake_features

    fake_feature_engine = types.ModuleType("ztb.features.feature_engine")

    def _compute_features_batch(df, feature_names=None, **_kwargs):
        """Mock feature computation that returns empty DataFrame."""
        return pd.DataFrame(index=df.index)

    fake_feature_engine.compute_features_batch = _compute_features_batch
    sys.modules["ztb.features.feature_engine"] = fake_feature_engine


def create_mock_observability():
    """Create a mock observability module for testing."""
    fake_observability = types.ModuleType("ztb.utils.observability")
    sys.modules["ztb.utils.observability"] = fake_observability

    def generate_correlation_id() -> str:
        return "test-correlation-id"

    fake_observability.generate_correlation_id = generate_correlation_id


def setup_test_modules():
    """Setup all mock modules needed for testing."""
    create_mock_feature_engine()
    create_mock_observability()