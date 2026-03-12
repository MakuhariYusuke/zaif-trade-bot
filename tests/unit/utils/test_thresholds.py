"""
Tests for threshold management utilities.
"""

from unittest.mock import patch
from pathlib import Path

from ztb.utils.thresholds import AdaptiveThresholdManager


class TestAdaptiveThresholdManager:
    """Test cases for AdaptiveThresholdManager."""

    def test_inheritance(self):
        """Test that AdaptiveThresholdManager implements ThresholdManagerProtocol."""
        # This test ensures the class properly implements the protocol
        assert isinstance(AdaptiveThresholdManager, type)
        # Check if it has the required methods
        manager = AdaptiveThresholdManager.__new__(AdaptiveThresholdManager)
        assert hasattr(manager, 'get_adaptive_gates')
        assert hasattr(manager, 'update_thresholds')

    @patch('ztb.utils.thresholds.Path.exists')
    @patch('ztb.utils.thresholds.CoverageValidator.load_coverage_files')
    def test_initialization_with_historical_data(self, mock_load_coverage, mock_exists):
        """Test initialization with historical data."""
        mock_exists.return_value = True
        mock_load_coverage.return_value = {
            'events': [
                {
                    'type': 'feature_promoted',
                    'to_status': 'verified',
                    'feature': 'test_feature',
                    'details': {'criterion_details': {'metric1': 0.8}}
                }
            ]
        }

        manager = AdaptiveThresholdManager('dummy_path')

        assert manager.historical_data_path == Path('dummy_path')
        assert 'test_feature' in manager.historical_successes

    def test_get_adaptive_gates(self):
        """Test get_adaptive_gates method."""
        manager = AdaptiveThresholdManager('dummy_path')

        gates = manager.get_adaptive_gates()

        assert isinstance(gates, dict)
        assert 'nan_rate_threshold' in gates
        assert 'correlation_threshold' in gates
        assert 'skew_threshold' in gates
        assert 'kurtosis_threshold' in gates

    def test_update_thresholds(self):
        """Test update_thresholds method."""
        manager = AdaptiveThresholdManager('dummy_path')
        evaluation_results = {'some_metric': 'data'}

        # Should not raise an exception
        manager.update_thresholds(evaluation_results)

        # The method currently does nothing, but we test it doesn't break
        assert True