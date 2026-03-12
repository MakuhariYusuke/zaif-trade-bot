"""
Unit tests for signal common utilities

Tests the shared utility functions used across signal processing components.
"""


import pandas as pd
import pytest

from ztb.trading.signal.common.utilities import (
    calculate_confidence_score,
    normalize_weights,
    validate_market_data,
)


class TestValidateMarketData:
    """Test validate_market_data function"""

    def test_valid_market_data(self):
        """Test validation of valid market data"""
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        assert validate_market_data(data) == True

    def test_empty_dataframe(self):
        """Test validation of empty dataframe"""
        data = pd.DataFrame()

        assert validate_market_data(data) == False

    def test_missing_required_columns(self):
        """Test validation with missing required columns"""
        # Missing 'close' column
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "volume": [1000, 1100, 1200],
            }
        )

        assert validate_market_data(data) == False

    def test_partial_missing_columns(self):
        """Test validation with some missing columns"""
        # Missing 'high' and 'low'
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        assert validate_market_data(data) == False

    def test_extra_columns_allowed(self):
        """Test that extra columns are allowed"""
        data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
                "extra_col": [1, 2, 3],  # Extra column
            }
        )

        assert validate_market_data(data) == True

    def test_non_numeric_data(self):
        """Test validation with non-numeric data in required columns"""
        data = pd.DataFrame(
            {
                "open": ["a", "b", "c"],  # Non-numeric
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        # Function doesn't check data types, only column presence
        assert validate_market_data(data) == True

    def test_case_sensitive_columns(self):
        """Test that column names are case sensitive"""
        data = pd.DataFrame(
            {
                "OPEN": [100, 101, 102],  # Wrong case
                "HIGH": [105, 106, 107],
                "LOW": [95, 96, 97],
                "CLOSE": [102, 103, 104],
                "VOLUME": [1000, 1100, 1200],
            }
        )

        assert validate_market_data(data) == False


class TestCalculateConfidenceScore:
    """Test calculate_confidence_score function"""

    def test_high_confidence_score(self):
        """Test confidence calculation for high score"""
        score = 85.0
        confidence = calculate_confidence_score(score)

        assert 0.8 <= confidence <= 1.0

    def test_medium_confidence_score(self):
        """Test confidence calculation for medium score"""
        score = 65.0
        confidence = calculate_confidence_score(score)

        assert 0.4 <= confidence <= 0.7

    def test_low_confidence_score(self):
        """Test confidence calculation for low score"""
        score = 25.0
        confidence = calculate_confidence_score(score)

        assert 0.0 <= confidence <= 0.3

    def test_boundary_values(self):
        """Test confidence calculation at boundaries"""
        # Score = 100 (perfect)
        confidence = calculate_confidence_score(100.0)
        assert confidence == 1.0

        # Score = 0 (worst)
        confidence = calculate_confidence_score(0.0)
        assert confidence == 0.0

    def test_neutral_score(self):
        """Test confidence calculation for neutral score (50)"""
        confidence = calculate_confidence_score(50.0)
        assert 0.4 <= confidence <= 0.6

    @pytest.mark.parametrize(
        "score,expected_range",
        [
            (90, (0.8, 1.0)),
            (75, (0.6, 0.8)),
            (60, (0.4, 0.7)),
            (40, (0.2, 0.5)),
            (20, (0.0, 0.3)),
        ],
    )
    def test_confidence_ranges(self, score, expected_range):
        """Test confidence calculation across different score ranges"""
        confidence = calculate_confidence_score(score)
        assert expected_range[0] <= confidence <= expected_range[1]

    def test_negative_score(self):
        """Test confidence calculation for negative score"""
        confidence = calculate_confidence_score(-10.0)
        assert confidence == 0.0

    def test_score_above_100(self):
        """Test confidence calculation for score above 100"""
        confidence = calculate_confidence_score(110.0)
        assert confidence == 1.0


class TestNormalizeWeights:
    """Test normalize_weights function"""

    def test_normalize_equal_weights(self):
        """Test normalization of equal weights"""
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        normalized = normalize_weights(weights)

        expected = 1.0 / 3.0
        assert normalized["a"] == pytest.approx(expected)
        assert normalized["b"] == pytest.approx(expected)
        assert normalized["c"] == pytest.approx(expected)

    def test_normalize_unequal_weights(self):
        """Test normalization of unequal weights"""
        weights = {"a": 2.0, "b": 3.0, "c": 1.0}
        normalized = normalize_weights(weights)

        total = 2.0 + 3.0 + 1.0
        assert normalized["a"] == pytest.approx(2.0 / total)
        assert normalized["b"] == pytest.approx(3.0 / total)
        assert normalized["c"] == pytest.approx(1.0 / total)

        # Sum should be 1.0
        assert sum(normalized.values()) == pytest.approx(1.0)

    def test_normalize_zero_weights(self):
        """Test normalization with zero weights"""
        weights = {"a": 0.0, "b": 1.0, "c": 0.0}
        normalized = normalize_weights(weights)

        assert normalized["a"] == 0.0
        assert normalized["b"] == 1.0
        assert normalized["c"] == 0.0

    def test_normalize_single_weight(self):
        """Test normalization with single weight"""
        weights = {"a": 5.0}
        normalized = normalize_weights(weights)

        assert normalized["a"] == 1.0

    def test_normalize_empty_weights(self):
        """Test normalization of empty weights dict"""
        weights = {}
        normalized = normalize_weights(weights)

        assert normalized == {}

    def test_normalize_negative_weights(self):
        """Test normalization with negative weights (should handle gracefully)"""
        weights = {"a": -1.0, "b": 2.0, "c": 1.0}
        normalized = normalize_weights(weights)

        total = 3.0  # Sum of positive weights (2.0 + 1.0)
        assert normalized["a"] == 0.0  # Negative weights become 0
        assert normalized["b"] == pytest.approx(2.0 / total)
        assert normalized["c"] == pytest.approx(1.0 / total)

    def test_normalize_float_precision(self):
        """Test normalization maintains float precision"""
        weights = {"a": 1.0, "b": 1.0}
        normalized = normalize_weights(weights)

        assert isinstance(normalized["a"], float)
        assert isinstance(normalized["b"], float)
        assert normalized["a"] == 0.5
        assert normalized["b"] == 0.5

    def test_normalize_large_numbers(self):
        """Test normalization with large numbers"""
        weights = {"a": 1000.0, "b": 2000.0}
        normalized = normalize_weights(weights)

        assert normalized["a"] == pytest.approx(1.0 / 3.0)
        assert normalized["b"] == pytest.approx(2.0 / 3.0)

    def test_original_dict_unchanged(self):
        """Test that original weights dict is not modified"""
        original_weights = {"a": 1.0, "b": 2.0}
        weights_copy = original_weights.copy()

        normalize_weights(original_weights)

        assert original_weights == weights_copy

    @pytest.mark.parametrize(
        "weights_input,expected_sum",
        [
            ({"a": 1.0, "b": 1.0}, 1.0),
            ({"a": 3.0, "b": 1.0, "c": 1.0}, 1.0),
            ({"a": 0.0, "b": 0.0}, 0.0),  # All zeros
            ({"a": 1.0}, 1.0),
            ({}, 0.0),
        ],
    )
    def test_normalized_sum(self, weights_input, expected_sum):
        """Test that normalized weights sum to expected value"""
        normalized = normalize_weights(weights_input)
        total = sum(normalized.values())
        assert total == pytest.approx(expected_sum)
