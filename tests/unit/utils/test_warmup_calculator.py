"""Tests for warmup_calculator module."""

import math
import pytest

from ztb.utils.warmup_calculator import (
    get_max_lookback,
    calculate_warmup,
    get_warmup_with_metadata,
    validate_warmup,
)


class TestGetMaxLookback:
    """Test get_max_lookback function."""
    
    def test_returns_positive_integer(self):
        """Max lookback should be positive."""
        max_lookback = get_max_lookback()
        assert isinstance(max_lookback, int)
        assert max_lookback > 0
    
    def test_includes_common_indicators(self):
        """Max lookback should account for common long-period indicators."""
        max_lookback = get_max_lookback()
        # Should be at least 200 for SMA_200, or 52 for Ichimoku
        assert max_lookback >= 52
    
    def test_deterministic(self):
        """Should return same value on repeated calls."""
        lookback1 = get_max_lookback()
        lookback2 = get_max_lookback()
        assert lookback1 == lookback2


class TestCalculateWarmup:
    """Test calculate_warmup function."""
    
    def test_default_margin(self):
        """Default 10% safety margin."""
        max_lookback = get_max_lookback()
        warmup = calculate_warmup()
        
        # Should be ceiling of max_lookback * 1.1
        expected = math.ceil(max_lookback * 1.1)
        assert warmup == expected
    
    def test_custom_margin(self):
        """Custom safety margin."""
        max_lookback = get_max_lookback()
        warmup = calculate_warmup(safety_margin=0.2)
        
        # Should be ceiling of max_lookback * 1.2
        expected = math.ceil(max_lookback * 1.2)
        assert warmup == expected
    
    def test_zero_margin(self):
        """Zero safety margin."""
        max_lookback = get_max_lookback()
        warmup = calculate_warmup(safety_margin=0.0)
        
        # Should be exactly max_lookback
        assert warmup == max_lookback
    
    def test_warmup_greater_than_lookback(self):
        """Warmup should be >= max_lookback."""
        max_lookback = get_max_lookback()
        warmup = calculate_warmup()
        assert warmup >= max_lookback


class TestGetWarmupWithMetadata:
    """Test get_warmup_with_metadata function."""
    
    def test_returns_dict_with_required_keys(self):
        """Should return dict with max_lookback, safety_margin, warmup."""
        metadata = get_warmup_with_metadata()
        
        assert isinstance(metadata, dict)
        assert "max_lookback" in metadata
        assert "safety_margin" in metadata
        assert "warmup" in metadata
    
    def test_metadata_consistency(self):
        """Metadata values should be consistent."""
        metadata = get_warmup_with_metadata(safety_margin=0.15)
        
        expected_warmup = math.ceil(metadata["max_lookback"] * (1 + metadata["safety_margin"]))
        assert metadata["warmup"] == expected_warmup
        assert metadata["safety_margin"] == 0.15
    
    def test_default_values(self):
        """Test default safety margin value."""
        metadata = get_warmup_with_metadata()
        assert metadata["safety_margin"] == 0.1


class TestValidateWarmup:
    """Test validate_warmup function."""
    
    def test_sufficient_warmup(self):
        """Warmup greater than or equal to required should pass."""
        required = calculate_warmup()
        
        assert validate_warmup(required) is True
        assert validate_warmup(required + 50) is True
    
    def test_insufficient_warmup(self):
        """Warmup less than required should fail."""
        required = calculate_warmup()
        
        assert validate_warmup(required - 1) is False
        assert validate_warmup(50) is False  # Definitely too small
    
    def test_custom_margin_validation(self):
        """Validation with custom safety margin."""
        # With 20% margin
        required = calculate_warmup(safety_margin=0.2)
        
        # Should fail with default margin warmup (10%)
        default_warmup = calculate_warmup(safety_margin=0.1)
        if default_warmup < required:
            assert validate_warmup(default_warmup, safety_margin=0.2) is False
        
        # Should pass with sufficient warmup
        assert validate_warmup(required, safety_margin=0.2) is True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_large_margin(self):
        """Very large safety margin should still work."""
        max_lookback = get_max_lookback()
        warmup = calculate_warmup(safety_margin=2.0)  # 200% margin
        
        expected = math.ceil(max_lookback * 3.0)
        assert warmup == expected
    
    def test_warmup_always_positive(self):
        """Warmup should always be positive."""
        warmup = calculate_warmup()
        assert warmup > 0
        
        warmup = calculate_warmup(safety_margin=0.0)
        assert warmup > 0
    
    def test_exact_values(self):
        """Test with known max_lookback value."""
        max_lookback = get_max_lookback()
        # From our lookback_map, max should be 200 (SMA_200)
        assert max_lookback == 200
        
        # With 10% margin: 200 * 1.1 = 220, but ceil(220.0) = 220
        # Actually: 200 * 1.1 = 220.0, ceil(220.0) = 220
        # But if there's floating point precision: ceil(200 * 1.1) might be 221
        warmup = calculate_warmup()
        # Accept both 220 and 221 due to floating point
        assert warmup in (220, 221)
