"""Tests for validation utility functions in errors module."""

import pytest

from ztb.utils.errors import (
    ValidationError,
    validate_batch,
    validate_non_negative,
    validate_portfolio_value,
    validate_positive,
    validate_price,
    validate_price_range,
    validate_quantity,
    validate_quantity_range,
    validate_range,
    validate_type,
    validate_volatility,
)


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_positive_valid_values(self) -> None:
        """Test validate_positive with valid values."""
        # Should not raise
        validate_positive(1.0)
        validate_positive(100.5)
        validate_positive(0.001)

    def test_validate_positive_invalid_values(self) -> None:
        """Test validate_positive with invalid values."""
        with pytest.raises(ValidationError, match="test_value must be positive"):
            validate_positive(0.0, "test_value")

        with pytest.raises(ValidationError, match="price must be positive"):
            validate_positive(-1.0, "price")

        with pytest.raises(ValidationError, match="amount must be positive"):
            validate_positive(-100.0, "amount")

    def test_validate_non_negative_valid_values(self) -> None:
        """Test validate_non_negative with valid values."""
        # Should not raise
        validate_non_negative(0.0)
        validate_non_negative(1.0)
        validate_non_negative(100.5)

    def test_validate_non_negative_invalid_values(self) -> None:
        """Test validate_non_negative with invalid values."""
        with pytest.raises(ValidationError, match="portfolio must be non-negative"):
            validate_non_negative(-1.0, "portfolio")

        with pytest.raises(ValidationError, match="value must be non-negative"):
            validate_non_negative(-0.001, "value")

    def test_validate_price_valid_values(self) -> None:
        """Test validate_price with valid values."""
        # Should not raise
        validate_price(100000.0)
        validate_price(50000.5)
        validate_price(0.01)

    def test_validate_price_invalid_values(self) -> None:
        """Test validate_price with invalid values."""
        with pytest.raises(ValidationError, match="price must be positive"):
            validate_price(0.0)

        with pytest.raises(ValidationError, match="current_price must be positive"):
            validate_price(-50000.0, "current_price")

    def test_validate_quantity_valid_values(self) -> None:
        """Test validate_quantity with valid values."""
        # Should not raise
        validate_quantity(0.001)
        validate_quantity(1.0)
        validate_quantity(0.000001)

    def test_validate_quantity_invalid_values(self) -> None:
        """Test validate_quantity with invalid values."""
        with pytest.raises(ValidationError, match="quantity must be positive"):
            validate_quantity(0.0)

        with pytest.raises(ValidationError, match="amount must be positive"):
            validate_quantity(-0.001, "amount")

    def test_validate_portfolio_value_valid_values(self) -> None:
        """Test validate_portfolio_value with valid values."""
        # Should not raise
        validate_portfolio_value(0.0)
        validate_portfolio_value(1000000.0)
        validate_portfolio_value(500000.5)

    def test_validate_portfolio_value_invalid_values(self) -> None:
        """Test validate_portfolio_value with invalid values."""
        with pytest.raises(
            ValidationError, match="portfolio_value must be non-negative"
        ):
            validate_portfolio_value(-1.0)

        with pytest.raises(ValidationError, match="balance must be non-negative"):
            validate_portfolio_value(-1000.0, "balance")

    def test_validate_volatility_valid_values(self) -> None:
        """Test validate_volatility with valid values."""
        # Should not raise
        validate_volatility(0.0)
        validate_volatility(0.15)
        validate_volatility(0.05)

    def test_validate_volatility_invalid_values(self) -> None:
        """Test validate_volatility with invalid values."""
        with pytest.raises(ValidationError, match="volatility must be non-negative"):
            validate_volatility(-0.01)

        with pytest.raises(ValidationError, match="market_vol must be non-negative"):
            validate_volatility(-0.05, "market_vol")

    def test_validate_range_valid_values(self) -> None:
        """Test validate_range with valid values."""
        # Should not raise
        validate_range(5.0, 0.0, 10.0)
        validate_range(0.0, min_val=0.0)  # No upper bound
        validate_range(10.0, max_val=10.0)  # No lower bound
        validate_range(5.0)  # No bounds

    def test_validate_range_invalid_values(self) -> None:
        """Test validate_range with invalid values."""
        with pytest.raises(ValidationError, match="value must be >= 0"):
            validate_range(-1.0, 0.0, 10.0)

        with pytest.raises(ValidationError, match="price must be <= 100"):
            validate_range(150.0, 0.0, 100.0, "price")

    def test_validate_type_valid_values(self) -> None:
        """Test validate_type with valid values."""
        # Should not raise
        validate_type(5.0, float)
        validate_type("test", str)
        validate_type(42, int)

    def test_validate_type_invalid_values(self) -> None:
        """Test validate_type with invalid values."""
        with pytest.raises(ValidationError, match=r"value must be of type int"):
            validate_type(5.0, int, "value")

        with pytest.raises(ValidationError, match="price must be of type float"):
            validate_type("100", float, "price")

    def test_validate_price_range_valid_values(self) -> None:
        """Test validate_price_range with valid values."""
        # Should not raise
        validate_price_range(100000.0)
        validate_price_range(50000.5)
        validate_price_range(1000.0)

    def test_validate_price_range_invalid_values(self) -> None:
        """Test validate_price_range with invalid values."""
        with pytest.raises(ValidationError, match="price must be positive"):
            validate_price_range(0.0)

        with pytest.raises(ValidationError, match="current_price must be >= 1"):
            validate_price_range(0.5, name="current_price")

        with pytest.raises(ValidationError, match="price must be <= 100000000"):
            validate_price_range(200000000.0)

    def test_validate_quantity_range_valid_values(self) -> None:
        """Test validate_quantity_range with valid values."""
        # Should not raise
        validate_quantity_range(0.001)
        validate_quantity_range(1.0)
        validate_quantity_range(0.1)

    def test_validate_quantity_range_invalid_values(self) -> None:
        """Test validate_quantity_range with invalid values."""
        with pytest.raises(ValidationError, match="quantity must be positive"):
            validate_quantity_range(0.0)

        with pytest.raises(ValidationError, match="amount must be >= 1e-06"):
            validate_quantity_range(0.0000001, name="amount")

        with pytest.raises(ValidationError, match="quantity must be <= 1000"):
            validate_quantity_range(2000.0)

    def test_validate_batch_valid_values(self) -> None:
        """Test validate_batch with valid values."""
        values = {"price": 100000.0, "quantity": 0.001, "volatility": 0.05}
        validators = {
            "price": lambda x: validate_price(x),
            "quantity": lambda x: validate_quantity(x),
            "volatility": lambda x: validate_volatility(x),
        }
        # Should not raise
        validate_batch(values, validators)

    def test_validate_batch_invalid_values(self) -> None:
        """Test validate_batch with invalid values."""
        values = {"price": -1000.0, "quantity": 0.001}
        validators = {
            "price": lambda x: validate_price(x),
            "quantity": lambda x: validate_quantity(x),
        }

        with pytest.raises(ValidationError, match="Validation failed for price"):
            validate_batch(values, validators)
