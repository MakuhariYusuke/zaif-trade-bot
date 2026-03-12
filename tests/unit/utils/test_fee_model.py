"""
Tests for fee model utilities.
"""

from ztb.utils.fee_model import FeeModel, FixedFeeModel, TieredFeeModel, ExchangeFeeModel
from ztb.utils.types import FeeModelProtocol


class TestFeeModelProtocol:
    """Test cases for FeeModelProtocol implementation."""

    def test_fee_model_inheritance(self):
        """Test that FeeModel implements FeeModelProtocol."""
        # Test abstract base class
        assert issubclass(FeeModel, FeeModelProtocol)

    def test_fixed_fee_model(self):
        """Test FixedFeeModel implementation."""
        model = FixedFeeModel()

        fee = model.calculate_fee(1000.0, 'buy')
        assert isinstance(fee, float)
        assert fee >= 0

        rate = model.get_fee_rate('buy')
        assert isinstance(rate, float)
        assert 0 <= rate <= 1

    def test_tiered_fee_model(self):
        """Test TieredFeeModel implementation."""
        model = TieredFeeModel()

        fee = model.calculate_fee(1000.0, 'buy')
        assert isinstance(fee, float)
        assert fee >= 0

        rate = model.get_fee_rate('buy')
        assert isinstance(rate, float)
        assert 0 <= rate <= 1

    def test_exchange_fee_model(self):
        """Test ExchangeFeeModel implementation."""
        model = ExchangeFeeModel()

        fee = model.calculate_fee(1000.0, 'buy')
        assert isinstance(fee, float)
        assert fee >= 0

        rate = model.get_fee_rate('buy')
        assert isinstance(rate, float)
        assert 0 <= rate <= 1


class TestFixedFeeModel:
    """Detailed tests for FixedFeeModel."""

    def test_default_rates(self):
        """Test default fee rates."""
        model = FixedFeeModel()

        assert model.buy_fee_rate > 0
        assert model.sell_fee_rate > 0

    def test_custom_rates(self):
        """Test custom fee rates."""
        model = FixedFeeModel(buy_fee_rate=0.001, sell_fee_rate=0.002)

        assert model.buy_fee_rate == 0.001
        assert model.sell_fee_rate == 0.002

    def test_fee_calculation(self):
        """Test fee calculation."""
        model = FixedFeeModel(buy_fee_rate=0.001, sell_fee_rate=0.002)

        buy_fee = model.calculate_fee(100000.0, 'buy')
        sell_fee = model.calculate_fee(100000.0, 'sell')

        assert buy_fee == 100.0  # 100000 * 0.001
        assert sell_fee == 200.0  # 100000 * 0.002