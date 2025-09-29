"""Unit tests for symbol normalization and precision policies."""

from decimal import Decimal

from ztb.live.precision_policy import PrecisionPolicyManager
from ztb.live.symbols import SymbolNormalizer


class TestSymbolNormalizer:
    """Test symbol normalization functionality."""

    def test_normalize_coincheck_standard(self):
        """Test Coincheck standard symbol formats."""
        normalizer = SymbolNormalizer()

        # BTC/JPY format
        assert normalizer.normalize("coincheck", "BTC/JPY") == ("BTC", "JPY")

        # BTC_JPY format
        assert normalizer.normalize("coincheck", "BTC_JPY") == ("BTC", "JPY")

        # Lowercase
        assert normalizer.normalize("coincheck", "btc/jpy") == ("BTC", "JPY")

    def test_normalize_zaif_formats(self):
        """Test Zaif symbol formats."""
        normalizer = SymbolNormalizer()

        # btc_jpy format
        assert normalizer.normalize("zaif", "btc_jpy") == ("BTC", "JPY")

        # btcfx_jpy format
        assert normalizer.normalize("zaif", "btcfx_jpy") == ("BTCFX", "JPY")

    def test_normalize_unknown_venue(self):
        """Test unknown venue defaults to coincheck."""
        normalizer = SymbolNormalizer()

        assert normalizer.normalize("unknown", "BTC/JPY") == ("BTC", "JPY")


class TestPrecisionPolicyManager:
    """Test precision policy quantization."""

    def test_quantize_price_coincheck(self):
        """Test price quantization for Coincheck."""
        manager = PrecisionPolicyManager()

        # BTC/JPY tick size = 0.01
        price = Decimal("12345.67")
        quantized = manager.quantize_price("coincheck", "BTC/JPY", price)
        assert quantized == Decimal("12345.67")  # Already compliant with 0.01 tick

        # Test rounding: 12345.678 -> 12345.68
        price = Decimal("12345.678")
        quantized = manager.quantize_price("coincheck", "BTC/JPY", price)
        assert quantized == Decimal("12345.68")

        # ETH/JPY tick size = 0.1
        price = Decimal("2345.678")
        quantized = manager.quantize_price("coincheck", "ETH/JPY", price)
        assert quantized == Decimal("2345.7")  # Rounded to nearest 0.1

    def test_quantize_quantity_coincheck(self):
        """Test quantity quantization for Coincheck."""
        manager = PrecisionPolicyManager()

        # BTC/JPY step size = 0.0001
        quantity = Decimal("1.234567")
        quantized = manager.quantize_quantity("coincheck", "BTC/JPY", quantity)
        assert quantized == Decimal("1.2345")  # Rounded down to 4 decimal places

        # ETH/JPY step size = 0.001
        quantity = Decimal("10.234567")
        quantized = manager.quantize_quantity("coincheck", "ETH/JPY", quantity)
        assert quantized == Decimal("10.234")  # Rounded down to 3 decimal places

    def test_quantize_price_zaif(self):
        """Test price quantization for Zaif."""
        manager = PrecisionPolicyManager()

        # btc_jpy tick size = 1
        price = Decimal("12347")
        quantized = manager.quantize_price("zaif", "btc_jpy", price)
        assert quantized == Decimal("12347")  # Already compliant with tick size 1

        # Test rounding: 12347.7 -> 12348
        price = Decimal("12347.7")
        quantized = manager.quantize_price("zaif", "btc_jpy", price)
        assert quantized == Decimal("12348")  # Rounded to nearest 1

    def test_quantize_quantity_zaif(self):
        """Test quantity quantization for Zaif."""
        manager = PrecisionPolicyManager()

        # btc_jpy step size = 0.0001
        quantity = Decimal("1.234567")
        quantized = manager.quantize_quantity("zaif", "btc_jpy", quantity)
        assert quantized == Decimal("1.2345")  # Rounded down to 4 decimal places

    def test_unknown_venue_defaults(self):
        """Test unknown venue defaults to coincheck policies."""
        manager = PrecisionPolicyManager()

        price = Decimal("12345.67")
        quantized = manager.quantize_price("unknown", "BTC/JPY", price)
        assert quantized == Decimal("12345.67")  # Coincheck default tick size = 0.01
