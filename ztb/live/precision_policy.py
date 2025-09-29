"""
Precision policies for price and quantity rounding across exchanges.

Ensures orders comply with exchange-specific tick sizes and step sizes.
"""

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal
from enum import Enum
from typing import Dict, Optional


class RoundingMode(Enum):
    """Rounding modes for price/quantity."""

    DOWN = ROUND_DOWN
    UP = ROUND_UP
    HALF_UP = ROUND_HALF_UP


@dataclass
class PrecisionPolicy:
    """Precision policy for a trading pair."""

    price_tick: Decimal  # Minimum price increment (e.g., 0.01 for JPY pairs)
    quantity_step: Decimal  # Minimum quantity increment (e.g., 0.0001 for BTC)
    price_rounding: RoundingMode = RoundingMode.HALF_UP
    quantity_rounding: RoundingMode = RoundingMode.DOWN
    min_quantity: Optional[Decimal] = None
    max_quantity: Optional[Decimal] = None
    min_price: Optional[Decimal] = None
    max_price: Optional[Decimal] = None


class PrecisionPolicyManager:
    """Manages precision policies for different venues and symbols."""

    # Default policies - can be loaded from config
    _DEFAULT_POLICIES: Dict[str, Dict[str, PrecisionPolicy]] = {
        "coincheck": {
            "BTC_JPY": PrecisionPolicy(
                price_tick=Decimal("0.01"),
                quantity_step=Decimal("0.0001"),
                min_quantity=Decimal("0.0001"),
                max_quantity=Decimal("100.0"),
            ),
            "ETH_JPY": PrecisionPolicy(
                price_tick=Decimal("0.1"),
                quantity_step=Decimal("0.001"),
                min_quantity=Decimal("0.001"),
                max_quantity=Decimal("1000.0"),
            ),
            "default": PrecisionPolicy(
                price_tick=Decimal("0.01"), quantity_step=Decimal("0.0001")
            ),
        },
        "zaif": {
            "BTC_JPY": PrecisionPolicy(
                price_tick=Decimal("1"),
                quantity_step=Decimal("0.0001"),
                min_quantity=Decimal("0.0001"),
                max_quantity=Decimal("10.0"),
            ),
            "MONA_JPY": PrecisionPolicy(
                price_tick=Decimal("0.01"),
                quantity_step=Decimal("0.1"),
                min_quantity=Decimal("0.1"),
                max_quantity=Decimal("10000.0"),
            ),
            "default": PrecisionPolicy(
                price_tick=Decimal("0.01"), quantity_step=Decimal("0.0001")
            ),
        },
    }

    def __init__(self):
        self._policies = self._DEFAULT_POLICIES.copy()

    def get_policy(self, venue: str, symbol: str) -> PrecisionPolicy:
        """Get precision policy for a venue and symbol."""
        # Normalize symbol first
        from ztb.live.symbols import SymbolNormalizer, Venue

        try:
            venue_enum = Venue(venue.lower())
        except ValueError:
            # Fallback to default if venue not recognized
            venue_enum = Venue.COINCHECK  # Default fallback
        normalizer = SymbolNormalizer()
        base, quote = normalizer.normalize(venue_enum, symbol)
        normalized_symbol = f"{base}_{quote}"

        venue_policies = self._policies.get(venue.lower(), {})
        return venue_policies.get(
            normalized_symbol.upper(),
            venue_policies.get(
                "default", PrecisionPolicy(Decimal("0.01"), Decimal("0.0001"))
            ),
        )

    def set_policy(self, venue: str, symbol: str, policy: PrecisionPolicy):
        """Set precision policy for a venue and symbol."""
        venue = venue.lower()
        symbol = symbol.upper()
        if venue not in self._policies:
            self._policies[venue] = {}
        self._policies[venue][symbol] = policy

    def quantize_price(self, venue: str, symbol: str, price: Decimal) -> Decimal:
        """Quantize price according to venue/symbol policy."""
        policy = self.get_policy(venue, symbol)

        # Apply minimum/maximum bounds
        if policy.min_price and price < policy.min_price:
            price = policy.min_price
        if policy.max_price and price > policy.max_price:
            price = policy.max_price

        # Quantize to tick size
        quantized = (price / policy.price_tick).quantize(
            Decimal("1"), rounding=policy.price_rounding.value
        ) * policy.price_tick
        return quantized

    def quantize_quantity(self, venue: str, symbol: str, quantity: Decimal) -> Decimal:
        """Quantize quantity according to venue/symbol policy."""
        policy = self.get_policy(venue, symbol)

        # Apply minimum/maximum bounds
        if policy.min_quantity and quantity < policy.min_quantity:
            quantity = policy.min_quantity
        if policy.max_quantity and quantity > policy.max_quantity:
            quantity = policy.max_quantity

        # Quantize to step size
        quantized = (quantity / policy.quantity_step).quantize(
            Decimal("1"), rounding=policy.quantity_rounding.value
        ) * policy.quantity_step
        return quantized

    def validate_order(
        self, venue: str, symbol: str, price: Decimal, quantity: Decimal
    ) -> bool:
        """Validate that price and quantity comply with policy."""
        policy = self.get_policy(venue, symbol)

        # Check bounds
        if policy.min_price and price < policy.min_price:
            return False
        if policy.max_price and price > policy.max_price:
            return False
        if policy.min_quantity and quantity < policy.min_quantity:
            return False
        if policy.max_quantity and quantity > policy.max_quantity:
            return False

        # Check quantization
        quantized_price = self.quantize_price(venue, symbol, price)
        quantized_quantity = self.quantize_quantity(venue, symbol, quantity)

        return price == quantized_price and quantity == quantized_quantity


# Global instance
_precision_manager = PrecisionPolicyManager()


def get_precision_manager() -> PrecisionPolicyManager:
    """Get global precision policy manager."""
    return _precision_manager


def quantize_price(venue: str, symbol: str, price: Decimal) -> Decimal:
    """Convenience function to quantize price."""
    return _precision_manager.quantize_price(venue, symbol, price)


def quantize_quantity(venue: str, symbol: str, quantity: Decimal) -> Decimal:
    """Convenience function to quantize quantity."""
    return _precision_manager.quantize_quantity(venue, symbol, quantity)
