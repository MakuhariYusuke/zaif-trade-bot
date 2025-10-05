"""
Symbol normalization utilities for cross-exchange compatibility.

Handles various symbol formats and normalizes them to standard base/quote pairs.
"""

import re
from enum import Enum
from typing import Dict, Set, Tuple


class Venue(Enum):
    """Supported trading venues."""

    ZAIF = "zaif"
    COINCHECK = "coincheck"
    # TODO: Add more venues as needed


class SymbolNormalizer:
    """Normalizes trading symbols across different exchanges."""

    # Common symbol patterns
    _PATTERNS = [
        # BTC/JPY format
        re.compile(r"^([A-Z]{2,10})/([A-Z]{3})$"),
        # BTC_JPY format
        re.compile(r"^([A-Z]{2,10})_([A-Z]{3})$"),
        # btcjpy format (lowercase)
        re.compile(r"^([a-z]{2,10})([a-z]{3})$"),
        # BTCJPY format (no separator)
        re.compile(r"^([A-Z]{2,10})([A-Z]{3})$"),
    ]

    # Known base currencies
    _BASE_CURRENCIES: Set[str] = {
        "BTC",
        "ETH",
        "XRP",
        "LTC",
        "BCH",
        "MONA",
        "XEM",
        "XLM",
        "BAT",
        "OMG",
    }

    # Known quote currencies
    _QUOTE_CURRENCIES: Set[str] = {"JPY", "USD", "EUR", "BTC"}

    # Venue-specific symbol mappings
    _VENUE_MAPPINGS: Dict[str, Dict[str, str]] = {
        "zaif": {
            "xem_jpy": "XEM_JPY",
            "mona_jpy": "MONA_JPY",
            "btc_jpy": "BTC_JPY",
        },
        "coincheck": {
            "btc_jpy": "BTC_JPY",
            "eth_jpy": "ETH_JPY",
            "etc_jpy": "ETC_JPY",
        },
    }

    @classmethod
    def normalize(cls, venue: Venue, symbol: str) -> Tuple[str, str]:
        """Normalize a symbol for a specific venue.

        Args:
            venue: Trading venue (Venue enum, e.g., Venue.COINCHECK, Venue.ZAIF)
            symbol: Raw symbol string

        Returns:
            Tuple of (base_currency, quote_currency)

        Raises:
            ValueError: If symbol cannot be normalized
        """
        # Convert to uppercase for consistency
        symbol = symbol.upper()

        # Check venue-specific mappings first
        venue_mappings = cls._VENUE_MAPPINGS.get(venue.value, {})
        if symbol in venue_mappings:
            symbol = venue_mappings[symbol]
        else:
            symbol = symbol.upper()

        # Try regex patterns
        for pattern in cls._PATTERNS:
            match = pattern.match(symbol)
            if match:
                base, quote = match.groups()
                base = base.upper()
                quote = quote.upper()

                # Validate currencies
                if cls._is_valid_symbol(base, quote):
                    return base, quote

        # Try to parse concatenated format (e.g., BTCJPY)
        if len(symbol) > 3:
            # Try different splits
            for i in range(3, len(symbol)):
                base = symbol[:i]
                quote = symbol[i:]
                if cls._is_valid_symbol(base, quote):
                    return base, quote

        raise ValueError(f"Cannot normalize symbol '{symbol}' for venue '{venue}'")

    @classmethod
    def _is_valid_symbol(cls, base: str, quote: str) -> bool:
        """Check if base/quote pair is valid."""
        return base in cls._BASE_CURRENCIES and quote in cls._QUOTE_CURRENCIES

    @classmethod
    def get_standard_symbol(cls, venue: str, base: str, quote: str) -> str:
        """Get the standard symbol format for a venue.

        現状はBTC_JPY形式で返却しますが、将来的に変更される可能性があります。

        Args:
            venue: Trading venue
            base: Base currency
            quote: Quote currency

        Returns:
            Standardized symbol string (currently BTC_JPY format)
        """
        # For now, use BTC_JPY format as standard
        return f"{base}_{quote}"

    @classmethod
    def is_supported_symbol(cls, venue: str, symbol: str) -> bool:
        """Check if a symbol is supported for a venue."""
        try:
            cls.normalize(Venue(venue.lower()), symbol)
            return True
        except ValueError:
            return False
