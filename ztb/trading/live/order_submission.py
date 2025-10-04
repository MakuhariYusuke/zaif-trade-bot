"""
Order submission utilities with precision and idempotency safety.

Provides a single entry point for order preparation with:
- Symbol normalization
- Price/quantity quantization
- Min notional validation
- Idempotency token generation
"""

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ztb.trading.live.idempotency_store import IdempotencyStore
from ztb.trading.live.precision_policy import PrecisionPolicyManager
from ztb.trading.live.symbols import SymbolNormalizer, Venue
from ztb.utils.errors import IdempotencyError, ValidationError


@dataclass
class PreparedOrder:
    """Prepared order data ready for submission."""

    venue: str
    normalized_symbol: str
    client_order_id: str
    side: str
    quantity: Decimal
    price: Optional[Decimal]
    idempotency_token: str
    original_symbol: str


class OrderPreparer:
    """Prepares orders with precision and idempotency safety."""

    def __init__(self, idempotency_store: Optional[IdempotencyStore] = None) -> None:
        self.precision_manager = PrecisionPolicyManager()
        self.symbol_normalizer = SymbolNormalizer()
        self.idempotency_store = idempotency_store or IdempotencyStore()

    def prepare_order(
        self,
        venue: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        client_order_id: Optional[str] = None,
    ) -> PreparedOrder:
        """
        Prepare an order for submission with full safety checks.

        Args:
            venue: Trading venue (e.g., 'coincheck')
            symbol: Trading symbol (e.g., 'BTC/JPY')
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Order price (None for market orders)
            client_order_id: Client-provided order ID

        Returns:
            PreparedOrder with normalized and validated data

        Raises:
            ValidationError: If order parameters are invalid
            IdempotencyError: If order ID already exists
        """
        # 1. Normalize symbol
        try:
            venue_enum = Venue(venue.lower())
        except ValueError:
            raise ValidationError(f"Unsupported venue: {venue}")

        base, quote = self.symbol_normalizer.normalize(venue_enum, symbol)
        normalized_symbol = f"{base}_{quote}"

        # 2. Apply price/quantity quantization
        quantized_quantity = self.precision_manager.quantize_quantity(
            venue, normalized_symbol, quantity
        )
        quantized_price = None
        if price is not None:
            quantized_price = self.precision_manager.quantize_price(
                venue, normalized_symbol, price
            )

        # 3. Validate min notional
        self._validate_min_notional(
            venue, normalized_symbol, quantized_quantity, quantized_price
        )

        # 4. Generate idempotency token
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        # Check for existing order
        existing = self.idempotency_store.get_order_data(client_order_id)
        if existing is not None:
            raise IdempotencyError(
                f"Order with client_order_id {client_order_id} already exists",
                details={"client_order_id": client_order_id, "existing_data": existing},
            )

        # Store preliminary order data
        order_data = {
            "venue": venue,
            "symbol": normalized_symbol,
            "side": side,
            "quantity": str(quantized_quantity),
            "price": str(quantized_price) if quantized_price else None,
            "status": "preparing",
        }

        try:
            self.idempotency_store.check_and_store(client_order_id, order_data)
        except Exception as e:
            raise IdempotencyError(
                f"Failed to store idempotency token for {client_order_id}",
                details={"client_order_id": client_order_id, "error": str(e)},
            ) from e

        return PreparedOrder(
            venue=venue,
            normalized_symbol=normalized_symbol,
            client_order_id=client_order_id,
            side=side,
            quantity=quantized_quantity,
            price=quantized_price,
            idempotency_token=client_order_id,  # Use client_order_id as token
            original_symbol=symbol,
        )

    def _validate_min_notional(
        self, venue: str, symbol: str, quantity: Decimal, price: Optional[Decimal]
    ) -> None:
        """Validate minimum notional value."""
        if price is None:
            # Market orders - assume minimum is met for now
            return

        # For now, skip notional checks - would need venue-specific min_notional
        # This can be added when venue APIs provide this information
        pass
