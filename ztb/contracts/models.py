"""
Pydantic models for API contracts and data validation.

This module defines the core data models used throughout the application
for type-safe data handling and API contract validation.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class Exchange(str, Enum):
    """Supported exchanges."""

    ZAIF = "zaif"
    COINCHECK = "coincheck"
    PAPER = "paper"


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""

    LIMIT = "limit"
    MARKET = "market"


class PositionStatus(str, Enum):
    """Position status enumeration."""

    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


class Trade(BaseModel):
    """Trade model representing a completed trade."""

    id: str = Field(..., description="Unique trade identifier")
    pair: str = Field(..., description="Trading pair (e.g., 'btc_jpy')")
    side: OrderSide = Field(..., description="Trade side")
    price: Decimal = Field(..., gt=0, description="Execution price")
    quantity: Decimal = Field(..., gt=0, description="Executed quantity")
    timestamp: datetime = Field(..., description="Trade execution timestamp")
    fee: Optional[Decimal] = Field(None, ge=0, description="Trading fee")
    exchange_order_id: Optional[str] = Field(
        None, description="Exchange-specific order ID"
    )

    @field_validator("pair")
    @classmethod
    def validate_pair(cls, v: str) -> str:
        """Validate trading pair format."""
        if not v or "_" not in v:
            raise ValueError('Pair must be in format "base_quote"')
        return v.lower()


class Order(BaseModel):
    """Order model representing a trading order."""

    id: str = Field(..., description="Unique order identifier")
    pair: str = Field(..., description="Trading pair")
    side: OrderSide = Field(..., description="Order side")
    type: OrderType = Field(..., description="Order type")
    price: Optional[Decimal] = Field(
        None, gt=0, description="Limit price (None for market orders)"
    )
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    filled_quantity: Decimal = Field(
        default=Decimal("0"), ge=0, description="Filled quantity"
    )
    status: str = Field(..., description="Order status")
    timestamp: datetime = Field(..., description="Order creation timestamp")
    exchange: Exchange = Field(..., description="Exchange where order is placed")

    @field_validator("price", mode="before")
    @classmethod
    def validate_price(cls, v: Any, info: ValidationInfo) -> Any:
        """Validate price based on order type."""
        order_type = info.data.get("type")
        if order_type == OrderType.MARKET and v is not None:
            raise ValueError("Market orders cannot have a price")
        if order_type == OrderType.LIMIT and v is None:
            raise ValueError("Limit orders must have a price")
        return v


class Position(BaseModel):
    """Position model representing a trading position."""

    id: str = Field(..., description="Unique position identifier")
    pair: str = Field(..., description="Trading pair")
    side: OrderSide = Field(..., description="Position side")
    entry_price: Decimal = Field(..., gt=0, description="Average entry price")
    quantity: Decimal = Field(..., gt=0, description="Position size")
    current_price: Optional[Decimal] = Field(
        None, gt=0, description="Current market price"
    )
    pnl: Optional[Decimal] = Field(None, description="Unrealized P&L")
    pnl_percentage: Optional[Decimal] = Field(None, description="P&L percentage")
    status: PositionStatus = Field(..., description="Position status")
    entry_timestamp: datetime = Field(..., description="Position entry timestamp")
    exit_timestamp: Optional[datetime] = Field(
        None, description="Position exit timestamp"
    )
    stop_loss_price: Optional[Decimal] = Field(
        None, ge=0, description="Stop loss price"
    )
    take_profit_price: Optional[Decimal] = Field(
        None, ge=0, description="Take profit price"
    )
    exchange: Exchange = Field(..., description="Exchange")

    @field_validator("exit_timestamp", mode="before")
    @classmethod
    def validate_exit_timestamp(cls, v: Any, info: ValidationInfo) -> Any:
        """Validate exit timestamp is after entry."""
        if v and info.data.get("entry_timestamp") and v < info.data["entry_timestamp"]:
            raise ValueError("Exit timestamp must be after entry timestamp")
        return v


class Balance(BaseModel):
    """Account balance model."""

    currency: str = Field(..., description="Currency code")
    total: Decimal = Field(..., ge=0, description="Total balance")
    available: Decimal = Field(..., ge=0, description="Available balance")
    locked: Decimal = Field(..., ge=0, description="Locked balance")
    exchange: Exchange = Field(..., description="Exchange")

    @field_validator("available")
    @classmethod
    def validate_available(cls, v: Decimal, info: ValidationInfo) -> Decimal:
        """Validate available + locked <= total."""
        locked = info.data.get("locked")
        total = info.data.get("total")
        if locked is not None and total is not None and v + locked > total:
            raise ValueError("Available + locked cannot exceed total")
        return v

    @field_validator("locked")
    @classmethod
    def validate_locked(cls, v: Decimal, info: ValidationInfo) -> Decimal:
        """Validate available + locked <= total."""
        available = info.data.get("available")
        total = info.data.get("total")
        if available is not None and total is not None and available + v > total:
            raise ValueError("Available + locked cannot exceed total")
        return v


class MarketData(BaseModel):
    """Market data model."""

    pair: str = Field(..., description="Trading pair")
    bid: Decimal = Field(..., gt=0, description="Best bid price")
    ask: Decimal = Field(..., gt=0, description="Best ask price")
    last: Optional[Decimal] = Field(None, gt=0, description="Last traded price")
    volume: Optional[Decimal] = Field(None, ge=0, description="24h volume")
    timestamp: datetime = Field(..., description="Data timestamp")
    exchange: Exchange = Field(..., description="Exchange")


class IndicatorData(BaseModel):
    """Technical indicator data model."""

    pair: str = Field(..., description="Trading pair")
    timestamp: datetime = Field(..., description="Data timestamp")
    indicators: Dict[str, Any] = Field(..., description="Indicator values")
    exchange: Exchange = Field(..., description="Exchange")


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_positions: int = Field(..., gt=0, description="Maximum open positions")
    max_dca_per_pair: int = Field(..., ge=0, description="Maximum DCA orders per pair")
    position_size_pct: Decimal = Field(
        ..., gt=0, le=1, description="Position size as % of portfolio"
    )
    stop_loss_pct: Decimal = Field(..., gt=0, le=1, description="Stop loss percentage")
    take_profit_pct: Decimal = Field(..., gt=0, description="Take profit percentage")
    max_slippage_pct: Decimal = Field(
        ..., ge=0, le=1, description="Maximum slippage percentage"
    )
    min_trade_size: Decimal = Field(..., gt=0, description="Minimum trade size")


class ApiResponse(BaseModel):
    """Generic API response wrapper."""

    success: bool = Field(..., description="Response success status")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )


class PaginatedResponse(ApiResponse):
    """Paginated API response."""

    data: List[Any] = Field(default_factory=list, description="List of items")
    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(default=50, ge=1, le=1000, description="Items per page")
    total: int = Field(default=0, ge=0, description="Total number of items")
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")
