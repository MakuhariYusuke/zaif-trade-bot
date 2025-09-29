"""
Serialization utilities for contract validation and data handling.

This module provides utilities for validating, serializing, and
deserializing data using Pydantic contract models.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from ztb.contracts.models import (
    ApiResponse,
    Balance,
    IndicatorData,
    MarketData,
    Order,
    PaginatedResponse,
    Position,
    RiskConfig,
    Trade,
)

T = TypeVar("T", bound=BaseModel)


class ContractValidationError(Exception):
    """Raised when contract validation fails."""

    pass


class ContractSerializer:
    """Serializer for contract models with validation."""

    @staticmethod
    def validate_and_serialize(model: BaseModel) -> str:
        """Validate a model and return its JSON representation."""
        try:
            return model.model_dump_json()
        except Exception as e:
            raise ContractValidationError(f"Serialization failed: {e}")

    @staticmethod
    def deserialize_and_validate(
        model_class: Type[T], data: Union[str, Dict[str, Any]]
    ) -> T:
        """Deserialize JSON/data and validate against a contract model."""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ContractValidationError(
                f"Validation failed for {model_class.__name__}: {e}"
            )

    @staticmethod
    def validate_trade(data: Union[str, Dict[str, Any]]) -> Trade:
        """Validate trade data."""
        return ContractSerializer.deserialize_and_validate(Trade, data)

    @staticmethod
    def validate_order(data: Union[str, Dict[str, Any]]) -> Order:
        """Validate order data."""
        return ContractSerializer.deserialize_and_validate(Order, data)

    @staticmethod
    def validate_position(data: Union[str, Dict[str, Any]]) -> Position:
        """Validate position data."""
        return ContractSerializer.deserialize_and_validate(Position, data)

    @staticmethod
    def validate_balance(data: Union[str, Dict[str, Any]]) -> Balance:
        """Validate balance data."""
        return ContractSerializer.deserialize_and_validate(Balance, data)

    @staticmethod
    def validate_market_data(data: Union[str, Dict[str, Any]]) -> MarketData:
        """Validate market data."""
        return ContractSerializer.deserialize_and_validate(MarketData, data)

    @staticmethod
    def validate_indicator_data(data: Union[str, Dict[str, Any]]) -> IndicatorData:
        """Validate indicator data."""
        return ContractSerializer.deserialize_and_validate(IndicatorData, data)

    @staticmethod
    def validate_risk_config(data: Union[str, Dict[str, Any]]) -> RiskConfig:
        """Validate risk configuration."""
        return ContractSerializer.deserialize_and_validate(RiskConfig, data)

    @staticmethod
    def create_api_response(
        success: bool, data: Optional[Any] = None, error: Optional[str] = None
    ) -> ApiResponse:
        """Create a validated API response."""
        return ApiResponse(success=success, data=data, error=error)

    @staticmethod
    def create_paginated_response(
        data: List[Any],
        page: int = 1,
        page_size: int = 50,
        total: int = 0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> PaginatedResponse:
        """Create a validated paginated response."""
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        return PaginatedResponse(
            success=success,
            data=data,
            error=error,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal objects."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """JSON serialize with support for Decimal and datetime."""
    kwargs.setdefault("cls", DecimalEncoder)
    return json.dumps(obj, **kwargs)


def safe_json_loads(s: str) -> Any:
    """JSON deserialize with Decimal support."""
    return json.loads(s, parse_float=Decimal)
