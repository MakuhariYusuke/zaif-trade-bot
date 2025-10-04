#!/usr/bin/env python3
"""
Dump JSON schemas for contract models.

This script generates JSON Schema documents for all Pydantic models
defined in ztb.contracts.models, useful for API documentation and
client-side validation.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Type

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def generate_schema(model_class: Type[Any]) -> Dict[str, Any]:
    """Generate JSON schema for a Pydantic model."""
    return model_class.model_json_schema()  # type: ignore[no-any-return]


def main() -> None:
    """Main entry point."""
    schemas = {
        "Trade": generate_schema(Trade),
        "Order": generate_schema(Order),
        "Position": generate_schema(Position),
        "Balance": generate_schema(Balance),
        "MarketData": generate_schema(MarketData),
        "IndicatorData": generate_schema(IndicatorData),
        "RiskConfig": generate_schema(RiskConfig),
        "ApiResponse": generate_schema(ApiResponse),
        "PaginatedResponse": generate_schema(PaginatedResponse),
    }

    # Output to stdout as JSON
    print(json.dumps(schemas, indent=2, default=str))


if __name__ == "__main__":
    main()
