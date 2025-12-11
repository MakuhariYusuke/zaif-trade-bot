"""
Exchange Profile Configuration.

This module defines the ExchangeProfile class, which encapsulates
exchange-specific characteristics such as fee models, liquidity, and latency.
"""

import dataclasses
from typing import Any, Dict

from ztb.utils.fee_model import FeeModel, FeeModelFactory, FixedFeeModel


@dataclasses.dataclass
class ExchangeProfile:
    """
    Configuration for exchange-specific characteristics.

    Attributes:
        name: Name of the exchange (e.g., "zaif", "binance").
        fee_model: The fee model to use for calculating transaction costs.
        slippage_rate: Estimated slippage rate (0.0 to 1.0).
        latency_ms: Estimated network/execution latency in milliseconds.
        maker_fee_rate: Base maker fee rate (used if fee_model is not provided).
        taker_fee_rate: Base taker fee rate (used if fee_model is not provided).
    """

    name: str = "generic"
    fee_model: FeeModel = dataclasses.field(default_factory=FixedFeeModel)
    slippage_rate: float = 0.0
    latency_ms: float = 0.0

    # Convenience fields for simple configuration
    maker_fee_rate: float = 0.001
    taker_fee_rate: float = 0.001

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ExchangeProfile":
        """
        Create an ExchangeProfile from a dictionary configuration.

        Args:
            config: Dictionary containing exchange configuration.

        Returns:
            ExchangeProfile instance.
        """
        name = config.get("name", "generic")

        # Handle fee model creation
        fee_config = config.get("fee_model", {})
        if isinstance(fee_config, dict):
            model_type = fee_config.get("type", "fixed")
            # If rates are provided at top level, override/use them in fee config
            if "maker_fee_rate" in config:
                fee_config["maker_fee_rate"] = config["maker_fee_rate"]
            if "taker_fee_rate" in config:
                fee_config["taker_fee_rate"] = config["taker_fee_rate"]

            # For FixedFeeModel, map maker/taker to buy/sell if needed,
            # though usually we treat them differently.
            # FeeModel currently uses buy/sell.
            # Let's assume buy=taker, sell=taker for simplicity unless specified.
            if "buy_fee_rate" not in fee_config and "taker_fee_rate" in config:
                fee_config["buy_fee_rate"] = config["taker_fee_rate"]
            if "sell_fee_rate" not in fee_config and "taker_fee_rate" in config:
                fee_config["sell_fee_rate"] = config["taker_fee_rate"]

            fee_model = FeeModelFactory.create_fee_model(model_type, fee_config)
        else:
            # Default to fixed fee model with provided rates
            maker = config.get("maker_fee_rate", 0.001)
            taker = config.get("taker_fee_rate", 0.001)
            fee_model = FixedFeeModel(buy_fee_rate=taker, sell_fee_rate=taker)

        return cls(
            name=name,
            fee_model=fee_model,
            slippage_rate=config.get("slippage_rate", 0.0),
            latency_ms=config.get("latency_ms", 0.0),
            maker_fee_rate=config.get("maker_fee_rate", 0.001),
            taker_fee_rate=config.get("taker_fee_rate", 0.001),
        )
