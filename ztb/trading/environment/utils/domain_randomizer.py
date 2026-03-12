"""
Domain Randomization for Trading Environment.

This module provides functionality to randomize environment parameters
(fees, slippage, latency, etc.) at the start of each episode to improve
agent robustness (Domain Randomization).
"""

import dataclasses
import logging
import random
from typing import Any

from ztb.trading.environment.utils.exchange_profile import ExchangeProfile
from ztb.utils.fee_model import FixedFeeModel

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class DomainRandomizationConfig:
    """
    Configuration for domain randomization.

    Attributes:
        enabled: Whether domain randomization is enabled.
        maker_fee_range: tuple of (min, max) for maker fee rate.
        taker_fee_range: tuple of (min, max) for taker fee rate.
        slippage_range: tuple of (min, max) for slippage rate.
        latency_range: tuple of (min, max) for latency in ms.
    """

    enabled: bool = False
    maker_fee_range: tuple[float, float] = (0.0005, 0.002)  # 0.05% - 0.2%
    taker_fee_range: tuple[float, float] = (0.0005, 0.002)  # 0.05% - 0.2%
    slippage_range: tuple[float, float] = (0.0, 0.005)  # 0.0% - 0.5%
    latency_range: tuple[float, float] = (0.0, 500.0)  # 0 - 500ms
    intensity: float = 1.0  # 0.0 - 1.0 (Scaling factor)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DomainRandomizationConfig":
        """Create config from dictionary."""
        return cls(
            enabled=config.get("enabled", False),
            maker_fee_range=tuple(config.get("maker_fee_range", (0.0005, 0.002))),
            taker_fee_range=tuple(config.get("taker_fee_range", (0.0005, 0.002))),
            slippage_range=tuple(config.get("slippage_range", (0.0, 0.005))),
            latency_range=tuple(config.get("latency_range", (0.0, 500.0))),
            intensity=float(config.get("intensity", 1.0)),
        )

class DomainRandomizer:
    """
    Handles randomization of environment parameters.
    """

    def __init__(self, config: DomainRandomizationConfig):
        self.config = config

    def randomize_profile(
        self, base_profile: ExchangeProfile, intensity: float = 1.0
    ) -> ExchangeProfile:
        """
        Create a randomized version of the exchange profile.

        Args:
            base_profile: The base exchange profile to start from.
            intensity: Scaling factor for randomization (0.0 to 1.0).
                      0.0 = No randomization (Base profile values).
                      1.0 = Full randomization (Values from config ranges).

        Returns:
            A new ExchangeProfile with randomized parameters.
        """
        if not self.config.enabled:
            return base_profile

        # Clamp intensity
        intensity = max(0.0, min(1.0, intensity))

        # Helper to interpolate
        def interpolate(base_val: float, target_val: float) -> float:
            return base_val + (target_val - base_val) * intensity

        # Randomize fees
        target_maker_fee = random.uniform(*self.config.maker_fee_range)
        target_taker_fee = random.uniform(*self.config.taker_fee_range)

        # Randomize slippage
        target_slippage = random.uniform(*self.config.slippage_range)

        # Randomize latency
        target_latency = random.uniform(*self.config.latency_range)

        # Apply intensity
        maker_fee = interpolate(base_profile.maker_fee_rate, target_maker_fee)
        taker_fee = interpolate(base_profile.taker_fee_rate, target_taker_fee)
        slippage = interpolate(base_profile.slippage_rate, target_slippage)
        latency = interpolate(base_profile.latency_ms, target_latency)

        # Create new fee model with randomized rates
        # Note: We currently only support randomizing FixedFeeModel easily.
        # For complex models, we might need more sophisticated logic.
        fee_model = FixedFeeModel(buy_fee_rate=taker_fee, sell_fee_rate=taker_fee)

        randomized_profile = ExchangeProfile(
            name=f"{base_profile.name}_randomized_i{intensity:.2f}",
            fee_model=fee_model,
            slippage_rate=slippage,
            latency_ms=latency,
            maker_fee_rate=maker_fee,
            taker_fee_rate=taker_fee,
        )

        logger.debug(
            f"Domain Randomization (i={intensity:.2f}): "
            f"Fee={taker_fee:.5f} (base={base_profile.taker_fee_rate}), "
            f"Slip={slippage:.5f} (base={base_profile.slippage_rate})"
        )

        return randomized_profile
