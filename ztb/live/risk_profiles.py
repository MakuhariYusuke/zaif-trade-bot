"""
Risk profile management for trading strategies.

Provides preset risk management configurations and validation.
"""

from typing import Dict, Optional

from ztb.config.schema import RiskProfileConfig


class RiskProfileManager:
    """Manages risk profiles for trading strategies."""

    def __init__(self):
        self._profiles: Dict[str, RiskProfileConfig] = {}

    def add_profile(self, profile: RiskProfileConfig):
        """Add a risk profile."""
        self._profiles[profile.name] = profile

    def get_profile(self, name: str) -> Optional[RiskProfileConfig]:
        """Get a risk profile by name."""
        return self._profiles.get(name)

    def list_profiles(self) -> Dict[str, RiskProfileConfig]:
        """List all available risk profiles."""
        return self._profiles.copy()

    def validate_position_size(
        self, profile_name: str, position_size: float, portfolio_value: float
    ) -> bool:
        """Validate position size against risk profile."""
        profile = self.get_profile(profile_name)
        if not profile:
            return False

        max_size = profile.max_position_size * portfolio_value
        return position_size <= max_size

    def validate_daily_loss(
        self, profile_name: str, daily_loss: float, portfolio_value: float
    ) -> bool:
        """Validate daily loss against risk profile."""
        profile = self.get_profile(profile_name)
        if not profile:
            return False

        max_loss = profile.max_daily_loss * portfolio_value
        return abs(daily_loss) <= max_loss

    def calculate_position_size(
        self, profile_name: str, portfolio_value: float, volatility: float
    ) -> float:
        """Calculate recommended position size based on risk profile and volatility."""
        profile = self.get_profile(profile_name)
        if not profile:
            return 0.0

        # Kelly criterion inspired sizing with risk limits
        risk_amount = profile.risk_per_trade * portfolio_value
        stop_loss_amount = risk_amount / profile.stop_loss_pct

        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjustment = min(1.0, 0.02 / volatility) if volatility > 0 else 1.0

        position_size = stop_loss_amount * volatility_adjustment
        max_size = profile.max_position_size * portfolio_value

        return min(position_size, max_size)


# Global instance
_risk_manager = RiskProfileManager()


def get_risk_manager() -> RiskProfileManager:
    """Get global risk profile manager."""
    return _risk_manager


def get_profile(name: str) -> Optional[RiskProfileConfig]:
    """Convenience function to get a risk profile."""
    return _risk_manager.get_profile(name)
