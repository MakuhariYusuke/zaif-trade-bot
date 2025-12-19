"""
Coincheck exchange configuration management.

This module handles API credentials and configuration for Coincheck exchange.
"""

from ztb.trading.live.exchanges.base.config import BaseExchangeConfig


class CoincheckConfig(BaseExchangeConfig):
    """
    Configuration management for Coincheck exchange.
    """

    def _get_env_vars(self) -> tuple[str, str]:
        """
        Get the environment variable names for Coincheck.

        Returns:
            Tuple of (api_key_env_var, api_secret_env_var)
        """
        return "COINCHECK_API_KEY", "COINCHECK_API_SECRET"


# Create a singleton instance for backward compatibility
_config = CoincheckConfig()


# Backward compatibility functions
def get_coincheck_credentials():
    """Get Coincheck API credentials (backward compatibility)."""
    return _config.get_credentials()


def get_coincheck_credentials_optional():
    """Get Coincheck API credentials optionally (backward compatibility)."""
    return _config.get_credentials_optional()


