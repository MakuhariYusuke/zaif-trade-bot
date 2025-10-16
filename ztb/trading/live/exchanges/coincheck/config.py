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


def validate_credentials(api_key=None, api_secret=None):
    """Validate credentials (backward compatibility)."""
    if api_key is None and api_secret is None:
        # If no args provided, validate current credentials
        creds = _config.get_credentials_optional()
        return _config.validate_credentials(creds[0], creds[1])
    return _config.validate_credentials(api_key, api_secret)