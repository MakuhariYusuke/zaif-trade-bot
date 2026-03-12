"""
bitFlyer exchange configuration management.

This module handles API credentials and configuration for bitFlyer exchange.
"""

from ztb.trading.live.exchanges.base.config import BaseExchangeConfig

class BitflyerConfig(BaseExchangeConfig):
    """
    Configuration management for bitFlyer exchange.
    """

    def _get_env_vars(self) -> tuple[str, str]:
        """
        Get the environment variable names for bitFlyer.

        Returns:
            tuple of (api_key_env_var, api_secret_env_var)
        """
        return "BITFLYER_API_KEY", "BITFLYER_API_SECRET"

# Create a singleton instance for backward compatibility
_config = BitflyerConfig()

# Backward compatibility functions
def get_bitflyer_credentials():
    """Get bitFlyer API credentials (backward compatibility)."""
    return _config.get_credentials()

def get_bitflyer_credentials_optional():
    """Get bitFlyer API credentials optionally (backward compatibility)."""
    return _config.get_credentials_optional()

def validate_credentials(api_key=None, api_secret=None):
    """Validate credentials (backward compatibility)."""
    if api_key is None and api_secret is None:
        # If no args provided, validate current credentials
        creds = _config.get_credentials_optional()
        return _config.validate_credentials(creds[0], creds[1])
    return _config.validate_credentials(api_key, api_secret)
