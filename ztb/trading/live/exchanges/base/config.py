"""
Base exchange configuration management.

This module provides a base class for exchange-specific configuration management.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from dotenv import load_dotenv


class BaseExchangeConfig(ABC):
    """
    Base class for exchange configuration management.

    This class provides common functionality for managing API credentials
    and configuration for different exchanges.
    """

    def __init__(self) -> None:
        """Initialize the config and load environment variables from .env file."""
        load_dotenv()

    @abstractmethod
    def _get_env_vars(self) -> Tuple[str, str]:
        """
        Get the environment variable names for this exchange.

        Returns:
            Tuple of (api_key_env_var, api_secret_env_var)
        """
        pass

    def get_credentials(self) -> Tuple[str, str]:
        """
        Get API credentials from environment variables.

        Returns:
            Tuple of (api_key, api_secret)

        Raises:
            ValueError: If credentials are not found or empty
        """
        api_key_env, api_secret_env = self._get_env_vars()
        exchange_name = self.__class__.__name__.replace("Config", "").lower()

        api_key = os.getenv(api_key_env, "").strip()
        api_secret = os.getenv(api_secret_env, "").strip()

        if not api_key or not api_secret:
            raise ValueError(
                f"{exchange_name.capitalize()} API credentials not found. Please set environment variables:\n"
                f"{api_key_env}=your_api_key\n"
                f"{api_secret_env}=your_api_secret\n"
                "Or create a .env file with these variables."
            )

        return api_key, api_secret

    def get_credentials_optional(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API credentials from environment variables (optional).

        Returns:
            Tuple of (api_key, api_secret) - may be None if not set
        """
        api_key_env, api_secret_env = self._get_env_vars()

        api_key = os.getenv(api_key_env, "").strip()
        api_secret = os.getenv(api_secret_env, "").strip()

        # Return None if either is empty
        if not api_key or not api_secret:
            return None, None

        return api_key, api_secret

    @staticmethod
    def validate_credentials(api_key: Optional[str], api_secret: Optional[str]) -> bool:
        """
        Validate that API credentials are present and non-empty.

        Args:
            api_key: API key string
            api_secret: API secret string

        Returns:
            True if credentials are valid, False otherwise
        """
        return bool(api_key and api_secret and api_key.strip() and api_secret.strip())
