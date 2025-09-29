"""
Log redaction utilities for sensitive data protection.

This module provides functions to redact sensitive information from log messages
and data structures before they are written to logs.
"""

import re
from typing import Any, Dict, List, Union


class LogRedactor:
    """Redacts sensitive information from log data."""

    # Patterns for sensitive data
    SENSITIVE_PATTERNS = [
        # API keys (common formats)
        (r"api[_-]?key\s*[:=]\s*([a-zA-Z0-9_-]{15,})", "***API_KEY_REDACTED***"),
        (r"key\s*[:=]\s*([a-zA-Z0-9_-]{15,})", "***KEY_REDACTED***"),
        # Secret tokens
        (r"secret\s*[:=]\s*([a-zA-Z0-9_-]{15,})", "***SECRET_REDACTED***"),
        (r"token\s*[:=]\s*([a-zA-Z0-9_-]{15,})", "***TOKEN_REDACTED***"),
        # Passwords
        (r'password\s*[:=]\s*([^"\s]{3,})', "***PASSWORD_REDACTED***"),
        (r'passwd\s*[:=]\s*([^"\s]{3,})', "***PASSWORD_REDACTED***"),
        # Email addresses
        (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "***EMAIL_REDACTED***",
        ),
        # Credit card numbers (basic pattern)
        (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "***CARD_NUMBER_REDACTED***"),
        # Social security numbers (US)
        (r"\b\d{3}[\s-]?\d{2}[\s-]?\d{4}\b", "***SSN_REDACTED***"),
        # Private keys (PEM format headers)
        (
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            "***PRIVATE_KEY_REDACTED***",
        ),
        # Database connection strings
        (r'(mongodb|mysql|postgresql)://[^"\s]+', "***DB_CONNECTION_REDACTED***"),
        # Generic sensitive values in JSON-like structures
        (
            r'("password"|"secret"|"token"|"key"|"api_key")\s*:\s*"[^"]*"',
            r'\1: "***REDACTED***"',
        ),
        (
            r"('password'|'secret'|'token'|'key'|'api_key')\s*:\s*'[^']*'",
            r"\1: '***REDACTED***'",
        ),
    ]

    @staticmethod
    def redact_text(text: str) -> str:
        """
        Redact sensitive information from a text string.

        Args:
            text: The text to redact

        Returns:
            The redacted text
        """
        if not text:
            return text

        redacted = str(text)
        for pattern, replacement in LogRedactor.SENSITIVE_PATTERNS:
            redacted = re.sub(
                pattern, replacement, redacted, flags=re.IGNORECASE | re.DOTALL
            )

        return redacted

    @staticmethod
    def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive information from a dictionary recursively.

        Args:
            data: The dictionary to redact

        Returns:
            A new dictionary with sensitive data redacted
        """
        if not isinstance(data, dict):
            return data

        redacted = {}
        sensitive_keys = {
            "password",
            "passwd",
            "secret",
            "token",
            "key",
            "api_key",
            "private_key",
            "access_token",
            "refresh_token",
            "auth_token",
            "email",
            "ssn",
            "social_security",
            "credit_card",
            "card_number",
        }

        for key, value in data.items():
            lower_key = str(key).lower()

            # Redact known sensitive keys
            if any(sensitive in lower_key for sensitive in sensitive_keys):
                redacted[key] = "***REDACTED***"
            elif isinstance(value, dict):
                redacted[key] = LogRedactor.redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = LogRedactor.redact_list(value)
            elif isinstance(value, str):
                redacted[key] = LogRedactor.redact_text(value)
            else:
                redacted[key] = value

        return redacted

    @staticmethod
    def redact_list(data: List[Any]) -> List[Any]:
        """
        Redact sensitive information from a list recursively.

        Args:
            data: The list to redact

        Returns:
            A new list with sensitive data redacted
        """
        if not isinstance(data, list):
            return data

        return [
            LogRedactor.redact_dict(item)
            if isinstance(item, dict)
            else LogRedactor.redact_list(item)
            if isinstance(item, list)
            else LogRedactor.redact_text(item)
            if isinstance(item, str)
            else item
            for item in data
        ]

    @staticmethod
    def redact_log_data(
        data: Union[str, Dict[str, Any], List[Any]],
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Redact sensitive information from log data.

        Args:
            data: The data to redact (string, dict, or list)

        Returns:
            The redacted data
        """
        if isinstance(data, str):
            return LogRedactor.redact_text(data)
        elif isinstance(data, dict):
            return LogRedactor.redact_dict(data)
        elif isinstance(data, list):
            return LogRedactor.redact_list(data)
        else:
            return data


def redact_for_logging(data: Any) -> Any:
    """
    Convenience function to redact data before logging.

    Args:
        data: The data to prepare for logging

    Returns:
        Redacted data safe for logging
    """
    return LogRedactor.redact_log_data(data)
