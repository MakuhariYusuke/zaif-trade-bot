"""
Secrets and safety hygiene utilities for redacting sensitive data.

This module provides utilities to redact sensitive information from logs,
outputs, and data structures to prevent accidental exposure of secrets.
"""

import re
from typing import Any, Dict, List, Union


# Common patterns for sensitive data
SENSITIVE_PATTERNS = [
    # API keys (various formats)
    r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
    # Secret keys
    r'(?i)(secret[_-]?key|secret)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
    # Passwords
    r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?([^"\']{3,})["\']?',
    # Tokens
    r'(?i)(token|bearer)\s*[:=]\s*["\']?([a-zA-Z0-9_.-]{20,})["\']?',
    # Private keys (basic pattern)
    r'(?i)-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----.*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
    # Database URLs with credentials
    r'(?i)(postgresql|mysql|mongodb)://([^:]+):([^@]+)@',
    # Generic base64-like strings (long alphanumeric)
    r'\b[A-Za-z0-9+/]{32,}\b',
]

REDACTED_PLACEHOLDER = "***REDACTED***"


def redact_sensitive_data(text: str) -> str:
    """
    Redact sensitive information from a text string.

    Args:
        text: Input text that may contain sensitive data

    Returns:
        Text with sensitive data redacted
    """
    if not isinstance(text, str):
        return str(text)

    redacted = text
    for pattern in SENSITIVE_PATTERNS:
        redacted = re.sub(pattern, lambda m: m.group(0).replace(m.group(0), REDACTED_PLACEHOLDER), redacted)

    return redacted


def redact_dict(data: Dict[str, Any], sensitive_keys: List[str] = None) -> Dict[str, Any]:
    """
    Redact sensitive values from a dictionary.

    Args:
        data: Dictionary that may contain sensitive data
        sensitive_keys: Additional keys to redact (case-insensitive)

    Returns:
        Dictionary with sensitive values redacted
    """
    if sensitive_keys is None:
        sensitive_keys = ['password', 'secret', 'key', 'token', 'api_key', 'apikey']

    redacted = {}
    for key, value in data.items():
        if any(sensitive_key.lower() in key.lower() for sensitive_key in sensitive_keys):
            redacted[key] = REDACTED_PLACEHOLDER
        elif isinstance(value, dict):
            redacted[key] = redact_dict(value, sensitive_keys)
        elif isinstance(value, list):
            redacted[key] = [redact_dict(item, sensitive_keys) if isinstance(item, dict) else redact_sensitive_data(str(item)) for item in value]
        else:
            redacted[key] = redact_sensitive_data(str(value))

    return redacted


def is_safe_content(content: Union[str, Dict, List]) -> bool:
    """
    Check if content appears to be free of sensitive data.

    Args:
        content: Content to check

    Returns:
        True if content appears safe, False if sensitive data detected
    """
    if isinstance(content, str):
        return not any(re.search(pattern, content, re.IGNORECASE | re.DOTALL) for pattern in SENSITIVE_PATTERNS)
    elif isinstance(content, dict):
        return all(is_safe_content(value) for value in content.values())
    elif isinstance(content, list):
        return all(is_safe_content(item) for item in content)
    else:
        return True


def sanitize_log_message(message: str, context: Dict[str, Any] = None) -> str:
    """
    Sanitize a log message by redacting sensitive data.

    Args:
        message: Log message to sanitize
        context: Additional context dictionary to sanitize

    Returns:
        Sanitized log message
    """
    sanitized = redact_sensitive_data(message)

    if context:
        sanitized_context = redact_dict(context)
        sanitized += f" | context: {sanitized_context}"

    return sanitized