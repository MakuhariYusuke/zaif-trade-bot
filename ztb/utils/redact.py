"""
Redact sensitive data from content.
"""
import re

def is_safe_content(content: str) -> bool:
    """
    Check if content is safe (no sensitive data).

    Args:
        content: Content to check

    Returns:
        True if content is safe, False otherwise
    """
    # Check for API keys, passwords, tokens, etc.
    patterns = [
        r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
        r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?([^"\s]{8,})["\']?',
        r'(?i)(token|secret)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
        r'(?i)(bearer|authorization)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
        r"[a-zA-Z0-9_-]{32,}",  # Long random strings that might be keys
    ]

    for pattern in patterns:
        if re.search(pattern, content):
            return False

    return True
