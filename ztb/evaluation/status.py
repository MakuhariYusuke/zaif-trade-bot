"""Compatibility shim for evaluation status utilities.

Provides a CoverageValidator placeholder when ztb.analysis.coverage_validator
is not available. Uses standardized error handling from ztb.utils.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_COVERAGE_VALIDATOR_AVAILABLE = False

try:
    from ztb.analysis.coverage_validator import CoverageValidator

    _COVERAGE_VALIDATOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"ztb.analysis.coverage_validator not available: {e}")

    # Stub implementation when module is not available
    class CoverageValidator:  # type: ignore[no-redef]
        """Placeholder for CoverageValidator.

        Used when the real implementation from ztb.analysis.coverage_validator
        is not available. Provides a no-op interface for compatibility.
        """

        def __init__(self, config: Dict[str, Any] | None = None) -> None:
            """Initialize validator with optional config.

            Args:
                config: Optional configuration dictionary.
            """
            self.config = config or {}

        def validate(self, data: Any | None = None) -> bool:
            """Validate data.

            Args:
                data: Data to validate (ignored in placeholder).

            Returns:
                Always returns True (no validation in placeholder).
            """
            return True


__all__ = ["CoverageValidator"]


