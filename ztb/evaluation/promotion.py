"""Minimal promotion-related stubs for legacy tests.

This module provides fallback implementations when the full ztb.analysis.promotion
module is not available. Uses standardized error handling from ztb.utils.safety.
"""

import logging
from collections.abc import Mapping

from ztb.utils.safety import safe_to_float

logger = logging.getLogger(__name__)

# Try importing the real implementations from analysis
_PROMOTION_AVAILABLE = False
try:
    from ztb.analysis.promotion import (
        DistributionCriterion,
        DurationCriterion,
        NumericCriterion,
        PromotionEngine,
        PromotionNotifier,
        PromotionResult,
        RatioCriterion,
        YamlPromotionEngine,
    )

    _PROMOTION_AVAILABLE = True
except Exception as e:
    logger.debug(f"ztb.analysis.promotion not available, using fallback stubs: {e}")

    # Lightweight fallbacks used only in tests when the full analysis module
    # is not importable. These use standardized error handling from ztb.utils.safety

    class PromotionEngine:  # type: ignore[no-redef]
        """Base promotion engine placeholder."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize engine."""
            pass

    class DistributionCriterion:  # type: ignore[no-redef]
        """Distribution-based evaluation criterion."""

        def __init__(
            self, name: str, operator: str, value: float, weight: float = 1.0
        ) -> None:
            """Initialize with parameter name and threshold.

            Args:
                name: Name of the parameter to evaluate.
                operator: Comparison operator (">" or "<").
                value: Threshold value.
                weight: Weight for scoring (default 1.0).
            """
            self.name = name
            self.operator = operator
            self.value = value
            self.weight = weight

        def evaluate(self, data: object) -> tuple[bool, float]:
            """Evaluate if data meets criteria.

            Args:
                data: Data object with attributes to evaluate.

            Returns:
                tuple of (passed: bool, score: float).
            """
            stat: float = safe_to_float(getattr(data, self.name, 0.0))
            passed = False
            if self.operator == ">":
                passed = stat > self.value
            elif self.operator == "<":
                passed = stat < self.value
            else:
                passed = False

            ratio: float = min(1.0, stat / max(self.value, 1.0))
            score: float = ratio if passed else 0.0
            return passed, score

    class DurationCriterion(DistributionCriterion):  # type: ignore[no-redef]
        """Duration-based evaluation criterion."""

        pass

    class NumericCriterion:  # type: ignore[no-redef]
        """Numeric evaluation criterion."""

        def __init__(
            self,
            name: str,
            operator: str,
            value: float,
            weight: float = 1.0,
        ) -> None:
            """Initialize with numeric parameter.

            Args:
                name: Name of the parameter to evaluate.
                operator: Comparison operator (">" or "<").
                value: Threshold value.
                weight: Weight for scoring (default 1.0).
            """
            self.name = name
            self.operator = operator
            self.value = value
            self.weight = weight

        def evaluate(self, data: object) -> tuple[bool, float]:
            """Evaluate if data meets criteria.

            Args:
                data: Data object or dict to evaluate.

            Returns:
                tuple of (passed: bool, score: float).
            """
            val: float = safe_to_float(
                getattr(data, self.name, None)
                or (data.get(self.name) if isinstance(data, Mapping) else None)
            )
            passed = False
            if self.operator == ">":
                passed = val > self.value
            elif self.operator == "<":
                passed = val < self.value
            else:
                passed = False

            score: float = (
                min(1.0, val / max(self.value, 1.0)) if passed else 0.0
            )
            return passed, score

    class RatioCriterion(NumericCriterion):  # type: ignore[no-redef]
        """Ratio-based evaluation criterion."""

        pass

    class PromotionNotifier:  # type: ignore[no-redef]
        """Promotion notifier placeholder."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize notifier."""
            pass

    class PromotionResult:  # type: ignore[no-redef]
        """Promotion result constants."""

        ACCEPTED = "accepted"
        REJECTED = "rejected"
        PENDING = "pending"

    class YamlPromotionEngine(PromotionEngine):  # type: ignore[no-redef]
        """YAML-based promotion engine."""

        def __init__(self, config: dict[str, object] | None = None) -> None:
            """Initialize with optional config.

            Args:
                config: Optional configuration dictionary.
            """
            super().__init__()
            self.config = config or {}

__all__ = [
    "PromotionEngine",
    "DistributionCriterion",
    "DurationCriterion",
    "NumericCriterion",
    "RatioCriterion",
    "PromotionNotifier",
    "PromotionResult",
    "YamlPromotionEngine",
]
