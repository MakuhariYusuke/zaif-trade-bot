"""Backward compatibility shim re-exporting TradingEvaluator.

Re-exports the classes from ztb.analysis.evaluator.evaluator
to support legacy imports. Uses standardized error handling
from ztb.utils when imports fail.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_EVALUATOR_AVAILABLE = False

try:
    from ztb.analysis.evaluator.evaluator import (  # type: ignore[attr-defined]
        EvaluationResult,
        SingleEpisodeResultDict,
        TradingEvaluator,
    )

    _EVALUATOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"ztb.analysis.evaluator.evaluator not available: {e}")

    # Stub implementations when module is not available
    class EvaluationResult(dict):  # type: ignore[no-redef,type-arg]
        """Placeholder for EvaluationResult."""

        pass

    class SingleEpisodeResultDict(dict):  # type: ignore[no-redef,type-arg]
        """Placeholder for SingleEpisodeResultDict."""

        pass

    class TradingEvaluator:  # type: ignore[no-redef]
        """Placeholder for TradingEvaluator.

        Used when ztb.analysis.evaluator.evaluator is not available.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Initialize evaluator."""
            pass

        def evaluate(self, *args: Any, **kwargs: Any) -> "EvaluationResult":
            """Evaluate and return empty result."""
            return EvaluationResult()

__all__ = ["TradingEvaluator", "EvaluationResult", "SingleEpisodeResultDict"]

