"""
Type definitions for ActionSignalGuide.

This module provides specific type aliases and definitions to replace generic Any types
used in the ActionSignalGuide system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from ztb.types.common import ConfigSection, ObjectMap

# Type aliases for configuration dictionaries
SignalConfig = dict[
    str,
    Union[bool, int, float, str, list[str], dict[str, Union[int, float, None]]],
]
PatternConfig = dict[
    str,
    Union[bool, int, float, str, list[str], dict[str, Union[int, float, None]]],
]

# Type aliases for signals and results (using string forward references)
SignalList = list["ActionSignal"]  # type: ignore
SignalHistory = list["ActionSignal"]  # type: ignore

# Type aliases for statistics and metrics structures
PerformanceStats = dict[str, Union[int, float, dict[str, Union[int, float]]]]
PatternStats = dict[str, Union[int, float, dict[str, Union[int, float]]]]
CacheStats = dict[str, Union[int, float]]

# Type aliases for signal metadata and statistics metadata
SignalMetadata = dict[str, Union[int, float, str, bool, list[Union[int, float]]]]
StatisticsMetadata = ObjectMap

# Type aliases for recognizer status structures
RecognizerGroupStatus = dict[str, Union[bool, int, list[str]]]
RecognizerStatus = dict[str, Union[int, str, dict[str, RecognizerGroupStatus]]]

# Union types for flexible inputs
ConfigInput = Union["ActionSignalGuideConfig", ConfigSection, None]  # type: ignore
GuidanceInput = Union["GuidanceLevel", str, None]  # type: ignore

# Type aliases for multi-timeframe data structures
MultiTimeframePayload = dict[str, object]
MultiTimeframeData = dict[str, MultiTimeframePayload]  # timeframe -> {'data': DataFrame, ...}

# More specific type aliases for pattern recognition components
PatternThresholds = dict[str, Union[int, float, None]]
PatternMetrics = dict[str, Union[int, float, str]]
PatternResult = dict[str, Union[int, float, str, PatternMetrics]]

# Type aliases for analysis results and multi-timeframe analysis structures
AnalysisResult = dict[str, Union[float, str, bool, dict[str, float]]]
MultiTimeframeAnalysis = dict[str, AnalysisResult]

# Type aliases for regime adjustments structures
RegimeAdjustment = dict[str, Union[int, float, str]]

# Type aliases for validation and error handling structures
ValidationResult = dict[str, Union[bool, str, list[str]]]
ErrorInfo = dict[str, Union[str, int, dict[str, object]]]


if TYPE_CHECKING:
    # Import concrete classes for static type checking only to satisfy forward
    # references used throughout the package. These imports are guarded to
    # avoid runtime import side effects during test collection.
    from .action_signal_guide import (
        ActionSignal,
        ActionSignalGuideConfig,
        GuidanceLevel,
    )
