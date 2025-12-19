"""
Type definitions for ActionSignalGuide.

This module provides specific type aliases and definitions to replace generic Any types
used in the ActionSignalGuide system.
"""

from typing import Any, Dict, List, Union, TYPE_CHECKING

# Type aliases for configuration dictionaries
SignalConfig = Dict[str, Union[bool, int, float, str, List[str], Dict[str, Union[int, float, None]]]]
PatternConfig = Dict[str, Union[bool, int, float, str, List[str], Dict[str, Union[int, float, None]]]]

# Type aliases for signals and results (using string forward references)
SignalList = List["ActionSignal"]  # type: ignore
SignalHistory = List["ActionSignal"]  # type: ignore

# Type aliases for statistics and metrics structures
PerformanceStats = Dict[str, Union[int, float, Dict[str, Union[int, float]]]]
PatternStats = Dict[str, Union[int, float, Dict[str, Union[int, float]]]]
CacheStats = Dict[str, Union[int, float]]

# Type aliases for signal metadata and statistics metadata
SignalMetadata = Dict[str, Union[int, float, str, bool, List[Union[int, float]]]]
StatisticsMetadata = Dict[str, Any]

# Type aliases for recognizer status structures
RecognizerGroupStatus = Dict[str, Union[bool, int, List[str]]]
RecognizerStatus = Dict[str, Union[int, str, Dict[str, RecognizerGroupStatus]]]

# Union types for flexible inputs
ConfigInput = Union["ActionSignalGuideConfig", Dict[str, Any], None]  # type: ignore
GuidanceInput = Union["GuidanceLevel", str, None]  # type: ignore

# Type aliases for multi-timeframe data structures
MultiTimeframeData = Dict[str, Dict[str, Any]]  # timeframe -> {'data': DataFrame, ...}

# More specific type aliases for pattern recognition components
PatternThresholds = Dict[str, Union[int, float, None]]
PatternMetrics = Dict[str, Union[int, float, str]]
PatternResult = Dict[str, Union[int, float, str, PatternMetrics]]

# Type aliases for analysis results and multi-timeframe analysis structures
AnalysisResult = Dict[str, Union[float, str, bool, Dict[str, float]]]
MultiTimeframeAnalysis = Dict[str, AnalysisResult]

# Type aliases for regime adjustments structures
RegimeAdjustment = Dict[str, Union[int, float, str]]

# Type aliases for validation and error handling structures
ValidationResult = Dict[str, Union[bool, str, List[str]]]
ErrorInfo = Dict[str, Union[str, int, Dict[str, Any]]]


if TYPE_CHECKING:
	# Import concrete classes for static type checking only to satisfy forward
	# references used throughout the package. These imports are guarded to
	# avoid runtime import side effects during test collection.
	from .action_signal_guide import (
		ActionSignal,
		ActionSignalGuideConfig,
		GuidanceLevel,
	)
