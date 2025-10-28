"""
Type definitions for ActionSignalGuide.

This module provides specific type aliases and definitions to replace generic Any types
used in the ActionSignalGuide system.
"""

from typing import Any, Dict, List, Union

# Type aliases for configuration
SignalConfig = Dict[str, Any]  # Generic signal configuration
PatternConfig = Dict[str, Any]  # Pattern-specific configuration

# Type aliases for signals and results (using string forward references)
SignalList = List["ActionSignal"]  # type: ignore
SignalHistory = List["ActionSignal"]  # type: ignore

# Type aliases for statistics and metrics
PerformanceStats = Dict[str, Union[int, float, Dict[str, Union[int, float]]]]
PatternStats = Dict[str, Union[int, float, Dict[str, Union[int, float]]]]
CacheStats = Dict[str, Union[int, float]]

# Type aliases for metadata
SignalMetadata = Dict[str, Any]
StatisticsMetadata = Dict[str, Any]

# Union types for flexible inputs
ConfigInput = Union["ActionSignalGuideConfig", Dict[str, Any], None]  # type: ignore
GuidanceInput = Union["GuidanceLevel", str, None]  # type: ignore
