#!/usr/bin/env python3
"""
Common Analysis Interfaces

Provides standardized analysis interfaces and base classes for various analysis tasks.
Ensures type safety and consistent behavior across analysis components.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Type for analysis input
U = TypeVar("U")  # Type for analysis output


class AnalysisError(Exception):
    """Exception raised when analysis fails."""

    pass


class Analyzer(Protocol[T, U]):
    """Protocol for analysis interfaces."""

    def analyze(self, data: T) -> U:
        """Perform analysis on the given data."""
        ...

    def validate_input(self, data: T) -> bool:
        """Validate input data."""
        ...


class BaseAnalyzer(ABC, Generic[T, U], Analyzer[T, U]):
    """Base class for analysis components with common functionality."""

    def __init__(self, name: str = "BaseAnalyzer"):
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def analyze(self, data: T) -> U:
        """Perform analysis on the given data."""
        pass

    def validate_input(self, data: T) -> bool:
        """Validate input data. Override in subclasses for specific validation."""
        return data is not None

    def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle analysis errors with logging."""
        self.logger.error(f"Analysis error in {self.name}{context}: {error}")
        raise AnalysisError(f"Analysis failed: {error}") from error


class AnalysisPipeline(Generic[T, U]):
    """Pipeline for chaining multiple analysis steps."""

    def __init__(self, analyzers: List[Analyzer]):
        self.analyzers = analyzers
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, data: T) -> Dict[str, Any]:
        """
        Execute the analysis pipeline.

        Args:
            data: Input data for analysis

        Returns:
            Dictionary containing results from all analyzers
        """
        results = {}

        for i, analyzer in enumerate(self.analyzers):
            try:
                step_result = analyzer.analyze(data)
                results[f"step_{i}_{analyzer.__class__.__name__}"] = step_result
                self.logger.info(
                    f"Completed analysis step {i}: {analyzer.__class__.__name__}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed analysis step {i}: {analyzer.__class__.__name__}"
                )
                raise AnalysisError(f"Pipeline failed at step {i}: {e}") from e

        return results


class AnalysisResultFormatter(ABC):
    """Base class for formatting analysis results."""

    @abstractmethod
    def format_results(self, results: Any) -> str:
        """Format analysis results for display."""
        pass


class BaseResultFormatter(Generic[U]):
    """Base class for result formatters."""

    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata

    def format_results(self, results: U) -> str:
        """Format results with optional metadata."""
        formatted = self._format_core_results(results)

        if self.include_metadata:
            formatted = self._add_metadata(formatted, results)

        return formatted

    @abstractmethod
    def _format_core_results(self, results: U) -> str:
        """Format the core analysis results."""
        pass

    def _add_metadata(self, formatted: str, results: U) -> str:
        """Add metadata to formatted results."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Generated: {timestamp}\n\n{formatted}"


class AnalysisValidator(Generic[T]):
    """Validator for analysis inputs and outputs."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_input(
        self, data: T, required_fields: Optional[List[str]] = None
    ) -> bool:
        """
        Validate input data structure.

        Args:
            data: Input data to validate
            required_fields: List of required field names

        Returns:
            True if validation passes
        """
        if data is None:
            if self.strict_mode:
                raise ValueError("Input data cannot be None")
            return False

        if required_fields:
            if isinstance(data, dict):
                missing_fields = [
                    field for field in required_fields if field not in data
                ]
                if missing_fields:
                    error_msg = f"Missing required fields: {missing_fields}"
                    if self.strict_mode:
                        raise ValueError(error_msg)
                    self.logger.warning(error_msg)
                    return False

        return True

    def validate_output(
        self, results: Any, expected_type: Optional[type] = None
    ) -> bool:
        """
        Validate analysis output.

        Args:
            results: Analysis results to validate
            expected_type: Expected type of results

        Returns:
            True if validation passes
        """
        if results is None:
            if self.strict_mode:
                raise ValueError("Analysis results cannot be None")
            return False

        if expected_type and not isinstance(results, expected_type):
            error_msg = f"Expected {expected_type}, got {type(results)}"
            if self.strict_mode:
                raise TypeError(error_msg)
            self.logger.warning(error_msg)
            return False

        return True


# Type-safe analysis result containers
from dataclasses import dataclass
from typing import List as TypingList


@dataclass
class AnalysisSummary:
    """Container for analysis summary information."""

    name: str
    description: str
    metrics: Dict[str, Any]
    warnings: Optional[TypingList[str]] = None
    errors: Optional[TypingList[str]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


@dataclass
class ComparativeAnalysisResult:
    """Container for comparative analysis results."""

    baseline_name: str
    comparison_name: str
    metrics_comparison: Dict[str, Dict[str, Any]]
    summary: str
    recommendations: Optional[TypingList[str]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class DisplayManagerProtocol(Protocol):
    """Protocol for display managers."""

    def display_backtest_results(
        self,
        results: Dict[str, Any],
        title: str = "Backtest Results",
        show_plots: bool = True,
        save_plots: bool = True,
    ) -> None:
        """Display backtest results."""
        ...

    def display_analysis_results(
        self, results: Dict[str, Any], title: str = "Analysis Results"
    ) -> None:
        """Display analysis results."""
        ...
