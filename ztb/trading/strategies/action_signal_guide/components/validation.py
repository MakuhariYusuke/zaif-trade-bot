"""
Utility Components for Action Signal Guide.

This module provides validation, helper functions, and utility classes
to support Action Signal Guide operations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Callable, TypedDict

import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Import ActionSignal only for static type checking to avoid runtime import
    # side-effects during test collection.
    from ..action_signal_guide import ActionSignal  # type: ignore

class ValidationMetadata(TypedDict):
    validation_timestamp: pd.Timestamp
    rules_checked: list[str]
    total_issues: int
    total_warnings: int

class ValidationRuleResult(TypedDict, total=False):
    passed: bool
    issues: list[str]
    warnings: list[str]
    penalty: float

class ValidationHistoryEntry(TypedDict):
    signal_id: str
    timestamp: pd.Timestamp
    result: "ValidationResult"

class SanitizationResult(TypedDict, total=False):
    data: pd.DataFrame
    issues: list[str]
    changes: str
    quality_penalty: float

class SanitizationStep(TypedDict, total=False):
    rule: str
    applied: bool
    changes: str
    error: str

class SanitizationReport(TypedDict):
    original_rows: int
    sanitization_steps: list[SanitizationStep]
    issues_found: list[str]
    data_quality_score: float
    final_rows: int

class PerformanceRecord(TypedDict):
    signal_id: str
    entry_price: float
    exit_price: float
    price_change_pct: float
    holding_time_hours: float
    is_profitable: bool
    risk_adjusted_return: float
    pattern_type: str
    market_regime: str | None
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    recorded_at: pd.Timestamp

class PerformanceTimeRange(TypedDict):
    start: pd.Timestamp
    end: pd.Timestamp

class PerformanceMetrics(TypedDict):
    total_signals: int
    win_rate: float
    avg_return_pct: float
    total_return_pct: float
    volatility: float
    sharpe_ratio: float
    max_drawdown_pct: float
    max_profit_pct: float
    avg_holding_time_hours: float
    pattern_performance: dict[str, float]
    time_range: PerformanceTimeRange | None

class CachedPerformanceMetrics(TypedDict):
    metrics: PerformanceMetrics
    calculated_at: pd.Timestamp

@dataclass
class ValidationResult:
    """Result of signal validation."""

    is_valid: bool
    confidence_score: float
    issues: list[str]
    warnings: list[str]
    metadata: ValidationMetadata

class SignalValidator:
    """
    Validates ActionSignal objects for consistency and correctness.
    """

    def __init__(self) -> None:
        """Initialize signal validator."""
        self.validation_rules: dict[
            str,
            Callable[["ActionSignal"], ValidationRuleResult],
        ] = {
            "required_fields": self._validate_required_fields,
            "data_types": self._validate_data_types,
            "value_ranges": self._validate_value_ranges,
            "temporal_consistency": self._validate_temporal_consistency,
            "logical_consistency": self._validate_logical_consistency,
        }
        self.validation_history: deque[ValidationHistoryEntry] = deque(maxlen=1000)

    def validate_signal(self, signal: "ActionSignal") -> ValidationResult:
        """
        Validate a complete ActionSignal object.

        Args:
            signal: Signal to validate

        Returns:
            Validation result
        """
        issues: list[str] = []
        warnings: list[str] = []
        confidence_score = 1.0

        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(signal)
                if not result["passed"]:
                    issues.extend(result.get("issues", []))
                    confidence_score *= result.get("penalty", 0.8)
                warnings.extend(result.get("warnings", []))
            except Exception as exc:
                issues.append(f"Validation error in {rule_name}: {exc}")
                confidence_score *= 0.9

        metadata: ValidationMetadata = {
            "validation_timestamp": pd.Timestamp.now(),
            "rules_checked": list(self.validation_rules.keys()),
            "total_issues": len(issues),
            "total_warnings": len(warnings),
        }

        result = ValidationResult(
            is_valid=not issues,
            confidence_score=confidence_score,
            issues=issues,
            warnings=warnings,
            metadata=metadata,
        )

        signal_id = getattr(signal, "id", "unknown")
        self.validation_history.append(
            {
                "signal_id": str(signal_id),
                "timestamp": pd.Timestamp.now(),
                "result": result,
            }
        )

        return result

    def _validate_required_fields(self, signal: "ActionSignal") -> ValidationRuleResult:
        """Validate required fields are present."""
        required_fields = ["action", "confidence", "timestamp", "pattern_type"]
        issues: list[str] = []
        warnings: list[str] = []

        for field in required_fields:
            if not hasattr(signal, field) or getattr(signal, field) is None:
                issues.append(f"Missing required field: {field}")

        # Check for recommended fields
        recommended_fields = ["price", "stop_loss", "take_profit"]
        for field in recommended_fields:
            if not hasattr(signal, field) or getattr(signal, field) is None:
                warnings.append(f"Missing recommended field: {field}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "penalty": 0.9 if issues else 1.0,
        }

    def _validate_data_types(self, signal: "ActionSignal") -> ValidationRuleResult:
        """Validate field data types."""
        type_requirements: dict[str, tuple[type[object], ...]] = {
            "action": (str, type(None)),
            "confidence": (int, float, type(None)),
            "timestamp": (pd.Timestamp, datetime, str, type(None)),
            "pattern_type": (str, type(None)),
            "price": (int, float, type(None)),
            "stop_loss": (int, float, type(None)),
            "take_profit": (int, float, type(None)),
        }

        issues: list[str] = []

        for field, expected_types in type_requirements.items():
            if hasattr(signal, field):
                value = getattr(signal, field)
                if value is not None and not isinstance(value, expected_types):
                    issues.append(
                        (
                            f"Field '{field}' has wrong type. "
                            f"Expected {expected_types}, got {type(value)}"
                        )
                    )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "penalty": 0.95 if issues else 1.0,
        }

    def _validate_value_ranges(self, signal: "ActionSignal") -> ValidationRuleResult:
        """Validate field value ranges."""
        range_requirements = {
            "confidence": (0.0, 1.0),
            "price": (0.0, float("inf")),
            "stop_loss": (0.0, float("inf")),
            "take_profit": (0.0, float("inf")),
        }

        issues: list[str] = []

        for field, (min_val, max_val) in range_requirements.items():
            if hasattr(signal, field):
                value = getattr(signal, field)
                if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                    issues.append(
                        (
                            f"Field '{field}' value {value} "
                            f"is outside valid range [{min_val}, {max_val}]"
                        )
                    )

        # Special validation for action field
        if hasattr(signal, "action"):
            raw_action = getattr(signal, "action", "")
            action = str(raw_action).upper() if raw_action is not None else ""
            valid_actions = ["BUY", "SELL", "LONG", "SHORT", "HOLD", "CLOSE"]
            if action and action not in valid_actions:
                issues.append(f"Invalid action '{action}'. Must be one of {valid_actions}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "penalty": 0.9 if issues else 1.0,
        }

    def _validate_temporal_consistency(self, signal: "ActionSignal") -> ValidationRuleResult:
        """Validate temporal aspects of the signal."""
        issues: list[str] = []
        warnings: list[str] = []

        if hasattr(signal, "timestamp"):
            timestamp = getattr(signal, "timestamp")
            if timestamp is not None:
                # Convert to pandas timestamp if needed
                if isinstance(timestamp, str):
                    try:
                        timestamp = pd.to_datetime(timestamp)
                    except Exception:
                        issues.append("Invalid timestamp format")
                        return {"passed": False, "issues": issues, "penalty": 0.9}

                if isinstance(timestamp, (pd.Timestamp, datetime)):
                    now = pd.Timestamp.now()
                    # Check if timestamp is not too far in the future
                    if timestamp > now + timedelta(hours=1):
                        issues.append("Signal timestamp is too far in the future")
                    # Check if timestamp is not too old (more than 1 hour ago)
                    elif timestamp < now - timedelta(hours=1):
                        warnings.append("Signal timestamp is more than 1 hour old")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "penalty": 0.95 if issues else 1.0,
        }

    def _validate_logical_consistency(self, signal: "ActionSignal") -> ValidationRuleResult:
        """Validate logical consistency between fields."""
        issues: list[str] = []

        # Check stop loss and take profit relationship with price
        if hasattr(signal, "price") and hasattr(signal, "action"):
            price = getattr(signal, "price")
            raw_action = getattr(signal, "action", "")
            action = str(raw_action).upper() if raw_action is not None else ""

            if isinstance(price, (int, float)) and action:
                if action in ["BUY", "LONG"]:
                    # For buy signals, stop loss should be below price, take profit above
                    if hasattr(signal, "stop_loss"):
                        stop_loss = getattr(signal, "stop_loss")
                        if isinstance(stop_loss, (int, float)) and stop_loss >= price:
                            issues.append(
                                "For BUY signals, stop loss should be below entry price"
                            )

                    if hasattr(signal, "take_profit"):
                        take_profit = getattr(signal, "take_profit")
                        if (
                            isinstance(take_profit, (int, float))
                            and take_profit <= price
                        ):
                            issues.append(
                                "For BUY signals, take profit should be above entry price"
                            )

                elif action in ["SELL", "SHORT"]:
                    # For sell signals, stop loss should be above price, take profit below
                    if hasattr(signal, "stop_loss"):
                        stop_loss = getattr(signal, "stop_loss")
                        if isinstance(stop_loss, (int, float)) and stop_loss <= price:
                            issues.append(
                                "For SELL signals, stop loss should be above entry price"
                            )

                    if hasattr(signal, "take_profit"):
                        take_profit = getattr(signal, "take_profit")
                        if (
                            isinstance(take_profit, (int, float))
                            and take_profit >= price
                        ):
                            issues.append(
                                "For SELL signals, take profit should be below entry price"
                            )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "penalty": 1.0,  # Logical issues don't reduce confidence as much
        }

class DataSanitizer:
    """
    Sanitizes and cleans market data for signal processing.
    """

    def __init__(self) -> None:
        """Initialize data sanitizer."""
        self.sanitization_rules: dict[
            str,
            Callable[[pd.DataFrame], SanitizationResult],
        ] = {
            "remove_outliers": self._remove_outliers,
            "fill_missing_values": self._fill_missing_values,
            "normalize_data": self._normalize_data,
            "validate_ohlc": self._validate_ohlc_consistency,
        }

    def sanitize_market_data(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, SanitizationReport]:
        """
        Sanitize market data for signal processing.

        Args:
            data: Raw market data

        Returns:
            tuple of (sanitized_data, sanitization_report)
        """
        sanitized_data = data.copy()
        report: SanitizationReport = {
            "original_rows": len(data),
            "sanitization_steps": [],
            "issues_found": [],
            "data_quality_score": 1.0,
            "final_rows": len(data),
        }

        # Apply sanitization rules
        for rule_name, rule_func in self.sanitization_rules.items():
            try:
                result = rule_func(sanitized_data)
                result_data = result.get("data")
                if isinstance(result_data, pd.DataFrame):
                    sanitized_data = result_data

                issues = result.get("issues", [])
                if issues:
                    report["issues_found"].extend(issues)
                    report["data_quality_score"] *= result.get("quality_penalty", 0.95)

                report["sanitization_steps"].append(
                    {
                        "rule": rule_name,
                        "applied": True,
                        "changes": result.get("changes", "None"),
                    }
                )

            except Exception as exc:
                report["issues_found"].append(f"Error in {rule_name}: {exc}")
                report["data_quality_score"] *= 0.9
                report["sanitization_steps"].append(
                    {
                        "rule": rule_name,
                        "applied": False,
                        "error": str(exc),
                    }
                )

        report["final_rows"] = len(sanitized_data)
        return sanitized_data, report

    def _remove_outliers(self, data: pd.DataFrame) -> SanitizationResult:
        """Remove statistical outliers from price data."""
        working_data = data.copy()
        result: SanitizationResult = {
            "data": working_data,
            "issues": [],
            "changes": "None",
        }

        if len(working_data) < 10:
            return result

        change_messages: list[str] = []

        # Check for price outliers using IQR method
        for column in ["open", "high", "low", "close"]:
            if column in working_data.columns:
                prices = working_data[column].dropna()
                if len(prices) > 0:
                    q1 = prices.quantile(0.25)
                    q3 = prices.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
                    if len(outliers) > 0:
                        result["issues"].append(
                            f"Found {len(outliers)} outliers in {column} column"
                        )
                        # Replace outliers with median
                        median_price = prices.median()
                        working_data.loc[outliers.index, column] = median_price
                        change_messages.append(
                            f"Replaced {len(outliers)} outliers in {column}"
                        )

        if change_messages:
            result["changes"] = "; ".join(change_messages)

        return result

    def _fill_missing_values(self, data: pd.DataFrame) -> SanitizationResult:
        """Fill missing values in market data."""
        result: SanitizationResult = {
            "data": data.copy(),
            "issues": [],
            "changes": "None",
        }

        # Check for missing values
        missing_counts = data.isnull().sum()
        total_missing = int(missing_counts.sum())

        if total_missing > 0:
            result["issues"].append(f"Found {total_missing} missing values")

            # Fill missing values using forward fill, then backward fill
            filled_data = data.ffill().bfill()

            # For any remaining missing values, use column median
            for column in filled_data.columns:
                if filled_data[column].isnull().any():
                    median_value = filled_data[column].median()
                    if pd.notnull(median_value):
                        filled_data[column] = filled_data[column].fillna(median_value)

            result["data"] = filled_data
            result["changes"] = f"Filled {total_missing} missing values"

        return result

    def _normalize_data(self, data: pd.DataFrame) -> SanitizationResult:
        """Normalize data formats and types."""
        normalized_data = data.copy()
        result: SanitizationResult = {
            "data": normalized_data,
            "issues": [],
            "changes": "None",
        }

        change_messages: list[str] = []

        # Ensure numeric columns are properly typed
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for column in numeric_columns:
            if column in normalized_data.columns:
                try:
                    # Convert to numeric, coerce errors to NaN
                    original_dtype = normalized_data[column].dtype
                    normalized_data[column] = pd.to_numeric(
                        normalized_data[column], errors="coerce"
                    )
                    new_dtype = normalized_data[column].dtype

                    if original_dtype != new_dtype:
                        change_messages.append(
                            f"Converted {column} from {original_dtype} to {new_dtype}"
                        )

                except Exception as exc:
                    result["issues"].append(f"Error normalizing {column}: {exc}")

        # Ensure datetime index if present
        if not pd.api.types.is_datetime64_any_dtype(normalized_data.index):
            try:
                has_timestamp_index = normalized_data.index.name == "timestamp"
                has_timestamp_column = "timestamp" in normalized_data.columns
                if has_timestamp_index or has_timestamp_column:
                    timestamp_col = (
                        normalized_data.index.name
                        if has_timestamp_index
                        else "timestamp"
                    )
                    if timestamp_col in normalized_data.columns:
                        converted_index = pd.to_datetime(
                            normalized_data[timestamp_col], errors="coerce"
                        )
                        if converted_index.notna().any():
                            normalized_data.index = converted_index
                            change_messages.append("Converted index to datetime")
            except Exception as exc:
                result["issues"].append(f"Error converting timestamp: {exc}")

        if change_messages:
            result["changes"] = "; ".join(change_messages)

        return result

    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> SanitizationResult:
        """Validate OHLC data consistency."""
        result: SanitizationResult = {
            "data": data.copy(),
            "issues": [],
            "changes": "None",
        }

        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            result["issues"].append(f"Missing required OHLC columns: {missing_columns}")
            return result

        # Check OHLC logical consistency
        inconsistencies: list[str] = []

        # High should be >= max(open, close)
        invalid_high = data["high"] < data[["open", "close"]].max(axis=1)
        if invalid_high.any():
            inconsistencies.append(f"{int(invalid_high.sum())} rows have high < max(open, close)")

        # Low should be <= min(open, close)
        invalid_low = data["low"] > data[["open", "close"]].min(axis=1)
        if invalid_low.any():
            inconsistencies.append(f"{int(invalid_low.sum())} rows have low > min(open, close)")

        if inconsistencies:
            result["issues"].extend(inconsistencies)
            result["quality_penalty"] = 0.8

        return result

class PerformanceTracker:
    """
    Tracks and analyzes signal performance metrics.
    """

    def __init__(self) -> None:
        """Initialize performance tracker."""
        self.performance_history: list[PerformanceRecord] = []
        self.metrics_cache: dict[
            tuple[str | None, str | None, timedelta | None],
            CachedPerformanceMetrics,
        ] = {}
        self.update_interval = timedelta(minutes=5)

    def record_signal_performance(
        self,
        signal_id: str,
        entry_price: float,
        exit_price: float,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        pattern_type: str,
        market_regime: str | None = None,
    ) -> PerformanceRecord:
        """
        Record performance of a completed signal.

        Args:
            signal_id: Unique signal identifier
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            pattern_type: Type of pattern used
            market_regime: Market regime during signal

        Returns:
            Performance metrics for the signal
        """
        if entry_price == 0:
            raise ZeroDivisionError("entry_price must be non-zero")
        price_change = (exit_price - entry_price) / entry_price

        holding_time = exit_time - entry_time
        holding_hours = max(holding_time.total_seconds() / 3600, 0.0)

        # Determine if profitable
        is_profitable = price_change > 0

        # Calculate risk-adjusted return (simplified Sharpe-like ratio)
        volatility = abs(price_change) / max(holding_hours, 1.0)
        risk_adjusted_return = price_change / volatility if volatility > 0 else 0.0

        now = pd.Timestamp.now()
        performance_record: PerformanceRecord = {
            "signal_id": signal_id,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "price_change_pct": price_change,
            "holding_time_hours": holding_hours,
            "is_profitable": is_profitable,
            "risk_adjusted_return": risk_adjusted_return,
            "pattern_type": pattern_type,
            "market_regime": market_regime,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "recorded_at": now,
        }

        self.performance_history.append(performance_record)

        # Keep only recent history
        cutoff_date = now - timedelta(days=30)
        self.performance_history = [
            record
            for record in self.performance_history
            if record["recorded_at"] > cutoff_date
        ]

        # Clear metrics cache to force recalculation
        self.metrics_cache.clear()

        return performance_record

    def get_performance_metrics(
        self,
        pattern_type: str | None = None,
        market_regime: str | None = None,
        time_window: timedelta | None = None,
    ) -> PerformanceMetrics:
        """
        Get performance metrics, optionally filtered.

        Args:
            pattern_type: Filter by pattern type
            market_regime: Filter by market regime
            time_window: Time window to analyze

        Returns:
            Performance metrics dictionary
        """
        cache_key = (pattern_type, market_regime, time_window)
        now = pd.Timestamp.now()

        cached_result = self.metrics_cache.get(cache_key)
        if cached_result and now - cached_result["calculated_at"] < self.update_interval:
            return cached_result["metrics"]

        filtered_data = list(self.performance_history)

        if pattern_type:
            filtered_data = [
                record
                for record in filtered_data
                if record["pattern_type"] == pattern_type
            ]

        if market_regime:
            filtered_data = [
                record
                for record in filtered_data
                if record["market_regime"] == market_regime
            ]

        if time_window:
            cutoff_time = now - time_window
            filtered_data = [
                record
                for record in filtered_data
                if record["recorded_at"] > cutoff_time
            ]

        if not filtered_data:
            return self._get_empty_metrics()

        metrics = self._calculate_metrics(filtered_data)

        self.metrics_cache[cache_key] = {
            "metrics": metrics,
            "calculated_at": now,
        }

        return metrics

    def _calculate_metrics(
        self,
        performance_data: list[PerformanceRecord],
    ) -> PerformanceMetrics:
        """Calculate performance metrics from data."""
        if not performance_data:
            return self._get_empty_metrics()

        # Basic metrics
        total_signals = len(performance_data)
        profitable_signals = sum(1 for record in performance_data if record["is_profitable"])
        win_rate = profitable_signals / total_signals if total_signals > 0 else 0.0

        # Profit metrics
        price_changes = [record["price_change_pct"] for record in performance_data]
        avg_return = sum(price_changes) / len(price_changes) if price_changes else 0.0
        total_return = float(sum(price_changes))

        # Risk metrics
        returns_std = float(pd.Series(price_changes, dtype="float64").std(ddof=0))
        if pd.isna(returns_std):
            returns_std = 0.0
        max_drawdown = float(min(price_changes)) if price_changes else 0.0
        max_profit = float(max(price_changes)) if price_changes else 0.0

        # Sharpe ratio (simplified)
        sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0.0

        # Holding time analysis
        holding_times = [record["holding_time_hours"] for record in performance_data]
        avg_holding_time = (
            sum(holding_times) / len(holding_times) if holding_times else 0.0
        )

        # Pattern analysis
        pattern_performance: dict[str, list[float]] = {}
        for record in performance_data:
            pattern = record["pattern_type"]
            pattern_performance.setdefault(pattern, []).append(record["price_change_pct"])

        pattern_avg_returns = {
            pattern: (sum(returns) / len(returns) if returns else 0.0)
            for pattern, returns in pattern_performance.items()
        }

        time_range: PerformanceTimeRange = {
            "start": min(record["recorded_at"] for record in performance_data),
            "end": max(record["recorded_at"] for record in performance_data),
        }

        return {
            "total_signals": total_signals,
            "win_rate": float(win_rate),
            "avg_return_pct": float(avg_return),
            "total_return_pct": total_return,
            "volatility": returns_std,
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown_pct": max_drawdown,
            "max_profit_pct": max_profit,
            "avg_holding_time_hours": float(avg_holding_time),
            "pattern_performance": pattern_avg_returns,
            "time_range": time_range,
        }

    def _get_empty_metrics(self) -> PerformanceMetrics:
        """Get empty metrics structure."""
        return {
            "total_signals": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "total_return_pct": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_profit_pct": 0.0,
            "avg_holding_time_hours": 0.0,
            "pattern_performance": {},
            "time_range": None,
        }
