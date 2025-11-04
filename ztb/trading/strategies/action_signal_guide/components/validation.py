"""
Utility Components for Action Signal Guide.

This module provides validation, helper functions, and utility classes
to support Action Signal Guide operations.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of signal validation."""
    is_valid: bool
    confidence_score: float
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class SignalValidator:
    """
    Validates ActionSignal objects for consistency and correctness.
    """

    def __init__(self):
        """Initialize signal validator."""
        self.validation_rules = {
            "required_fields": self._validate_required_fields,
            "data_types": self._validate_data_types,
            "value_ranges": self._validate_value_ranges,
            "temporal_consistency": self._validate_temporal_consistency,
            "logical_consistency": self._validate_logical_consistency,
        }

        self.validation_history = []

    def validate_signal(self, signal: "ActionSignal") -> ValidationResult:
        """
        Validate a complete ActionSignal object.

        Args:
            signal: Signal to validate

        Returns:
            Validation result
        """
        issues = []
        warnings = []
        confidence_score = 1.0

        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(signal)
                if not result["passed"]:
                    issues.extend(result.get("issues", []))
                    confidence_score *= result.get("penalty", 0.8)
                if result.get("warnings"):
                    warnings.extend(result["warnings"])
            except Exception as e:
                issues.append(f"Validation error in {rule_name}: {str(e)}")
                confidence_score *= 0.9

        # Overall validity
        is_valid = len(issues) == 0

        # Create metadata
        metadata = {
            "validation_timestamp": pd.Timestamp.now(),
            "rules_checked": list(self.validation_rules.keys()),
            "total_issues": len(issues),
            "total_warnings": len(warnings),
        }

        result = ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues=issues,
            warnings=warnings,
            metadata=metadata,
        )

        # Store validation result
        self.validation_history.append({
            "signal_id": getattr(signal, 'id', 'unknown'),
            "timestamp": pd.Timestamp.now(),
            "result": result,
        })

        # Keep only recent history
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-500:]

        return result

    def _validate_required_fields(self, signal: "ActionSignal") -> Dict[str, Any]:
        """Validate required fields are present."""
        required_fields = ['action', 'confidence', 'timestamp', 'pattern_type']
        issues = []
        warnings = []

        for field in required_fields:
            if not hasattr(signal, field) or getattr(signal, field) is None:
                issues.append(f"Missing required field: {field}")

        # Check for recommended fields
        recommended_fields = ['price', 'stop_loss', 'take_profit']
        for field in recommended_fields:
            if not hasattr(signal, field) or getattr(signal, field) is None:
                warnings.append(f"Missing recommended field: {field}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "penalty": 0.9 if issues else 1.0,
        }

    def _validate_data_types(self, signal: "ActionSignal") -> Dict[str, Any]:
        """Validate field data types."""
        type_requirements = {
            'action': (str, type(None)),
            'confidence': (int, float, type(None)),
            'timestamp': (pd.Timestamp, datetime, str, type(None)),
            'pattern_type': (str, type(None)),
            'price': (int, float, type(None)),
            'stop_loss': (int, float, type(None)),
            'take_profit': (int, float, type(None)),
        }

        issues = []

        for field, expected_types in type_requirements.items():
            if hasattr(signal, field):
                value = getattr(signal, field)
                if value is not None and not isinstance(value, expected_types):
                    issues.append(
                        f"Field '{field}' has wrong type. Expected {expected_types}, got {type(value)}"
                    )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "penalty": 0.95 if issues else 1.0,
        }

    def _validate_value_ranges(self, signal: "ActionSignal") -> Dict[str, Any]:
        """Validate field value ranges."""
        range_requirements = {
            'confidence': (0.0, 1.0),
            'price': (0.0, float('inf')),
            'stop_loss': (0.0, float('inf')),
            'take_profit': (0.0, float('inf')),
        }

        issues = []

        for field, (min_val, max_val) in range_requirements.items():
            if hasattr(signal, field):
                value = getattr(signal, field)
                if value is not None:
                    if not (min_val <= value <= max_val):
                        issues.append(
                            f"Field '{field}' value {value} is outside valid range [{min_val}, {max_val}]"
                        )

        # Special validation for action field
        if hasattr(signal, 'action'):
            action = getattr(signal, 'action', '').upper()
            valid_actions = ['BUY', 'SELL', 'LONG', 'SHORT', 'HOLD', 'CLOSE']
            if action and action not in valid_actions:
                issues.append(f"Invalid action '{action}'. Must be one of {valid_actions}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "penalty": 0.9 if issues else 1.0,
        }

    def _validate_temporal_consistency(self, signal: "ActionSignal") -> Dict[str, Any]:
        """Validate temporal aspects of the signal."""
        issues = []
        warnings = []

        if hasattr(signal, 'timestamp'):
            timestamp = getattr(signal, 'timestamp')
            if timestamp is not None:
                # Convert to pandas timestamp if needed
                if isinstance(timestamp, str):
                    try:
                        timestamp = pd.to_datetime(timestamp)
                    except:
                        issues.append("Invalid timestamp format")
                        return {"passed": False, "issues": issues}

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

    def _validate_logical_consistency(self, signal: "ActionSignal") -> Dict[str, Any]:
        """Validate logical consistency between fields."""
        issues = []

        # Check stop loss and take profit relationship with price
        if hasattr(signal, 'price') and hasattr(signal, 'action'):
            price = getattr(signal, 'price')
            action = getattr(signal, 'action', '').upper()

            if price is not None and action:
                if action in ['BUY', 'LONG']:
                    # For buy signals, stop loss should be below price, take profit above
                    if hasattr(signal, 'stop_loss'):
                        stop_loss = getattr(signal, 'stop_loss')
                        if stop_loss is not None and stop_loss >= price:
                            issues.append("For BUY signals, stop loss should be below entry price")

                    if hasattr(signal, 'take_profit'):
                        take_profit = getattr(signal, 'take_profit')
                        if take_profit is not None and take_profit <= price:
                            issues.append("For BUY signals, take profit should be above entry price")

                elif action in ['SELL', 'SHORT']:
                    # For sell signals, stop loss should be above price, take profit below
                    if hasattr(signal, 'stop_loss'):
                        stop_loss = getattr(signal, 'stop_loss')
                        if stop_loss is not None and stop_loss <= price:
                            issues.append("For SELL signals, stop loss should be above entry price")

                    if hasattr(signal, 'take_profit'):
                        take_profit = getattr(signal, 'take_profit')
                        if take_profit is not None and take_profit >= price:
                            issues.append("For SELL signals, take profit should be below entry price")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "penalty": 1.0,  # Logical issues don't reduce confidence as much
        }


class DataSanitizer:
    """
    Sanitizes and cleans market data for signal processing.
    """

    def __init__(self):
        """Initialize data sanitizer."""
        self.sanitization_rules = {
            "remove_outliers": self._remove_outliers,
            "fill_missing_values": self._fill_missing_values,
            "normalize_data": self._normalize_data,
            "validate_ohlc": self._validate_ohlc_consistency,
        }

    def sanitize_market_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Sanitize market data for signal processing.

        Args:
            data: Raw market data

        Returns:
            Tuple of (sanitized_data, sanitization_report)
        """
        sanitized_data = data.copy()
        report = {
            "original_rows": len(data),
            "sanitization_steps": [],
            "issues_found": [],
            "data_quality_score": 1.0,
        }

        # Apply sanitization rules
        for rule_name, rule_func in self.sanitization_rules.items():
            try:
                result = rule_func(sanitized_data)
                sanitized_data = result.get("data", sanitized_data)

                if result.get("issues"):
                    report["issues_found"].extend(result["issues"])
                    report["data_quality_score"] *= result.get("quality_penalty", 0.95)

                report["sanitization_steps"].append({
                    "rule": rule_name,
                    "applied": True,
                    "changes": result.get("changes", "None"),
                })

            except Exception as e:
                report["issues_found"].append(f"Error in {rule_name}: {str(e)}")
                report["data_quality_score"] *= 0.9
                report["sanitization_steps"].append({
                    "rule": rule_name,
                    "applied": False,
                    "error": str(e),
                })

        report["final_rows"] = len(sanitized_data)
        return sanitized_data, report

    def _remove_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Remove statistical outliers from price data."""
        result = {"data": data.copy(), "issues": [], "changes": "None"}

        if len(data) < 10:
            return result

        # Check for price outliers using IQR method
        for column in ['open', 'high', 'low', 'close']:
            if column in data.columns:
                prices = data[column].dropna()
                if len(prices) > 0:
                    Q1 = prices.quantile(0.25)
                    Q3 = prices.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
                    if len(outliers) > 0:
                        result["issues"].append(
                            f"Found {len(outliers)} outliers in {column} column"
                        )
                        # Replace outliers with median
                        median_price = prices.median()
                        data.loc[outliers.index, column] = median_price
                        result["changes"] = f"Replaced {len(outliers)} outliers in {column}"

        return result

    def _fill_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fill missing values in market data."""
        result = {"data": data.copy(), "issues": [], "changes": "None"}

        # Check for missing values
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()

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

    def _normalize_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Normalize data formats and types."""
        result = {"data": data.copy(), "issues": [], "changes": "None"}
        normalized_data = result["data"]  # Work on the copy

        # Ensure numeric columns are properly typed
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in numeric_columns:
            if column in normalized_data.columns:
                try:
                    # Convert to numeric, coerce errors to NaN
                    original_dtype = normalized_data[column].dtype
                    normalized_data[column] = pd.to_numeric(normalized_data[column], errors='coerce')
                    new_dtype = normalized_data[column].dtype

                    if original_dtype != new_dtype:
                        result["changes"] = f"Converted {column} from {original_dtype} to {new_dtype}"

                except Exception as e:
                    result["issues"].append(f"Error normalizing {column}: {str(e)}")

        # Ensure datetime index if present
        if hasattr(data.index, 'dtype') and not pd.api.types.is_datetime64_any_dtype(data.index):
            try:
                if data.index.name == 'timestamp' or 'timestamp' in data.columns:
                    timestamp_col = data.index.name if data.index.name == 'timestamp' else 'timestamp'
                    if timestamp_col in data.columns:
                        data.index = pd.to_datetime(data[timestamp_col])
                        result["changes"] = "Converted index to datetime"
            except Exception as e:
                result["issues"].append(f"Error converting timestamp: {str(e)}")

        return result

    def _validate_ohlc_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate OHLC data consistency."""
        result = {"data": data.copy(), "issues": [], "changes": "None"}

        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            result["issues"].append(f"Missing required OHLC columns: {missing_columns}")
            return result

        # Check OHLC logical consistency
        inconsistencies = []

        # High should be >= max(open, close)
        invalid_high = data['high'] < data[['open', 'close']].max(axis=1)
        if invalid_high.any():
            inconsistencies.append(f"{invalid_high.sum()} rows have high < max(open, close)")

        # Low should be <= min(open, close)
        invalid_low = data['low'] > data[['open', 'close']].min(axis=1)
        if invalid_low.any():
            inconsistencies.append(f"{invalid_low.sum()} rows have low > min(open, close)")

        if inconsistencies:
            result["issues"].extend(inconsistencies)
            result["quality_penalty"] = 0.8

        return result


class PerformanceTracker:
    """
    Tracks and analyzes signal performance metrics.
    """

    def __init__(self):
        """Initialize performance tracker."""
        self.performance_history = []
        self.metrics_cache = {}
        self.update_interval = timedelta(minutes=5)

    def record_signal_performance(
        self,
        signal_id: str,
        entry_price: float,
        exit_price: float,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        pattern_type: str,
        market_regime: str = None
    ) -> Dict[str, Any]:
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
        # Calculate basic metrics
        price_change = (exit_price - entry_price) / entry_price
        holding_time = exit_time - entry_time
        holding_hours = holding_time.total_seconds() / 3600

        # Determine if profitable
        is_profitable = price_change > 0

        # Calculate risk-adjusted return (simplified Sharpe-like ratio)
        volatility = abs(price_change) / max(holding_hours, 1)  # Rough volatility measure
        risk_adjusted_return = price_change / volatility if volatility > 0 else 0

        performance_record = {
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
            "recorded_at": pd.Timestamp.now(),
        }

        self.performance_history.append(performance_record)

        # Keep only recent history
        cutoff_date = pd.Timestamp.now() - timedelta(days=30)
        self.performance_history = [
            p for p in self.performance_history
            if p["recorded_at"] > cutoff_date
        ]

        # Clear metrics cache to force recalculation
        self.metrics_cache = {}

        return performance_record

    def get_performance_metrics(
        self,
        pattern_type: str = None,
        market_regime: str = None,
        time_window: timedelta = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics, optionally filtered.

        Args:
            pattern_type: Filter by pattern type
            market_regime: Filter by market regime
            time_window: Time window to analyze

        Returns:
            Performance metrics dictionary
        """
        # Create cache key
        cache_key = f"{pattern_type}_{market_regime}_{time_window}"

        # Check cache
        if cache_key in self.metrics_cache:
            cached_result = self.metrics_cache[cache_key]
            if pd.Timestamp.now() - cached_result["calculated_at"] < self.update_interval:
                return cached_result["metrics"]

        # Filter data
        filtered_data = self.performance_history

        if pattern_type:
            filtered_data = [p for p in filtered_data if p["pattern_type"] == pattern_type]

        if market_regime:
            filtered_data = [p for p in filtered_data if p["market_regime"] == market_regime]

        if time_window:
            cutoff_time = pd.Timestamp.now() - time_window
            filtered_data = [p for p in filtered_data if p["recorded_at"] > cutoff_time]

        if not filtered_data:
            return self._get_empty_metrics()

        # Calculate metrics
        metrics = self._calculate_metrics(filtered_data)

        # Cache result
        self.metrics_cache[cache_key] = {
            "metrics": metrics,
            "calculated_at": pd.Timestamp.now(),
        }

        return metrics

    def _calculate_metrics(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Calculate performance metrics from data."""
        if not performance_data:
            return self._get_empty_metrics()

        # Basic metrics
        total_signals = len(performance_data)
        profitable_signals = sum(1 for p in performance_data if p["is_profitable"])
        win_rate = profitable_signals / total_signals if total_signals > 0 else 0

        # Profit metrics
        price_changes = [p["price_change_pct"] for p in performance_data]
        avg_return = sum(price_changes) / len(price_changes) if price_changes else 0
        total_return = sum(price_changes)

        # Risk metrics
        returns_std = pd.Series(price_changes).std()
        max_drawdown = min(price_changes) if price_changes else 0
        max_profit = max(price_changes) if price_changes else 0

        # Sharpe ratio (simplified)
        sharpe_ratio = avg_return / returns_std if returns_std > 0 else 0

        # Holding time analysis
        holding_times = [p["holding_time_hours"] for p in performance_data]
        avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

        # Pattern analysis
        pattern_performance = {}
        for record in performance_data:
            pattern = record["pattern_type"]
            if pattern not in pattern_performance:
                pattern_performance[pattern] = []
            pattern_performance[pattern].append(record["price_change_pct"])

        pattern_avg_returns = {
            pattern: sum(returns) / len(returns)
            for pattern, returns in pattern_performance.items()
        }

        return {
            "total_signals": total_signals,
            "win_rate": win_rate,
            "avg_return_pct": avg_return,
            "total_return_pct": total_return,
            "volatility": returns_std,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "max_profit_pct": max_profit,
            "avg_holding_time_hours": avg_holding_time,
            "pattern_performance": pattern_avg_returns,
            "time_range": {
                "start": min(p["recorded_at"] for p in performance_data),
                "end": max(p["recorded_at"] for p in performance_data),
            },
        }

    def _get_empty_metrics(self) -> Dict[str, Any]:
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