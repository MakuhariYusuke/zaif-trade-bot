"""
Promotion Engine for feature status advancement.

This module provides a flexible promotion system that evaluates features
against configurable criteria to determine status advancement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml
from enum import Enum
import json
from datetime import datetime
import numpy as np
from .status import CoverageValidator
import requests
import os
import time


class PromotionResult(Enum):
    """Result of promotion evaluation"""
    PROMOTE = "promote"
    KEEP = "keep"
    DEMOTE = "demote"


class Criterion(ABC):
    """Abstract base class for promotion criteria"""

    def __init__(self, name: str, operator: str, value: float, weight: float):
        self.name = name
        self.operator = operator
        self.value = value
        self.weight = weight

    @abstractmethod
    def evaluate(self, feature_results: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate criterion against feature results.

        Returns:
            Tuple of (passed: bool, score: float)
        """
        pass


class NumericCriterion(Criterion):
    """Numeric comparison criterion (sharpe_ratio, win_rate, etc.)"""

    def evaluate(self, feature_results: Dict[str, Any]) -> Tuple[bool, float]:
        actual_value = feature_results.get(self.name)
        if actual_value is None:
            return False, 0.0

        # Apply operator
        if self.operator == ">":
            passed = actual_value > self.value
        elif self.operator == ">=":
            passed = actual_value >= self.value
        elif self.operator == "<":
            passed = actual_value < self.value
        elif self.operator == "<=":
            passed = actual_value <= self.value
        elif self.operator == "==":
            passed = actual_value == self.value
        else:
            return False, 0.0

        # Calculate score contribution
        if passed:
            # Perfect score for passing
            return True, self.weight
        else:
            # Partial score based on how close we are
            if self.operator in [">", ">="]:
                # For "greater than" criteria, score based on ratio
                ratio = min(1.0, actual_value / self.value) if self.value != 0 else 0.0
                return False, self.weight * ratio
            elif self.operator in ["<", "<="]:
                # For "less than" criteria (like drawdown), score based on inverse ratio
                ratio = min(1.0, self.value / abs(actual_value)) if actual_value != 0 else 0.0
                return False, self.weight * ratio
            else:
                return False, 0.0


class RatioCriterion(Criterion):
    """Ratio-based criterion (sortino_ratio, calmar_ratio, etc.)"""

    def evaluate(self, feature_results: Dict[str, Any]) -> Tuple[bool, float]:
        actual_value = feature_results.get(self.name)
        if actual_value is None:
            return False, 0.0

        # For ratios, we want higher values to be better
        if self.operator == ">":
            passed = actual_value > self.value
        elif self.operator == ">=":
            passed = actual_value >= self.value
        elif self.operator == "<":
            passed = actual_value < self.value
        elif self.operator == "<=":
            passed = actual_value <= self.value
        else:
            return False, 0.0

        # Calculate score contribution
        if passed:
            return True, self.weight
        else:
            # For ratios, partial score based on how close we are
            if self.operator in [">", ">="]:
                ratio = min(1.0, actual_value / self.value) if self.value != 0 else 0.0
                return False, self.weight * ratio
            else:
                ratio = min(1.0, self.value / abs(actual_value)) if actual_value != 0 else 0.0
                return False, self.weight * ratio


class DurationCriterion(Criterion):
    """Duration-based criterion (max_drawdown_duration_days, etc.)"""

    def evaluate(self, feature_results: Dict[str, Any]) -> Tuple[bool, float]:
        actual_value = feature_results.get(self.name)
        if actual_value is None:
            return False, 0.0

        # For durations, lower values are generally better
        if self.operator == ">":
            passed = actual_value > self.value
        elif self.operator == ">=":
            passed = actual_value >= self.value
        elif self.operator == "<":
            passed = actual_value < self.value
        elif self.operator == "<=":
            passed = actual_value <= self.value
        else:
            return False, 0.0

        # Calculate score contribution
        if passed:
            return True, self.weight
        else:
            # For durations, partial score based on inverse ratio
            if self.operator in ["<", "<="]:
                ratio = min(1.0, self.value / max(actual_value, 1))  # Avoid division by zero
                return False, self.weight * ratio
            else:
                ratio = min(1.0, actual_value / max(self.value, 1)) if self.value != 0 else 0.0
                return False, self.weight * ratio


class DistributionCriterion(Criterion):
    """Distribution quality criterion (skew, kurtosis, etc.)"""

    def evaluate(self, feature_results: Dict[str, Any]) -> Tuple[bool, float]:
        actual_value = feature_results.get(self.name)
        if actual_value is None:
            return False, 0.0

        # For distribution metrics, lower absolute values are generally better
        # (prefer normal-like distributions)
        abs_value = abs(actual_value)

        if self.operator == "<":
            passed = abs_value < self.value
        elif self.operator == "<=":
            passed = abs_value <= self.value
        elif self.operator == ">":
            passed = abs_value > self.value
        elif self.operator == ">=":
            passed = abs_value >= self.value
        else:
            return False, 0.0

        # Calculate score contribution
        if passed:
            return True, self.weight
        else:
            # For distribution metrics, partial score based on how close to ideal (0)
            if self.operator in ["<", "<="]:
                # Lower values are better, score based on inverse ratio
                ratio = min(1.0, self.value / max(abs_value, 0.1))  # Avoid division by zero
                return False, self.weight * ratio
            else:
                # Higher values are better (less common for distribution metrics)
                ratio = min(1.0, abs_value / max(self.value, 0.1))
                return False, self.weight * ratio


class PromotionEngine(ABC):
    """Abstract base class for promotion engines"""

    @abstractmethod
    def evaluate_promotion(self, feature_name: str, feature_results: Dict[str, Any],
                          current_status: str, category: Optional[str] = None) -> Tuple[PromotionResult, Dict[str, Any]]:
        """
        Evaluate if a feature should be promoted, kept, or demoted.

        Args:
            feature_name: Name of the feature
            feature_results: Evaluation results for the feature
            current_status: Current status ('pending', 'staging', 'verified')
            category: Feature category (optional)

        Returns:
            Tuple of (result, details_dict)
        """
        pass


class YamlPromotionEngine(PromotionEngine):
    """YAML-based promotion engine"""

    def __init__(self, config_path: str = "config/promotion_criteria.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.criteria_cache = {}  # Cache compiled criteria
        self.notifier = PromotionNotifier(self.config.get('notifications', {}))

    def _load_config(self) -> Dict[str, Any]:
        """Load promotion criteria from YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Promotion criteria config not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_category_config(self, category: Optional[str]) -> Dict[str, Any]:
        """Get configuration for a specific category"""
        if category and category in self.config.get('categories', {}):
            return self.config['categories'][category]
        return self.config.get('default', {})

    def _compile_criteria(self, category_config: Dict[str, Any]) -> Tuple[List[Criterion], List[Criterion]]:
        """Compile criteria and hard requirements from configuration"""
        cache_key = json.dumps(category_config, sort_keys=True)
        if cache_key in self.criteria_cache:
            return self.criteria_cache[cache_key]

        criteria = []
        hard_requirements = []

        # Compile regular criteria
        for criterion_config in category_config.get('criteria', []):
            criterion = self._create_criterion(criterion_config)
            criteria.append(criterion)

        # Compile hard requirements
        for requirement_config in category_config.get('hard_requirements', []):
            requirement = self._create_criterion(requirement_config)
            hard_requirements.append(requirement)

        result = (criteria, hard_requirements)
        self.criteria_cache[cache_key] = result
        return result

    def _create_criterion(self, criterion_config: Dict[str, Any]) -> Criterion:
        """Create a criterion instance based on configuration"""
        name = criterion_config['name']
        operator = criterion_config['operator']
        value = criterion_config['value']
        weight = criterion_config.get('weight', 1.0)
        criterion_type = criterion_config.get('type', 'numeric')

        if criterion_type == 'ratio':
            return RatioCriterion(name, operator, value, weight)
        elif criterion_type == 'duration':
            return DurationCriterion(name, operator, value, weight)
        elif criterion_type == 'distribution':
            return DistributionCriterion(name, operator, value, weight)
        else:
            return NumericCriterion(name, operator, value, weight)

    def _optimize_required_score(self, category: str, historical_results: List[Dict[str, Any]]) -> float:
        """Optimize required score based on historical promotion success patterns"""
        if not historical_results:
            return self.config.get('categories', {}).get(category, {}).get('required_score', 0.7)

        # Analyze successful vs failed promotions
        successful_scores = []
        failed_scores = []

        for result in historical_results:
            if result.get('promotion_result') == 'promote':
                successful_scores.append(result.get('achieved_score', 0))
            else:
                failed_scores.append(result.get('achieved_score', 0))

        if not successful_scores:
            return self.config.get('categories', {}).get(category, {}).get('required_score', 0.7)

        # Find optimal threshold using percentile analysis
        # Use 25th percentile of successful scores as minimum threshold
        optimal_threshold = float(np.percentile(successful_scores, 25))

        # Ensure threshold is reasonable (between 0.5 and 0.9)
        optimal_threshold = max(0.5, min(0.9, optimal_threshold))

        return round(optimal_threshold, 2)

    def evaluate_promotion(self, feature_name: str, feature_results: Dict[str, Any],
                          current_status: str, category: Optional[str] = None) -> Tuple[PromotionResult, Dict[str, Any]]:
        """
        Evaluate promotion based on YAML criteria with hard requirements
        """
        category_config = self._get_category_config(category)
        criteria, hard_requirements = self._compile_criteria(category_config)

        if not criteria and not hard_requirements:
            return PromotionResult.KEEP, {"error": "No criteria defined"}

        logic = category_config.get('logic', 'AND')
        required_score = category_config.get('required_score', 0.5)

        # Evaluate hard requirements first (all must pass)
        hard_requirement_details = []
        all_hard_requirements_passed = True

        for requirement in hard_requirements:
            passed, score = requirement.evaluate(feature_results)
            detail = {
                "name": requirement.name,
                "operator": requirement.operator,
                "expected": requirement.value,
                "actual": feature_results.get(requirement.name),
                "passed": passed,
                "type": "hard_requirement"
            }
            hard_requirement_details.append(detail)

            if not passed:
                all_hard_requirements_passed = False

        # If hard requirements fail, handle based on mode
        hard_requirement_mode = self.config.get('staging', {}).get('hard_requirement_mode', 'strict')

        if not all_hard_requirements_passed:
            if hard_requirement_mode == 'warning':
                # Warning mode: Allow promotion but record warning
                # Continue with regular criteria evaluation but mark as warning
                pass
            else:
                # Strict mode: Cannot promote
                return PromotionResult.KEEP, {
                    "category": category,
                    "logic": logic,
                    "required_score": required_score,
                    "hard_requirements_passed": False,
                    "hard_requirement_details": hard_requirement_details,
                    "reason": "Hard requirements not met"
                }

        # Evaluate regular criteria
        total_score = 0.0
        max_score = sum(c.weight for c in criteria)
        passed_criteria = []
        failed_criteria = []
        criterion_details = []

        for criterion in criteria:
            passed, score = criterion.evaluate(feature_results)
            total_score += score

            detail = {
                "name": criterion.name,
                "operator": criterion.operator,
                "expected": criterion.value,
                "actual": feature_results.get(criterion.name),
                "passed": passed,
                "score": score,
                "weight": criterion.weight,
                "type": "criterion"
            }
            criterion_details.append(detail)

            if passed:
                passed_criteria.append(criterion.name)
            else:
                failed_criteria.append(criterion.name)

        # Calculate normalized score
        normalized_score = total_score / max_score if max_score > 0 else 0.0

        # Determine result based on logic and required score
        if logic == "AND":
            all_passed = len(failed_criteria) == 0
            score_met = normalized_score >= required_score
            should_promote = all_passed and score_met
        elif logic == "OR":
            any_passed = len(passed_criteria) > 0
            score_met = normalized_score >= required_score
            should_promote = any_passed and score_met
        else:
            # Default to AND logic
            should_promote = len(failed_criteria) == 0 and normalized_score >= required_score

        # Determine promotion result based on current status
        if current_status == "pending":
            if should_promote:
                result = PromotionResult.PROMOTE  # pending -> staging
            else:
                result = PromotionResult.KEEP
        elif current_status == "staging":
            # Check staging requirements
            staging_config = self.config.get('staging', {})
            min_samples = staging_config.get('min_samples_required', 1000)
            current_samples = feature_results.get('sample_count', 0)

            if should_promote and current_samples >= min_samples:
                result = PromotionResult.PROMOTE  # staging -> verified
            elif not should_promote:
                result = PromotionResult.DEMOTE  # staging -> pending
            else:
                result = PromotionResult.KEEP
        elif current_status == "verified":
            # Check for demotion
            demotion_mode = self.config.get('staging', {}).get('demotion_mode', 'direct')
            if not should_promote:
                if demotion_mode == 'graceful':
                    result = PromotionResult.DEMOTE  # verified -> staging (再テスト)
                else:
                    result = PromotionResult.DEMOTE  # verified -> pending or failed (直接降格)
            else:
                result = PromotionResult.KEEP
        else:
            result = PromotionResult.KEEP

        details = {
            "category": category,
            "logic": logic,
            "required_score": required_score,
            "achieved_score": normalized_score,
            "total_score": total_score,
            "max_score": max_score,
            "passed_criteria": passed_criteria,
            "failed_criteria": failed_criteria,
            "criterion_details": criterion_details,
            "hard_requirements_passed": all_hard_requirements_passed,
            "hard_requirement_details": hard_requirement_details,
            "staging_samples": feature_results.get('sample_count', 0) if current_status == 'staging' else None,
            "staging_min_samples": self.config.get('staging', {}).get('min_samples_required', 1000) if current_status == 'staging' else None
        }

        # Send notifications based on result
        if result == PromotionResult.PROMOTE:
            self.notifier.notify_promotion_success(feature_name, current_status,
                                                 "staging" if current_status == "pending" else "verified",
                                                 normalized_score)
        elif result == PromotionResult.DEMOTE or (result == PromotionResult.KEEP and not all_hard_requirements_passed):
            failed_criteria_details = [d for d in criterion_details if not d["passed"]]
            hard_req_failures = [d for d in hard_requirement_details if not d["passed"]]
            self.notifier.notify_criterion_failure(feature_name, failed_criteria_details, hard_req_failures)

        return result, details


class PromotionNotifier:
    """Handles notifications for promotion events"""

    def __init__(self, notification_config: Dict[str, Any]):
        self.config = notification_config
        self.enabled = notification_config.get('enabled', False)
        # Load webhook configuration
        self.webhook_config = self._load_webhook_config()

    def _load_webhook_config(self) -> Dict[str, Any]:
        """Load webhook configuration from config/notifier.yaml"""
        config_path = Path("config/notifier.yaml")
        if not config_path.exists():
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load webhook config: {e}")
            return {}

    def notify_criterion_failure(self, feature: str, failed_criteria: List[Dict[str, Any]],
                                hard_requirement_failures: List[Dict[str, Any]]) -> None:
        """Notify about criterion failures"""
        if not self.enabled:
            return

        message = f"🚨 Feature '{feature}' failed promotion criteria:\n"

        if hard_requirement_failures:
            message += "\n**Hard Requirements Failed:**\n"
            for failure in hard_requirement_failures:
                message += f"• {failure['name']}: {failure['actual']} {failure['operator']} {failure['expected']}\n"

        if failed_criteria:
            message += "\n**Criteria Failed:**\n"
            for failure in failed_criteria:
                message += f"• {failure['name']}: {failure['actual']} {failure['operator']} {failure['expected']}\n"

        self._send_notification(message, priority="high")

        # Send webhook notification for validation failure
        issues = []
        if hard_requirement_failures:
            issues.extend([f"{f['name']}: {f['actual']} {f['operator']} {f['expected']}" for f in hard_requirement_failures])
        if failed_criteria:
            issues.extend([f"{f['name']}: {f['actual']} {f['operator']} {f['expected']}" for f in failed_criteria])

        self._send_webhook_notification('validation_failed', {
            'feature_name': feature,
            'current_status': 'unknown',  # We don't have this info here
            'category': 'unknown',  # We don't have this info here
            'issues': ', '.join(issues),
            'score': 0.0  # We don't have this info here
        })

    def notify_promotion_success(self, feature: str, from_status: str, to_status: str,
                                score: float) -> None:
        """Notify about successful promotion"""
        if not self.enabled:
            return

        message = f"✅ Feature '{feature}' promoted: {from_status} → {to_status} (Score: {score:.2f})"
        self._send_notification(message, priority="normal")

        # Send webhook notification for promotion
        self._send_webhook_notification('feature_promoted', {
            'feature_name': feature,
            'from_status': from_status,
            'to_status': to_status,
            'category': 'unknown',  # We don't have this info here
            'score': score,
            'reason': f'Promotion score: {score:.2f}'
        })

    def _send_notification(self, message: str, priority: str) -> None:
        """Send notification via configured channels"""
        # Implementation would depend on notification service (Slack, Discord, etc.)
        # For now, just print to console
        print(f"[{priority.upper()}] {message}")

    def _send_webhook_notification(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Send webhook notification for specific event"""
        if not self.webhook_config:
            return

        event_config = self.webhook_config.get('events', {}).get(event_type, {})
        if not event_config.get('enabled', False):
            return

        message = self._format_message(event_config.get('template', ''), event_data)

        # Send to all enabled channels
        channels = self.webhook_config.get('channels', {})
        for channel_name, channel_config in channels.items():
            if channel_config.get('enabled', False) and channel_config.get('webhook_url'):
                self._send_to_webhook(channel_name, channel_config, message)

    def _format_message(self, template: str, data: Dict[str, Any]) -> str:
        """Format message using template and data"""
        if not template:
            return str(data)

        try:
            return template.format(**data)
        except KeyError as e:
            # If formatting fails, return template with available data
            return f"Notification formatting error: {e}\nData: {data}"

    def _send_to_webhook(self, channel_name: str, channel_config: Dict[str, Any], message: str) -> None:
        """Send message to specific webhook channel with retry logic"""
        webhook_url = channel_config['webhook_url']
        retry_config = self.webhook_config.get('retry', {})
        max_attempts = retry_config.get('max_attempts', 3)
        backoff_seconds = retry_config.get('backoff_seconds', 1)

        for attempt in range(max_attempts):
            try:
                if channel_name == 'slack':
                    payload = {
                        'text': message,
                        'username': channel_config.get('username', 'Feature Bot'),
                        'icon_emoji': channel_config.get('icon_emoji', ':robot_face:')
                    }
                elif channel_name == 'discord':
                    payload = {
                        'content': message,
                        'username': channel_config.get('username', 'Feature Bot')
                    }
                else:
                    print(f"Unknown channel type: {channel_name}")
                    return

                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                return  # Success

            except requests.RequestException as e:
                if attempt < max_attempts - 1:
                    time.sleep(backoff_seconds * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"Failed to send {channel_name} webhook after {max_attempts} attempts: {e}")


class AdaptiveThresholdManager:
    """Manages adaptive thresholds based on historical market data"""

    def __init__(self, historical_data_path: str):
        self.historical_data_path = Path(historical_data_path)
        self.thresholds_cache = {}
        self._load_historical_data()

    def _load_historical_data(self) -> None:
        """Load historical evaluation results"""
        if not self.historical_data_path.exists():
            return

        # Load historical coverage data
        coverage_data = CoverageValidator.load_coverage_files(str(self.historical_data_path))

        # Extract successful feature metrics
        successful_features = {}
        for event in coverage_data.get("events", []):
            if event.get("type") == "feature_promoted" and event.get("to_status") == "verified":
                feature = event.get("feature")
                if "details" in event and "criterion_details" in event["details"]:
                    successful_features[feature] = event["details"]

        self.historical_successes = successful_features

    def get_adaptive_threshold(self, metric_name: str, percentile: float = 20.0) -> float:
        """Get adaptive threshold based on historical successful features"""
        if metric_name not in self.thresholds_cache:
            if not self.historical_successes:
                # Fallback to default values
                defaults = {
                    "sharpe_ratio": 0.3,
                    "max_drawdown": -0.2,
                    "win_rate": 0.5
                }
                self.thresholds_cache[metric_name] = defaults.get(metric_name, 0.0)
            else:
                # Calculate percentile from successful features
                values = []
                for feature_data in self.historical_successes.values():
                    for criterion in feature_data.get("criterion_details", []):
                        if criterion.get("name") == metric_name and criterion.get("passed"):
                            values.append(criterion.get("actual", 0))

                if values:
                    threshold = float(np.percentile(values, percentile))
                    self.thresholds_cache[metric_name] = threshold
                else:
                    self.thresholds_cache[metric_name] = 0.0

        return self.thresholds_cache[metric_name]

    def get_adaptive_gates(self) -> Dict[str, float]:
        """Get adaptive quality gates based on historical data"""
        gates = {
            'nan_rate_threshold': 0.8,
            'correlation_threshold': 0.05,
            'skew_threshold': 3.0,
            'kurtosis_threshold': 8.0
        }

        # Try to adapt based on historical harmful features
        # Analyze failed features to adjust thresholds
        failed_features = {}
        coverage_data = CoverageValidator.load_coverage_files(str(self.historical_data_path))
        for event in coverage_data.get("events", []):
            # Exclude synthetic data events from learning
            if event.get("details", {}).get("dataset") == "synthetic":
                continue
            if event.get("type") == "feature_tested" and event.get("to_status") == "discarded":
                feature = event.get("feature")
                if "details" in event and "quality_results" in event["details"]:
                    failed_features[feature] = event["details"]["quality_results"]

        # Require minimum events for adaptation
        if len(failed_features) < 10:
            return gates  # Return default if insufficient data

        if failed_features:
            # Adjust thresholds based on failed features
            nan_rates = [f.get('nan_rate', 0) for f in failed_features.values() if f.get('nan_rate', 0) > 0]
            if nan_rates:
                # Set threshold slightly above the maximum failed NaN rate
                gates['nan_rate_threshold'] = min(0.9, max(nan_rates) + 0.05)

            correlations = [f.get('correlation', 0) for f in failed_features.values() if f.get('correlation') is not None]
            if correlations:
                # Set threshold slightly above the maximum failed correlation
                gates['correlation_threshold'] = min(0.1, max(correlations) + 0.01)

            skews = [abs(f.get('skew', 0)) for f in failed_features.values() if f.get('skew') is not None]
            if skews:
                # Set threshold slightly above the maximum failed skew, with bounds
                adaptive_skew = max(skews) + 0.5
                gates['skew_threshold'] = max(1.5, min(5.0, adaptive_skew))

            kurtoses = [abs(f.get('kurtosis', 0)) for f in failed_features.values() if f.get('kurtosis') is not None]
            if kurtoses:
                # Set threshold slightly above the maximum failed kurtosis, with bounds
                adaptive_kurtosis = max(kurtoses) + 1.0
                gates['kurtosis_threshold'] = max(3.0, min(12.0, adaptive_kurtosis))

        return gates

    def update_thresholds(self, new_evaluation_results: Dict[str, Any]) -> None:
        """Update thresholds with new evaluation results"""
        # This would be called after each evaluation cycle
        # Implementation depends on how we want to update the adaptive thresholds
        pass


class CriterionPluginManager:
    """Manages pluggable criterion implementations"""

    def __init__(self):
        self.criterion_types = {
            'numeric': NumericCriterion,
            'ratio': RatioCriterion,
            'duration': DurationCriterion,
            'distribution': DistributionCriterion
        }
        self.custom_criterion_types = {}

    def register_criterion_type(self, name: str, criterion_class: type) -> None:
        """Register a custom criterion type"""
        if not issubclass(criterion_class, Criterion):
            raise ValueError(f"Custom criterion class must inherit from Criterion: {criterion_class}")
        self.custom_criterion_types[name] = criterion_class

    def create_criterion(self, criterion_config: Dict[str, Any]) -> Criterion:
        """Create a criterion instance, supporting custom types"""
        name = criterion_config['name']
        operator = criterion_config['operator']
        value = criterion_config['value']
        weight = criterion_config.get('weight', 1.0)
        criterion_type = criterion_config.get('type', 'numeric')

        # Check custom types first, then built-in types
        criterion_class = self.custom_criterion_types.get(criterion_type) or self.criterion_types.get(criterion_type)

        if criterion_class is None:
            raise ValueError(f"Unknown criterion type: {criterion_type}")

        return criterion_class(name, operator, value, weight)

    def get_available_types(self) -> List[str]:
        """Get list of all available criterion types"""
        return list(self.criterion_types.keys()) + list(self.custom_criterion_types.keys())


def create_promotion_engine(config_path: str = "config/promotion_criteria.yaml") -> PromotionEngine:
    """Factory function to create promotion engine"""
    return YamlPromotionEngine(config_path)