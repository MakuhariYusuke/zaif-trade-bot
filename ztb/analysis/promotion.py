"""
Promotion Engine for feature status advancement.

This module provides a flexible promotion system that evaluates features
against configurable criteria to determine status advancement.
"""

import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from ztb.types.common import ObjectMap, ObjectRecords
from ztb.utils.config_loader import ConfigLoader


def _as_object_map(value: object) -> ObjectMap:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def _as_object_records(value: object) -> ObjectRecords:
    if not isinstance(value, list):
        return []
    records: ObjectRecords = []
    for item in value:
        if isinstance(item, dict):
            records.append({str(k): v for k, v in item.items()})
    return records


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_metric(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, str):
        try:
            numeric = float(value.strip())
            return numeric if np.isfinite(numeric) else None
        except ValueError:
            return None
    return None


def _compare_values(actual: float, operator: str, expected: float) -> bool:
    if operator == ">":
        return actual > expected
    if operator == ">=":
        return actual >= expected
    if operator == "<":
        return actual < expected
    if operator == "<=":
        return actual <= expected
    if operator == "==":
        return actual == expected
    return False


def _bounded_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    raw = numerator / denominator
    return max(0.0, min(1.0, raw))


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
        self.weight = max(0.0, weight)

    @abstractmethod
    def evaluate(self, feature_results: ObjectMap) -> tuple[bool, float]:
        """
        Evaluate criterion against feature results.

        Returns:
            Tuple of (passed: bool, score: float)
        """


class NumericCriterion(Criterion):
    """Numeric comparison criterion (sharpe_ratio, win_rate, etc.)"""

    def evaluate(self, feature_results: ObjectMap) -> tuple[bool, float]:
        actual_value = _coerce_metric(feature_results.get(self.name))
        if actual_value is None:
            return False, 0.0

        passed = _compare_values(actual_value, self.operator, self.value)
        if passed:
            return True, self.weight

        if self.operator in {">", ">="}:
            ratio = _bounded_ratio(actual_value, self.value)
            return False, self.weight * ratio
        if self.operator in {"<", "<="}:
            ratio = _bounded_ratio(self.value, abs(actual_value))
            return False, self.weight * ratio
        return False, 0.0


class RatioCriterion(NumericCriterion):
    """Ratio-based criterion (sortino_ratio, calmar_ratio, etc.)"""
    # Ratio comparison semantics are identical to numeric criteria.
    def evaluate(self, feature_results: ObjectMap) -> tuple[bool, float]:
        return NumericCriterion.evaluate(self, feature_results)


class DurationCriterion(Criterion):
    """Duration-based criterion (max_drawdown_duration_days, etc.)"""

    def evaluate(self, feature_results: ObjectMap) -> tuple[bool, float]:
        actual_value = _coerce_metric(feature_results.get(self.name))
        if actual_value is None:
            return False, 0.0

        passed = _compare_values(actual_value, self.operator, self.value)
        if passed:
            return True, self.weight

        if self.operator in {"<", "<="}:
            ratio = _bounded_ratio(self.value, max(actual_value, 1.0))
            return False, self.weight * ratio

        ratio = _bounded_ratio(actual_value, max(self.value, 1.0))
        return False, self.weight * ratio


class DistributionCriterion(Criterion):
    """Distribution quality criterion (skew, kurtosis, etc.)"""

    def evaluate(self, feature_results: ObjectMap) -> tuple[bool, float]:
        actual_value = _coerce_metric(feature_results.get(self.name))
        if actual_value is None:
            return False, 0.0

        abs_value = abs(actual_value)
        passed = _compare_values(abs_value, self.operator, self.value)

        if passed:
            return True, self.weight

        if self.operator in {"<", "<="}:
            ratio = _bounded_ratio(self.value, max(abs_value, 0.1))
            return False, self.weight * ratio

        ratio = _bounded_ratio(abs_value, max(self.value, 0.1))
        return False, self.weight * ratio


class PromotionEngine(ABC):
    """Abstract base class for promotion engines"""

    @abstractmethod
    def evaluate_promotion(
        self,
        feature_name: str,
        feature_results: ObjectMap,
        current_status: str,
        category: Optional[str] = None,
    ) -> tuple[PromotionResult, ObjectMap]:
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


class YamlPromotionEngine(PromotionEngine):
    """YAML-based promotion engine"""

    def __init__(self, config_path: str = "config/promotion_criteria.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.plugin_manager = CriterionPluginManager()
        self.criteria_cache: dict[str, tuple[list[Criterion], list[Criterion]]] = {}
        self.notifier = PromotionNotifier(
            _as_object_map(self.config.get("notifications", {}))
        )

    def _load_config(self) -> ObjectMap:
        """Load promotion criteria from YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Promotion criteria config not found: {self.config_path}"
            )

        loaded = ConfigLoader.load(self.config_path)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Promotion criteria config must be a mapping: {self.config_path}"
            )
        return {str(k): v for k, v in loaded.items()}

    def _get_category_config(self, category: Optional[str]) -> ObjectMap:
        """Get configuration for a specific category"""
        categories = _as_object_map(self.config.get("categories", {}))
        if category and category in categories:
            return _as_object_map(categories.get(category))
        return _as_object_map(self.config.get("default", {}))

    def _compile_criteria(
        self, category_config: ObjectMap
    ) -> tuple[list[Criterion], list[Criterion]]:
        """Compile criteria and hard requirements from configuration"""
        cache_key = json.dumps(category_config, sort_keys=True, default=str)
        cached = self.criteria_cache.get(cache_key)
        if cached is not None:
            return cached

        criteria: list[Criterion] = []
        hard_requirements: list[Criterion] = []

        for criterion_config in _as_object_records(category_config.get("criteria", [])):
            try:
                criteria.append(self._create_criterion(criterion_config))
            except (KeyError, ValueError, TypeError):
                continue

        for requirement_config in _as_object_records(
            category_config.get("hard_requirements", [])
        ):
            try:
                hard_requirements.append(self._create_criterion(requirement_config))
            except (KeyError, ValueError, TypeError):
                continue

        result = (criteria, hard_requirements)
        self.criteria_cache[cache_key] = result
        return result

    def _create_criterion(self, criterion_config: ObjectMap) -> Criterion:
        """Create a criterion instance based on configuration"""
        return self.plugin_manager.create_criterion(criterion_config)

    def _build_criterion_detail(
        self,
        criterion: Criterion,
        feature_results: ObjectMap,
        passed: bool,
        score: float,
        detail_type: str,
    ) -> ObjectMap:
        """Build normalized detail payload for criterion evaluation logs."""
        return {
            "name": criterion.name,
            "operator": criterion.operator,
            "expected": criterion.value,
            "actual": feature_results.get(criterion.name),
            "passed": passed,
            "score": score,
            "weight": criterion.weight,
            "type": detail_type,
        }

    def _evaluate_criterion_group(
        self,
        criteria: list[Criterion],
        feature_results: ObjectMap,
        detail_type: str,
    ) -> tuple[float, ObjectRecords, list[str], list[str]]:
        """Evaluate a criterion group and return score/details/pass-fail names."""
        total_score = 0.0
        details: ObjectRecords = []
        passed_names: list[str] = []
        failed_names: list[str] = []

        for criterion in criteria:
            passed, score = criterion.evaluate(feature_results)
            total_score += score
            details.append(
                self._build_criterion_detail(
                    criterion=criterion,
                    feature_results=feature_results,
                    passed=passed,
                    score=score,
                    detail_type=detail_type,
                )
            )

            if passed:
                passed_names.append(criterion.name)
            else:
                failed_names.append(criterion.name)

        return total_score, details, passed_names, failed_names

    def _should_promote(
        self,
        logic: str,
        passed_criteria: list[str],
        failed_criteria: list[str],
        normalized_score: float,
        required_score: float,
    ) -> bool:
        """Determine promotion eligibility according to configured logic."""
        score_met = normalized_score >= required_score
        if logic == "OR":
            return bool(passed_criteria) and score_met
        return (not failed_criteria) and score_met

    def _resolve_status_result(
        self,
        current_status: str,
        should_promote: bool,
        feature_results: ObjectMap,
        staging_config: ObjectMap,
    ) -> PromotionResult:
        """Resolve final PromotionResult from status transition rules."""
        if current_status == "pending":
            return PromotionResult.PROMOTE if should_promote else PromotionResult.KEEP

        if current_status == "staging":
            min_samples = max(
                0, _as_int(staging_config.get("min_samples_required", 1000), 1000)
            )
            current_samples = max(0, _as_int(feature_results.get("sample_count", 0), 0))
            if should_promote and current_samples >= min_samples:
                return PromotionResult.PROMOTE
            if not should_promote:
                return PromotionResult.DEMOTE
            return PromotionResult.KEEP

        if current_status == "verified":
            return PromotionResult.KEEP if should_promote else PromotionResult.DEMOTE

        return PromotionResult.KEEP

    def _build_evaluation_details(
        self,
        *,
        category: Optional[str],
        logic: str,
        required_score: float,
        normalized_score: float,
        total_score: float,
        max_score: float,
        passed_criteria: list[str],
        failed_criteria: list[str],
        criterion_details: ObjectRecords,
        all_hard_requirements_passed: bool,
        hard_requirement_details: ObjectRecords,
        current_status: str,
        feature_results: ObjectMap,
        staging_config: ObjectMap,
    ) -> ObjectMap:
        """Build normalized response payload for promotion decisions."""
        return {
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
            "staging_samples": (
                _as_int(feature_results.get("sample_count", 0), 0)
                if current_status == "staging"
                else None
            ),
            "staging_min_samples": (
                _as_int(staging_config.get("min_samples_required", 1000), 1000)
                if current_status == "staging"
                else None
            ),
        }

    def _notify_promotion_result(
        self,
        feature_name: str,
        current_status: str,
        result: PromotionResult,
        normalized_score: float,
        all_hard_requirements_passed: bool,
        criterion_details: ObjectRecords,
        hard_requirement_details: ObjectRecords,
    ) -> None:
        """Dispatch notifier events for promotion outcomes."""
        if result == PromotionResult.PROMOTE:
            self.notifier.notify_promotion_success(
                feature_name,
                current_status,
                "staging" if current_status == "pending" else "verified",
                normalized_score,
            )
            return

        if result == PromotionResult.DEMOTE or (
            result == PromotionResult.KEEP and not all_hard_requirements_passed
        ):
            failed_criteria_details = [d for d in criterion_details if not d["passed"]]
            hard_req_failures = [d for d in hard_requirement_details if not d["passed"]]
            self.notifier.notify_criterion_failure(
                feature_name, failed_criteria_details, hard_req_failures
            )

    def evaluate_promotion(
        self,
        feature_name: str,
        feature_results: ObjectMap,
        current_status: str,
        category: Optional[str] = None,
    ) -> tuple[PromotionResult, ObjectMap]:
        """Evaluate promotion based on YAML criteria with hard requirements."""
        category_config = self._get_category_config(category)
        criteria, hard_requirements = self._compile_criteria(category_config)

        if not criteria and not hard_requirements:
            return PromotionResult.KEEP, {"error": "No criteria defined"}

        logic = str(category_config.get("logic", "AND")).upper()
        required_score = max(
            0.0, min(1.0, _as_float(category_config.get("required_score", 0.5), 0.5))
        )

        (
            _hard_total_score,
            hard_requirement_details,
            _hard_passed_names,
            hard_failed_names,
        ) = self._evaluate_criterion_group(
            hard_requirements, feature_results, detail_type="hard_requirement"
        )
        all_hard_requirements_passed = not hard_failed_names

        staging_config = _as_object_map(self.config.get("staging", {}))
        hard_requirement_mode = str(staging_config.get("hard_requirement_mode", "strict"))

        if not all_hard_requirements_passed and hard_requirement_mode != "warning":
            return PromotionResult.KEEP, {
                "category": category,
                "logic": logic,
                "required_score": required_score,
                "hard_requirements_passed": False,
                "hard_requirement_details": hard_requirement_details,
                "reason": "Hard requirements not met",
            }

        max_score = sum(c.weight for c in criteria)
        total_score, criterion_details, passed_criteria, failed_criteria = (
            self._evaluate_criterion_group(
                criteria, feature_results, detail_type="criterion"
            )
        )

        normalized_score = total_score / max_score if max_score > 0 else 0.0
        should_promote = self._should_promote(
            logic=logic,
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            normalized_score=normalized_score,
            required_score=required_score,
        )
        result = self._resolve_status_result(
            current_status=current_status,
            should_promote=should_promote,
            feature_results=feature_results,
            staging_config=staging_config,
        )
        details = self._build_evaluation_details(
            category=category,
            logic=logic,
            required_score=required_score,
            normalized_score=normalized_score,
            total_score=total_score,
            max_score=max_score,
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            criterion_details=criterion_details,
            all_hard_requirements_passed=all_hard_requirements_passed,
            hard_requirement_details=hard_requirement_details,
            current_status=current_status,
            feature_results=feature_results,
            staging_config=staging_config,
        )
        self._notify_promotion_result(
            feature_name=feature_name,
            current_status=current_status,
            result=result,
            normalized_score=normalized_score,
            all_hard_requirements_passed=all_hard_requirements_passed,
            criterion_details=criterion_details,
            hard_requirement_details=hard_requirement_details,
        )

        return result, details


class PromotionNotifier:
    """Handles notifications for promotion events"""

    def __init__(self, notification_config: ObjectMap):
        self.config = notification_config
        self.enabled = _as_bool(notification_config.get("enabled", False), False)
        self.webhook_config = self._load_webhook_config()

    def _load_webhook_config(self) -> ObjectMap:
        """Load webhook configuration from config/notifier.yaml"""
        config_path = Path("config/notifier.yaml")
        if not config_path.exists():
            return {}

        try:
            loaded = ConfigLoader.load(config_path)
            return _as_object_map(loaded)
        except Exception as e:
            print(f"Failed to load webhook config: {e}")
            return {}

    def notify_criterion_failure(
        self,
        feature: str,
        failed_criteria: ObjectRecords,
        hard_requirement_failures: ObjectRecords,
    ) -> None:
        """Notify about criterion failures"""
        if not self.enabled:
            return

        message = f"🚨 Feature '{feature}' failed promotion criteria:\n"

        if hard_requirement_failures:
            message += "\n**Hard Requirements Failed:**\n"
            for failure in hard_requirement_failures:
                message += (
                    f"• {failure['name']}: {failure['actual']} "
                    f"{failure['operator']} {failure['expected']}\n"
                )

        if failed_criteria:
            message += "\n**Criteria Failed:**\n"
            for failure in failed_criteria:
                message += (
                    f"• {failure['name']}: {failure['actual']} "
                    f"{failure['operator']} {failure['expected']}\n"
                )

        self._send_notification(message, priority="high")

        issues: list[str] = []
        if hard_requirement_failures:
            issues.extend(
                [
                    f"{f['name']}: {f['actual']} {f['operator']} {f['expected']}"
                    for f in hard_requirement_failures
                ]
            )
        if failed_criteria:
            issues.extend(
                [
                    f"{f['name']}: {f['actual']} {f['operator']} {f['expected']}"
                    for f in failed_criteria
                ]
            )

        self._send_webhook_notification(
            "validation_failed",
            {
                "feature_name": feature,
                "current_status": "unknown",
                "category": "unknown",
                "issues": ", ".join(issues),
                "score": 0.0,
            },
        )

    def notify_promotion_success(
        self, feature: str, from_status: str, to_status: str, score: float
    ) -> None:
        """Notify about successful promotion"""
        if not self.enabled:
            return

        message = (
            f"✅ Feature '{feature}' promoted: {from_status} → "
            f"{to_status} (Score: {score:.2f})"
        )
        self._send_notification(message, priority="normal")

        self._send_webhook_notification(
            "feature_promoted",
            {
                "feature_name": feature,
                "from_status": from_status,
                "to_status": to_status,
                "category": "unknown",
                "score": score,
                "reason": f"Promotion score: {score:.2f}",
            },
        )

    def _send_notification(self, message: str, priority: str) -> None:
        """Send notification via configured channels"""
        print(f"[{priority.upper()}] {message}")

    def _send_webhook_notification(
        self, event_type: str, event_data: ObjectMap
    ) -> None:
        """Send webhook notification for specific event"""
        if not self.webhook_config:
            return

        events_config = _as_object_map(self.webhook_config.get("events", {}))
        event_config = _as_object_map(events_config.get(event_type, {}))
        if not _as_bool(event_config.get("enabled", False), False):
            return

        template = str(event_config.get("template", ""))
        message = self._format_message(template, event_data)

        channels = _as_object_map(self.webhook_config.get("channels", {}))
        for channel_name, channel_config in channels.items():
            channel_map = _as_object_map(channel_config)
            webhook_url = str(channel_map.get("webhook_url", ""))
            if _as_bool(channel_map.get("enabled", False), False) and webhook_url:
                self._send_to_webhook(channel_name, channel_map, message)

    def _format_message(self, template: str, data: ObjectMap) -> str:
        """Format message using template and data"""
        if not template:
            return str(data)

        format_data = {str(k): v for k, v in data.items()}
        try:
            return template.format(**format_data)
        except KeyError as e:
            return f"Notification formatting error: {e}\\nData: {format_data}"

    def _send_to_webhook(
        self, channel_name: str, channel_config: ObjectMap, message: str
    ) -> None:
        """Send message to specific webhook channel with retry logic"""
        webhook_url = str(channel_config.get("webhook_url", ""))
        if not webhook_url:
            return

        retry_config = _as_object_map(self.webhook_config.get("retry", {}))
        max_attempts = max(1, _as_int(retry_config.get("max_attempts", 3), 3))
        backoff_seconds = max(0.0, _as_float(retry_config.get("backoff_seconds", 1), 1.0))

        for attempt in range(max_attempts):
            try:
                if channel_name == "slack":
                    payload: ObjectMap = {
                        "text": message,
                        "username": channel_config.get("username", "Feature Bot"),
                        "icon_emoji": channel_config.get(
                            "icon_emoji", ":robot_face:"
                        ),
                    }
                elif channel_name == "discord":
                    payload = {
                        "content": message,
                        "username": channel_config.get("username", "Feature Bot"),
                    }
                else:
                    print(f"Unknown channel type: {channel_name}")
                    return

                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                return

            except requests.RequestException as e:
                if attempt < max_attempts - 1:
                    time.sleep(backoff_seconds * (2**attempt))
                else:
                    print(
                        f"Failed to send {channel_name} webhook after "
                        f"{max_attempts} attempts: {e}"
                    )


class CriterionPluginManager:
    """Manages pluggable criterion implementations"""

    def __init__(self) -> None:
        self.criterion_types: dict[str, type[Criterion]] = {
            "numeric": NumericCriterion,
            "ratio": RatioCriterion,
            "duration": DurationCriterion,
            "distribution": DistributionCriterion,
        }
        self.custom_criterion_types: dict[str, type[Criterion]] = {}

    def register_criterion_type(
        self, name: str, criterion_class: type[Criterion]
    ) -> None:
        """Register a custom criterion type"""
        if not issubclass(criterion_class, Criterion):
            raise ValueError(
                f"Custom criterion class must inherit from Criterion: {criterion_class}"
            )
        self.custom_criterion_types[name] = criterion_class

    def create_criterion(self, criterion_config: ObjectMap) -> Criterion:
        """Create a criterion instance, supporting custom types"""
        name = str(criterion_config["name"])
        operator = str(criterion_config["operator"])
        value = _as_float(criterion_config["value"], 0.0)
        weight = _as_float(criterion_config.get("weight", 1.0), 1.0)
        criterion_type = str(criterion_config.get("type", "numeric"))

        criterion_class = self.custom_criterion_types.get(
            criterion_type
        ) or self.criterion_types.get(criterion_type)

        if criterion_class is None:
            raise ValueError(f"Unknown criterion type: {criterion_type}")

        return criterion_class(name, operator, value, weight)

    def get_available_types(self) -> list[str]:
        """Get list of all available criterion types"""
        return list(self.criterion_types.keys()) + list(
            self.custom_criterion_types.keys()
        )


def create_promotion_engine(
    config_path: str = "config/promotion_criteria.yaml",
) -> PromotionEngine:
    """Factory function to create promotion engine"""
    return YamlPromotionEngine(config_path)
