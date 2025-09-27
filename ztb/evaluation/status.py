"""
Feature status and reason enums for consistent validation.
"""
from enum import Enum
from ztb.utils.core.stats import count_features_by_category
from typing import Dict, Any, Optional, List, Union, cast, Tuple
from pathlib import Path
import json
from datetime import datetime


class FeatureStatus(Enum):
    """Feature validation status enum"""
    VERIFIED = "verified"
    STAGING = "staging"
    PENDING = "pending"
    PENDING_DUE_TO_GATE_FAIL = "pending_due_to_gate_fail"
    UNVERIFIED = "unverified"
    FAILED = "failed"


class FeatureReason(Enum):
    """Feature validation reason enum"""
    # Pending reasons
    INSUFFICIENT_DATA = "insufficient_data"
    HIGH_NAN_RATE = "high_nan_rate"
    ALIGNMENT_MISMATCH = "alignment_mismatch"

    # Unverified reasons
    NOT_TESTED = "not_tested"

    # Failed reasons
    COMPUTATION_ERROR = "computation_error"
    TYPE_MISMATCH = "type_mismatch"
    INVALID_RESULT = "invalid_result"


# Status to allowed reasons mapping
STATUS_REASONS: Dict[FeatureStatus, set] = {
    FeatureStatus.PENDING: {
        FeatureReason.INSUFFICIENT_DATA,
        FeatureReason.HIGH_NAN_RATE,
        FeatureReason.ALIGNMENT_MISMATCH
    },
    FeatureStatus.PENDING_DUE_TO_GATE_FAIL: {
        FeatureReason.INSUFFICIENT_DATA,
        FeatureReason.HIGH_NAN_RATE,
        FeatureReason.ALIGNMENT_MISMATCH
    },
    FeatureStatus.STAGING: set(),  # Staging features don't need reasons
    FeatureStatus.UNVERIFIED: {
        FeatureReason.NOT_TESTED
    },
    FeatureStatus.FAILED: {
        FeatureReason.COMPUTATION_ERROR,
        FeatureReason.TYPE_MISMATCH,
        FeatureReason.INVALID_RESULT
    },
    FeatureStatus.VERIFIED: set()  # Verified features don't need reasons
}


class CoverageValidator:
    """Validates coverage.json structure and consistency"""

    @staticmethod
    def load_coverage_files(base_path: str = "ztb/coverage") -> Dict[str, Any]:
        """Load and merge multiple coverage files"""
        from pathlib import Path
        import glob

        base_path_obj = Path(base_path)
        merged_coverage = {
            "events": [],  # Event sourcing
            "current_state": {  # Current snapshot
                "verified": [],
                "staging": [],
                "pending": [],
                "pending_due_to_gate_fail": [],
                "failed": [],
                "unverified": [],
                "discarded": [],
                "quality_gates": {
                    "discarded_features": []
                }
            },
            "metadata": {
                "last_updated": None,
                "total_verified": 0,
                "total_staging": 0,
                "total_pending": 0,
                "total_pending_due_to_gate_fail": 0,
                "total_failed": 0,
                "total_unverified": 0,
                "total_discarded": 0,
                "source_files": []
            }
        }

        # Load main coverage.json
        main_file = base_path_obj / "coverage.json"
        if main_file.exists():
            with open(main_file, 'r', encoding='utf-8') as f:
                main_data = json.load(f)
                CoverageValidator._merge_coverage_data(merged_coverage, main_data, str(main_file))

        # Load yearly coverage files (coverage_2024.json, etc.)
        yearly_pattern = str(base_path_obj / "coverage_*.json")
        yearly_files = glob.glob(yearly_pattern)

        for file_path in sorted(yearly_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                year_data = json.load(f)
                CoverageValidator._merge_coverage_data(merged_coverage, year_data, file_path)

        # Update totals
        current_state = cast(Dict[str, List[str]], merged_coverage["current_state"])
        merged_coverage["metadata"]["total_verified"] = len(current_state["verified"])  # type: ignore
        merged_coverage["metadata"]["total_staging"] = len(current_state["staging"])  # type: ignore
        merged_coverage["metadata"]["total_pending"] = len(current_state["pending"])  # type: ignore
        merged_coverage["metadata"]["total_pending_due_to_gate_fail"] = len(current_state["pending_due_to_gate_fail"])  # type: ignore
        merged_coverage["metadata"]["total_failed"] = len(current_state["failed"])  # type: ignore
        merged_coverage["metadata"]["total_unverified"] = len(current_state["unverified"])  # type: ignore

        return merged_coverage

    @staticmethod
    def _merge_coverage_data(target: Dict[str, Any], source: Dict[str, Any], source_file: str) -> None:
        """Merge source coverage data into target with strictness priority"""
        # Handle event sourcing structure
        if "events" in source:
            target["events"].extend(source["events"])

        # Handle current state merging
        source_current = cast(Dict[str, Union[List[str], List[Dict[str, Any]]]], source.get("current_state", source))  # Backward compatibility
        target_current = cast(Dict[str, Union[List[str], List[Dict[str, Any]]]], target["current_state"])

        # Status priority: VERIFIED > STAGING > PENDING > UNVERIFIED > FAILED
        status_priority = {
            FeatureStatus.VERIFIED: 5,
            FeatureStatus.STAGING: 4,
            FeatureStatus.PENDING: 3,
            FeatureStatus.PENDING_DUE_TO_GATE_FAIL: 3,
            FeatureStatus.UNVERIFIED: 2,
            FeatureStatus.FAILED: 1
        }
        
        # Track existing features by name
        existing_features: Dict[str, Tuple[FeatureStatus, Optional[Dict[str, Any]]]] = {}
        for status in FeatureStatus:
            status_key = status.value
            if status_key in target_current:
                if status == FeatureStatus.VERIFIED:
                    for name in cast(List[str], target_current[status_key]):
                        existing_features[name] = (status, None)
                else:
                    for item in cast(List[Dict[str, Any]], target_current[status_key]):
                        existing_features[item["name"]] = (status, item)
        
        # Merge status sections with conflict resolution
        for status in FeatureStatus:
            status_key = status.value
            if status_key in source_current:
                if status == FeatureStatus.VERIFIED:
                    # Verified is list of feature names
                    for name in cast(List[str], source_current[status_key]):
                        if name in existing_features:
                            existing_status, _ = existing_features[name]
                            if status_priority[status] > status_priority[existing_status]:
                                # Remove from old status and add to new
                                CoverageValidator._remove_feature_from_status(target_current, name, existing_status)
                                cast(List[str], target_current[status_key]).append(name)
                        else:
                            cast(List[str], target_current[status_key]).append(name)
                elif status == FeatureStatus.STAGING:
                    # STAGING status: Similar to PENDING but for staging evaluation
                    for item in cast(List[Dict[str, Any]], source_current[status_key]):
                        name = item["name"]
                        if name in existing_features:
                            existing_status, _ = existing_features[name]
                            if status_priority[status] > status_priority[existing_status]:
                                # Remove from old status and add to new
                                CoverageValidator._remove_feature_from_status(target_current, name, existing_status)
                                cast(List[Dict[str, Any]], target_current[status_key]).append(item)
                        else:
                            cast(List[Dict[str, Any]], target_current[status_key]).append(item)
                elif status == FeatureStatus.FAILED:
                    # FAILED status: Only add if feature doesn't exist in higher priority statuses
                    # FAILED should not override VERIFIED, STAGING, PENDING, or UNVERIFIED
                    for item in cast(List[Dict[str, Any]], source_current[status_key]):
                        name = item["name"]
                        if name not in existing_features:
                            # New failed feature - add it
                            cast(List[Dict[str, Any]], target_current[status_key]).append(item)
                        else:
                            # Feature exists - check if it's already FAILED, and merge dependency info
                            existing_status, existing_item = existing_features[name]
                            if existing_status == FeatureStatus.FAILED and existing_item is not None:
                                # Merge dependency information
                                if "dependency_chain" in item and "dependency_chain" not in existing_item:
                                    existing_item["dependency_chain"] = item["dependency_chain"]
                                if "blocked_children" in item and "blocked_children" not in existing_item:
                                    existing_item["blocked_children"] = item["blocked_children"]
                                # Merge error details if present
                                if "error" in item and "error" not in existing_item:
                                    existing_item["error"] = item["error"]
                            # If feature exists in higher priority status, skip (don't override)
                else:
                    # PENDING and UNVERIFIED: normal priority-based merging
                    for item in cast(List[Dict[str, Any]], source_current[status_key]):
                        name = item["name"]
                        if name in existing_features:
                            existing_status, _ = existing_features[name]
                            if status_priority[status] > status_priority[existing_status]:
                                # Remove from old status and add to new
                                CoverageValidator._remove_feature_from_status(target_current, name, existing_status)
                                cast(List[Dict[str, Any]], target_current[status_key]).append(item)
                        else:
                            cast(List[Dict[str, Any]], target_current[status_key]).append(item)

        # Update metadata
        target["metadata"]["source_files"].append(source_file)
        if "last_updated" in source.get("metadata", {}):
            source_updated = source["metadata"]["last_updated"]
            if target["metadata"]["last_updated"] is None or source_updated > target["metadata"]["last_updated"]:
                target["metadata"]["last_updated"] = source_updated

    @staticmethod
    def record_event(coverage_data: Dict[str, Any], event_type: str, feature: str,
                    from_status: Optional[str] = None, to_status: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Record an event in the coverage data"""
        event: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "feature": feature
        }

        if from_status is not None:
            event["from_status"] = from_status
        if to_status is not None:
            event["to_status"] = to_status
        if details:
            event["details"] = details

        if "events" not in coverage_data:
            coverage_data["events"] = []

        coverage_data["events"].append(event)

        # Update current_state based on event type
        if "current_state" not in coverage_data:
            coverage_data["current_state"] = {}

        if event_type == "feature_promoted":
            # Update current_state for promotion
            coverage_data["current_state"][feature] = {
                "status": to_status or "unknown",
                "last_updated": event["timestamp"]
            }
        elif event_type == "feature_tested":
            # Update current_state for test results
            if details and details.get("status") == "harmful":
                coverage_data["current_state"][feature] = {
                    "status": "discarded",
                    "last_updated": event["timestamp"]
                }
                # Add to discarded list
                if "discarded" not in coverage_data:
                    coverage_data["discarded"] = []
                coverage_data["discarded"].append({
                    "name": feature,
                    "reason": details.get("reason", "harmful"),
                    "discarded_at": event["timestamp"]
                })
            elif details and details.get("status") == "error":
                coverage_data["current_state"][feature] = {
                    "status": "failed",
                    "last_updated": event["timestamp"]
                }
                # Add to failed list
                if "failed" not in coverage_data:
                    coverage_data["failed"] = []
                coverage_data["failed"].append({
                    "name": feature,
                    "reason": details.get("error", "unknown_error"),
                    "failed_at": event["timestamp"]
                })

        # Update last_updated
        coverage_data["metadata"]["last_updated"] = event["timestamp"]

    @staticmethod
    def archive_coverage_data(coverage_data: Dict[str, Any], archive_dir: str = "ztb/coverage/archive") -> None:
        """Archive coverage data by year"""
        from pathlib import Path

        if "events" not in coverage_data or not coverage_data["events"]:
            return

        # Get current year
        current_year = datetime.now().year
        archive_file = Path(archive_dir) / f"coverage_{current_year}.json"

        # Load existing archive or create new
        archive_data: Dict[str, Any] = {"events": [], "metadata": {"year": current_year}}

        if archive_file.exists():
            try:
                with open(archive_file, 'r', encoding='utf-8') as f:
                    archive_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        # Add current events to archive
        archive_data["events"].extend(coverage_data["events"])

        # Update metadata
        archive_data["metadata"]["last_updated"] = datetime.now().isoformat()
        archive_data["metadata"]["total_events"] = len(archive_data["events"])

        # Save archive
        archive_file.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(archive_data, f, indent=2, ensure_ascii=False)

        # Clear events from current coverage (keep recent events)
        recent_events = [e for e in coverage_data["events"]
                        if datetime.fromisoformat(e["timestamp"]).year == current_year]
        coverage_data["events"] = recent_events

    @staticmethod
    def _remove_feature_from_status(target_current: Dict[str, Union[List[str], List[Dict[str, Any]]]], 
                                   feature_name: str, status: FeatureStatus) -> None:
        """Remove a feature from a specific status section"""
        status_key = status.value
        if status_key not in target_current:
            return
            
        if status == FeatureStatus.VERIFIED:
            # VERIFIED is list of strings
            status_list: List[str] = target_current[status_key]  # type: ignore
            if feature_name in status_list:
                status_list.remove(feature_name)
        else:
            # Other statuses are list of dicts with "name" key
            status_list: List[Dict[str, Any]] = target_current[status_key]  # type: ignore
            status_list[:] = [item for item in status_list if item.get("name") != feature_name]  # type: ignore

    @staticmethod
    def validate_coverage_structure(coverage_data: Dict[str, Any]) -> List[str]:
        """Validate coverage.json structure"""
        errors = []

        # Check required top-level keys
        required_statuses = {status.value for status in FeatureStatus}
        status_keys = set(coverage_data.keys()) - {"metadata", "business_rules"}

        if not status_keys.issubset(required_statuses):
            invalid_keys = status_keys - required_statuses
            errors.append(f"Invalid status keys: {invalid_keys}")

        # Validate each status section
        for status_key, items in coverage_data.items():
            if status_key in ["metadata", "business_rules"]:
                continue

            try:
                status = FeatureStatus(status_key)
                errors.extend(CoverageValidator._validate_status_section(status, items))
            except ValueError:
                errors.append(f"Invalid status: {status_key}")

        return errors

    @staticmethod
    def _validate_status_section(status: FeatureStatus, items: Any) -> List[str]:
        """Validate a status section"""
        errors = []

        if status == FeatureStatus.VERIFIED:
            if not isinstance(items, list):
                errors.append(f"VERIFIED section must be a list, got {type(items)}")
        else:
            if not isinstance(items, list):
                errors.append(f"{status.value} section must be a list, got {type(items)}")
            else:
                for item in items:
                    if not isinstance(item, dict) or "name" not in item or "reason" not in item:
                        errors.append(f"Invalid item in {status.value}: {item}")
                    else:
                        try:
                            reason = FeatureReason(item["reason"])
                            if not validate_status_reason(status, reason):
                                errors.append(f"Invalid reason '{item['reason']}' for status '{status.value}'")
                        except ValueError:
                            errors.append(f"Invalid reason: {item['reason']}")

        return errors


    @staticmethod
    def validate_coverage_comprehensive(coverage_data: Dict[str, Any]) -> List[str]:
        """Comprehensive validation including business rules"""
        errors = CoverageValidator.validate_coverage_structure(coverage_data)
        
        # Business rule validations
        errors.extend(CoverageValidator._validate_business_rules(coverage_data))
        
        return errors

    @staticmethod
    def _validate_business_rules(coverage_data: Dict[str, Any]) -> List[str]:
        """Validate business rules for coverage data"""
        errors = []
        
        # Validate timestamp format in metadata
        if "metadata" in coverage_data:
            metadata = coverage_data["metadata"]
            if "last_updated" in metadata and metadata["last_updated"]:
                last_updated = metadata["last_updated"]
                try:
                    # Validate ISO format timestamp
                    datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                except ValueError:
                    errors.append(f"Invalid timestamp format in metadata.last_updated: {last_updated} (expected ISO format)")
        
        # Check feature name uniqueness across all statuses
        all_features = set()
        duplicate_features = set()
        
        for status in FeatureStatus:
            status_key = status.value
            if status_key in coverage_data:
                if status == FeatureStatus.VERIFIED:
                    for name in coverage_data[status_key]:
                        if name in all_features:
                            duplicate_features.add(name)
                        all_features.add(name)
                else:
                    for item in coverage_data[status_key]:
                        name = item["name"]
                        if name in all_features:
                            duplicate_features.add(name)
                        all_features.add(name)
        
        for dup in duplicate_features:
            errors.append(f"Duplicate feature name across statuses: {dup}")
        
        # Status-specific business rules
        for status in FeatureStatus:
            status_key = status.value
            if status_key not in coverage_data:
                continue
                
            if status == FeatureStatus.VERIFIED:
                # VERIFIED features must have verified_at timestamp
                # Note: This would be checked when features are promoted
                pass
            elif status in [FeatureStatus.PENDING, FeatureStatus.UNVERIFIED, FeatureStatus.FAILED]:
                for item in coverage_data[status_key]:
                    name = item["name"]
                    
                    # Must have reason
                    if "reason" not in item:
                        errors.append(f"Feature '{name}' in {status_key} missing 'reason' field")
                    elif not item["reason"]:
                        errors.append(f"Feature '{name}' in {status_key} has empty 'reason'")
                    
                    # Status-specific validations
                    if status == FeatureStatus.PENDING:
                        if item.get("reason") not in [r.value for r in [FeatureReason.INSUFFICIENT_DATA, FeatureReason.HIGH_NAN_RATE, FeatureReason.ALIGNMENT_MISMATCH]]:
                            errors.append(f"Invalid reason '{item.get('reason')}' for PENDING status on feature '{name}'")
                    elif status == FeatureStatus.UNVERIFIED:
                        if item.get("reason") != FeatureReason.NOT_TESTED.value:
                            errors.append(f"Invalid reason '{item.get('reason')}' for UNVERIFIED status on feature '{name}'")
                    elif status == FeatureStatus.FAILED:
                        if item.get("reason") not in [r.value for r in [FeatureReason.COMPUTATION_ERROR, FeatureReason.TYPE_MISMATCH, FeatureReason.INVALID_RESULT]]:
                            errors.append(f"Invalid reason '{item.get('reason')}' for FAILED status on feature '{name}'")
                        # Must have error field
                        if "error" not in item or not item["error"]:
                            errors.append(f"Feature '{name}' in FAILED status missing 'error' field")
        
        # Additional quality metrics validation (if business_rules section exists)
        if "business_rules" in coverage_data:
            rules = coverage_data["business_rules"]
            
            # Validate minimum series length requirement
            if "min_series_length" in rules:
                _ = rules["min_series_length"]
                # Note: Actual series length validation would require data access
                # This is a placeholder for future implementation
                pass
            
            # Validate maximum skew tolerance
            if "max_skew_tolerance" in rules:
                _ = rules["max_skew_tolerance"]
                # Note: Actual skew validation would require statistical analysis
                # This is a placeholder for future implementation
                pass
        
        return errors


class CoverageReporter:
    """Unified reporting for coverage.json in multiple formats"""

    @staticmethod
    def generate_json_report(coverage_data: Dict[str, Any]) -> str:
        """Generate JSON format report"""
        return json.dumps(coverage_data, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_markdown_report(coverage_data: Dict[str, Any]) -> str:
        """Generate Markdown format report"""
        lines = ["# Feature Coverage Report\n"]

        # Summary
        metadata = coverage_data.get("metadata", {})
        lines.append("## Summary\n")
        lines.append(f"- **Last Updated**: {metadata.get('last_updated', 'N/A')}")
        lines.append(f"- **Verified Features**: {metadata.get('total_verified', 0)}")
        lines.append(f"- **Pending Features**: {metadata.get('total_pending', 0)}")
        lines.append(f"- **Failed Features**: {metadata.get('total_failed', 0)}")
        lines.append(f"- **Unverified Features**: {metadata.get('total_unverified', 0)}\n")

        # Details by status
        for status in FeatureStatus:
            status_key = status.value
            if status_key in coverage_data:
                items = coverage_data[status_key]
                lines.append(f"## {status_key.title()} Features\n")

                if status == FeatureStatus.VERIFIED:
                    for feature in items:
                        lines.append(f"- {feature}")
                else:
                    for item in items:
                        lines.append(f"- **{item['name']}**: {item['reason']}")

                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_text_report(coverage_data: Dict[str, Any]) -> str:
        """Generate plain text format report"""
        lines = ["Feature Coverage Report", "=" * 50, ""]

        metadata = coverage_data.get("metadata", {})
        lines.append(f"Last Updated: {metadata.get('last_updated', 'N/A')}")
        lines.append(f"Verified: {metadata.get('total_verified', 0)}")
        lines.append(f"Pending: {metadata.get('total_pending', 0)}")
        lines.append(f"Failed: {metadata.get('total_failed', 0)}")
        lines.append(f"Unverified: {metadata.get('total_unverified', 0)}")
        lines.append("")

        for status in FeatureStatus:
            status_key = status.value
            if status_key in coverage_data:
                items = coverage_data[status_key]
                lines.append(f"{status_key.title()} Features:")
                lines.append("-" * 30)

                if status == FeatureStatus.VERIFIED:
                    for feature in items:
                        lines.append(f"  - {feature}")
                else:
                    for item in items:
                        lines.append(f"  - {item['name']}: {item['reason']}")

                lines.append("")

        return "\n".join(lines)


class StatusTransitionManager:
    """Manages status transitions for features"""

    # Valid transitions: from -> to
    VALID_TRANSITIONS = {
        FeatureStatus.UNVERIFIED: {FeatureStatus.PENDING, FeatureStatus.VERIFIED, FeatureStatus.FAILED},
        FeatureStatus.PENDING: {FeatureStatus.VERIFIED, FeatureStatus.FAILED},
        FeatureStatus.VERIFIED: {FeatureStatus.FAILED},  # Can be demoted if issues found
        FeatureStatus.FAILED: {FeatureStatus.PENDING, FeatureStatus.VERIFIED},  # Can be fixed and re-tested
    }

    @staticmethod
    def can_transition(from_status: FeatureStatus, to_status: FeatureStatus) -> bool:
        """Check if transition is valid"""
        return to_status in StatusTransitionManager.VALID_TRANSITIONS.get(from_status, set())

    @staticmethod
    def transition_feature(coverage_data: Dict[str, Any], feature_name: str,
                          new_status: FeatureStatus, reason: Optional[FeatureReason] = None) -> bool:
        """Transition a feature to new status"""
        # Find current status
        current_status = None
        current_item = None

        for status in FeatureStatus:
            status_key = status.value
            if status_key in coverage_data:
                items = coverage_data[status_key]

                if status == FeatureStatus.VERIFIED:
                    if feature_name in items:
                        current_status = status
                        current_item = feature_name
                        break
                else:
                    for item in items:
                        if item.get("name") == feature_name:
                            current_status = status  # type: ignore
                            current_item = item
                            break

            if current_status:
                break

        if not current_status:
            return False  # Feature not found

        # Check if transition is valid
        if not StatusTransitionManager.can_transition(current_status, new_status):
            return False

        # Remove from current status
        if current_status == FeatureStatus.VERIFIED:
            coverage_data[current_status.value].remove(feature_name)
        elif current_item is not None:  # type: ignore[unreachable]
            coverage_data[current_status.value].remove(current_item)

        # Add to new status
        new_status_key = new_status.value
        if new_status_key not in coverage_data:
            coverage_data[new_status_key] = []

        if new_status == FeatureStatus.VERIFIED:
            coverage_data[new_status_key].append(feature_name)
        else:
            if reason is None:
                return False
            coverage_data[new_status_key].append({
                "name": feature_name,
                "reason": reason.value
            })

        return True


def validate_coverage_comprehensive(coverage_data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Comprehensive validation including business rules and category requirements.

    Args:
        coverage_data: Coverage data to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Basic structure validation
    structure_errors = CoverageValidator.validate_coverage_structure(coverage_data)
    errors.extend(structure_errors)

    # Business rules validation
    if "business_rules" not in coverage_data:
        errors.append("Missing business_rules section in coverage data")
        return False, errors

    rules = coverage_data["business_rules"]

    # Check minimum requirements
    verified_features = coverage_data.get("verified", [])
    total_verified = len(verified_features)

    if "total_verified_min" in rules:
        min_total = rules["total_verified_min"]
        if total_verified < min_total:
            errors.append(f"Insufficient total verified features: {total_verified}/{min_total}")

    # Category-based validation
    category_counts = count_features_by_category(verified_features)

    category_reqs = {
        "trend_features_min": "trend",
        "oscillator_features_min": "oscillator",
        "volume_features_min": "volume",
        "channel_features_min": "channel"
    }

    for rule_key, category in category_reqs.items():
        if rule_key in rules:
            min_count = rules[rule_key]
            actual_count = category_counts.get(category, 0)
            if actual_count < min_count:
                errors.append(f"Insufficient {category} features: {actual_count}/{min_count}")

    return len(errors) == 0, errors


def validate_status_reason(status: FeatureStatus, reason: Optional[FeatureReason]) -> bool:
    """
    Validate that the reason is appropriate for the given status.

    Args:
        status: Feature status
        reason: Feature reason (optional for VERIFIED)

    Returns:
        True if valid combination, False otherwise
    """
    if status == FeatureStatus.VERIFIED:
        return reason is None

    if reason is None:
        return False

    return reason in STATUS_REASONS[status]


def serialize_status(status: FeatureStatus) -> str:
    """Serialize FeatureStatus to string for JSON output"""
    return status.value


def serialize_reason(reason: FeatureReason) -> str:
    """Serialize FeatureReason to string for JSON output"""
    return reason.value