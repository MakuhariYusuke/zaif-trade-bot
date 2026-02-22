"""
Deprecated shim for report catalog utilities.

Use ztb.reporting.services.catalog.
"""

from ztb.reporting.services.catalog import (
    clear_report_cache,
    extract_action_distribution,
    extract_action_distribution_from_payload,
    extract_reward_components,
    extract_reward_components_from_payload,
    find_reports_for_model,
    get_latest_report_for_model,
    get_recent_training_reports,
    list_training_reports,
    load_training_report,
)

__all__ = [
    "clear_report_cache",
    "extract_action_distribution",
    "extract_action_distribution_from_payload",
    "extract_reward_components",
    "extract_reward_components_from_payload",
    "find_reports_for_model",
    "get_latest_report_for_model",
    "get_recent_training_reports",
    "list_training_reports",
    "load_training_report",
]
