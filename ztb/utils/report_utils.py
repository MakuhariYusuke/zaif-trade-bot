"""
Deprecated shim for report catalog utilities.

Use ztb.reporting.services.catalog.
"""

from ztb.reporting.services.catalog import (
    extract_action_distribution,
    find_reports_for_model,
    get_latest_report_for_model,
)

__all__ = [
    "extract_action_distribution",
    "find_reports_for_model",
    "get_latest_report_for_model",
]
