"""
Long-running operation confirmation utility.
"""

import sys
from typing import Optional


def confirm_long_running_operation(
    operation_name: str,
    estimated_time: str,
    risk_description: Optional[str] = None
) -> bool:
    """
    Confirm before starting a long-running operation.

    Args:
        operation_name: Name of the operation
        estimated_time: Estimated duration (e.g., "2-4 hours")
        risk_description: Optional risk description

    Returns:
        True if user confirms, False otherwise
    """
    print(f"\n⚠️  LONG-RUNNING OPERATION WARNING ⚠️")
    print(f"Operation: {operation_name}")
    print(f"Estimated time: {estimated_time}")

    if risk_description:
        print(f"Risks: {risk_description}")

    print("\nThis operation may:")
    print("- Consume significant CPU/memory resources")
    print("- Take a long time to complete")
    print("- Generate large amounts of log data")
    print("- Require manual intervention if issues occur")

    while True:
        response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            print("✅ Proceeding with operation...")
            return True
        elif response in ['no', 'n']:
            print("❌ Operation cancelled by user.")
            return False
        else:
            print("Please answer 'yes' or 'no'.")


def require_confirmation_for_experiments() -> bool:
    """
    Require confirmation for ML experiments.
    """
    return confirm_long_running_operation(
        operation_name="Machine Learning Experiment",
        estimated_time="Several hours to days",
        risk_description="High memory usage, potential system instability, large disk usage"
    )