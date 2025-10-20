#!/usr/bin/env python3
"""
Training Configuration Update Script

Update common parameters across all training configuration files.
Supports bulk updates for reward_scaling, transaction_cost, max_position_size, etc.

Usage:
    # Update reward_scaling for all configs
    python scripts/update_training_configs.py --reward-scaling 6.0

    # Update multiple parameters
    python scripts/update_training_configs.py --reward-scaling 6.0 --transaction-cost 0.001

    # Dry run (preview changes without applying)
    python scripts/update_training_configs.py --reward-scaling 6.0 --dry-run

    # Target specific configs
    python scripts/update_training_configs.py --reward-scaling 6.0 --pattern "ppo_*.json"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_config_files(pattern: str = "*.json") -> List[Path]:
    """Find all training configuration files.

    Args:
        pattern: Glob pattern for config files

    Returns:
        List of config file paths
    """
    configs_dir = Path("configs/training")
    if not configs_dir.exists():
        print(f"Error: {configs_dir} directory not found")
        sys.exit(1)

    config_files = list(configs_dir.glob(pattern))
    return sorted(config_files)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config_path: Path, config: Dict[str, Any]) -> None:
    """Save JSON configuration file with pretty formatting.

    Args:
        config_path: Path to config file
        config: Configuration dictionary
    """
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")  # Add trailing newline


def update_config(
    config: Dict[str, Any],
    reward_scaling: Optional[float] = None,
    transaction_cost: Optional[float] = None,
    max_position_size: Optional[float] = None,
    learning_rate: Optional[float] = None,
) -> Dict[str, str]:
    """Update configuration parameters.

    Args:
        config: Configuration dictionary to update
        reward_scaling: New reward_scaling value (if specified)
        transaction_cost: New transaction_cost value (if specified)
        max_position_size: New max_position_size value (if specified)
        learning_rate: New learning_rate value (if specified)

    Returns:
        Dictionary of changes made {param: "old_value -> new_value"}
    """
    changes = {}

    if reward_scaling is not None:
        old_value = config.get("reward_scaling", "NOT_SET")
        if old_value != reward_scaling:
            config["reward_scaling"] = reward_scaling
            changes["reward_scaling"] = f"{old_value} -> {reward_scaling}"

    if transaction_cost is not None:
        old_value = config.get("transaction_cost", "NOT_SET")
        if old_value != transaction_cost:
            config["transaction_cost"] = transaction_cost
            changes["transaction_cost"] = f"{old_value} -> {transaction_cost}"

    if max_position_size is not None:
        old_value = config.get("max_position_size", "NOT_SET")
        if old_value != max_position_size:
            config["max_position_size"] = max_position_size
            changes["max_position_size"] = f"{old_value} -> {max_position_size}"

    if learning_rate is not None:
        old_value = config.get("learning_rate", "NOT_SET")
        if old_value != learning_rate:
            config["learning_rate"] = learning_rate
            changes["learning_rate"] = f"{old_value} -> {learning_rate}"

    return changes


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update training configuration parameters across multiple files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--reward-scaling",
        type=float,
        help="Set reward_scaling value (default: 6.0)",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        help="Set transaction_cost value (default: 0.001)",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        help="Set max_position_size value (default: 1.0)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Set learning_rate value",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for config files (default: *.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying",
    )

    args = parser.parse_args()

    # Check if at least one parameter is specified
    if not any(
        [
            args.reward_scaling,
            args.transaction_cost,
            args.max_position_size,
            args.learning_rate,
        ]
    ):
        parser.print_help()
        print("\nError: At least one parameter must be specified")
        sys.exit(1)

    # Find config files
    config_files = find_config_files(args.pattern)
    if not config_files:
        print(f"No config files found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(config_files)} config file(s):")
    for config_file in config_files:
        print(f"  - {config_file.name}")
    print()

    # Update configs
    total_changes = 0
    for config_file in config_files:
        config = load_config(config_file)

        changes = update_config(
            config,
            reward_scaling=args.reward_scaling,
            transaction_cost=args.transaction_cost,
            max_position_size=args.max_position_size,
            learning_rate=args.learning_rate,
        )

        if changes:
            total_changes += len(changes)
            print(f"📝 {config_file.name}:")
            for param, change in changes.items():
                print(f"   {param}: {change}")

            if not args.dry_run:
                save_config(config_file, config)
                print("   ✅ Saved")
            else:
                print("   🔍 (dry run - not saved)")
            print()

    if total_changes == 0:
        print("✨ No changes needed - all configs already have the specified values")
    elif args.dry_run:
        print(f"🔍 Dry run complete - {total_changes} change(s) would be applied")
        print("Run without --dry-run to apply changes")
    else:
        print(f"✅ Complete - {total_changes} change(s) applied")


if __name__ == "__main__":
    main()
