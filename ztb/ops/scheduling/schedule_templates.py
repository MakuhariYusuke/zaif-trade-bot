#!/usr/bin/env python3
"""
schedule_templates.py - Generate JSON templates for cronish.py presets

This script generates cross-platform schedule templates in JSON format
that can be used as presets for cronish.py scheduling.

Usage:
    python schedule_templates.py --template daily --output templates/daily.json
    python schedule_templates.py --list
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def get_template(name: str) -> dict[str, Any]:
    """Get the schedule template for the given name."""
    templates = {
        "daily": {
            "description": "Run daily at 9:00 AM",
            "windows": "0 9 * * *",
            "linux": "0 9 * * *",
            "timezone": "Asia/Tokyo",
        },
        "hourly": {
            "description": "Run every hour at minute 0",
            "windows": "0 * * * *",
            "linux": "0 * * * *",
            "timezone": "Asia/Tokyo",
        },
        "weekly": {
            "description": "Run weekly on Monday at 9:00 AM",
            "windows": "0 9 * * 1",
            "linux": "0 9 * * 1",
            "timezone": "Asia/Tokyo",
        },
        "monthly": {
            "description": "Run monthly on the 1st at 9:00 AM",
            "windows": "0 9 1 * *",
            "linux": "0 9 1 * *",
            "timezone": "Asia/Tokyo",
        },
        "training-daily": {
            "description": "Daily training run at 2:00 AM",
            "windows": "0 2 * * *",
            "linux": "0 2 * * *",
            "timezone": "Asia/Tokyo",
        },
        "backup-hourly": {
            "description": "Hourly backup at minute 30",
            "windows": "30 * * * *",
            "linux": "30 * * * *",
            "timezone": "Asia/Tokyo",
        },
    }

    if name not in templates:
        raise ValueError(f"Unknown template: {name}")

    return templates[name]


def list_templates() -> None:
    """List all available templates."""
    templates = [
        "daily",
        "hourly",
        "weekly",
        "monthly",
        "training-daily",
        "backup-hourly",
    ]
    print("Available templates:")
    for template in templates:
        desc = get_template(template)["description"]
        print(f"  {template}: {desc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cronish.py schedule templates"
    )
    parser.add_argument("--template", "-t", help="Template name to generate")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available templates"
    )

    args = parser.parse_args()

    if args.list:
        list_templates()
        return

    if not args.template:
        parser.error("--template is required unless --list is used")

    try:
        template_data = get_template(args.template)
        output_data = {"template": args.template, "schedule": template_data}

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Template written to {output_path}")
        else:
            print(json.dumps(output_data, indent=2, ensure_ascii=False))

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
