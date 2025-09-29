#!/usr/bin/env python3
"""
Print effective configuration after merging all sources.

Usage: python ztb/ztb/ztb/scripts/print_effective_config.py --config config/example.yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.config.loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Print effective configuration")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    args = parser.parse_args()

    try:
        config = load_config(config_path=args.config)
        config_dict = config.model_dump()

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            print(f"Configuration written to {args.output}")
        else:
            print(json.dumps(config_dict, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
