#!/usr/bin/env python3
"""
Dump configuration JSON schema to schema/config_schema.json
"""

import json
import sys
from pathlib import Path

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.config.schema import GlobalConfig


def main():
    schema = GlobalConfig.model_json_schema()
    output_path = Path("schema/config_schema.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"Schema dumped to {output_path}")


if __name__ == "__main__":
    main()
