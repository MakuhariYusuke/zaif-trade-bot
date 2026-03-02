#!/usr/bin/env python3
"""
Dump configuration JSON schema to schema/config_schema.json
"""

import sys
from pathlib import Path

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.config.schemas.zaif import GlobalConfig
from ztb.io.json_io import write_json

def main() -> None:
    schema = GlobalConfig.model_json_schema()
    output_path = Path("schema/config_schema.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(output_path, schema, indent=2, ensure_ascii=False)

    print(f"Schema dumped to {output_path}")

if __name__ == "__main__":
    main()
