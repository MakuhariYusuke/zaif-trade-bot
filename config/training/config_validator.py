"""
Configuration validation module for RL project
設定検証モジュール
"""

import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigValidator:
    """Configuration validator with JSON schema validation"""

    def __init__(self, schema_path: Optional[str] = None):
        """Initialize validator with schema"""
        if schema_path is None:
            schema_path = str((Path(__file__).parent / "config_schema.json").resolve())

        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        try:
            jsonschema.validate(instance=config, schema=self.schema)
            logging.info("Configuration validation passed")
            return True
        except jsonschema.ValidationError as e:
            logging.error(f"Configuration validation failed: {e.message}")
            logging.error(f"Failed at: {' -> '.join(str(x) for x in e.absolute_path)}")
            return False
        except jsonschema.SchemaError as e:
            logging.error(f"Schema error: {e.message}")
            return False

    def validate_config_file(self, config_path: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Validate configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            if self.validate_config(config):
                return True, config
            else:
                return False, None

        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")
            return False, None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in configuration file: {e}")
            return False, None

def load_and_validate_config(config_path: str = "rl_config.json") -> Optional[Dict[str, Any]]:
    """Load and validate configuration file"""
    validator = ConfigValidator()

    # Try different possible paths
    possible_paths = [
        Path(config_path),  # Direct path (e.g., project root or current working directory)
        Path(__file__).parent / config_path,  # Same directory as this script
        Path(__file__).parent.parent / "config" / config_path  # Parent directory's config folder
    ]

    for path in possible_paths:
        if path.exists():
            success, config = validator.validate_config_file(str(path))
            if success:
                return config

    logging.error(
        f"Could not find or validate configuration file: {config_path}. "
        f"Tried paths: {[str(p) for p in possible_paths]}"
    )
    return None