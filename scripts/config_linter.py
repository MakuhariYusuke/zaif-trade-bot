#!/usr/bin/env python3
"""
Configuration linter for trading bot.

Validates configuration files against schemas and performs go/no-go checks.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import jsonschema
import requests
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    go_nogo: bool


class ConfigLinter:
    """Configuration validation and linting."""

    def __init__(self, schema_dir: Path, config_dir: Path):
        self.schema_dir = schema_dir
        self.config_dir = config_dir
        self.schemas: Dict[str, Dict[str, Any]] = {}

        # Load schemas
        self._load_schemas()

    def _load_schemas(self):
        """Load JSON schemas from schema directory."""
        for schema_file in self.schema_dir.glob("*.json"):
            with open(schema_file, 'r', encoding='utf-8') as f:
                self.schemas[schema_file.stem] = json.load(f)

    def validate_config(self, config_path: Path, schema_name: str,
                       strict_mode: bool = False, connectivity_skip: bool = False) -> ValidationResult:
        """
        Validate a configuration file against a schema.

        Args:
            config_path: Path to configuration file
            schema_name: Name of schema to validate against
            strict_mode: Whether to treat warnings as errors

        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []

        # Load config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            errors.append(f"Failed to load config {config_path}: {e}")
            return ValidationResult(False, errors, warnings, False)

        # Get schema
        schema = self.schemas.get(schema_name)
        if not schema:
            errors.append(f"Schema '{schema_name}' not found")
            return ValidationResult(False, errors, warnings, False)

        # Validate against schema
        try:
            jsonschema.validate(config, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
            return ValidationResult(False, errors, warnings, False)
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
            return ValidationResult(False, errors, warnings, False)

        # Additional venue-specific validations
        if schema_name == "venue_schema":
            venue_warnings = self._validate_venue_config(config, connectivity_skip)
            warnings.extend(venue_warnings)

        # Perform go/no-go check
        go_nogo = self._go_nogo_check(config, errors, warnings, connectivity_skip)

        is_valid = len(errors) == 0 and (not strict_mode or len(warnings) == 0)

        return ValidationResult(is_valid, errors, warnings, go_nogo)

    def _validate_venue_config(self, config: Dict[str, Any], connectivity_skip: bool = False) -> List[str]:
        """Additional validation for venue configurations."""
        warnings = []

        venue = config.get('venue', {})

        # Check API connectivity if enabled and not skipped
        if not connectivity_skip and config.get('validation', {}).get('check_connectivity', True):
            try:
                response = requests.get(venue['api_url'], timeout=10)
                if response.status_code != 200:
                    warnings.append(f"API endpoint {venue['api_url']} returned status {response.status_code}")
            except requests.RequestException as e:
                warnings.append(f"Failed to connect to API {venue['api_url']}: {e}")

        # Check for weak API keys (placeholder check)
        if len(venue.get('api_key', '')) < 20:
            warnings.append("API key appears to be too short (< 20 characters)")

        # Check fee configuration
        fees = venue.get('fees', {})
        if fees.get('maker_fee_percent', 0) > fees.get('taker_fee_percent', 0):
            warnings.append("Maker fee is higher than taker fee - unusual configuration")

        return warnings

    def _go_nogo_check(self, config: Dict[str, Any], errors: List[str],
                      warnings: List[str], connectivity_skip: bool = False) -> bool:
        """
        Perform go/no-go assessment based on validation results.

        Go criteria:
        - No schema validation errors
        - API connectivity works (if checked)
        - No critical configuration issues

        No-go criteria:
        - Schema validation failures
        - API connectivity failures
        - Missing required credentials
        """
        # No-go if there are any errors
        if errors:
            return False

        venue = config.get('venue', {})

        # No-go if missing API credentials
        if not venue.get('api_key') or not venue.get('api_secret'):
            return False

        # No-go if API connectivity failed and checking is enabled and not skipped
        if not connectivity_skip and config.get('validation', {}).get('check_connectivity', True):
            for warning in warnings:
                if "Failed to connect to API" in warning:
                    return False

        # Go if all checks pass
        return True

    def lint_configs(self, config_files: List[Path], strict_mode: bool = False, connectivity_skip: bool = False) -> Dict[str, ValidationResult]:
        """
        Lint multiple configuration files.

        Args:
            config_files: List of config files to validate
            strict_mode: Whether to treat warnings as errors

        Returns:
            Dictionary mapping config paths to validation results
        """
        results = {}

        for config_file in config_files:
            # Determine schema based on filename
            if 'venue' in config_file.name:
                schema_name = 'venue_schema'
            else:
                # Skip files without known schemas
                continue

            result = self.validate_config(config_file, schema_name, strict_mode, connectivity_skip)
            results[str(config_file)] = result

        return results


def main():
    """CLI entry point for config linting."""
    parser = argparse.ArgumentParser(description='Lint trading bot configurations')
    parser.add_argument('--config-dir', type=Path, default=Path('config'),
                       help='Configuration directory')
    parser.add_argument('--schema-dir', type=Path, default=Path('config'),
                       help='Schema directory')
    parser.add_argument('--files', nargs='*', type=Path,
                       help='Specific config files to validate')
    parser.add_argument('--strict', action='store_true',
                       help='Treat warnings as errors')
    parser.add_argument('--connectivity-skip', action='store_true',
                       help='Skip API connectivity checks')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')

    args = parser.parse_args()

    # Default to skip connectivity in CI
    import os
    if os.getenv('CI') and not args.connectivity_skip:
        args.connectivity_skip = True

    # Initialize linter
    linter = ConfigLinter(args.schema_dir, args.config_dir)

    # Determine files to validate
    if args.files:
        config_files = args.files
    else:
        # Find all JSON config files
        config_files = list(args.config_dir.glob("*.json"))

    # Validate configs
    results = linter.lint_configs(config_files, args.strict, args.connectivity_skip)

    # Output results
    if args.format == 'json':
        output = {}
        for path, result in results.items():
            output[path] = {
                'valid': result.is_valid,
                'go_nogo': result.go_nogo,
                'errors': result.errors,
                'warnings': result.warnings
            }
        print(json.dumps(output, indent=2))
    else:
        # Text output
        all_valid = True
        all_go = True

        for path, result in results.items():
            print(f"\nValidating {path}:")
            print(f"  Valid: {'✓' if result.is_valid else '✗'}")
            print(f"  Go/No-Go: {'GO' if result.go_nogo else 'NO-GO'}")

            if result.errors:
                print("  Errors:")
                for error in result.errors:
                    print(f"    - {error}")

            if result.warnings:
                print("  Warnings:")
                for warning in result.warnings:
                    print(f"    - {warning}")

            all_valid &= result.is_valid
            all_go &= result.go_nogo

        print(f"\nSummary:")
        print(f"  All configs valid: {'✓' if all_valid else '✗'}")
        print(f"  Go for deployment: {'GO' if all_go else 'NO-GO'}")

    # Exit with appropriate code
    if not all(result.go_nogo for result in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()