"""
Unified Results Schema Validator

Provides validation for results from backtesting, training, and live trading operations
against the unified JSON schema. Used in CI/CD pipelines for result consistency.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema

logger = logging.getLogger(__name__)


class ResultsValidator:
    """Validator for unified results schema."""

    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize validator with schema.

        Args:
            schema_path: Path to schema file. Defaults to schema/results_schema.json
        """
        if schema_path is None:
            schema_path = Path(__file__).parent / "results_schema.json"

        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)

        # Compile validator for performance
        self.validator = jsonschema.Draft7Validator(self.schema)

    def validate(self, results: Dict[str, Any]) -> bool:
        """
        Validate results against schema.

        Args:
            results: Results dictionary to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValidationError: If validation fails and raise_exception=True
        """
        try:
            self.validator.validate(results)
            logger.info("Results validation passed")
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Results validation failed: {e.message}")
            logger.error(f"Failed at path: {e.absolute_path}")
            return False
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return False

    def validate_file(self, results_path: Path) -> bool:
        """
        Validate results from file.

        Args:
            results_path: Path to results JSON file

        Returns:
            True if valid, False otherwise
        """
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return self.validate(results)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load results file {results_path}: {e}")
            return False

    def get_validation_errors(self, results: Dict[str, Any]) -> list:
        """
        Get detailed validation errors.

        Args:
            results: Results dictionary to validate

        Returns:
            List of validation error messages
        """
        errors = []
        for error in self.validator.iter_errors(results):
            errors.append({
                'message': error.message,
                'path': str(error.absolute_path),
                'schema_path': str(error.absolute_schema_path)
            })
        return errors


def validate_results_cli():
    """CLI entry point for result validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate results against unified schema")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--schema", help="Path to schema file (optional)")

    args = parser.parse_args()

    schema_path = Path(args.schema) if args.schema else None
    validator = ResultsValidator(schema_path)

    results_path = Path(args.results_file)
    if validator.validate_file(results_path):
        print("✓ Results validation passed")
        return 0
    else:
        print("✗ Results validation failed")
        return 1


if __name__ == "__main__":
    exit(validate_results_cli())