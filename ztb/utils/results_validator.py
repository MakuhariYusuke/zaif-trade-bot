"""
Results validation utilities for unified schema compliance.

This module provides validation for results from backtesting, training,
and live trading operations against the unified results schema.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    jsonschema = None

from ztb.utils.observability import get_logger

logger = get_logger(__name__)

# Schema paths
SCHEMA_DIR = Path(__file__).parent.parent.parent / "schema"
RESULTS_SCHEMA_PATH = SCHEMA_DIR / "results_schema.json"


class ResultsValidator:
    """Validator for results data against unified schema."""

    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize validator with schema.

        Args:
            schema_path: Path to schema file. Defaults to results_schema.json
        """
        if not HAS_JSONSCHEMA:
            raise ImportError("jsonschema package is required for validation. Install with: pip install jsonschema")

        self.schema_path = schema_path or RESULTS_SCHEMA_PATH
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Load schema from file."""
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Schema file not found: {self.schema_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid schema JSON: {e}")
            return {}

    def validate(self, data: Dict[str, Any], raise_on_error: bool = True) -> bool:
        """Validate data against schema.

        Args:
            data: Data to validate
            raise_on_error: Whether to raise exception on validation failure

        Returns:
            True if valid, False otherwise

        Raises:
            jsonschema.ValidationError: If validation fails and raise_on_error is True
        """
        if not self.schema:
            logger.warning("No schema loaded, skipping validation")
            return True

        try:
            jsonschema.validate(instance=data, schema=self.schema)
            logger.debug("Results validation passed")
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Results validation failed: {e.message}")
            if raise_on_error:
                raise
            return False
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            if raise_on_error:
                raise
            return False

    def validate_file(self, file_path: Union[str, Path], raise_on_error: bool = True) -> bool:
        """Validate JSON file against schema.

        Args:
            file_path: Path to JSON file
            raise_on_error: Whether to raise exception on validation failure

        Returns:
            True if valid, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.validate(data, raise_on_error)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load results file {file_path}: {e}")
            if raise_on_error:
                raise
            return False

    def get_validation_errors(self, data: Dict[str, Any]) -> List[str]:
        """Get list of validation error messages.

        Args:
            data: Data to validate

        Returns:
            List of error messages
        """
        if not self.schema:
            return []

        errors = []
        try:
            jsonschema.validate(instance=data, schema=self.schema)
        except jsonschema.ValidationError as e:
            errors.append(e.message)
            # Add nested errors
            for sub_error in e.context:
                errors.append(f"  {sub_error.message}")
        except Exception as e:
            errors.append(str(e))

        return errors

    def is_valid_structure(self, data: Dict[str, Any]) -> bool:
        """Check if data has basic required structure without full validation.

        Args:
            data: Data to check

        Returns:
            True if basic structure is valid
        """
        if not isinstance(data, dict):
            return False

        # Check for required metadata
        if "metadata" not in data:
            return False

        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            return False

        required_meta_fields = ["version", "timestamp", "run_id", "type"]
        if not all(field in metadata for field in required_meta_fields):
            return False

        return True


# Global validator instance
_default_validator: Optional[ResultsValidator] = None


def get_validator() -> ResultsValidator:
    """Get default results validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = ResultsValidator()
    return _default_validator


def validate_results(data: Dict[str, Any], raise_on_error: bool = True) -> bool:
    """Validate results data using default validator.

    Args:
        data: Results data to validate
        raise_on_error: Whether to raise exception on validation failure

    Returns:
        True if valid
    """
    return get_validator().validate(data, raise_on_error)


def validate_results_file(file_path: Union[str, Path], raise_on_error: bool = True) -> bool:
    """Validate results file using default validator.

    Args:
        file_path: Path to results file
        raise_on_error: Whether to raise exception on validation failure

    Returns:
        True if valid
    """
    return get_validator().validate_file(file_path, raise_on_error)