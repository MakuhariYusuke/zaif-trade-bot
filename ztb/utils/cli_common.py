"""
Common CLI utilities for consistent argument parsing and validation.

Provides standardized argument definitions, help text formatting, and validation
functions across all trading bot CLI tools.
"""

import argparse
import os
from pathlib import Path
from typing import Any, List, Optional


class CLIFormatter:
    """Standardized CLI help text formatter."""

    @staticmethod
    def format_help(
        description: str, default: Any = None, choices: Optional[List[str]] = None
    ) -> str:
        """Format help text with consistent style."""
        help_parts = [description]
        if default is not None:
            help_parts.append(f"(default: {default})")
        if choices:
            help_parts.append(f"(choices: {', '.join(choices)})")
        return " ".join(help_parts)

    @staticmethod
    @staticmethod
    def format_required_help(
        description: str, choices: Optional[List[str]] = None
    ) -> str:
        """Format help text for required arguments."""
        help_parts = [description]
        if choices:
            help_parts.append(f"(choices: {', '.join(choices)})")
        return " ".join(help_parts)


class CLIValidator:
    """Common validation functions for CLI arguments."""

    @staticmethod
    def validate_positive_int(value: str, name: str) -> int:
        """Validate that a string represents a positive integer."""
        try:
            int_val = int(value)
            if int_val <= 0:
                raise ValueError(f"{name} must be positive, got {int_val}")
            return int_val
        except ValueError as e:
            if "positive" in str(e):
                raise
            raise ValueError(f"{name} must be a positive integer, got '{value}'")

    @staticmethod
    def validate_positive_float(value: str, name: str) -> float:
        """Validate that a string represents a positive float."""
        try:
            float_val = float(value)
            if float_val <= 0:
                raise ValueError(f"{name} must be positive, got {float_val}")
            return float_val
        except ValueError as e:
            if "positive" in str(e):
                raise
            raise ValueError(f"{name} must be a positive float, got '{value}'")

    @staticmethod
    def validate_path_exists(value: str, name: str) -> Path:
        """Validate that a path exists."""
        path = Path(value)
        if not path.exists():
            raise ValueError(f"{name} path does not exist: {value}")
        return path

    @staticmethod
    def validate_venue(venue: str) -> str:
        """Validate venue name."""
        supported_venues = ["coincheck"]
        if venue.lower() not in supported_venues:
            raise ValueError(
                f"Unsupported venue: {venue}. Supported: {', '.join(supported_venues)}"
            )
        return venue.lower()


class CommonArgs:
    """Common argument definitions for CLI tools."""

    @staticmethod
    def add_artifacts_dir(
        parser: argparse.ArgumentParser, default: str = "artifacts"
    ) -> None:
        """Add artifacts directory argument."""
        parser.add_argument(
            "--artifacts-dir",
            default=default,
            help=CLIFormatter.format_help("Artifacts directory", default),
        )

    @staticmethod
    def add_correlation_id(
        parser: argparse.ArgumentParser, required: bool = True
    ) -> None:
        """Add correlation ID argument."""
        parser.add_argument(
            "--correlation-id",
            required=required,
            help=(
                CLIFormatter.format_required_help("Session correlation ID")
                if required
                else CLIFormatter.format_help("Session correlation ID")
            ),
        )

    @staticmethod
    def add_venue(parser: argparse.ArgumentParser, default: str = "coincheck") -> None:
        """Add venue argument."""
        parser.add_argument(
            "--venue",
            default=default,
            help=CLIFormatter.format_help("Trading venue", default, ["coincheck"]),
        )

    @staticmethod
    def add_timeout(parser: argparse.ArgumentParser, default: int = 5) -> None:
        """Add timeout argument."""
        parser.add_argument(
            "--timeout",
            type=lambda x: CLIValidator.validate_positive_int(x, "timeout"),
            default=default,
            help=CLIFormatter.format_help("Timeout in seconds", default),
        )

    @staticmethod
    def add_output_dir(
        parser: argparse.ArgumentParser, default: str = "results"
    ) -> None:
        """Add output directory argument."""
        parser.add_argument(
            "--output-dir",
            default=default,
            help=CLIFormatter.format_help("Output directory", default),
        )

    @staticmethod
    def add_verbose(parser: argparse.ArgumentParser) -> None:
        """Add verbose flag."""
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output"
        )

    @staticmethod
    def add_dry_run(parser: argparse.ArgumentParser) -> None:
        """Add dry run flag."""
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be done without making changes",
        )


def create_standard_parser(description: str, **kwargs: Any) -> argparse.ArgumentParser:
    """Create a parser with standard formatting and behavior."""
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        **kwargs,
    )


def get_env_default(env_var: str, default: Any) -> Any:
    """Get value from environment variable or default."""
    return os.getenv(env_var, default)
