#!/usr/bin/env python3
"""
Artifact validation for Zaif Trade Bot.

Validates session artifacts against expectations and schema.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def load_expectations() -> Dict[str, Any]:
    """Load validation expectations."""
    schema_path = Path("schema/artifacts_expectations.json")
    if not schema_path.exists():
        return {
            "required_files": ["summary.json", "metrics.json", "reports/"],
            "optional_files": ["checkpoints/", "logs/", "tensorboard/"],
            "summary_schema": {},
            "metrics_schema": {},
        }

    with open(schema_path, "r") as f:
        return cast(Dict[str, Any], json.load(f))


def validate_file_presence(
    artifacts_dir: Path, expectations: Dict[str, Any]
) -> List[str]:
    """Validate required files exist."""
    errors = []

    for req_file in expectations["required_files"]:
        path = artifacts_dir / req_file
        if req_file.endswith("/"):
            if not path.is_dir():
                errors.append(f"Required directory missing: {req_file}")
        else:
            if not path.is_file():
                errors.append(f"Required file missing: {req_file}")

    return errors


def validate_file_sizes(artifacts_dir: Path) -> List[str]:
    """Validate file sizes are reasonable."""
    warnings = []

    # Check summary.json size
    summary_path = artifacts_dir / "summary.json"
    if summary_path.exists() and summary_path.stat().st_size < 10:
        warnings.append("summary.json is very small (<10 bytes)")

    # Check metrics.json size
    metrics_path = artifacts_dir / "metrics.json"
    if metrics_path.exists() and metrics_path.stat().st_size < 10:
        warnings.append("metrics.json is very small (<10 bytes)")

    # Check checkpoints directory
    checkpoints_dir = artifacts_dir / "checkpoints"
    if checkpoints_dir.exists():
        total_size = sum(
            f.stat().st_size for f in checkpoints_dir.rglob("*") if f.is_file()
        )
        if total_size < 1000:  # Less than 1KB
            warnings.append("checkpoints directory is very small (<1KB)")

    return warnings


def validate_json_schema(file_path: Path, schema: Dict[str, Any]) -> List[str]:
    """Validate JSON file against schema."""
    if not HAS_JSONSCHEMA or not schema:
        return []

    errors = []
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation failed for {file_path.name}: {e.message}")
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {file_path.name}: {e}")
    except Exception as e:
        errors.append(f"Error validating {file_path.name}: {e}")

    return errors


def validate_artifacts(correlation_id: str, strict: bool) -> Dict[str, Any]:
    """Validate artifacts for a session."""
    artifacts_dir = Path("artifacts") / correlation_id
    if not artifacts_dir.exists():
        return {
            "valid": False,
            "errors": [f"Artifacts directory not found: {artifacts_dir}"],
            "warnings": [],
            "correlation_id": correlation_id,
        }

    expectations = load_expectations()

    errors = []
    warnings = []

    # File presence
    errors.extend(validate_file_presence(artifacts_dir, expectations))

    # File sizes
    warnings.extend(validate_file_sizes(artifacts_dir))

    # Schema validation
    summary_path = artifacts_dir / "summary.json"
    if summary_path.exists():
        schema_errors = validate_json_schema(
            summary_path, expectations.get("summary_schema", {})
        )
        if strict:
            errors.extend(schema_errors)
        else:
            warnings.extend(schema_errors)

    metrics_path = artifacts_dir / "metrics.json"
    if metrics_path.exists():
        schema_errors = validate_json_schema(
            metrics_path, expectations.get("metrics_schema", {})
        )
        if strict:
            errors.extend(schema_errors)
        else:
            warnings.extend(schema_errors)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "correlation_id": correlation_id,
        "artifacts_dir": str(artifacts_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate artifacts for Zaif Trade Bot session"
    )
    parser.add_argument(
        "--correlation-id", required=True, help="Session correlation ID"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat schema warnings as errors"
    )

    args = parser.parse_args()

    result = validate_artifacts(args.correlation_id, args.strict)

    # Write report
    reports_dir = Path("artifacts") / args.correlation_id / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "validation_report.json"

    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    if result["valid"]:
        print(f"✓ Artifacts valid for {args.correlation_id}")
    else:
        print(f"✗ Artifacts invalid for {args.correlation_id}")
        for error in result["errors"]:
            print(f"  ERROR: {error}")

    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  WARN: {warning}")

    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
