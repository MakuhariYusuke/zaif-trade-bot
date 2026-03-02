"""
Run Manifest Manager for Training Sessions.

Generates and validates manifest.json files that contain complete metadata
about training runs, including git state, configuration, and data fingerprints.
"""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import cast

from ztb.io.json_io import read_json_object, write_json
from ztb.types.common import ConfigDict
from ztb.utils.git_utils import (
    get_git_dirty_status as _get_git_dirty_status,
    get_git_sha as _get_git_sha,
)
from ztb.utils.safety import ensure_dict

def _as_object_map(value: object) -> dict[str, object]:
    """Safely coerce object to mapping."""
    return ensure_dict(value)

def _as_string_list(value: object) -> list[str]:
    """Safely coerce object to list[str]."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]

def _as_short_text(value: object, limit: int = 16) -> str:
    """Convert value to a short printable text."""
    if isinstance(value, str):
        return value[:limit]
    return "N/A"

def inference_config_to_dict(config: object) -> dict[str, object]:
    """
    Convert InferenceConfig to dictionary for serialization.

    Args:
        config: InferenceConfig instance or dict

    Returns:
        Dictionary representation
    """
    if config is None:
        return {}

    if isinstance(config, dict):
        return {str(k): v for k, v in config.items() if isinstance(k, str)}

    if is_dataclass(config):
        return cast(dict[str, object], asdict(config))

    # Fallback: try to extract attributes
    config_dict = getattr(config, "__dict__", None)
    if not isinstance(config_dict, dict):
        return {}
    return {
        str(k): v
        for k, v in config_dict.items()
        if isinstance(k, str) and not k.startswith("_")
    }

def get_git_sha() -> str:
    """
    Get current git commit SHA.

    Returns:
        Git SHA (40-character hex string) or "unknown" if not in git repo
    """
    return _get_git_sha()

def get_git_dirty_status() -> bool:
    """
    Check if git working directory has uncommitted changes.

    Returns:
        True if there are uncommitted changes, False otherwise
    """
    # Manifest generation is called frequently; tracked changes are sufficient.
    return _get_git_dirty_status(include_untracked=False)

def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex string of SHA256 hash
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def compute_dataset_metadata(dataset_path: Path) -> dict[str, object]:
    """
    Compute dataset metadata for reproducibility.

    Args:
        dataset_path: Path to dataset file (CSV or pickle)

    Returns:
        Dictionary with dataset metadata:
        - sha256: Dataset file SHA256 hash
        - rows: Number of rows
        - time_range: [start_timestamp, end_timestamp] (if timestamp column exists)
        - timezone: Timezone of timestamps (if applicable)
        - missing_ratio: Ratio of missing values
    """
    import pandas as pd

    # Compute file hash
    dataset_sha256 = compute_file_hash(dataset_path)

    # Load dataset to extract metadata
    if dataset_path.suffix == ".csv":
        from ztb.io.data_loader import DataLoader

        df = DataLoader.load_csv_strict(dataset_path)
    elif dataset_path.suffix in [".pkl", ".pickle"]:
        df = pd.read_pickle(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    rows = len(df)

    # Extract time range if timestamp column exists
    time_range = None
    timezone = None
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        time_range = [
            df["timestamp"].min().isoformat(),
            df["timestamp"].max().isoformat(),
        ]
        # Check timezone
        try:
            tz = getattr(df["timestamp"].dtype, "tz", None)
            if tz:
                timezone = str(tz)
        except AttributeError:
            pass

    # Compute missing ratio
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_ratio = float(missing_cells) / max(total_cells, 1)

    return {
        "sha256": dataset_sha256,
        "rows": rows,
        "time_range": time_range,
        "timezone": timezone,
        "missing_ratio": round(missing_ratio, 6),
    }

def generate_manifest(
    model_dir: Path,
    config: ConfigDict,
    feature_names: list[str],
    warmup: int,
    schema_hash: str | None = None,
    scaler_hash: str | None = None,
    fingerprint: str | None = None,
    additional_metadata: dict[str, object] | None = None,
    inference_config: dict[str, object] | None = None,
    dataset_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Generate complete manifest for a training run.

    Args:
        model_dir: Directory where model is saved
        config: Training configuration dictionary
        feature_names: list of feature names used
        warmup: Warmup period used
        schema_hash: Hash of feature schema (optional, will compute if schema file exists)
        scaler_hash: Hash of normalization scaler (optional, will compute if scaler file exists)
        fingerprint: Configuration fingerprint (optional, will compute from config)
        additional_metadata: Additional metadata to include (optional)
        inference_config: Inference configuration (temperature, tau, tiebreaker, etc.) for reproducibility
        dataset_metadata: Dataset metadata (sha256, rows, time_range, timezone, missing_ratio) for reproducibility

    Returns:
        Dictionary with complete manifest
    """
    # Get git information
    git_sha = get_git_sha()
    git_dirty = get_git_dirty_status()

    # Compute hashes if not provided
    if schema_hash is None:
        schema_file = model_dir / "feature_schema.json"
        if schema_file.exists():
            schema_hash = compute_file_hash(schema_file)
        else:
            schema_hash = "not_available"

    if scaler_hash is None:
        scaler_file = model_dir / "normalization_stats.pkl"
        if scaler_file.exists():
            scaler_hash = compute_file_hash(scaler_file)
        else:
            scaler_hash = "not_available"

    # Compute config fingerprint if not provided
    if fingerprint is None:
        # Simple fingerprint: hash of sorted config JSON
        config_str = json.dumps(config, sort_keys=True)
        fingerprint = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    # Build manifest
    manifest = {
        "version": "1.0",
        "git": {
            "sha": git_sha,
            "dirty": git_dirty,
        },
        "hashes": {
            "schema": schema_hash,
            "scaler": scaler_hash,
            "config_fingerprint": fingerprint,
        },
        "training": {
            "config": config,
            "feature_names": feature_names,
            "warmup": warmup,
            "n_features": len(feature_names),
        },
    }

    # Add dataset metadata if provided (for data version control)
    if dataset_metadata:
        manifest["dataset"] = dataset_metadata

    # Add inference config if provided (for reproducibility in sweeps/evaluations)
    if inference_config:
        manifest["inference"] = inference_config

    # Add additional metadata if provided
    if additional_metadata:
        manifest["additional"] = additional_metadata

    return manifest

def save_manifest(manifest: dict[str, object], output_path: Path) -> None:
    """
    Save manifest to JSON file.

    Args:
        manifest: Manifest dictionary
        output_path: Path to save manifest.json
    """
    write_json(output_path, manifest, indent=2)

def load_manifest(manifest_path: Path) -> dict[str, object]:
    """
    Load manifest from JSON file.

    Args:
        manifest_path: Path to manifest.json

    Returns:
        Manifest dictionary

    Raises:
        FileNotFoundError: If manifest file does not exist
        json.JSONDecodeError: If manifest file is invalid JSON
    """
    return read_json_object(manifest_path)

def validate_manifest(manifest: dict[str, object]) -> tuple[bool, list[str]]:
    """
    Validate manifest structure and required fields.

    Args:
        manifest: Manifest dictionary

    Returns:
        tuple of (is_valid, list_of_errors)
    """
    errors: list[str] = []

    # Check version
    if "version" not in manifest:
        errors.append("Missing 'version' field")

    # Check git info
    git_info = _as_object_map(manifest.get("git"))
    if not git_info:
        errors.append("Missing 'git' field")
    else:
        if "sha" not in git_info:
            errors.append("Missing 'git.sha' field")
        if "dirty" not in git_info:
            errors.append("Missing 'git.dirty' field")

    # Check hashes
    hashes = _as_object_map(manifest.get("hashes"))
    if not hashes:
        errors.append("Missing 'hashes' field")
    else:
        required_hashes = ["schema", "scaler", "config_fingerprint"]
        for hash_type in required_hashes:
            if hash_type not in hashes:
                errors.append(f"Missing 'hashes.{hash_type}' field")

    # Check training info
    training = _as_object_map(manifest.get("training"))
    if not training:
        errors.append("Missing 'training' field")
    else:
        required_training = ["config", "feature_names", "warmup", "n_features"]
        for field in required_training:
            if field not in training:
                errors.append(f"Missing 'training.{field}' field")

    is_valid = len(errors) == 0
    return is_valid, errors

def compare_manifests(
    manifest1: dict[str, object],
    manifest2: dict[str, object],
    ignore_git: bool = True,
) -> tuple[bool, list[str]]:
    """
    Compare two manifests for compatibility.

    Args:
        manifest1: First manifest
        manifest2: Second manifest
        ignore_git: If True, ignore git SHA differences (default: True)

    Returns:
        tuple of (are_compatible, list_of_differences)
    """
    differences: list[str] = []
    hashes1 = _as_object_map(manifest1.get("hashes"))
    hashes2 = _as_object_map(manifest2.get("hashes"))

    # Compare hashes
    if hashes1.get("schema") != hashes2.get("schema"):
        differences.append("Schema hash mismatch")

    if hashes1.get("scaler") != hashes2.get("scaler"):
        differences.append("Scaler hash mismatch")

    # Compare feature names
    training1 = _as_object_map(manifest1.get("training"))
    training2 = _as_object_map(manifest2.get("training"))
    features1 = set(_as_string_list(training1.get("feature_names", [])))
    features2 = set(_as_string_list(training2.get("feature_names", [])))

    if features1 != features2:
        differences.append(f"Feature names mismatch: {features1 ^ features2}")

    # Compare warmup
    warmup1 = training1.get("warmup")
    warmup2 = training2.get("warmup")

    if warmup1 != warmup2:
        differences.append(f"Warmup mismatch: {warmup1} vs {warmup2}")

    # Compare git (optional)
    if not ignore_git:
        git1 = _as_object_map(manifest1.get("git")).get("sha")
        git2 = _as_object_map(manifest2.get("git")).get("sha")

        if git1 != git2:
            differences.append(f"Git SHA mismatch: {git1} vs {git2}")

    are_compatible = len(differences) == 0
    return are_compatible, differences

def preflight_dataset_check(
    dataset_path: Path,
    expected_manifest: dict[str, object],
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Preflight check: Verify dataset matches expected manifest.

    Args:
        dataset_path: Path to dataset file
        expected_manifest: Manifest with expected dataset metadata
        strict: If True, fail on any mismatch (default: True)

    Returns:
        tuple of (is_valid, list_of_errors)
    """
    errors: list[str] = []

    # Check if dataset metadata exists in manifest
    expected_dataset = _as_object_map(expected_manifest.get("dataset"))
    if not expected_dataset:
        if strict:
            errors.append("No dataset metadata in manifest (strict mode)")
        return not strict, errors

    # Compute current dataset metadata
    try:
        current_metadata = compute_dataset_metadata(dataset_path)
    except Exception as e:
        errors.append(f"Failed to compute dataset metadata: {e}")
        return False, errors

    # Compare SHA256
    expected_sha = expected_dataset.get("sha256", "")
    current_sha = current_metadata.get("sha256", "")
    if expected_sha != current_sha:
        errors.append(
            f"Dataset SHA256 mismatch: "
            f"expected={_as_short_text(expected_sha)}..., "
            f"actual={_as_short_text(current_sha)}..."
        )

    # Compare row count
    if expected_dataset.get("rows") != current_metadata.get("rows"):
        errors.append(
            f"Dataset row count mismatch: "
            f"expected={expected_dataset.get('rows')}, "
            f"actual={current_metadata.get('rows')}"
        )

    # Compare time range (if exists)
    if expected_dataset.get("time_range") and current_metadata.get("time_range"):
        expected_time_range = expected_dataset.get("time_range")
        current_time_range = current_metadata.get("time_range")
        if expected_time_range != current_time_range:
            errors.append(
                f"Dataset time range mismatch: "
                f"expected={expected_time_range}, "
                f"actual={current_time_range}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors
