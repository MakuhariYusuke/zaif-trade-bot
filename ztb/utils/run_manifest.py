"""
Run Manifest Manager for Training Sessions.

Generates and validates manifest.json files that contain complete metadata
about training runs, including git state, configuration, and data fingerprints.
"""

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast


def inference_config_to_dict(config: Any) -> Dict[str, Any]:
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
        return config

    if is_dataclass(config):
        return asdict(cast(Any, config))

    # Fallback: try to extract attributes
    return {k: v for k, v in config.__dict__.items() if not k.startswith("_")}


def get_git_sha() -> str:
    """
    Get current git commit SHA.

    Returns:
        Git SHA (40-character hex string) or "unknown" if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_dirty_status() -> bool:
    """
    Check if git working directory has uncommitted changes.

    Returns:
        True if there are uncommitted changes, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


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


def compute_dataset_metadata(dataset_path: Path) -> Dict[str, Any]:
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
        df = pd.read_csv(dataset_path)
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
    config: Dict[str, Any],
    feature_names: List[str],
    warmup: int,
    schema_hash: Optional[str] = None,
    scaler_hash: Optional[str] = None,
    fingerprint: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    inference_config: Optional[Dict[str, Any]] = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate complete manifest for a training run.

    Args:
        model_dir: Directory where model is saved
        config: Training configuration dictionary
        feature_names: List of feature names used
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


def save_manifest(manifest: Dict[str, Any], output_path: Path) -> None:
    """
    Save manifest to JSON file.

    Args:
        manifest: Manifest dictionary
        output_path: Path to save manifest.json
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
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
    with open(manifest_path, "r") as f:
        return cast(Dict[str, Any], json.load(f))


def validate_manifest(manifest: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate manifest structure and required fields.

    Args:
        manifest: Manifest dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check version
    if "version" not in manifest:
        errors.append("Missing 'version' field")

    # Check git info
    if "git" not in manifest:
        errors.append("Missing 'git' field")
    else:
        if "sha" not in manifest["git"]:
            errors.append("Missing 'git.sha' field")
        if "dirty" not in manifest["git"]:
            errors.append("Missing 'git.dirty' field")

    # Check hashes
    if "hashes" not in manifest:
        errors.append("Missing 'hashes' field")
    else:
        required_hashes = ["schema", "scaler", "config_fingerprint"]
        for hash_type in required_hashes:
            if hash_type not in manifest["hashes"]:
                errors.append(f"Missing 'hashes.{hash_type}' field")

    # Check training info
    if "training" not in manifest:
        errors.append("Missing 'training' field")
    else:
        required_training = ["config", "feature_names", "warmup", "n_features"]
        for field in required_training:
            if field not in manifest["training"]:
                errors.append(f"Missing 'training.{field}' field")

    is_valid = len(errors) == 0
    return is_valid, errors


def compare_manifests(
    manifest1: Dict[str, Any],
    manifest2: Dict[str, Any],
    ignore_git: bool = True,
) -> tuple[bool, List[str]]:
    """
    Compare two manifests for compatibility.

    Args:
        manifest1: First manifest
        manifest2: Second manifest
        ignore_git: If True, ignore git SHA differences (default: True)

    Returns:
        Tuple of (are_compatible, list_of_differences)
    """
    differences = []

    # Compare hashes
    if manifest1.get("hashes", {}).get("schema") != manifest2.get("hashes", {}).get(
        "schema"
    ):
        differences.append("Schema hash mismatch")

    if manifest1.get("hashes", {}).get("scaler") != manifest2.get("hashes", {}).get(
        "scaler"
    ):
        differences.append("Scaler hash mismatch")

    # Compare feature names
    features1 = set(manifest1.get("training", {}).get("feature_names", []))
    features2 = set(manifest2.get("training", {}).get("feature_names", []))

    if features1 != features2:
        differences.append(f"Feature names mismatch: {features1 ^ features2}")

    # Compare warmup
    warmup1 = manifest1.get("training", {}).get("warmup")
    warmup2 = manifest2.get("training", {}).get("warmup")

    if warmup1 != warmup2:
        differences.append(f"Warmup mismatch: {warmup1} vs {warmup2}")

    # Compare git (optional)
    if not ignore_git:
        git1 = manifest1.get("git", {}).get("sha")
        git2 = manifest2.get("git", {}).get("sha")

        if git1 != git2:
            differences.append(f"Git SHA mismatch: {git1} vs {git2}")

    are_compatible = len(differences) == 0
    return are_compatible, differences


def preflight_dataset_check(
    dataset_path: Path,
    expected_manifest: Dict[str, Any],
    strict: bool = True,
) -> tuple[bool, List[str]]:
    """
    Preflight check: Verify dataset matches expected manifest.

    Args:
        dataset_path: Path to dataset file
        expected_manifest: Manifest with expected dataset metadata
        strict: If True, fail on any mismatch (default: True)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check if dataset metadata exists in manifest
    if "dataset" not in expected_manifest:
        if strict:
            errors.append("No dataset metadata in manifest (strict mode)")
        return not strict, errors

    expected_dataset = expected_manifest["dataset"]

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
            f"expected={expected_sha[:16] if expected_sha else 'N/A'}..., "
            f"actual={current_sha[:16] if current_sha else 'N/A'}..."
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
        if expected_dataset["time_range"] != current_metadata["time_range"]:
            errors.append(
                f"Dataset time range mismatch: "
                f"expected={expected_dataset['time_range']}, "
                f"actual={current_metadata['time_range']}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors
