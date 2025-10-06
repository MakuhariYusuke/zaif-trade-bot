"""
Run Metadata Manager for Training Sessions.

Generates and validates manifest.json files that contain complete metadata
about training runs, including git state, configuration, and data fingerprints.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib


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


def generate_manifest(
    model_dir: Path,
    config: Dict[str, Any],
    feature_names: List[str],
    warmup: int,
    schema_hash: Optional[str] = None,
    scaler_hash: Optional[str] = None,
    fingerprint: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
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
        return json.load(f)


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
    if manifest1.get("hashes", {}).get("schema") != manifest2.get("hashes", {}).get("schema"):
        differences.append("Schema hash mismatch")
    
    if manifest1.get("hashes", {}).get("scaler") != manifest2.get("hashes", {}).get("scaler"):
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
