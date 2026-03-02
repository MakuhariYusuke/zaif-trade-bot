#!/usr/bin/env python3
"""
Normalization Statistics Persistence for Training/Evaluation Consistency.

Saves and loads VecNormalize/Scaler statistics to ensure training and evaluation
use identical normalization parameters.

Critical Requirements:
1. Save normalization stats during training: mean, std, feature_order, version
2. Load and validate during evaluation: FAIL immediately on any mismatch
3. SHA256 hash for quick integrity verification
4. Supports both VecNormalize (SB3) and StandardScaler (sklearn)

Usage:
    # During training
    stats = NormalizationStats.from_vec_normalize(vec_env)
    # OR
    stats = NormalizationStats.from_scaler(scaler, feature_names)
    stats.save(model_dir / "scaler.npz")

    # During evaluation
    stats = NormalizationStats.load(model_dir / "scaler.npz")
    stats.apply_to_vec_normalize(vec_env, strict=True)
    # OR
    scaler = stats.to_scaler(strict=True)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from ztb.analysis.common.types import NormalizerProtocol

import numpy as np
from numpy.typing import NDArray

@dataclass
class NormalizationStats(NormalizerProtocol):
    """Normalization statistics with validation capabilities."""

    feature_names: list[str]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    n_samples: int = 0
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate data consistency after initialization."""
        if len(self.feature_names) != len(self.mean):
            raise ValueError(
                f"Feature names count ({len(self.feature_names)}) != "
                f"mean count ({len(self.mean)})"
            )
        if len(self.mean) != len(self.std):
            raise ValueError(
                f"Mean count ({len(self.mean)}) != std count ({len(self.std)})"
            )

    @classmethod
    def from_vec_normalize(
        cls,
        vec_env: Any,
        feature_names: list[str] | None = None,
    ) -> NormalizationStats:
        """
        Create normalization stats from VecNormalize environment.

        Args:
            vec_env: VecNormalize environment from SB3
            feature_names: Feature names (auto-detect if None)

        Returns:
            NormalizationStats instance
        """
        # Extract running mean/std from VecNormalize
        if not hasattr(vec_env, "obs_rms"):
            raise ValueError("Environment does not have obs_rms (not a VecNormalize?)")

        obs_rms = vec_env.obs_rms
        mean = obs_rms.mean.copy()
        var = obs_rms.var.copy()
        std = np.sqrt(var)
        n_samples = int(obs_rms.count) if hasattr(obs_rms, "count") else 0

        # Auto-detect feature names if not provided
        if feature_names is None:
            n_features = len(mean)
            feature_names = [f"feature_{i}" for i in range(n_features)]

        metadata = {
            "normalization_type": "VecNormalize",
            "clip_obs": getattr(vec_env, "clip_obs", 10.0),
            "clip_reward": getattr(vec_env, "clip_reward", 10.0),
            "normalize_obs": getattr(vec_env, "norm_obs", True),
            "normalize_reward": getattr(vec_env, "norm_reward", True),
        }

        return cls(
            feature_names=feature_names,
            mean=mean,
            std=std,
            n_samples=n_samples,
            metadata=metadata,
        )

    @classmethod
    def from_scaler(
        cls,
        scaler: Any,
        feature_names: list[str],
    ) -> NormalizationStats:
        """
        Create normalization stats from sklearn StandardScaler.

        Args:
            scaler: Fitted StandardScaler instance
            feature_names: Feature names

        Returns:
            NormalizationStats instance
        """
        if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
            raise ValueError("Scaler not fitted or missing mean_/scale_ attributes")

        mean = scaler.mean_.copy()
        std = scaler.scale_.copy()
        n_samples = (
            int(scaler.n_samples_seen_) if hasattr(scaler, "n_samples_seen_") else 0
        )

        metadata = {
            "normalization_type": "StandardScaler",
            "with_mean": getattr(scaler, "with_mean", True),
            "with_std": getattr(scaler, "with_std", True),
        }

        return cls(
            feature_names=feature_names,
            mean=mean,
            std=std,
            n_samples=n_samples,
            metadata=metadata,
        )

    def compute_hash(self) -> str:
        """
        Compute SHA256 hash of normalization statistics.

        Returns:
            SHA256 hex digest
        """
        # Hash feature order and statistics
        data_to_hash = {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        data_str = json.dumps(data_to_hash, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def validate(
        self,
        other: NormalizationStats,
        strict: bool = True,
        tolerance: float = 1e-6,
    ) -> tuple[bool, list[str]]:
        """
        Validate normalization stats against another instance.

        Args:
            other: Another NormalizationStats to compare
            strict: If True, raise on mismatch; if False, return errors
            tolerance: Numerical tolerance for float comparisons

        Returns:
            (is_valid, error_messages)
        """
        errors: list[str] = []

        # Check feature names
        if self.feature_names != other.feature_names:
            errors.append(
                f"Feature names mismatch: {len(self.feature_names)} features vs "
                f"{len(other.feature_names)} features"
            )
            if len(self.feature_names) == len(other.feature_names):
                diffs = [
                    (i, a, b)
                    for i, (a, b) in enumerate(
                        zip(self.feature_names, other.feature_names)
                    )
                    if a != b
                ]
                if diffs:
                    errors.append(f"First 5 name differences: {diffs[:5]}")

        # Check mean values
        if not np.allclose(self.mean, other.mean, rtol=tolerance, atol=tolerance):
            max_diff = np.max(np.abs(self.mean - other.mean))
            errors.append(
                f"Mean mismatch: max absolute difference = {max_diff:.6e} "
                f"(tolerance = {tolerance:.6e})"
            )

        # Check std values
        if not np.allclose(self.std, other.std, rtol=tolerance, atol=tolerance):
            max_diff = np.max(np.abs(self.std - other.std))
            errors.append(
                f"Std mismatch: max absolute difference = {max_diff:.6e} "
                f"(tolerance = {tolerance:.6e})"
            )

        is_valid = len(errors) == 0

        if strict and not is_valid:
            error_msg = "Normalization statistics validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            raise ValueError(error_msg)

        return is_valid, errors

    def save(self, path: Path) -> None:
        """
        Save normalization stats to .npz file.

        Args:
            path: Output file path (e.g., model_dir/scaler.npz)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save arrays and metadata
        np.savez_compressed(
            path,
            feature_names=np.array(self.feature_names, dtype=object),
            mean=self.mean,
            std=self.std,
            n_samples=np.array([self.n_samples]),
            version=np.array([self.version], dtype=object),
            metadata=np.array([json.dumps(self.metadata)], dtype=object),
            hash=np.array([self.compute_hash()], dtype=object),
        )

        print(
            f"Normalization stats saved to {path} "
            f"({len(self.feature_names)} features, hash: {self.compute_hash()[:16]}...)"
        )

    @classmethod
    def load(cls, path: Path) -> NormalizationStats:
        """
        Load normalization stats from .npz file.

        Args:
            path: Input file path

        Returns:
            NormalizationStats instance

        Raises:
            FileNotFoundError: If stats file does not exist
            ValueError: If hash verification fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {path}")

        data = np.load(path, allow_pickle=True)

        feature_names = data["feature_names"].tolist()
        mean = data["mean"]
        std = data["std"]
        n_samples = int(data["n_samples"][0])
        version = str(data["version"][0])
        metadata = json.loads(str(data["metadata"][0]))
        saved_hash = str(data["hash"][0])

        stats = cls(
            feature_names=feature_names,
            mean=mean,
            std=std,
            n_samples=n_samples,
            version=version,
            metadata=metadata,
        )

        # Verify hash integrity
        computed_hash = stats.compute_hash()
        if saved_hash != computed_hash:
            raise ValueError(
                f"Normalization stats hash mismatch: saved {saved_hash[:16]}..., "
                f"computed {computed_hash[:16]}..."
            )

        print(
            f"Normalization stats loaded from {path} "
            f"({len(feature_names)} features, hash: {computed_hash[:16]}...)"
        )
        return stats

    def apply_to_vec_normalize(
        self,
        vec_env: Any,
        strict: bool = True,
    ) -> None:
        """
        Apply normalization stats to VecNormalize environment.

        Args:
            vec_env: VecNormalize environment to update
            strict: If True, raise on validation failure

        Raises:
            ValueError: If strict=True and current stats don't match
        """
        # Get current stats
        current_stats = NormalizationStats.from_vec_normalize(
            vec_env, self.feature_names
        )

        # Validate
        is_valid, errors = self.validate(current_stats, strict=False)

        if not is_valid:
            if strict:
                error_msg = (
                    "VecNormalize stats mismatch with saved stats:\n"
                    + "\n".join(f"  - {err}" for err in errors)
                )
                raise ValueError(error_msg)
            else:
                print(f"WARNING: Normalization stats mismatch:\n{errors}")

        # Apply saved stats
        vec_env.obs_rms.mean = self.mean.copy()
        vec_env.obs_rms.var = (self.std**2).copy()
        if hasattr(vec_env.obs_rms, "count"):
            vec_env.obs_rms.count = self.n_samples

        print("Applied normalization stats to VecNormalize environment")

    def to_scaler(self, strict: bool = True) -> Any:
        """
        Convert to sklearn StandardScaler.

        Args:
            strict: If True, verify scaler type in metadata

        Returns:
            Fitted StandardScaler instance
        """
        from sklearn.preprocessing import StandardScaler

        if strict and self.metadata.get("normalization_type") != "StandardScaler":
            raise ValueError(
                f"Stats were created from {self.metadata.get('normalization_type')}, "
                "not StandardScaler"
            )

        scaler = StandardScaler()
        scaler.mean_ = self.mean.copy()
        scaler.scale_ = self.std.copy()
        scaler.var_ = (self.std**2).copy()
        scaler.n_features_in_ = len(self.feature_names)
        scaler.n_samples_seen_ = self.n_samples

        # Restore settings
        scaler.with_mean = self.metadata.get("with_mean", True)
        scaler.with_std = self.metadata.get("with_std", True)

        return scaler

def save_scaler(
    model_dir: Path,
    stats: NormalizationStats,
) -> None:
    """
    Convenience function: Save normalization stats to model directory.

    Args:
        model_dir: Model directory path
        stats: NormalizationStats instance
    """
    scaler_path = model_dir / "scaler.npz"
    stats.save(scaler_path)

def load_scaler(
    model_dir: Path,
    strict: bool = True,
) -> NormalizationStats:
    """
    Convenience function: Load normalization stats from model directory.

    Args:
        model_dir: Model directory path
        strict: If True, raise on file not found or validation failure

    Returns:
        NormalizationStats instance

    Raises:
        FileNotFoundError: If strict=True and scaler file missing
        ValueError: If hash verification fails
    """
    scaler_path = model_dir / "scaler.npz"

    if not scaler_path.exists():
        if strict:
            raise FileNotFoundError(
                f"Normalization stats not found: {scaler_path}. "
                "Please retrain model with normalization stats persistence enabled."
            )
        else:
            print(f"WARNING: Normalization stats not found: {scaler_path}")
            return None  # type: ignore

    return NormalizationStats.load(scaler_path)

# Protocol implementation methods
def fit(self, data: Any) -> None:
    """Fit normalizer to data (not implemented for stats-only class)."""
    raise NotImplementedError("NormalizationStats is read-only; use from_scaler() or from_vec_normalize() to create")

def transform(self, data: Any) -> Any:
    """Transform data using stored statistics."""
    scaler = self.to_scaler()
    return scaler.transform(data)

def inverse_transform(self, data: Any) -> Any:
    """Inverse transform data using stored statistics."""
    scaler = self.to_scaler()
    return scaler.inverse_transform(data)
