#!/usr/bin/env python3
"""Tests for normalization statistics persistence."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from ztb.utils.normalization import (
    NormalizationStats,
    load_scaler,
    save_scaler,
)


class TestNormalizationStats:
    """Tests for NormalizationStats class."""

    @pytest.fixture
    def sample_stats(self) -> NormalizationStats:
        """Create sample normalization stats."""
        return NormalizationStats(
            feature_names=["feature_0", "feature_1", "feature_2"],
            mean=np.array([0.5, 1.0, 1.5]),
            std=np.array([0.1, 0.2, 0.3]),
            n_samples=1000,
        )

    @pytest.fixture
    def fitted_scaler(self) -> tuple[StandardScaler, list[str]]:
        """Create fitted StandardScaler."""
        X = np.random.randn(100, 3)
        scaler = StandardScaler()
        scaler.fit(X)
        feature_names = ["feat_a", "feat_b", "feat_c"]
        return scaler, feature_names

    def test_from_scaler(self, fitted_scaler: tuple[StandardScaler, list[str]]) -> None:
        """Test creating stats from StandardScaler."""
        scaler, feature_names = fitted_scaler
        stats = NormalizationStats.from_scaler(scaler, feature_names)

        assert stats.feature_names == feature_names
        assert len(stats.mean) == 3
        assert len(stats.std) == 3
        assert stats.metadata["normalization_type"] == "StandardScaler"

    def test_compute_hash(self, sample_stats: NormalizationStats) -> None:
        """Test hash computation."""
        hash1 = sample_stats.compute_hash()
        hash2 = sample_stats.compute_hash()

        # Same stats should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

    def test_compute_hash_different_values(self) -> None:
        """Test that different stats produce different hashes."""
        stats1 = NormalizationStats(
            feature_names=["a", "b"],
            mean=np.array([1.0, 2.0]),
            std=np.array([0.5, 0.5]),
        )
        stats2 = NormalizationStats(
            feature_names=["a", "b"],
            mean=np.array([1.0, 2.1]),  # Different mean
            std=np.array([0.5, 0.5]),
        )

        assert stats1.compute_hash() != stats2.compute_hash()

    def test_validate_success(self, sample_stats: NormalizationStats) -> None:
        """Test successful validation."""
        # Same stats
        other = NormalizationStats(
            feature_names=sample_stats.feature_names.copy(),
            mean=sample_stats.mean.copy(),
            std=sample_stats.std.copy(),
        )

        is_valid, errors = sample_stats.validate(other, strict=False)
        assert is_valid
        assert len(errors) == 0

    def test_validate_feature_name_mismatch(
        self, sample_stats: NormalizationStats
    ) -> None:
        """Test validation with feature name mismatch."""
        other = NormalizationStats(
            feature_names=["wrong", "names", "here"],
            mean=sample_stats.mean.copy(),
            std=sample_stats.std.copy(),
        )

        is_valid, errors = sample_stats.validate(other, strict=False)
        assert not is_valid
        assert any("Feature names mismatch" in err for err in errors)

    def test_validate_mean_mismatch(self, sample_stats: NormalizationStats) -> None:
        """Test validation with mean value mismatch."""
        other = NormalizationStats(
            feature_names=sample_stats.feature_names.copy(),
            mean=sample_stats.mean + 0.1,  # Significant difference
            std=sample_stats.std.copy(),
        )

        is_valid, errors = sample_stats.validate(other, strict=False)
        assert not is_valid
        assert any("Mean mismatch" in err for err in errors)

    def test_validate_strict_mode(self, sample_stats: NormalizationStats) -> None:
        """Test strict validation mode (raises on failure)."""
        other = NormalizationStats(
            feature_names=["wrong", "names", "here"],
            mean=sample_stats.mean.copy(),
            std=sample_stats.std.copy(),
        )

        with pytest.raises(ValueError, match="Normalization statistics validation failed"):
            sample_stats.validate(other, strict=True)

    def test_save_and_load(
        self, sample_stats: NormalizationStats, tmp_path: Path
    ) -> None:
        """Test saving and loading stats."""
        stats_path = tmp_path / "scaler.npz"
        sample_stats.save(stats_path)

        assert stats_path.exists()

        # Load
        loaded_stats = NormalizationStats.load(stats_path)

        assert loaded_stats.feature_names == sample_stats.feature_names
        assert np.allclose(loaded_stats.mean, sample_stats.mean)
        assert np.allclose(loaded_stats.std, sample_stats.std)
        assert loaded_stats.n_samples == sample_stats.n_samples
        assert loaded_stats.compute_hash() == sample_stats.compute_hash()

    def test_load_missing_file(self, tmp_path: Path) -> None:
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            NormalizationStats.load(tmp_path / "nonexistent.npz")

    def test_to_scaler(self, fitted_scaler: tuple[StandardScaler, list[str]]) -> None:
        """Test converting stats back to StandardScaler."""
        original_scaler, feature_names = fitted_scaler
        stats = NormalizationStats.from_scaler(original_scaler, feature_names)

        # Convert back
        reconstructed_scaler = stats.to_scaler(strict=True)

        assert np.allclose(reconstructed_scaler.mean_, original_scaler.mean_)
        assert np.allclose(reconstructed_scaler.scale_, original_scaler.scale_)
        assert reconstructed_scaler.n_features_in_ == original_scaler.n_features_in_

    def test_convenience_functions(
        self, sample_stats: NormalizationStats, tmp_path: Path
    ) -> None:
        """Test convenience save/load functions."""
        # Save
        save_scaler(tmp_path, sample_stats)
        assert (tmp_path / "scaler.npz").exists()

        # Load
        loaded_stats = load_scaler(tmp_path, strict=True)
        assert loaded_stats.compute_hash() == sample_stats.compute_hash()

    def test_load_scaler_strict_missing(self, tmp_path: Path) -> None:
        """Test load_scaler with strict=True and missing file."""
        with pytest.raises(FileNotFoundError, match="Normalization stats not found"):
            load_scaler(tmp_path, strict=True)

    def test_load_scaler_non_strict_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test load_scaler with strict=False and missing file."""
        result = load_scaler(tmp_path, strict=False)
        assert result is None

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_post_init_validation(self) -> None:
        """Test __post_init__ validation."""
        # Mismatched lengths should raise
        with pytest.raises(ValueError, match="Feature names count"):
            NormalizationStats(
                feature_names=["a", "b"],
                mean=np.array([1.0, 2.0, 3.0]),  # Wrong length
                std=np.array([0.1, 0.2]),
            )

    def test_hash_integrity_check(
        self, sample_stats: NormalizationStats, tmp_path: Path
    ) -> None:
        """Test that hash integrity is verified on load."""
        stats_path = tmp_path / "scaler.npz"
        sample_stats.save(stats_path)

        # Corrupt the file by modifying mean
        data = np.load(stats_path, allow_pickle=True)
        corrupted_mean = data["mean"] + 1.0

        np.savez_compressed(
            stats_path,
            feature_names=data["feature_names"],
            mean=corrupted_mean,  # Corrupted
            std=data["std"],
            n_samples=data["n_samples"],
            version=data["version"],
            metadata=data["metadata"],
            hash=data["hash"],  # Old hash (won't match)
        )

        # Loading should fail
        with pytest.raises(ValueError, match="hash mismatch"):
            NormalizationStats.load(stats_path)

    def test_numerical_tolerance(self) -> None:
        """Test validation with small numerical differences."""
        stats1 = NormalizationStats(
            feature_names=["a", "b"],
            mean=np.array([1.0, 2.0]),
            std=np.array([0.5, 0.5]),
        )
        stats2 = NormalizationStats(
            feature_names=["a", "b"],
            mean=np.array([1.0 + 1e-9, 2.0]),  # Tiny difference
            std=np.array([0.5, 0.5]),
        )

        # Should pass with default tolerance
        is_valid, _ = stats1.validate(stats2, strict=False)
        assert is_valid

    def test_from_unfitted_scaler(self) -> None:
        """Test creating stats from unfitted scaler."""
        scaler = StandardScaler()
        feature_names = ["a", "b", "c"]

        with pytest.raises(ValueError, match="Scaler not fitted"):
            NormalizationStats.from_scaler(scaler, feature_names)
